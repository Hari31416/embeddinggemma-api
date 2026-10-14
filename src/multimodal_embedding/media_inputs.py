"""Resolve request-local media and fetch HTTPS resources without DNS rebinding."""

import asyncio
import base64
import binascii
import http.client
import ipaddress
import mimetypes
import socket
import ssl
import time
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

from starlette.datastructures import UploadFile

from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import InvalidInputError, PayloadTooLargeError
from multimodal_embedding.schemas import MediaPart
from multimodal_embedding.storage import StorageManager, UploadRecord


def public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv6Address):
        if ip not in ipaddress.ip_network("2000::/3") or ip.sixtofour or ip.teredo:
            return False
    return ip.is_global and not ip.is_multicast


async def download_url(url: str, cfg: Settings) -> tuple[bytes, str, str]:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, 443)
            or "\\" in url
            or "%" in host
        ):
            raise ValueError
        host = host.encode("idna").decode("ascii").lower()
    except (ValueError, UnicodeError):
        raise InvalidInputError(
            "file_url must be an HTTPS URL on port 443 without credentials or fragments."
        ) from None
    allowed = {value.lower().rstrip(".") for value in cfg.url_allowed_hosts}
    if allowed and host.rstrip(".") not in allowed:
        raise InvalidInputError("file_url hostname is not allowed.")

    limit = min(cfg.url_download_max_bytes, cfg.max_upload_bytes)
    deadline = time.monotonic() + cfg.url_download_timeout_seconds

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError
        return seconds

    def fetch(address: str) -> tuple[bytes, str, str]:
        conn = http.client.HTTPSConnection(
            host, timeout=remaining(), context=ssl.create_default_context()
        )
        # Pin TCP to the validated address, retaining the hostname for TLS SNI,
        # certificate verification and the HTTP Host header. No proxy environment.
        conn._create_connection = (
            lambda target, timeout, source_address: socket.create_connection(
                (address, 443), timeout=timeout, source_address=source_address
            )
        )
        response = None
        timer = None
        try:
            conn.connect()
            connection_socket = conn.sock

            def abort_download():
                # Also stop trickle-fed HTTP headers/chunk framing, where an
                # individual socket timeout alone is not a total deadline.
                try:
                    connection_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(remaining(), abort_download)
            timer.daemon = True
            timer.start()
            conn.sock.settimeout(remaining())
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            conn.request("GET", target, headers={"Accept-Encoding": "identity"})
            conn.sock.settimeout(remaining())
            response = conn.getresponse()
            if response.status != 200:
                raise InvalidInputError(
                    "file_url must return HTTP 200; redirects are not followed."
                )
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise InvalidInputError("Compressed URL responses are not supported.")
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    size = int(length)
                    if size < 0:
                        raise ValueError
                except ValueError:
                    raise InvalidInputError(
                        "Invalid download Content-Length."
                    ) from None
                if size > limit:
                    raise PayloadTooLargeError(
                        "URL download exceeds the media byte limit."
                    )
            chunks = []
            size = 0
            while True:
                # read1 returns after one underlying read, so a slow peer cannot
                # keep resetting the socket timeout indefinitely within read().
                connection_socket.settimeout(remaining())
                chunk = response.read1(min(65536, limit + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise PayloadTooLargeError(
                        "URL download exceeds the media byte limit."
                    )
                chunks.append(chunk)
            content_type = (
                response.getheader("Content-Type", "application/octet-stream")
                .split(";")[0]
                .strip()
            )
            return (
                b"".join(chunks),
                Path(unquote(parsed.path)).name or "media",
                content_type,
            )
        finally:
            if timer is not None:
                timer.cancel()
            if response is not None:
                response.close()
            conn.close()

    try:
        async with asyncio.timeout(cfg.url_download_timeout_seconds):
            addresses = await asyncio.to_thread(
                socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM
            )
            if not addresses or any(
                not public_address(result[4][0]) for result in addresses
            ):
                raise InvalidInputError(
                    "file_url must resolve only to public IP addresses."
                )
            return await asyncio.to_thread(fetch, addresses[0][4][0])
    except (OSError, TimeoutError, http.client.HTTPException, ValueError):
        # Never expose signed query strings, credentials or upstream error bodies.
        raise InvalidInputError(
            "Unable to download file_url within the configured timeout."
        ) from None


class MediaResolver:
    def __init__(
        self, storage: StorageManager, cfg: Settings, files: dict[str, UploadFile]
    ):
        self.storage = storage
        self.cfg = cfg
        self.files = files
        self.created: list[str] = []
        self.file_records: dict[str, UploadRecord] = {}

    async def resolve(self, part: MediaPart) -> UploadRecord:
        if part.upload_id is not None:
            return await self.storage.get_upload(part.upload_id)
        if part.file is not None:
            if part.file in self.file_records:
                return self.file_records[part.file]
            upload = self.files.get(part.file)
            if upload is None:
                raise InvalidInputError(
                    "Media file reference has no matching multipart field."
                )
            data = await upload.read(self.cfg.max_upload_bytes + 1)
            filename, content_type = upload.filename or "media", upload.content_type
        elif part.file_url is not None:
            data, filename, content_type = await download_url(part.file_url, self.cfg)
            filename = part.filename or filename
            content_type = part.content_type or content_type
        else:
            encoded = part.file_data
            content_type = part.content_type
            if encoded.startswith("data:"):
                header, separator, encoded = encoded.partition(",")
                if not separator or not header.endswith(";base64") or not header[5:-7]:
                    raise InvalidInputError(
                        "file_data must use data:<media-type>;base64,<data>."
                    )
                content_type = header[5:-7]
            if len(encoded) > 4 * ((self.cfg.max_upload_bytes + 2) // 3):
                raise PayloadTooLargeError("Base64 media exceeds the media byte limit.")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise InvalidInputError("file_data contains invalid base64.") from None
            filename = part.filename or "media"
        if not Path(filename).suffix and content_type:
            filename += mimetypes.guess_extension(content_type) or ""
        record = await self.storage.save_upload(filename, content_type, data)
        # Request-local files must survive queue waits and inference, even if
        # the reusable-upload TTL elapses in the meantime.
        record.expires_at = float("inf")
        self.created.append(record.upload_id)
        if part.file is not None:
            self.file_records[part.file] = record
        return record

    async def close(self) -> None:
        for upload_id in self.created:
            await self.storage.delete_upload(upload_id)
