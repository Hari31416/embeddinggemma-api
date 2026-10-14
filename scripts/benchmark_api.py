"""Measure warmed, sequential API requests against real local model weights.

Run each device/dtype in a fresh process. No remote API or uploads are used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from PIL import Image

from multimodal_embedding.adapter.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings
from scripts.create_fixtures import create_fixtures

logger = logging.getLogger("benchmark_api")


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def prepare_cases(directory: Path, dataset: Path | None):
    paths = create_fixtures(directory)
    sources = {}
    if dataset:
        image_source = dataset / "images/624073main_1969-04-04_full.jpg"
        audio_source = dataset / "audio/590333main_ringtone_eagleHasLanded_extended.mp3"
        with Image.open(image_source) as image:
            image = image.convert("RGB")
            image.thumbnail((1024, 1024))
            paths["apollo_image"] = directory / "apollo.png"
            image.save(paths["apollo_image"])
            sources["image_size"] = list(image.size)
        audio, rate = librosa.load(audio_source, sr=16000, mono=True, duration=10)
        paths["apollo_audio"] = directory / "apollo.wav"
        sf.write(paths["apollo_audio"], audio, rate, subtype="PCM_16")
        sources["audio_seconds"] = len(audio) / rate
        sources["originals"] = {
            str(path.relative_to(dataset)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (image_source, audio_source)
        }

    def text(value):
        return {"type": "text", "text": value}

    def media(name):
        return {"type": "media", "file": name}

    # Exactly 512 whitespace-separated words, without a manually added prefix.
    document = " ".join(
        (
            "Apollo 11 astronauts landed on the Moon and collected lunar samples. ".split()
            * 50
        )[:512]
    )
    image = "apollo_image" if dataset else "image"
    audio = "apollo_audio" if dataset else "audio"
    combined = [
        text("Apollo 11 mission footage and radio communication."),
        media(image),
        media(audio),
    ]
    cases = {
        "text_query": ("query", [[text("When did Apollo 11 land on the Moon?")]]),
        "text_document_512_words": ("document", [[text(document)]]),
        "text_batch_8": ("document", [[text(document)] for _ in range(8)]),
        "image_64": ("document", [[media("image")]]),
        "audio_1s": ("document", [[media("audio")]]),
        "video_2_frames": (
            "document",
            [[{"type": "video_frames", "files": ["image", "image"]}]],
        ),
        "combined": ("document", [combined]),
        "mixed_batch_5": (
            "document",
            [
                [text(document)],
                [media(image)],
                [media(audio)],
                [{"type": "video_frames", "files": ["image", "image"]}],
                combined,
            ],
        ),
    }
    if dataset:
        cases["apollo_image"] = ("document", [[media(image)]])
        cases["apollo_audio_10s"] = ("document", [[media(audio)]])
    contents = {
        name: (
            path.name,
            path.read_bytes(),
            "image/png" if path.suffix == ".png" else "audio/wav",
        )
        for name, path in paths.items()
        if name in {"image", "audio", "apollo_image", "apollo_audio"}
    }
    sources["prepared_sha256"] = {
        name: hashlib.sha256(value[1]).hexdigest() for name, value in contents.items()
    }
    return cases, contents, sources


def request_arguments(task, parts, contents):
    payload = {
        "task": task,
        "prompt_mode": "auto",
        "items": [
            {"id": str(index), "parts": value} for index, value in enumerate(parts)
        ],
    }
    names = set()
    for item in parts:
        for part in item:
            if "file" in part:
                names.add(part["file"])
            names.update(part.get("files", []))
    if names:
        return {
            "data": {"payload": json.dumps(payload)},
            "files": {name: contents[name] for name in sorted(names)},
        }
    return {"json": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.threads < 1:
        parser.error("warmup >= 1, repeats >= 2, and threads >= 1 are required")
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "sentence-transformers", "transformers", "fastapi"]
        },
        "device": args.device,
        "dtype": args.dtype,
        "threads": args.threads,
        "interop_threads": 1,
        "warmup_per_case": args.warmup,
        "measured_requests_per_case": args.repeats,
        "concurrency": 1,
        "cases": {},
    }
    if platform.system() == "Darwin":
        report["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        report["ram_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        )
    with tempfile.TemporaryDirectory(prefix="embedding-benchmark-") as temporary:
        directory = Path(temporary)
        cases, contents, report["inputs"] = prepare_cases(
            directory / "fixtures", args.dataset
        )
        settings = Settings(
            _env_file=None,
            device=args.device,
            torch_dtype=args.dtype,
            upload_dir=directory / "uploads",
            api_keys=[],
            use_fake_adapter=False,
        )
        report["model"] = settings.model_id
        report["revision"] = settings.model_revision
        report["vision_max_soft_tokens"] = settings.vision_max_soft_tokens
        report["video_max_soft_tokens"] = settings.video_max_soft_tokens
        adapter = SentenceTransformersAdapter(settings)
        if adapter.actual_device != args.device or adapter.device_fallback:
            raise RuntimeError(
                "Requested device is unavailable; refusing to benchmark a fallback"
            )
        synchronize(args.device)
        started = time.perf_counter()
        adapter.load()
        synchronize(args.device)
        report["model_load_ms"] = (time.perf_counter() - started) * 1000
        with TestClient(create_app(settings=settings, adapter=adapter)) as client:
            for name, (task, parts) in cases.items():
                arguments = request_arguments(task, parts, contents)
                samples = []
                max_norm_error = 0.0
                for iteration in range(args.warmup + args.repeats):
                    synchronize(args.device)
                    started = time.perf_counter()
                    response = client.post("/v1/embeddings", **arguments)
                    synchronize(args.device)
                    elapsed = (time.perf_counter() - started) * 1000
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"{name}: {response.status_code}: {response.text}"
                        )
                    vectors = np.asarray(
                        [item["embedding"] for item in response.json()["embeddings"]]
                    )
                    if (
                        vectors.shape != (len(parts), 768)
                        or not np.isfinite(vectors).all()
                    ):
                        raise RuntimeError(f"{name}: invalid embeddings")
                    error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))
                    if error > 1e-3:
                        raise RuntimeError(f"{name}: embeddings are not normalized")
                    max_norm_error = max(max_norm_error, error)
                    if iteration >= args.warmup:
                        samples.append(elapsed)
                report["cases"][name] = {
                    "items_per_request": len(parts),
                    "samples_ms": samples,
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "items_per_second": len(parts)
                    * len(samples)
                    / (sum(samples) / 1000),
                    "max_norm_error": max_norm_error,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                logger.info(
                    "%s/%s %s: median %.1f ms, p95 %.1f ms",
                    args.device,
                    args.dtype,
                    name,
                    report["cases"][name]["median_ms"],
                    report["cases"][name]["p95_ms"],
                )


if __name__ == "__main__":
    main()
