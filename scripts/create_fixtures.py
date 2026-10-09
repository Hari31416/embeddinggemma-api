import logging
import math
import os
import struct
import wave
from pathlib import Path
from PIL import Image
import av

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fixtures(base_dir: Path) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Text
    text_path = base_dir / "sample.txt"
    text_path.write_text("A fast brown fox jumps over the lazy dog.", encoding="utf-8")
    paths["text"] = text_path
    logger.info("Created text fixture at %s", text_path)

    # 2. Image (64x64 PNG)
    image_path = base_dir / "sample.png"
    img = Image.new("RGB", (64, 64), color=(73, 109, 137))
    img.save(image_path)
    paths["image"] = image_path
    logger.info("Created image fixture at %s", image_path)

    # 3. Audio (1s 16kHz mono WAV)
    audio_path = base_dir / "sample.wav"
    sample_rate = 16000
    duration_s = 1.0
    freq = 440.0
    num_samples = int(sample_rate * duration_s)
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            val = int(
                32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate))
            )
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    paths["audio"] = audio_path
    logger.info("Created audio fixture at %s", audio_path)

    # 4. Video (1s 2fps 64x64 MP4 using PyAV)
    video_path = base_dir / "sample.mp4"
    container = av.open(str(video_path), mode="w", format="mp4")
    stream = container.add_stream("h264", rate=2)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"

    for frame_idx in range(2):
        color = (100 + frame_idx * 50, 50, 150)
        pil_frame = Image.new("RGB", (64, 64), color=color)
        frame = av.VideoFrame.from_image(pil_frame)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    paths["video"] = video_path
    logger.info("Created video fixture at %s", video_path)

    return paths


if __name__ == "__main__":
    fixtures_dir = Path("tests/fixtures")
    create_fixtures(fixtures_dir)
