import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase1_verification")


def probe_device(requested_device: str = "auto") -> tuple[torch.device, str]:
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if requested_device == "mps":
        if mps_available:
            return torch.device("mps"), "mps"
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu"), "cpu (fallback from mps)"
    elif requested_device == "cuda":
        if cuda_available:
            return torch.device("cuda"), "cuda"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu"), "cpu (fallback from cuda)"
    elif requested_device == "cpu":
        return torch.device("cpu"), "cpu"
    elif requested_device == "auto":
        if mps_available:
            return torch.device("mps"), "mps"
        elif cuda_available:
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"
    else:
        logger.warning(
            "Unknown device '%s' requested. Defaulting to CPU.", requested_device
        )
        return torch.device("cpu"), "cpu"


def check_l2_normalized(emb: np.ndarray | torch.Tensor, tol: float = 1e-3) -> bool:
    if isinstance(emb, torch.Tensor):
        norm = torch.linalg.norm(emb, dim=-1).cpu().numpy()
    else:
        norm = np.linalg.norm(emb, axis=-1)
    return bool(np.all(np.abs(norm - 1.0) < tol))


def run_verification(device_name: str = "cpu") -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    device, device_desc = probe_device(device_name)
    logger.info("Using device: %s (%s)", device, device_desc)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        logger.info("Hugging Face authentication token detected in environment.")
    else:
        logger.info("No Hugging Face token detected in environment.")

    model_id = "google/embeddinggemma-2"
    logger.info("Loading model: %s on device: %s ...", model_id, device)

    t_start_load = time.time()
    dtype = torch.float32
    model = SentenceTransformer(
        model_id,
        device=str(device),
        model_kwargs={"torch_dtype": dtype},
    )
    load_duration = time.time() - t_start_load
    logger.info("Model loaded in %.2f seconds", load_duration)

    fixtures_dir = Path("tests/fixtures")
    img_path = str(fixtures_dir / "sample.png")
    wav_path = str(fixtures_dir / "sample.wav")
    mp4_path = str(fixtures_dir / "sample.mp4")

    results: dict[str, Any] = {
        "device": str(device),
        "device_desc": device_desc,
        "load_time_s": load_duration,
        "modalities": {},
    }

    # 1. Text Query & Document
    logger.info("Testing Text Query and Document encoding...")
    query_text = "What is the capital of France?"
    doc_text = "Paris is the capital of France."

    t0 = time.time()
    q_emb = model.encode(query_text, prompt_name="SearchQuery", convert_to_tensor=True)
    t_query = time.time() - t0

    t0 = time.time()
    d_emb = model.encode(doc_text, prompt_name="Document", convert_to_tensor=True)
    t_doc = time.time() - t0

    assert q_emb.shape[-1] == 768, f"Expected 768 dim, got {q_emb.shape}"
    assert d_emb.shape[-1] == 768, f"Expected 768 dim, got {d_emb.shape}"
    assert not torch.isnan(q_emb).any()
    assert not torch.isnan(d_emb).any()
    assert check_l2_normalized(q_emb)
    assert check_l2_normalized(d_emb)

    results["modalities"]["text_query"] = {
        "shape": list(q_emb.shape),
        "latency_s": t_query,
        "normalized": True,
    }
    results["modalities"]["text_doc"] = {
        "shape": list(d_emb.shape),
        "latency_s": t_doc,
        "normalized": True,
    }
    logger.info(
        "Text query shape: %s (%.3fs), doc shape: %s (%.3fs)",
        list(q_emb.shape),
        t_query,
        list(d_emb.shape),
        t_doc,
    )

    # 2. Image
    logger.info("Testing Image encoding...")
    t0 = time.time()
    img_emb = model.encode({"image": img_path}, convert_to_tensor=True)
    t_img = time.time() - t0

    assert img_emb.shape[-1] == 768, f"Expected 768 dim, got {img_emb.shape}"
    assert not torch.isnan(img_emb).any()
    assert check_l2_normalized(img_emb)
    results["modalities"]["image"] = {
        "shape": list(img_emb.shape),
        "latency_s": t_img,
        "normalized": True,
    }
    logger.info("Image shape: %s (%.3fs)", list(img_emb.shape), t_img)

    # 3. Audio
    logger.info("Testing Audio encoding...")
    t0 = time.time()
    audio_emb = model.encode({"audio": wav_path}, convert_to_tensor=True)
    t_audio = time.time() - t0

    assert audio_emb.shape[-1] == 768, f"Expected 768 dim, got {audio_emb.shape}"
    assert not torch.isnan(audio_emb).any()
    assert check_l2_normalized(audio_emb)
    results["modalities"]["audio"] = {
        "shape": list(audio_emb.shape),
        "latency_s": t_audio,
        "normalized": True,
    }
    logger.info("Audio shape: %s (%.3fs)", list(audio_emb.shape), t_audio)

    # 4. Video
    logger.info("Testing Video encoding...")
    t0 = time.time()
    video_emb = model.encode({"video": mp4_path}, convert_to_tensor=True)
    t_video = time.time() - t0

    assert video_emb.shape[-1] == 768, f"Expected 768 dim, got {video_emb.shape}"
    assert not torch.isnan(video_emb).any()
    assert check_l2_normalized(video_emb)
    results["modalities"]["video"] = {
        "shape": list(video_emb.shape),
        "latency_s": t_video,
        "normalized": True,
    }
    logger.info("Video shape: %s (%.3fs)", list(video_emb.shape), t_video)

    # 5. Combined (text + image + audio)
    logger.info("Testing Combined (text + image + audio) encoding...")
    combined_input = {
        "text": "A scene with sound and picture",
        "image": img_path,
        "audio": wav_path,
    }
    t0 = time.time()
    combined_emb = model.encode(combined_input, convert_to_tensor=True)
    t_comb = time.time() - t0

    assert combined_emb.shape[-1] == 768, f"Expected 768 dim, got {combined_emb.shape}"
    assert not torch.isnan(combined_emb).any()
    assert check_l2_normalized(combined_emb)
    results["modalities"]["combined"] = {
        "shape": list(combined_emb.shape),
        "latency_s": t_comb,
        "normalized": True,
    }
    logger.info("Combined shape: %s (%.3fs)", list(combined_emb.shape), t_comb)

    # 6. Heterogeneous Batch
    logger.info("Testing Heterogeneous batch encoding...")
    batch_inputs = [
        "A plain text document",
        {"image": img_path},
        {"audio": wav_path},
        {"video": mp4_path},
        combined_input,
    ]
    t0 = time.time()
    batch_embs = model.encode(batch_inputs, convert_to_tensor=True)
    t_batch = time.time() - t0

    assert batch_embs.shape == (
        5,
        768,
    ), f"Expected (5, 768), got {batch_embs.shape}"
    assert not torch.isnan(batch_embs).any()
    assert check_l2_normalized(batch_embs)
    results["batch"] = {"shape": list(batch_embs.shape), "latency_s": t_batch}
    logger.info(
        "Batch shape: %s for %d items (%.3fs)",
        list(batch_embs.shape),
        len(batch_inputs),
        t_batch,
    )

    # 7. Truncation dimensions (Matryoshka Representation Learning)
    logger.info("Testing MRL Truncation dimensions (512, 256, 128)...")
    for dim in [512, 256, 128]:
        truncated_emb = q_emb[:dim]
        renormed_emb = F.normalize(truncated_emb.float(), p=2, dim=-1)
        assert renormed_emb.shape[-1] == dim
        assert check_l2_normalized(renormed_emb)
        logger.info("Dimension %d truncated and re-normalized successfully.", dim)

    logger.info("Phase 1 verification PASSED on %s!", device_desc)
    return results


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    run_verification(dev)
