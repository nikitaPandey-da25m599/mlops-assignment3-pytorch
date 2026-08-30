"""
src/serve.py
------------
FastAPI inference server for the CIFAR-10 classifier.

Endpoints:
  GET  /health   — liveness / readiness check
  POST /predict  — accepts a multipart image file, returns class probabilities

Environment variables:
  MODEL_PATH   path to the .pt checkpoint  (default /app/checkpoints/classifier_v1.pt)
  CONFIG_PATH  path to training_config.yaml (default /app/configs/training_config.yaml)
  PORT         port to bind                 (default 8080)

Run locally:
  uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Allow running directly from repo root.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import CIFAR10_CLASSES, get_single_image_transform  # noqa: E402
from model import load_checkpoint  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = "/app/checkpoints/classifier_v1.pt"
_DEFAULT_CONFIG_PATH = "/app/configs/training_config.yaml"

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
CONFIG_PATH = os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG_PATH)

# Fallback to local paths when running outside Docker.
if not Path(MODEL_PATH).exists():
    _local = Path(__file__).parent.parent / "checkpoints" / "classifier_v1.pt"
    if _local.exists():
        MODEL_PATH = str(_local)

if not Path(CONFIG_PATH).exists():
    _local_cfg = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    if _local_cfg.exists():
        CONFIG_PATH = str(_local_cfg)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")
_transform = get_single_image_transform()
_start_time: float = time.time()


def _load_model() -> None:
    global _model, _device

    # Read architecture / num_classes from config so the server is config-driven.
    architecture = "resnet18"
    num_classes = 10
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        architecture = cfg.get("model", {}).get("architecture", architecture)
        num_classes = cfg.get("model", {}).get("num_classes", num_classes)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_checkpoint(
        checkpoint_path=MODEL_PATH,
        architecture=architecture,
        num_classes=num_classes,
        device=_device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CIFAR-10 Classifier API",
    description="Serves a PyTorch ResNet-18 model trained on CIFAR-10.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model when the server starts up."""
    if Path(MODEL_PATH).exists():
        _load_model()
    # If the checkpoint doesn't exist yet (e.g. first-start before training),
    # the /health endpoint will return 503 until the model is available.


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness & readiness probe.

    Returns HTTP 200 when the model is loaded, HTTP 503 otherwise.
    """
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "reason": "model not loaded",
                "uptime_seconds": round(time.time() - _start_time, 1),
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_path": MODEL_PATH,
            "device": str(_device),
            "uptime_seconds": round(time.time() - _start_time, 1),
        },
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict:
    """Run inference on an uploaded image.

    Accepts any standard image format (PNG, JPEG, …). Returns a dict with:
    - ``predicted_class``   — string label (e.g. ``"cat"``)
    - ``predicted_index``   — integer class index (0–9)
    - ``confidence``        — probability of the top class (0–1)
    - ``probabilities``     — dict mapping each class name to its probability
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Read and decode the uploaded image.
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    # Pre-process.
    tensor = _transform(pil_image).unsqueeze(0).to(_device)  # (1, 3, 32, 32)

    # Inference.
    with torch.no_grad():
        logits = _model(tensor)  # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    top_idx = int(probs.argmax())
    top_prob = float(probs[top_idx])

    class_probs = {
        CIFAR10_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CIFAR10_CLASSES))
    }

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx],
        "predicted_index": top_idx,
        "confidence": round(top_prob, 4),
        "probabilities": class_probs,
    }
