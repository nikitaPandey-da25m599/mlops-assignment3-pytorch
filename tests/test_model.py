"""
tests/test_model.py
-------------------
Unit tests for model.py, dataset.py, and the serving predict endpoint.
Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Ensure src/ is importable regardless of CWD.
# ─────────────────────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import get_model  # noqa: E402
from dataset import get_transforms, get_single_image_transform, CIFAR10_CLASSES  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def untrained_model():
    """A freshly initialised ResNet-18 (pretrained=False for offline tests)."""
    return get_model(architecture="resnet18", num_classes=10, pretrained=False)


# ─────────────────────────────────────────────────────────────────────────────
# model.py tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetModel:
    def test_output_shape(self, untrained_model, cpu_device):
        """Model should produce (batch, num_classes) tensors."""
        model = untrained_model.eval().to(cpu_device)
        batch = torch.randn(4, 3, 32, 32).to(cpu_device)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_no_nan_in_output(self, untrained_model, cpu_device):
        """Output should not contain NaN values."""
        model = untrained_model.eval().to(cpu_device)
        batch = torch.randn(2, 3, 32, 32).to(cpu_device)
        with torch.no_grad():
            out = model(batch)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_unsupported_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            get_model(architecture="vgg16", num_classes=10)

    @pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
    def test_supported_architectures(self, arch):
        """All three supported architectures should instantiate without error."""
        model = get_model(architecture=arch, num_classes=10, pretrained=False)
        assert model is not None

    def test_custom_num_classes(self):
        """Model with 5 output classes should have the right head dimension."""
        model = get_model(architecture="resnet18", num_classes=5, pretrained=False)
        batch = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(batch)
        assert out.shape[-1] == 5

    def test_model_is_in_train_mode_by_default(self):
        model = get_model(architecture="resnet18", num_classes=10, pretrained=False)
        assert model.training, "Model should be in training mode after instantiation"


# ─────────────────────────────────────────────────────────────────────────────
# dataset.py tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTransforms:
    def _fake_pil(self, size=(32, 32)):
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_train_transform_output_shape(self):
        img = self._fake_pil()
        t = get_transforms(train=True)
        tensor = t(img)
        assert tensor.shape == (3, 32, 32)

    def test_val_transform_output_shape(self):
        img = self._fake_pil()
        t = get_transforms(train=False)
        tensor = t(img)
        assert tensor.shape == (3, 32, 32)

    def test_single_image_transform_resizes(self):
        """Images of arbitrary size should be resized to (3, 32, 32)."""
        img = self._fake_pil(size=(128, 128))
        t = get_single_image_transform()
        tensor = t(img)
        assert tensor.shape == (3, 32, 32)

    def test_output_is_normalised(self):
        """Pixels should no longer be in the [0, 1] range after normalisation."""
        img = self._fake_pil()
        t = get_transforms(train=False)
        tensor = t(img)
        # At least some pixel values should be outside [0, 1] after normalisation.
        assert not torch.all((tensor >= 0) & (tensor <= 1))


class TestClassLabels:
    def test_ten_classes(self):
        assert len(CIFAR10_CLASSES) == 10

    def test_known_labels_present(self):
        for label in ("airplane", "cat", "dog", "automobile"):
            assert label in CIFAR10_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Integration: forward pass through model + predict pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestInferencePipeline:
    """End-to-end: image → transform → model → softmax → class index."""

    def test_full_pipeline(self, untrained_model, cpu_device):
        import torch.nn.functional as F

        model = untrained_model.eval().to(cpu_device)
        transform = get_single_image_transform()

        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)

        tensor = transform(pil_img).unsqueeze(0).to(cpu_device)  # (1, 3, 32, 32)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Sanity checks
        assert probs.shape == (10,)
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities should sum to 1"
        top_idx = int(probs.argmax())
        assert 0 <= top_idx < 10
        assert CIFAR10_CLASSES[top_idx] in CIFAR10_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# serve.py tests (uses FastAPI TestClient — no real checkpoint needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestServeHealthEndpoint:
    """Tests that only require the FastAPI app itself — no trained model."""

    @pytest.fixture(autouse=True)
    def _patch_model(self, monkeypatch):
        """Replace the module-level _model with None to simulate pre-training."""
        import serve  # type: ignore[import]
        monkeypatch.setattr(serve, "_model", None)

    def test_health_returns_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import serve  # type: ignore[import]

        client = TestClient(serve.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 503

    def test_predict_returns_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import serve  # type: ignore[import]

        client = TestClient(serve.app, raise_server_exceptions=False)

        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/predict", files={"image": ("test.png", buf, "image/png")})
        assert resp.status_code == 503


class TestServeWithMockedModel:
    """Inject a working (untrained) model and verify the predict endpoint."""

    @pytest.fixture(autouse=True)
    def _inject_model(self, monkeypatch, untrained_model, cpu_device):
        import serve  # type: ignore[import]

        m = untrained_model.eval().to(cpu_device)
        monkeypatch.setattr(serve, "_model", m)
        monkeypatch.setattr(serve, "_device", cpu_device)

    def test_health_returns_200(self):
        from fastapi.testclient import TestClient
        import serve  # type: ignore[import]

        client = TestClient(serve.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_returns_valid_json(self):
        from fastapi.testclient import TestClient
        import serve  # type: ignore[import]

        client = TestClient(serve.app)

        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/predict", files={"image": ("test.png", buf, "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        assert "predicted_class" in data
        assert "predicted_index" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["predicted_class"] in CIFAR10_CLASSES
        assert 0 <= data["predicted_index"] <= 9
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 10
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3
