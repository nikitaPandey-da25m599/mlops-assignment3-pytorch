# SUBMISSION ANSWER FILE
## Assignment 3: Deploying PyTorch ML Workloads with Docker & Kubernetes
**Student:** Nikita Pandey | **Roll No:** DA25M599 | **Course:** MLOps & Infrastructure for Machine Learning — IIT Madras M.Tech

---

## Part 1: PyTorch Model Training Pipeline

### Q1.1 — Model Architecture
**File:** `src/model.py`

The model is a `CIFAR10Classifier` — a custom CNN built with PyTorch's `nn.Module`:

- **3 Convolutional blocks** (`conv1→conv2→conv3`), each with:
  - `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`
  - Channels: 3→32→64→128
- **Adaptive average pooling** (2×2) to flatten spatial dims
- **Fully-connected head**: `512 → 256 → 10` (10 CIFAR-10 classes)
- **Dropout** (p=0.5) for regularisation

**Design rationale:** BatchNorm stabilises training; pooling controls overfitting; adaptive pooling makes the FC head input-size-independent.

---

### Q1.2 — Training Loop
**File:** `src/train.py`

Key training features:
| Feature | Implementation |
|---|---|
| Optimizer | `Adam` (lr=1e-3, weight_decay=1e-4) |
| Loss | `CrossEntropyLoss` |
| LR scheduler | `ReduceLROnPlateau` (patience=3) |
| Early stopping | Triggered after 5 epochs with no val-loss improvement |
| Checkpointing | Best model saved to `$CHECKPOINT_DIR/classifier_v1.pt` |
| Config | Loaded from `configs/training_config.yaml` via environment variable |

The training loop logs epoch-level train loss/acc and val loss/acc.

---

### Q1.3 — Dataset Pipeline
**File:** `src/dataset.py`

- Downloads CIFAR-10 via `torchvision.datasets.CIFAR10`
- **Train transforms:** RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
- **Val transforms:** Normalize only (deterministic)
- Returns `DataLoader` objects for train + validation splits

---

## Part 2: Docker Containerisation

### Q2.1 — Training Dockerfile
**File:** `docker/Dockerfile.train`

```dockerfile
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements/train.txt .
RUN pip install --no-cache-dir -r train.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY src/ ./src/
COPY configs/ ./configs/
RUN useradd -m mluser && chown -R mluser:mluser /app
USER mluser
ENTRYPOINT ["python", "src/train.py"]
```

**Key decisions:**
- Multi-stage build reduces final image size
- Non-root user (`mluser`) for security
- `python:3.10-slim` base (CPU-only PyTorch — no CUDA bloat)

---

### Q2.2 — Serving Dockerfile
**File:** `docker/Dockerfile.serve`

- FastAPI + Uvicorn serving on port 8080
- `HEALTHCHECK` on `/health` endpoint (30s interval, 3 retries)
- Same multi-stage, non-root pattern as training image

---

### Q2.3 — Build & Run Commands

```bash
# Build
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Train locally
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serve locally
docker run -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

---

## Part 3: Kubernetes Deployment

### Q3.1 — Namespace & ConfigMap
**Files:** `k8s/namespace.yaml`, `k8s/configmap.yaml`

```yaml
# namespace
apiVersion: v1
kind: Namespace
metadata:
  name: ml-training
```

ConfigMap stores `EPOCHS=20`, `BATCH_SIZE=64`, `LEARNING_RATE=0.001` — injected as env vars into both Job and Deployment pods.

---

### Q3.2 — Training Job
**File:** `k8s/training-job.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-training-job
  namespace: ml-training
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: mlops-train:v1
        resources:
          requests: {cpu: "2", memory: "4Gi"}
          limits:   {cpu: "4", memory: "8Gi"}
        volumeMounts:
        - name: model-storage
          mountPath: /app/checkpoints
```

`backoffLimit: 3` retries on failure; `restartPolicy: OnFailure` (required for Jobs).

---

### Q3.3 — Serving Deployment + HPA
**Files:** `k8s/serving-deployment.yaml`, `k8s/hpa.yaml`

- **2 replicas** (HA), liveness + readiness probes on `/health:8080`
- **HPA** scales 2→10 replicas when CPU > 50%
- **PVC** (`model-storage`, 5Gi) shared between training Job and serving Deployment via `ReadWriteOnce`

---

### Q3.4 — kubectl Commands

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/pytorch-training-job \
  -n ml-training --timeout=3600s
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## Part 4: CI/CD with GitHub Actions

### Q4.1 — Pipeline Design
**File:** `.github/workflows/ci.yml`

Triggered on every push/PR to `main` or `develop`:

| Step | Tool | What it checks |
|---|---|---|
| Lint | `flake8`, `black --check` | PEP8 + formatting |
| Unit tests | `pytest tests/` | Model forward pass, data loading, API responses |
| Docker build | `docker build` | Both images build cleanly |

---

### Q4.2 — Unit Tests
**File:** `tests/test_model.py` (249 lines)

Key test cases:
- `test_model_forward_pass` — output shape `[B, 10]`
- `test_model_output_range` — logits are finite
- `test_dataset_transforms` — tensor shape after transforms
- `test_training_step` — single batch gradient update runs
- `test_serve_health` — `/health` returns 200
- `test_serve_predict` — `/predict` returns class + confidence

---

## Part 5: Git Workflow

### Branch Structure
| Branch | Role |
|---|---|
| `main` | Stable production-ready code |
| `develop` | Integration branch |
| `feature/docker-training` | Docker images (PR #1 → develop) |
| `feature/k8s-deployment` | K8s manifests (PR #2 → develop) |
| `hotfix/training-docs` | Doc improvements (PR #4 → main) |

### Pull Request History
- **PR #1** — `feature/docker-training` → `develop`: "feat: Add multi-stage Dockerfiles"
- **PR #2** — `feature/k8s-deployment` → `develop`: "feat: Add Kubernetes manifests"
- **PR #3** — `develop` → `main`: "feat: Merge docker + k8s work to main"
- **PR #4** — `hotfix/training-docs` → `main`: "docs: add training loop docstring"

---

## GitHub Repository

**URL:** https://github.com/nikitaPandey-da25m599/mlops-assignment3-pytorch

**Branches:** main, develop, feature/docker-training, feature/k8s-deployment, hotfix/training-docs

---

## AI Assistance Disclosure

Per the academic integrity policy, AI assistance (Claude/AneMone) was used for:
- Boilerplate Dockerfile and YAML structure
- CI/CD pipeline template
- Test case scaffolding

All code was reviewed, understood, and adapted. Model architecture and training logic reflect independent design choices.
