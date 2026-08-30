# mlops-pytorch-pipeline

A production-style MLOps pipeline that trains a PyTorch image classifier on CIFAR-10 and serves predictions via a REST API — all containerised with Docker and orchestrated on Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│  Namespace: ml-training                                  │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Training Job    │    │  Serving Deployment      │   │
│  │  (mlops-train)   │    │  (mlops-serve, 2 replicas│   │
│  │  CPU:2 / Mem:4Gi │    │  CPU:1 / Mem:2Gi)        │   │
│  └────────┬─────────┘    └──────────┬───────────────┘   │
│           │                         │                    │
│           └──────┬──────────────────┘                    │
│                  ▼                                        │
│         PersistentVolumeClaim                            │
│         /app/data  &  /app/checkpoints                   │
│                                                          │
│  ConfigMap ──► training_config.yaml                      │
│  HPA       ──► auto-scales serving pods 2-10             │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop
- `kubectl` CLI
- A Kubernetes cluster (Minikube / kind / cloud)
- A GitHub account

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

### 2. Local training (no Docker)

```bash
pip install -r requirements/train.txt
python src/train.py
```

### 3. Docker — build & run training

```bash
# Build
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run (mounts local data & checkpoint dirs)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### 4. Docker — build & run serving

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test health
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

### 5. Kubernetes deployment

```bash
# Create namespace + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Persistent storage
kubectl apply -f k8s/pvc.yaml

# Train
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/pytorch-training-job -n ml-training --timeout=3600s

# Serve
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Port-forward and test
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Git Workflow

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code |
| `develop` | Integration branch |
| `feature/docker-training` | Docker training image |
| `feature/k8s-deployment` | Kubernetes manifests |

All features are merged to `develop` via Pull Requests, then `develop` → `main` via a release PR.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONFIG_PATH` | `/app/configs/training_config.yaml` | Path to YAML config |
| `CHECKPOINT_DIR` | `/app/checkpoints` | Where model checkpoints are saved |
| `MODEL_PATH` | `/app/checkpoints/classifier_v1.pt` | Path to checkpoint for serving |
| `PORT` | `8080` | Serving port |

## CI/CD

GitHub Actions runs on every push and PR to `main`/`develop`:

1. **Lint** — `flake8` + `black --check`
2. **Unit tests** — `pytest tests/`
3. **Docker build** — validates both images build without error
