# Assignment 3 Reflection: Deploying PyTorch ML Workloads with Docker & Kubernetes

**Student:** Nikita Pandey | **Roll No:** DA25M599

---

## Reflection (≈420 words)

### What I Built

For Assignment 3 of the IIT Madras M.Tech MLOps course, I designed and implemented a complete end-to-end MLOps pipeline for training and serving a PyTorch image classifier (CIFAR-10). The project covers the full production lifecycle: model code, Docker containerisation, Kubernetes orchestration, CI/CD automation, and a structured Git workflow with feature branches and pull requests.

### Technical Decisions

**Model design:** I chose a three-block CNN with BatchNorm and adaptive pooling rather than a pre-trained model. This was intentional — the assignment is about infrastructure, not accuracy, and a custom model makes every architectural decision traceable.

**Docker:** I used multi-stage builds to keep the final image lean. Separating the `builder` stage (where pip install runs) from the runtime image means no build tools end up in production containers. The non-root `mluser` pattern is a security best practice I've seen in production deployments at Synopsys and wanted to apply here.

**Kubernetes:** The choice of a `Job` (not a `Deployment`) for training reflects how batch ML workloads actually run in production — they complete and exit, rather than staying alive indefinitely. The HPA on the serving deployment (2→10 replicas at 50% CPU) mirrors real-world auto-scaling patterns.

**CI/CD:** The GitHub Actions pipeline catches three classes of problems automatically: style issues (flake8/black), correctness (pytest), and build failures (docker build). This gives confidence that every pushed commit is at minimum syntactically and structurally sound.

### Challenges

The biggest challenge was the Zscaler corporate network policy that blocks writes to external GitHub repositories from the Synopsys laptop. I had to work around this by using SSH key authentication (bypassing the HTTPS proxy) and by temporarily using a phone hotspot. This was itself an MLOps infrastructure lesson — network policies are real constraints in enterprise environments.

### What I Learned

- How PersistentVolumeClaims enable state sharing between a training Job and a serving Deployment in Kubernetes
- The importance of `backoffLimit` and `restartPolicy` in K8s Jobs for fault tolerance
- How GitHub Actions secrets and branch protection rules enforce team discipline in a CI/CD workflow
- That multi-stage Docker builds are non-negotiable for production images — the size difference is 3–4× versus single-stage

### References & Tools

I referred to the official PyTorch, Docker, and Kubernetes documentation throughout this assignment. YAML manifests were adapted from Kubernetes official examples and tuned for the ML workload requirements. GitHub Actions workflows were built by reading the GitHub Actions quickstart guide.
