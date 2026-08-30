# Kubernetes Deployment Notes 
 
## Manifests Overview 
- namespace.yaml: mlops-pipeline namespace 
- training-job.yaml: K8s Job with backoffLimit=3 
- serving-deployment.yaml: 2 replicas with probes 
- hpa.yaml: scales 2-10 replicas at 50% CPU 
