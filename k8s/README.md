# Kubernetes (kind): deployment, scaling, self-healing (spec §3.10)

The same API image used in Docker Compose and on AWS runs here as a Kubernetes **Deployment** with 3 replicas behind a **Service**, on a local **kind** cluster. kind was chosen per instructor guidance for 16 GB laptops. A production cluster on AWS (EKS) is not required by the spec and was skipped to save cost.

| File | Purpose |
|---|---|
| `kind-cluster.yaml` | One-node kind cluster; publishes NodePort 30500 as `http://127.0.0.1:5050` |
| `api-deployment.yaml` | 3 replicas, readiness and liveness probes, resource requests and limits, rolling updates with no downtime |
| `api-service.yaml` | NodePort Service that load-balances across ready pods |
| `aws-iam-role-policy.yaml` | The least-privilege IAM permissions used by the AWS deployment, as a reference manifest |

## Run it

Use PowerShell from the repo root, with Docker Desktop running.
```powershell
winget install -e --id Kubernetes.kind          # once; open a new terminal afterwards
kind create cluster --name telco --config k8s/kind-cluster.yaml
docker build -f docker/Dockerfile.api -t telco-churn-api:local .
kind load docker-image telco-churn-api:local --name telco
kubectl create secret generic dagshub-credentials --from-env-file=.env   # MLflow/DagsHub creds; never committed
kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml
kubectl wait --for=condition=Ready pod -l app=telco-churn-api --timeout=300s
Invoke-RestMethod http://127.0.0.1:5050/health
.venv\Scripts\python -m src.monitoring.traffic --url http://127.0.0.1:5050 --n 60
```

## Demonstrations

| Demo | Command | What to observe |
|---|---|---|
| 3 replicas ready | `kubectl get pods,deploy,svc` | `3/3` ready, Service `NodePort 5000:30500` |
| Probes | `kubectl describe pod <pod>` | `Readiness: http-get /health`, `Liveness: http-get /metrics` |
| Load balancing | Send traffic, then `kubectl logs <pod> \| Select-String "predict instances"` for each pod | Requests are spread over all pods |
| Self-healing | `kubectl delete pod <pod>` then `kubectl get pods -w` | A new pod is created right away, and becomes Ready after the model loads |
| Scaling | `kubectl scale deploy/telco-churn-api --replicas=5`, then `--replicas=3` | Pods are added, then removed |
| Rolling update | `kubectl rollout restart deploy/telco-churn-api` then `kubectl rollout status deploy/telco-churn-api` | Pods are replaced one at a time (`maxUnavailable: 0`), so no downtime |

**Measured on 2026-11-23:**
- 3 pods became Ready in 75 s with 0 restarts. 60 of 60 requests through the Service succeeded, distributed **26 / 21 / 13** across the 3 pods.
- **Self-healing:** after a pod was deleted, its replacement was created within 2 s and Ready after 22 s.
- **Scaling:** 3 → 5 replicas (the new pods were Ready in about 30 s, 37% of node memory reserved), then back to 3.
- **Rolling update** (`kubectl rollout restart`): pods were replaced one at a time with 0 restarts and no gap in capacity.

**Bug found by the probes (fixed):** the first deployment showed `Liveness probe failed: … /metrics … context deadline exceeded` during startup. Every `/health` probe started its own model download, the downloads used up all 4 gunicorn threads, and `/metrics` could not answer. Three failures in a row would have restarted the pod, possibly in a loop when DagsHub is slow. Fix in `src/serving/model_loader.py`: `ModelStore` now lets only one thread load the model (in the background from startup), and other callers get "not ready" immediately. After the fix, new pods showed **0 liveness failures**. This also affects Docker Compose and AWS, which use the same code.

## Design notes

- **Readiness vs liveness.** Readiness uses `/health`, which returns 200 only after the `production` model has loaded, so the Service only sends traffic to pods that can predict. Liveness uses `/metrics`, which answers whenever the process is alive. Using `/health` for liveness would make Kubernetes restart every pod during a short DagsHub/MLflow outage, even though restarting cannot fix an external outage.
- **Load balancing.** The Service has a stable virtual IP. kube-proxy forwards each new connection to one of the *ready* endpoints, so traffic spreads across pods and leaves out pods that are not ready. (`kubectl port-forward` would bypass this and pin all traffic to one pod, which is why the demo uses the NodePort.)
- **Secrets.** DagsHub credentials are a Kubernetes Secret created from the git-ignored `.env`. On AWS the same variables come from Secrets Manager.

## Limitations

- **Single node.** kind runs one node inside Docker, so there is no protection against node failure. A real cluster would spread replicas across nodes and zones (pod anti-affinity or topology spread).
- **No real cloud load balancer or Ingress.** A NodePort on localhost stands in for one. There is no TLS and no authentication on this local endpoint.
- **Every pod downloads the model at startup** from DagsHub (20–60 s), so scaling up is slow and depends on DagsHub being reachable. A production setup would bake the approved model into the image or use a shared cache.
- **No autoscaling.** A HorizontalPodAutoscaler would need metrics-server; replicas are scaled manually here.
- **Per-pod metrics.** Each pod has its own Prometheus counters. The Compose Prometheus scrapes one API container; on Kubernetes you would scrape every pod (Prometheus service discovery).
