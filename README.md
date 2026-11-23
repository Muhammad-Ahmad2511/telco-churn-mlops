# telco-churn-mlops

End-to-end MLOps system for **telecom customer churn prediction**. Given a customer's account details, the API predicts whether the customer is likely to leave, with a probability, so the business can act before they do.

The focus of the project is the **MLOps lifecycle around the model**, more than model accuracy: versioned data, tracked experiments, automated quality gates, containerised serving, cloud deployment, orchestration, monitoring, and controlled promotion and rollback.

> **Status (2026-11-23):** working and demonstrated: DVC pipeline, MLflow tracking and registry, promotion and rollback, CI/CD with gates, Flask API in Docker, Prometheus and Grafana with alerts, AWS ECS Fargate deployment, Kubernetes on kind.
> **In progress:** Kubeflow Pipelines, Airflow orchestration and retraining trigger, Evidently drift, Feast, BentoML, REPORT.md.

---

## Architecture

```
            ┌─────────────── GitHub ───────────────┐
 code push  │  PR → CI: ruff, pytest, bandit,      │  merge to main + manual approval
 ─────────► │  pip-audit, Docker build, Trivy,     │ ─────────────────────────────────┐
            │  smoke test  (required checks)       │                                  │
            └──────────────────────────────────────┘                                  ▼
 data/raw/telco_churn.csv ──DVC──► DagsHub remote              AWS ap-southeast-2 (OIDC, no keys)
        │                                                       ECR → ECS Fargate (1 task)
        ▼  dvc repro                                            Secrets Manager → DagsHub token
 validate (Pandera) → preprocess (Parquet) → tune (3 model     CloudWatch logs + alarms
 types, 11 candidates) ──► MLflow on DagsHub (runs, metrics)
                                   │
                                   ▼
             Model Registry: telco-churn-classifier
             @candidate (register.py) → @production (promote.py: compare, promote, rollback)
                                   │
                                   ▼
        Flask API (gunicorn) loads @production ── /health /model-info /predict /metrics
          ├─ Docker Compose: API + Prometheus + Grafana (dashboard, 5+2 alerts)
          ├─ Kubernetes (kind): 3 replicas, probes, NodePort Service
          └─ AWS ECS Fargate
```

## Repository layout

| Path | What it contains |
|---|---|
| `src/data/validation.py` | Pandera schema for the raw CSV (21 columns, categories, ranges, unique IDs) |
| `src/features/preprocessing.py` | Type fixes, train/val/test split, scikit-learn preprocessing pipeline |
| `src/models/` | `train.py` (baseline), `tune.py` (model comparison), `evaluate.py` (metrics and quality gate), `register.py`, `promote.py` |
| `src/serving/` | Flask app, request schema, model loader, Prometheus metrics |
| `src/monitoring/traffic.py` | Simulated traffic generator for dashboards and alert demos |
| `configs/` | Pipeline and model configuration (thresholds, tuning grid, promotion rule) |
| `dvc.yaml`, `dvc.lock`, `reports/` | Reproducible pipeline and its outputs |
| `docker/` | `Dockerfile.api`, `docker-compose.yml` |
| `monitoring/` | Prometheus config and alert rules, Grafana provisioning and dashboard |
| `k8s/` | kind cluster config, Deployment, Service ([k8s/README.md](k8s/README.md)) |
| `deploy/aws/`, `.github/workflows/deploy.yml` | AWS task definition, IAM policies, CD workflow |
| `.github/workflows/ci.yml` | CI quality gates |
| `docs/` | [AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md), [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| `tests/` | 101 tests: data, unit, model, API, metrics, promotion, traffic |

---

## 1. Setup

**Prerequisites:** Git, Python **3.12**, [uv](https://docs.astral.sh/uv/), Docker Desktop. Windows commands are shown in PowerShell. On Linux/macOS use `.venv/bin/` instead of `.venv\Scripts\`.

```powershell
git clone https://github.com/Muhammad-Ahmad2511/telco-churn-mlops.git
cd telco-churn-mlops
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

### Credentials (names only; never commit values)
The project uses **DagsHub** for the DVC data remote and the MLflow tracking server. You need read access to the DagsHub repository and a DagsHub access token (DagsHub → Settings → Tokens).

1. Copy `.env.example` to `.env` and fill it in, **without quotes**:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/Muhammad-Ahmad2511/telco-churn-mlops.mlflow
   MLFLOW_TRACKING_USERNAME=<your DagsHub username>
   MLFLOW_TRACKING_PASSWORD=<your DagsHub token>
   ```
2. Give DVC the same credentials. `--local` writes to the git-ignored `.dvc/config.local`:
   ```powershell
   .venv\Scripts\dvc remote modify origin --local auth basic
   .venv\Scripts\dvc remote modify origin --local user <your DagsHub username>
   .venv\Scripts\dvc remote modify origin --local password <your DagsHub token>
   ```
3. Download the versioned data:
   ```powershell
   .venv\Scripts\dvc pull
   ```

| Variable | Where | Used for |
|---|---|---|
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | `.env` (local), Kubernetes Secret, AWS Secrets Manager | MLflow tracking and Model Registry on DagsHub |
| `MODEL_ALIAS` | Container environment (default `production`) | Which registry alias the API serves |
| `LOG_LEVEL` | Container environment (default `INFO`) | API log level |
| `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` | GitHub Actions secrets | `dvc pull` and the smoke test in CI |
| `AWS_DEPLOY_ROLE_ARN`, `DAGSHUB_SECRET_ARN` | GitHub environment `production` secrets | AWS deployment ([docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md)) |
| `ECS_SUBNETS`, `ECS_SECURITY_GROUP`, `DEPLOY_ENABLED` | GitHub repository variables | AWS deployment |
| `GRAFANA_ADMIN_PASSWORD` | Optional, for Docker Compose | Grafana admin password (default `admin`, changed at first login) |

---

## 2. Run the tests and quality checks

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\pytest -q
```
Expected: `All checks passed!` and `101 passed`. The same checks, plus `bandit`, `pip-audit`, a Docker build, a Trivy scan and a container smoke test, run in GitHub Actions on every pull request. `main` is protected, so a failing check blocks the merge.

## 3. Train: the reproducible pipeline

```powershell
.venv\Scripts\dvc repro           # validate → preprocess → tune (logs to MLflow on DagsHub)
.venv\Scripts\dvc metrics show    # test metrics of the selected model
.venv\Scripts\dvc dag             # pipeline graph
```
- `tune` trains 11 candidates (Logistic Regression, Random Forest, HistGradientBoosting) on the training split, compares them on the validation split, refits the winner and scores it once on the test split. It **fails (exit 1) if the quality gate fails** (recall ≥ 0.60, F1 ≥ 0.55; see `configs/model_params.yaml`).
- Running `dvc repro` again without changes skips every stage. Changing data, code or params re-runs only the affected stages.
- To recover an earlier code version together with its dataset, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 4. Register and promote a model

```powershell
.venv\Scripts\python -m src.models.register                  # newest gate-passed run → alias @candidate
.venv\Scripts\python -m src.models.promote compare           # candidate vs production on the same test rows (changes nothing)
.venv\Scripts\python -m src.models.promote promote           # moves @production only if the rule passes
.venv\Scripts\python -m src.models.promote rollback          # @production back to the previous version
```
Promotion rule: the candidate passed the quality gate, **and** F1 ≥ production + 0.005, **and** recall ≥ production − 0.02. Every decision is logged in MLflow. Promotion is a **manual** step, never automatic.

**Current registry:** v1 Logistic Regression (recall 0.783, F1 0.613) · **v2 Random Forest @production** (recall 0.773, F1 0.629, ROC-AUC 0.839).

## 5. Run the API locally

```powershell
.venv\Scripts\python -m src.serving.app      # http://127.0.0.1:5000
```

| Endpoint | Description |
|---|---|
| `GET /health` | `200 {"status":"ok","model_loaded":true}` once the model is loaded; `503` before that |
| `GET /model-info` | Model name, version, alias, git commit, DVC data hash, features |
| `POST /predict` | One customer object, or `{"instances": [...]}` with up to 100. Returns `churn` (0/1) and `churn_probability` |
| `GET /metrics` | Prometheus metrics |

Example request (PowerShell):
```powershell
$body = @{SeniorCitizen=0; Partner="Yes"; Dependents="No"; tenure=12; PhoneService="Yes"; MultipleLines="No"; InternetService="DSL"; OnlineSecurity="Yes"; OnlineBackup="No"; DeviceProtection="No"; TechSupport="No"; StreamingTV="No"; StreamingMovies="No"; Contract="Month-to-month"; PaperlessBilling="Yes"; PaymentMethod="Electronic check"; MonthlyCharges=29.85; TotalCharges=358.2} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $body
```
Response:
```json
{"model_name": "telco-churn-classifier", "model_version": "2",
 "predictions": [{"churn": 1, "churn_probability": 0.5728}]}
```
Invalid input returns **422** with the reasons, never echoing the values. Malformed JSON returns 400, and a body over 64 KB returns 413. Request contents are never logged.

## 6. Docker Compose: API + Prometheus + Grafana

```powershell
docker compose -f docker/docker-compose.yml up -d --build
.venv\Scripts\python -m src.monitoring.traffic --n 300 --invalid 0.1     # simulated traffic
```
| Service | URL |
|---|---|
| API | http://localhost:5000 |
| Prometheus (targets, alerts) | http://localhost:9090 |
| Grafana, dashboard "Telco Churn API" (user `admin`) | http://localhost:3000 |

Alert demos: `--invalid 0.6` fires `TelcoHighValidationFailureRate`; `docker compose -f docker/docker-compose.yml stop api` fires `TelcoApiDown` after about 1 minute. Stop everything with `docker compose -f docker/docker-compose.yml down`.

## 7. Kubernetes (kind)

See [k8s/README.md](k8s/README.md). In short: `kind create cluster --name telco --config k8s/kind-cluster.yaml`, load the image, create the `dagshub-credentials` secret from `.env`, `kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml`, then use http://127.0.0.1:5050.

## 8. AWS deployment

The full runbook (one-time setup in AWS CloudShell, deploy, test, CloudWatch, rollback, **cleanup**) is in [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md). After the one-time setup, every merge to `main` that passes CI triggers the **Deploy** workflow, which **waits for manual approval** in the GitHub `production` environment.

Cost control: a budget alert ($100, notifications at 85% and 100%); the smallest Fargate size that fits (0.5 vCPU / 1 GB); the service is scaled to 0 between demos; everything is deleted after the final demo (runbook "Cleanup").

---

## Automated vs manual steps

| Step | Type |
|---|---|
| Lint, tests, security scans, Docker build and smoke test on every PR | **Automated**, blocks the merge on failure |
| Data validation and training (`dvc repro`) | Automated pipeline, started manually (Airflow scheduling in progress) |
| Registering a gate-passed model as `@candidate` | Manual command; refuses failed or uncommitted runs |
| Promoting `@candidate` to `@production` | **Manual approval** (`promote.py`, rule-based, logged) |
| Deploying to AWS | Automatic after CI on `main`, then **manual approval** in GitHub |
| Rollback | Manual command (`promote.py rollback`, or redeploy a previous image) |

## Dataset

IBM **Telco Customer Churn** sample dataset (7,043 customers, 21 columns, 26.5% churn), versioned with DVC. See [DATASET_LICENSE.md](DATASET_LICENSE.md). `customerID` and the sensitive attribute `gender` are **not** used as model inputs; gender is kept only in the processed data for a fairness check.

## Team

Course project for a team of three. Individual contributions are listed in REPORT.md.
