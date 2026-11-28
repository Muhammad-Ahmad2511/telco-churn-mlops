# BentoML serving path (spec 12.2)

The **Flask API stays the required API** (`src/serving/app.py`, spec 3.8). BentoML is
the second serving path: the approved model is packaged as a versioned *Bento*, built
into a container image, and served with the same request validation as the Flask API.

## Files

| File | Purpose |
|---|---|
| `src/serving/bentoml_package.py` | Fetches the approved model from the MLflow registry and saves it to the BentoML model store with its lineage |
| `src/serving/bentoml_service.py` | The BentoML service: `/predict`, `/model-info`, built-in `/livez` and `/readyz` |
| `bentofile.yaml` | Build definition: service, included files, pinned Python packages, model reference |
| `src/serving/bentoml_smoke_test.py` | Repeatable smoke test for a local server or a container |
| `tests/test_bentoml_service.py` | Integration tests that start a real `bentoml serve` process |
| `requirements-bento.txt` | `requirements-api.txt` plus `bentoml==1.4.40` |

## How the model is fetched and how the approved version is selected

1. `python -m src.serving.bentoml_package` calls the same `load_registered_model()` the
   Flask API uses. It reads the registry model name from `configs/model_params.yaml` and
   loads the version that the alias points to. The default alias is **`production`**
   (override with `MODEL_ALIAS` or `--alias`).
2. `@production` only moves through `python -m src.models.promote` (rule-based, logged,
   manual approval), so the packaged version is always an approved one.
3. The registry model name, version, alias, MLflow run id and lineage tags (`git_commit`,
   `dvc_data_md5` and, once present, `feast_version` and `feature_repo_commit`) are stored
   as labels and metadata of the BentoML model. `POST /model-info` returns them.
4. `bentoml build` pins the model version inside the Bento, so the container needs **no
   registry access and no credentials**; only the packaging step needs the DagsHub `.env`.

Selecting a specific older version is done by moving the alias (`promote rollback`) and
re-running the packaging step. `load_registered_model()` selects by alias only, not by
version number.

## Run it locally (PowerShell, repository root)

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-bento.txt
python -m src.serving.bentoml_package          # registry @production -> BentoML model store
bentoml build                                  # -> telco_churn:<version>
bentoml serve telco_churn:latest --port 3000   # in a second terminal
python -m src.serving.bentoml_smoke_test       # expects: all checks passed
```

Docker:

```powershell
bentoml containerize telco_churn:latest -t telco-churn-bento:local
docker run --rm -p 3000:3000 telco-churn-bento:local
python -m src.serving.bentoml_smoke_test --url http://localhost:3000
```

## API

| Endpoint | Method | Behaviour |
|---|---|---|
| `/livez`, `/readyz` | GET | Built-in health endpoints; 200 when the service is up / ready |
| `/model-info` | POST (empty `{}` body) | Model name, version, alias, run id, lineage tags, feature order |
| `/predict` | POST | `{"instances": [ ... ]}` with 1 to 100 customers; returns `churn` and `churn_probability` per customer plus model name and version |

Status codes: 200 valid; **422** invalid values, missing or unknown fields, empty list or
more than 100 instances (same validation code as the Flask API, `src/serving/schema.py`);
**400** malformed JSON or a body without `instances`; 405 wrong method.

Differences from the Flask API:

- `/model-info` is `POST` (BentoML API routes accept POST only). There is no `GET /health`;
  use `/livez` and `/readyz`.
- A single customer must be sent as `{"instances": [{...}]}`; the Flask-style bare record
  is rejected with 400.

## Tests

`tests/test_bentoml_service.py` creates a temporary BentoML store, packages a stand-in
model with the project's 18 input columns, starts `bentoml serve`, and calls it over HTTP:
health, model info and lineage, single and batch prediction, invalid values (422),
missing and unknown fields, instance limits, malformed requests (400, never 500), wrong
method, and the smoke-test script itself. The file is skipped when `bentoml` is not
installed (for example in CI, which installs only `requirements.txt`).

## Promotion through CI/CD (design, not yet implemented)

The existing deploy workflow builds the Flask API image, pushes it to ECR and waits for
manual approval in the `production` environment. The Bento image would follow the same
path: after tests pass and the approval gate is cleared, run the packaging step with the
DagsHub secrets, `bentoml build`, `bentoml containerize`, the smoke test against the
running container, then tag and push to ECR and update the ECS task definition. A
failed smoke test blocks the push, the same way failed checks block the Flask image.

## Limitations

- `bentoml containerize` and the Docker run were not part of the automated tests; verify
  them with the commands above on the Docker host.
- The Bento container is not deployed to AWS or Kubernetes in this project.
- Request validation is shared with the Flask API, but the two services are separate
  processes and are not load-tested against each other.
