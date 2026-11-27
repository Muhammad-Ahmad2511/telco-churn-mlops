"""Airflow DAG: scheduled ingest -> validate -> drift check -> conditional retrain.

Flow:
  ingest -> validate -> drift_check --(exit 0)--> skip_retrain --------+
                                     \\-(exit 2)--> retrain (dvc repro)  |
                                                    -> register_candidate
                                                    -> compare_candidate |
                                                                         v
                                                                    finish
Promotion to @production is NOT in this DAG: it stays a manual approval step
(`python -m src.models.promote promote`).
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.sdk import DAG, Param

# Where the repository is mounted inside the Airflow containers.
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")

DRIFT_EXIT_CODE = 2  # src.monitoring.monitor --exit-code returns 2 on drift


def _drift_branch(params: dict) -> str:
    """Run the drift check and choose the next task from its exit code.

    `simulate` ("none" or "drift") is chosen when the DAG is triggered; scheduled
    runs use "none", i.e. the real recent inputs.
    """
    simulate = params.get("simulate", "none")
    print(f"drift check: --simulate {simulate}")
    result = subprocess.run(
        [
            "python",
            "-m",
            "src.monitoring.monitor",
            "--simulate",
            simulate,
            "--exit-code",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode == 0:
        return "skip_retrain"
    if result.returncode == DRIFT_EXIT_CODE:
        return "retrain"
    raise RuntimeError(f"drift check failed with exit code {result.returncode}")


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="telco_churn_ml_orchestrator",
    description="Ingest, validate, drift-check and conditionally retrain",
    default_args=default_args,
    schedule="0 2 * * *",  # daily at 02:00; can also be triggered manually
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    catchup=False,
    params={"simulate": Param("none", type="string", enum=["none", "drift"])},
    tags=["mlops", "telco-churn"],
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"cd {PROJECT_DIR} && python -m src.data.ingestion",
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=(
            f"cd {PROJECT_DIR} && python -m src.data.validation "
            "data/raw/telco_churn.csv --report reports/validation.json"
        ),
    )

    drift_check = BranchPythonOperator(
        task_id="drift_check",
        python_callable=_drift_branch,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    # --force: drift means the model must be retrained even if DVC thinks nothing
    # changed. A quality-gate failure is deterministic, so do not retry blindly.
    retrain = BashOperator(
        task_id="retrain",
        bash_command=f"cd {PROJECT_DIR} && dvc repro --force",
        retries=0,
    )

    register_candidate = BashOperator(
        task_id="register_candidate",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.register",
        retries=0,
    )

    # Read-only: compares candidate vs production and changes nothing.
    compare_candidate = BashOperator(
        task_id="compare_candidate",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.promote compare",
        retries=0,
    )

    finish = EmptyOperator(
        task_id="finish",
        trigger_rule="none_failed_min_one_success",
    )

    ingest >> validate >> drift_check
    drift_check >> skip_retrain >> finish
    drift_check >> retrain >> register_candidate >> compare_candidate >> finish