# Airflow image for the telco-churn project (build context = repository root).
FROM apache/airflow:3.4.0

USER root
# git: DVC needs it to read the repository that is mounted into the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Project dependencies go into their own virtualenv so they cannot conflict
# with the package versions pinned by Airflow itself.
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/airflow/project-venv \
    && /opt/airflow/project-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# `python` and `dvc` inside DAG tasks resolve to the project virtualenv.
ENV PATH="/opt/airflow/project-venv/bin:${PATH}"

# The repo is bind-mounted from Windows, so git sees a different owner.
RUN git config --global --add safe.directory '*'