"""register.py copies lineage tags (including the Feast ones) to the model version."""

from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.models import register as reg

NEW_TAGS = {"feast_version": "0.68.0", "feature_repo_commit": "a" * 40}
OLD_TAGS = {"git_commit": "b" * 40, "dvc_data_md5": "c" * 32, "quality_gate": "passed"}


class FakeClient:
    instances: ClassVar[list["FakeClient"]] = []

    def __init__(self):
        self.version_tags: dict[str, str] = {}
        self.alias: tuple | None = None
        FakeClient.instances.append(self)

    def set_model_version_tag(self, name, version, key, value):
        self.version_tags[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.alias = (name, alias, version)


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "model_params.yaml"
    path.write_text(
        "registry:\n  model_name: test-model\nmlflow:\n  experiment_name: test-exp\n",
        encoding="utf-8",
    )
    return str(path)


def _install_fakes(monkeypatch, tags, with_model=True):
    FakeClient.instances.clear()
    outputs = [SimpleNamespace(model_id="m-1")] if with_model else []
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(tags=tags),
        outputs=SimpleNamespace(model_outputs=outputs),
    )
    monkeypatch.setattr(reg, "find_run", lambda experiment, run_id=None: run)
    monkeypatch.setattr(reg.mlflow, "get_run", lambda run_id: run)
    monkeypatch.setattr(
        reg.mlflow, "register_model", lambda source, name: SimpleNamespace(version=7)
    )
    monkeypatch.setattr(reg, "MlflowClient", FakeClient)


def test_feast_lineage_tags_are_copied_to_the_model_version(monkeypatch, cfg_path):
    _install_fakes(monkeypatch, {**OLD_TAGS, **NEW_TAGS})
    assert reg.register(cfg_path=cfg_path) == "7"
    tags = FakeClient.instances[0].version_tags
    assert tags["feast_version"] == "0.68.0"
    assert tags["feature_repo_commit"] == "a" * 40
    assert tags["git_commit"] == "b" * 40
    assert tags["dvc_data_md5"] == "c" * 32
    assert tags["quality_gate"] == "passed"


def test_older_run_without_feast_tags_gets_unknown(monkeypatch, cfg_path):
    _install_fakes(monkeypatch, dict(OLD_TAGS))
    assert reg.register(cfg_path=cfg_path) == "7"
    tags = FakeClient.instances[0].version_tags
    assert tags["feast_version"] == "unknown"
    assert tags["feature_repo_commit"] == "unknown"
    assert tags["git_commit"] == "b" * 40
    assert FakeClient.instances[0].alias == ("test-model", "candidate", 7)


def test_failed_quality_gate_is_not_registered(monkeypatch, cfg_path):
    _install_fakes(monkeypatch, {**OLD_TAGS, "quality_gate": "failed"})
    with pytest.raises(SystemExit):
        reg.register(cfg_path=cfg_path)
    assert FakeClient.instances == []
