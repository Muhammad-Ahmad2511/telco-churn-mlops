"""Tests for the Feast lineage tags recorded on MLflow runs."""

import re
import subprocess
from pathlib import Path

from src.models.lineage import (
    UNKNOWN,
    feast_version,
    feature_lineage,
    feature_repo_commit,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        text=True,
    ).strip()


def test_feast_version_is_read_from_the_pin(tmp_path):
    (tmp_path / "requirements-feast.txt").write_text(
        "# comment\nfeast==0.68.0\npytest\n", encoding="utf-8"
    )
    assert feast_version(tmp_path) == "0.68.0"


def test_feast_version_unknown_when_file_missing(tmp_path):
    assert feast_version(tmp_path) == UNKNOWN


def test_feast_version_unknown_when_not_pinned(tmp_path):
    (tmp_path / "requirements-feast.txt").write_text("feast>=0.60\n", encoding="utf-8")
    assert feast_version(tmp_path) == UNKNOWN


def test_feature_repo_commit_tracks_only_feature_store_changes(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "feature_store").mkdir()
    (tmp_path / "feature_store" / "features.py").write_text("A = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "feature repo")
    feature_commit = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "other.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "unrelated change")

    assert feature_repo_commit(tmp_path) == feature_commit

    (tmp_path / "feature_store" / "features.py").write_text("A = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "change features")
    assert feature_repo_commit(tmp_path) == _git(tmp_path, "rev-parse", "HEAD")


def test_feature_repo_commit_unknown_without_history(tmp_path):
    assert feature_repo_commit(tmp_path) == UNKNOWN


def test_feature_lineage_has_both_tags(tmp_path):
    (tmp_path / "requirements-feast.txt").write_text("feast==0.68.0\n", encoding="utf-8")
    tags = feature_lineage(tmp_path)
    assert set(tags) == {"feast_version", "feature_repo_commit"}
    assert tags["feast_version"] == "0.68.0"
    commit = tags["feature_repo_commit"]
    assert commit == UNKNOWN or re.fullmatch(r"[0-9a-f]{40}", commit)
