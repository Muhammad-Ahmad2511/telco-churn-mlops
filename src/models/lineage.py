"""Feature-store lineage for MLflow runs (spec 12.1).

Feast lives in its own environment (pandas 2.x), so the main environment cannot
import it. The version is read from the pin in requirements-feast.txt, and the
feature repository version is the last Git commit that touched feature_store/.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

UNKNOWN = "unknown"
PIN_FILE = "requirements-feast.txt"
FEATURE_DIR = "feature_store"


def feast_version(repo_root: Path | None = None, pin_file: str = PIN_FILE) -> str:
    """Return the Feast version pinned in requirements-feast.txt, or 'unknown'."""
    path = Path(repo_root or ".") / pin_file
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return UNKNOWN
    for line in lines:
        line = line.strip()
        if line.startswith("feast=="):
            return line.split("==", 1)[1].strip() or UNKNOWN
    return UNKNOWN


def feature_repo_commit(repo_root: Path | None = None, feature_dir: str = FEATURE_DIR) -> str:
    """Return the last commit that changed feature_store/, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", feature_dir],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return UNKNOWN
    return out or UNKNOWN


def feature_lineage(repo_root: Path | None = None) -> dict[str, str]:
    """Tags that identify the feature repository used by a run."""
    return {
        "feast_version": feast_version(repo_root),
        "feature_repo_commit": feature_repo_commit(repo_root),
    }
