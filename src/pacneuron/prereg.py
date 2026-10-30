"""Preregistration lock: confirmatory phases refuse to run unless the plan files are committed and unchanged.

Local: `require_frozen()` checks git for the preregistration, its addendum and results/candidates.json, and returns
their SHA-256 hashes plus the commit that froze them (written into every confirmatory result file).
Global: enforces "no held-out data before the candidates are frozen" (reports/preregistration.md §3, §6).
"""

from __future__ import annotations

import hashlib
import subprocess

from pacneuron.config import REPO_ROOT

FROZEN = ["reports/preregistration.md", "reports/preregistration_addendum.md", "results/candidates.json"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def require_frozen(extra: list[str] | None = None) -> dict:
    """Raise unless every frozen file (+ `extra`, e.g. rule-derived replication candidates) exists, is tracked and
    has no uncommitted changes; return SHA-256 hashes and the freezing commit."""
    lock = {}
    for rel in FROZEN + list(extra or []):
        path = REPO_ROOT / rel
        if not path.exists():
            raise RuntimeError(f"{rel} missing: freeze candidates (preregistration addendum) before confirmatory runs")
        if _git("ls-files", "--error-unmatch", rel).returncode != 0:
            raise RuntimeError(f"{rel} is not committed")
        if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            raise RuntimeError(f"{rel} has uncommitted changes")
        lock[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock["frozen_commit"] = _git("log", "-1", "--format=%H", "--", *FROZEN, *(extra or [])).stdout.strip()
    return lock
