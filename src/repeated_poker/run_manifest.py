"""Reproducibility manifest for analysis runs.

A run manifest records what produced an analysis output: the SHA-256 of the
scenario file (when the run started from a file), the scenario format version,
the package version, the git commit of the package source checkout (best
effort; ``None`` when git or the checkout is unavailable), a UTC timestamp,
and the effective analysis parameters.  It is descriptive metadata only: it
changes no analysis result and makes no correctness or profitability claim.

The git lookup is a single local ``git rev-parse HEAD`` for the directory
containing this package, cached per process.  It does no network work and
never raises: any failure (git missing, not a checkout, timeout) yields
``None`` in the manifest.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

PathLike = Union[str, Path]

# The single source of the package version exposed to manifests and to
# ``repeated_poker.__version__``.  Kept in sync with ``pyproject.toml`` by a
# test rather than by an install-time dependency, so a plain source checkout
# (no editable install) still reports the right version.
PACKAGE_VERSION = "0.1.0"


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility metadata attached to an analysis run.

    ``scenario_sha256`` and ``scenario_format_version`` are ``None`` when the
    run did not start from a scenario file (for example an in-memory
    ``RiverScenario``, or a batch-level manifest whose scenarios each carry
    their own).  ``parameters`` holds the effective analysis parameters of the
    run; for a batch-level manifest it holds the requested overrides, while
    each per-scenario manifest holds that scenario's resolved values.
    """

    package_version: str
    git_commit: Optional[str]
    timestamp_utc: str
    scenario_sha256: Optional[str] = None
    scenario_format_version: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable mapping with English keys."""

        return {
            "scenario_sha256": self.scenario_sha256,
            "scenario_format_version": self.scenario_format_version,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "timestamp_utc": self.timestamp_utc,
            "parameters": self.parameters,
        }


def sha256_of_file(path: PathLike) -> str:
    """Return the SHA-256 hex digest of the file's raw bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonicalize_filter_parameters(
    *,
    allowed_info_sets: Optional[Iterable[str]],
    max_l1_distance: Optional[float],
    min_required_observations: Optional[int],
) -> Dict[str, Any]:
    """Return deterministic, JSON-safe candidate-filter manifest fields.

    Callers pass the filter values forwarded to the candidate pipeline.  An
    unspecified allowed collection stays ``None``; an empty collection stays an
    empty list; and a non-empty collection is de-duplicated and sorted.  Numeric
    values are recorded unchanged so the manifest reflects the validated
    effective values used by the filter.
    """

    if allowed_info_sets is None:
        canonical_allowed = None
    else:
        allowed = set(allowed_info_sets)
        for value in allowed:
            if not isinstance(value, str):
                raise ValueError(
                    "allowed_info_sets must contain only strings, "
                    f"got {value!r}"
                )
        canonical_allowed = sorted(allowed)
    return {
        "filter_allowed_info_sets": canonical_allowed,
        "filter_max_l1_distance": max_l1_distance,
        "filter_min_required_observations": min_required_observations,
    }


@lru_cache(maxsize=1)
def _resolve_git_commit() -> Optional[str]:
    """Best-effort commit hash of the package source checkout.

    Returns ``None`` when git is unavailable, the package does not live in a
    git checkout, the call times out, or the output is not a commit hash.
    Cached per process (one subprocess call at most).
    """

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
        return commit
    return None


def _utc_timestamp() -> str:
    """The current UTC time as an ISO 8601 ``...Z`` string."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_manifest(
    scenario_path: Optional[PathLike] = None,
    scenario_format_version: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    """Build a :class:`RunManifest` for an analysis run.

    ``scenario_path`` is hashed (SHA-256 of the raw file bytes) when given;
    pass ``None`` for runs that did not start from a file.  ``parameters``
    should hold the effective analysis parameters (horizon, discount,
    response mode, and so on) and is stored as a shallow copy.
    """

    return RunManifest(
        package_version=PACKAGE_VERSION,
        git_commit=_resolve_git_commit(),
        timestamp_utc=_utc_timestamp(),
        scenario_sha256=(
            sha256_of_file(scenario_path) if scenario_path is not None else None
        ),
        scenario_format_version=scenario_format_version,
        parameters=dict(parameters) if parameters is not None else None,
    )
