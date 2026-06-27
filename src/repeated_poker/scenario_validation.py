"""Validate river scenario JSON inputs before running any analysis.

This is a parser/build-level pre-flight check, not an analysis run. For each
scenario file it loads the JSON, parses it with
:func:`repeated_poker.scenario_io.river_scenario_from_dict`, and builds the game
with :func:`repeated_poker.scenario_io.build_river_steal_game_from_scenario`, so
the user can confirm that an input is well formed and see *which kind* of
scenario it is interpreted as. It deliberately stops there: it never generates
candidates, runs the exact-response solver, or runs the analysis pipeline, so it
is cheap to run and adds no new analysis maths.

It adds no new solver or game-theory model. Input expansion (a single path, a
sequence of paths, or a directory whose ``*.json`` files are read in filename
order), the cwd-relative / file-name-only ``source_path`` display, and the
``model_kind`` label are reused from
:mod:`repeated_poker.scenario_batch` so validation and batch analysis report a
scenario the same way.

Errors are handled per the ``continue_on_error`` flag: fail-fast (the default)
re-raises the first failure with the offending display path, while
continue-on-error records the error on that scenario's row and proceeds. Failure
rows carry a short ``error_type`` / ``error_message`` instead of a Python
traceback, and never leak an absolute local path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .game import (
    ChanceNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    iter_terminals,
)
from .scenario_batch import (
    BatchInput,
    display_scenario_path,
    expand_scenario_inputs,
    model_kind_from_metadata,
)
from .scenario_io import (
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)


@dataclass(frozen=True)
class ScenarioValidationConfig:
    """Configuration for :func:`validate_river_scenario_inputs`.

    ``continue_on_error`` switches between fail-fast (raise on the first failing
    scenario) and continue-on-error (record the error on the row and proceed).
    """

    continue_on_error: bool = False


@dataclass(frozen=True)
class ScenarioValidationRow:
    """One validation result row per scenario, in input order.

    On success ``ok`` is true, the descriptive fields are filled from the parsed
    scenario and built game, and ``error_type`` / ``error_message`` are ``None``.
    On failure ``ok`` is false, the descriptive fields are ``None``, and
    ``error_type`` / ``error_message`` carry a short, traceback-free summary.

    ``chance_outcome_count`` is the total number of chance-node outcomes across
    the tree (the bucket / matchup count); it is ``None`` when the tree has no
    chance node (single-hand mode).
    """

    source_path: str
    ok: bool
    scenario_id: Optional[str] = None
    model_kind: Optional[str] = None
    horizons: Optional[List[int]] = None
    discount: Optional[float] = None
    shift_amounts_count: Optional[int] = None
    hero_info_set_count: Optional[int] = None
    villain_info_set_count: Optional[int] = None
    terminal_count: Optional[int] = None
    has_chance_node: Optional[bool] = None
    chance_outcome_count: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    # Appended last for positional-constructor compatibility; ``to_dict`` still
    # emits ``format_version`` near ``scenario_id`` for readable output.
    format_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "ok": self.ok,
            "scenario_id": self.scenario_id,
            "format_version": self.format_version,
            "model_kind": self.model_kind,
            "horizons": self.horizons,
            "discount": self.discount,
            "shift_amounts_count": self.shift_amounts_count,
            "hero_info_set_count": self.hero_info_set_count,
            "villain_info_set_count": self.villain_info_set_count,
            "terminal_count": self.terminal_count,
            "has_chance_node": self.has_chance_node,
            "chance_outcome_count": self.chance_outcome_count,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ScenarioValidationResult:
    """The validation rows of a run, one per input scenario in input order."""

    rows: List[ScenarioValidationRow] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for row in self.rows if row.ok)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if not row.ok)

    def to_dict(self) -> dict:
        return {
            "ok_count": self.ok_count,
            "error_count": self.error_count,
            "rows": [row.to_dict() for row in self.rows],
        }


def _chance_outcome_stats(tree) -> tuple:
    """Return ``(has_chance_node, chance_outcome_count)`` for ``tree``.

    ``chance_outcome_count`` sums the children of every chance node, and is
    ``None`` when the tree has no chance node at all.
    """

    total = 0
    has_chance = False
    for node in iter_nodes(tree.root):
        if isinstance(node, ChanceNode):
            has_chance = True
            total += len(node.children)
    return has_chance, (total if has_chance else None)


def _success_row(display: str, scenario, build) -> ScenarioValidationRow:
    has_chance, chance_outcomes = _chance_outcome_stats(build.tree)
    repeated = scenario.repeated
    return ScenarioValidationRow(
        source_path=display,
        ok=True,
        scenario_id=scenario.scenario_id,
        format_version=scenario.format_version,
        model_kind=model_kind_from_metadata(build.metadata),
        horizons=list(repeated.horizons) if repeated and repeated.horizons else None,
        discount=repeated.discount if repeated else None,
        shift_amounts_count=len(scenario.shift_amounts) if scenario.shift_amounts else 0,
        hero_info_set_count=len(collect_hero_info_sets(build.tree)),
        villain_info_set_count=len(collect_villain_info_sets(build.tree)),
        terminal_count=sum(1 for _ in iter_terminals(build.tree.root)),
        has_chance_node=has_chance,
        chance_outcome_count=chance_outcomes,
    )


def _sanitize_message(message: str, path: Path, display: str) -> str:
    """Replace any absolute spelling of ``path`` in ``message`` with ``display``.

    Defends against an exception string that embeds the input path -- a file read
    error such as ``FileNotFoundError`` includes the absolute path in ``str(exc)``
    -- so neither a row's ``error_message`` nor a fail-fast error leaks an absolute
    local path. Both the native (``str(path)``) and POSIX spellings of the raw and
    resolved path are replaced. Parser/build errors do not contain the path, so
    this is a no-op for them.
    """

    needles = [str(path), path.as_posix()]
    try:
        resolved = path.resolve()
        needles.extend([str(resolved), resolved.as_posix()])
    except OSError:  # pragma: no cover - resolve only fails in odd environments
        pass
    cleaned = message
    for needle in needles:
        if needle and needle != display:
            cleaned = cleaned.replace(needle, display)
    return cleaned


def _error_message(exc: BaseException, path: Path, display: str) -> str:
    """Return a short, path-safe error message for a failed scenario.

    File read errors (``OSError`` / ``FileNotFoundError``) are reported with the
    display name and the OS reason only (never the absolute path); parser/build
    errors keep their own message, passed through :func:`_sanitize_message` as a
    defensive measure.
    """

    if isinstance(exc, OSError):
        reason = exc.strerror or "could not be read"
        message = f"could not read scenario file {display!r}: {reason}"
    else:
        message = str(exc)
    return _sanitize_message(message, path, display)


def _error_row(display: str, exc: BaseException, message: str) -> ScenarioValidationRow:
    return ScenarioValidationRow(
        source_path=display,
        ok=False,
        error_type=type(exc).__name__,
        error_message=message,
    )


def validate_river_scenario_inputs(
    inputs: BatchInput,
    config: Optional[ScenarioValidationConfig] = None,
) -> ScenarioValidationResult:
    """Validate each scenario by parsing and building it (no analysis run).

    See the module docstring for input expansion and error handling. Returns a
    :class:`ScenarioValidationResult` whose ``rows`` are in input order.
    """

    config = config or ScenarioValidationConfig()
    paths = expand_scenario_inputs(inputs)

    rows: List[ScenarioValidationRow] = []
    for path in paths:
        display = display_scenario_path(path)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            scenario = river_scenario_from_dict(data)
            build = build_river_steal_game_from_scenario(scenario)
        except Exception as exc:  # noqa: BLE001 - surfaced via row or re-raised
            message = _error_message(exc, Path(path), display)
            if not config.continue_on_error:
                raise ValueError(
                    f"failed to validate scenario {display!r}: {message}"
                ) from exc
            rows.append(_error_row(display, exc, message))
            continue
        rows.append(_success_row(display, scenario, build))

    return ScenarioValidationResult(rows=rows)
