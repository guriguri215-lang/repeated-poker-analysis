"""Generate starter river-scenario JSON templates.

These are *abstract toy examples*, not strategic recommendations: each template
is a minimal, well-formed scenario that passes the parser/build validation and is
convenient to edit into a real abstract spot. The templates follow
``docs/scenario_format_reference.md`` and the bundled
``examples/scenarios/*.json`` samples; they add no new solver or game-theory
model.

One builder exists per supported ``kind`` (see
:data:`SCENARIO_TEMPLATE_KINDS`). :func:`create_scenario_template` returns a plain
``dict`` ready to serialise with ``json.dumps``; every template carries
``"format_version": "1"``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .scenario_io import DEFAULT_FORMAT_VERSION

# Supported template kinds, in display order.
SCENARIO_TEMPLATE_KINDS: List[str] = [
    "single-hand",
    "hero-range",
    "range-matrix-showdown",
    "range-matrix-equity",
    "range-matrix-equity-betting-tree",
]

_DEFAULT_SCENARIO_IDS: Dict[str, str] = {
    "single-hand": "template_single_hand",
    "hero-range": "template_hero_range",
    "range-matrix-showdown": "template_range_matrix_showdown",
    "range-matrix-equity": "template_range_matrix_equity",
    "range-matrix-equity-betting-tree": "template_range_matrix_equity_betting_tree",
}

# Shared abstract toy parameters, matching the bundled samples.
_RAKE = {"rate": 0.05, "cap": 4.0}
_INITIAL_COMMITMENT = {"hero": 1.0, "villain": 1.0}
_BET_SIZE = 98.0
_CANDIDATE_GENERATION = {"shift_amounts": [0.25, 0.5, 1.0]}
_REPEATED = {"horizons": [10, 20, 50, 100], "discount": 1.0}


def available_scenario_template_kinds() -> List[str]:
    """Return the supported template kinds, in display order."""

    return list(SCENARIO_TEMPLATE_KINDS)


def _base(scenario_id: str, description: str) -> dict:
    return {
        "format_version": DEFAULT_FORMAT_VERSION,
        "scenario_id": scenario_id,
        "description": description,
        "rake": dict(_RAKE),
        "initial_commitment": dict(_INITIAL_COMMITMENT),
    }


def _with_tail(template: dict) -> dict:
    template["candidate_generation"] = {
        "shift_amounts": list(_CANDIDATE_GENERATION["shift_amounts"])
    }
    template["repeated"] = {
        "horizons": list(_REPEATED["horizons"]),
        "discount": _REPEATED["discount"],
    }
    return template


def _single_hand(scenario_id: str) -> dict:
    template = _base(
        scenario_id,
        "Abstract single-hand river template (toy example, not strategic advice).",
    )
    template["bet_size"] = _BET_SIZE
    template["showdown"] = "chop"
    template["baseline_hero_strategy"] = {"IP_vs_bet": {"call": 0.0, "fold": 1.0}}
    return _with_tail(template)


def _hero_range(scenario_id: str) -> dict:
    template = _base(
        scenario_id,
        "Abstract Hero-range river template (toy example, not strategic advice).",
    )
    template["bet_size"] = _BET_SIZE
    template["hero_range"] = [
        {
            "hand_id": "hero_chop",
            "weight": 0.8,
            "showdown": "chop",
            "baseline_strategy": {"call": 0.0, "fold": 1.0},
        },
        {
            "hand_id": "hero_winner",
            "weight": 0.2,
            "showdown": "hero",
            "baseline_strategy": {"call": 1.0, "fold": 0.0},
        },
    ]
    return _with_tail(template)


def _range_matrix_showdown(scenario_id: str) -> dict:
    template = _base(
        scenario_id,
        "Abstract Hero/Villain showdown-matrix template "
        "(toy example, not strategic advice).",
    )
    template["bet_size"] = _BET_SIZE
    template["hero_range"] = [
        {"hand_id": "hero_chop", "weight": 0.8, "baseline_strategy": {"call": 0.0, "fold": 1.0}},
        {"hand_id": "hero_strong", "weight": 0.2, "baseline_strategy": {"call": 1.0, "fold": 0.0}},
    ]
    template["villain_range"] = [
        {"hand_id": "villain_chop", "weight": 0.7},
        {"hand_id": "villain_strong", "weight": 0.3},
    ]
    template["showdown_matrix"] = {
        "hero_chop": {"villain_chop": "chop", "villain_strong": "villain"},
        "hero_strong": {"villain_chop": "hero", "villain_strong": "chop"},
    }
    return _with_tail(template)


def _range_matrix_equity(scenario_id: str) -> dict:
    template = _base(
        scenario_id,
        "Abstract Hero/Villain equity-matrix template "
        "(toy example, not strategic advice).",
    )
    template["bet_size"] = _BET_SIZE
    template["hero_range"] = [
        {"hand_id": "hero_medium", "weight": 0.7, "baseline_strategy": {"call": 0.0, "fold": 1.0}},
        {"hand_id": "hero_strong", "weight": 0.3, "baseline_strategy": {"call": 1.0, "fold": 0.0}},
    ]
    template["villain_range"] = [
        {"hand_id": "villain_weak", "weight": 0.6},
        {"hand_id": "villain_strong", "weight": 0.4},
    ]
    template["equity_matrix"] = {
        "hero_medium": {"villain_weak": 0.65, "villain_strong": 0.35},
        "hero_strong": {"villain_weak": 0.90, "villain_strong": 0.55},
    }
    return _with_tail(template)


def _range_matrix_equity_betting_tree(scenario_id: str) -> dict:
    template = _base(
        scenario_id,
        "Abstract equity-matrix river betting-tree template "
        "(toy example, not strategic advice).",
    )
    template["hero_range"] = [
        {
            "hand_id": "hero_medium",
            "weight": 0.7,
            "baseline_strategies": {
                "after_oop_check": {"check": 1.0, "bet": 0.0},
                "vs_oop_bet": {"call": 0.0, "fold": 1.0, "raise": 0.0},
            },
        },
        {
            "hand_id": "hero_strong",
            "weight": 0.3,
            "baseline_strategies": {
                "after_oop_check": {"check": 0.0, "bet": 1.0},
                "vs_oop_bet": {"call": 1.0, "fold": 0.0, "raise": 0.0},
            },
        },
    ]
    template["villain_range"] = [
        {"hand_id": "villain_weak", "weight": 0.6},
        {"hand_id": "villain_strong", "weight": 0.4},
    ]
    template["equity_matrix"] = {
        "hero_medium": {"villain_weak": 0.65, "villain_strong": 0.35},
        "hero_strong": {"villain_weak": 0.90, "villain_strong": 0.55},
    }
    template["betting_tree"] = {
        "oop_bet_size": 98.0,
        "ip_bet_after_check_size": 98.0,
        "ip_raise_size": 196.0,
    }
    return _with_tail(template)


_BUILDERS: Dict[str, Callable[[str], dict]] = {
    "single-hand": _single_hand,
    "hero-range": _hero_range,
    "range-matrix-showdown": _range_matrix_showdown,
    "range-matrix-equity": _range_matrix_equity,
    "range-matrix-equity-betting-tree": _range_matrix_equity_betting_tree,
}


def create_scenario_template(kind: str, scenario_id: Optional[str] = None) -> dict:
    """Return a starter scenario ``dict`` for ``kind``.

    ``kind`` must be one of :data:`SCENARIO_TEMPLATE_KINDS`. ``scenario_id``
    overrides the per-kind default id when given; it must then be a non-empty
    string (matching the parser's ``scenario_id`` rule), so a non-string or an
    empty string is rejected rather than silently falling back to the default.
    The returned dict is an abstract toy example (not a strategic recommendation)
    that passes the parser/build validation and carries ``"format_version": "1"``.
    """

    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(
            f"unknown template kind {kind!r}; choose one of "
            f"{SCENARIO_TEMPLATE_KINDS}"
        )
    if scenario_id is None:
        scenario_id = _DEFAULT_SCENARIO_IDS[kind]
    elif not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(
            f"scenario_id must be a non-empty string, got {scenario_id!r}"
        )
    return builder(scenario_id)
