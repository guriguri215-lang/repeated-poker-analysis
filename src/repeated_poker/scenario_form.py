"""Form-friendly intermediate representation for single-hand scenarios.

This is the first, deliberately small step towards a future GUI/form input layer
(see ``docs/gui_input_design.md``). It is *not* a GUI and adds no new solver,
game-theory model, or analysis logic. It only offers a flat dataclass that is
convenient to bind to form fields, plus helpers to convert that form to and from
the existing JSON scenario format and to surface field-level validation messages
for display.

Scope of v1: single-hand mode only. The matrix / range / betting-tree modes are
out of scope here and are rejected by :func:`single_hand_form_from_dict`.

JSON stays the source of truth. :func:`single_hand_form_from_dict` reuses the
existing :func:`repeated_poker.scenario_io.river_scenario_from_dict` parser (so
loading a scenario into a form applies the same structural validation, with no
duplicated parsing), and :func:`single_hand_form_to_dict` produces a dict that
the same parser and ``build_river_steal_game_from_scenario`` accept. The separate
:func:`validate_single_hand_form` returns GUI-facing field-level messages instead
of raising, so a form being edited can be checked without throwing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .payoffs import CHOP, HERO, VILLAIN
from .scenario_io import (
    DEFAULT_FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    river_scenario_from_dict,
)

_TOLERANCE = 1e-9
_IP_INFO_SET = "IP_vs_bet"
_SHOWDOWN_RESULTS = (CHOP, HERO, VILLAIN)
# Top-level keys that select a non-single-hand mode; their presence means the
# input is not a single-hand scenario.
_NON_SINGLE_HAND_KEYS = (
    "hero_range",
    "villain_range",
    "showdown_matrix",
    "equity_matrix",
    "betting_tree",
)


@dataclass(frozen=True)
class FormValidationMessage:
    """One field-level validation message for display in a form.

    ``field`` is the form field name, ``message`` is a short human-readable
    explanation, and ``severity`` defaults to ``"error"``.
    """

    field: str
    message: str
    severity: str = "error"


@dataclass
class SingleHandScenarioForm:
    """A flat, form-friendly view of a single-hand river scenario.

    The fields map one-to-one to the single-hand scenario JSON (see
    ``docs/scenario_format_reference.md``): the baseline ``IP_vs_bet`` call/fold
    distribution is split into two probability fields, and the rake, initial
    commitment, candidate-generation, and repeated-game values are flattened.
    """

    format_version: str = DEFAULT_FORMAT_VERSION
    scenario_id: str = ""
    description: str = ""
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    initial_commitment_hero: float = 0.0
    initial_commitment_villain: float = 0.0
    bet_size: float = 0.0
    showdown: str = CHOP
    baseline_call_probability: float = 0.0
    baseline_fold_probability: float = 1.0
    shift_amounts: List[float] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    discount: float = 1.0


def single_hand_form_from_dict(data: dict) -> SingleHandScenarioForm:
    """Build a :class:`SingleHandScenarioForm` from a single-hand scenario dict.

    Raises :class:`ValueError` if ``data`` is not a single-hand scenario (for
    example it carries ``hero_range`` / ``villain_range`` / ``showdown_matrix`` /
    ``equity_matrix`` / ``betting_tree``) or fails the existing parser's
    validation (an unsupported baseline action, a bad ``showdown``, an
    unsupported ``format_version``, and so on).
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    forbidden = [key for key in _NON_SINGLE_HAND_KEYS if key in data]
    if forbidden:
        raise ValueError(
            "single-hand form supports single-hand mode only; remove "
            f"{sorted(forbidden)} (those select a range / matrix / betting-tree mode)"
        )

    # Reuse the existing parser for all structural validation and mode handling;
    # with no forbidden keys present this yields a single-hand scenario.
    scenario = river_scenario_from_dict(data)
    distribution = scenario.baseline_hero_strategy[_IP_INFO_SET]
    repeated = scenario.repeated
    return SingleHandScenarioForm(
        format_version=scenario.format_version,
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        rake_rate=scenario.rake.rate,
        rake_cap=scenario.rake.cap,
        initial_commitment_hero=scenario.initial_commitment.hero,
        initial_commitment_villain=scenario.initial_commitment.villain,
        bet_size=scenario.bet_size,
        showdown=scenario.showdown,
        baseline_call_probability=distribution["call"],
        baseline_fold_probability=distribution["fold"],
        shift_amounts=list(scenario.shift_amounts) if scenario.shift_amounts else [],
        horizons=list(repeated.horizons) if (repeated and repeated.horizons) else [],
        discount=repeated.discount if repeated else 1.0,
    )


def single_hand_form_to_dict(form: SingleHandScenarioForm) -> dict:
    """Return the single-hand scenario dict represented by ``form``.

    The result always includes ``"format_version"`` and the
    ``baseline_hero_strategy`` / ``candidate_generation`` / ``repeated`` sections.
    A valid form yields a dict accepted by ``river_scenario_from_dict`` and
    ``build_river_steal_game_from_scenario``; an invalid form may not (use
    :func:`validate_single_hand_form` first).
    """

    return {
        # Emit the form's value as-is: a valid form yields "1", and an invalid
        # value (for example "") is preserved so it is not silently corrected and
        # is still caught by validate_single_hand_form and the JSON parser.
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake": {"rate": form.rake_rate, "cap": form.rake_cap},
        "initial_commitment": {
            "hero": form.initial_commitment_hero,
            "villain": form.initial_commitment_villain,
        },
        "bet_size": form.bet_size,
        "showdown": form.showdown,
        "baseline_hero_strategy": {
            _IP_INFO_SET: {
                "call": form.baseline_call_probability,
                "fold": form.baseline_fold_probability,
            }
        },
        "candidate_generation": {"shift_amounts": list(form.shift_amounts)},
        "repeated": {"horizons": list(form.horizons), "discount": form.discount},
    }


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_single_hand_form(
    form: SingleHandScenarioForm,
) -> List[FormValidationMessage]:
    """Return field-level validation messages for ``form`` (empty when valid).

    These checks mirror the single-hand rules of the JSON scenario format but
    return messages for display rather than raising, so a form being edited can
    be validated field by field. The authoritative check remains
    ``river_scenario_from_dict`` + ``build_river_steal_game_from_scenario`` on
    :func:`single_hand_form_to_dict`'s output.
    """

    messages: List[FormValidationMessage] = []

    def add(field_name: str, message: str) -> None:
        messages.append(FormValidationMessage(field_name, message))

    if not isinstance(form.scenario_id, str) or not form.scenario_id:
        add("scenario_id", "scenario_id must be a non-empty string")

    if not isinstance(form.description, str):
        add("description", "description must be a string (it may be empty)")

    if form.format_version not in SUPPORTED_FORMAT_VERSIONS:
        add(
            "format_version",
            f"format_version must be one of {list(SUPPORTED_FORMAT_VERSIONS)}",
        )

    if not _is_finite_number(form.rake_rate) or not 0.0 <= form.rake_rate <= 1.0:
        add("rake_rate", "rake_rate must be a number within [0, 1]")

    if form.rake_cap is not None and (
        not _is_finite_number(form.rake_cap) or form.rake_cap < 0
    ):
        add("rake_cap", "rake_cap must be empty or a non-negative number")

    if not _is_finite_number(form.initial_commitment_hero) or form.initial_commitment_hero < 0:
        add("initial_commitment_hero", "must be a non-negative number")
    if (
        not _is_finite_number(form.initial_commitment_villain)
        or form.initial_commitment_villain < 0
    ):
        add("initial_commitment_villain", "must be a non-negative number")

    if not _is_finite_number(form.bet_size) or form.bet_size <= 0:
        add("bet_size", "bet_size must be a positive number")

    if form.showdown not in _SHOWDOWN_RESULTS:
        add("showdown", f"showdown must be one of {list(_SHOWDOWN_RESULTS)}")

    call_ok = (
        _is_finite_number(form.baseline_call_probability)
        and form.baseline_call_probability >= 0
    )
    fold_ok = (
        _is_finite_number(form.baseline_fold_probability)
        and form.baseline_fold_probability >= 0
    )
    if not call_ok:
        add("baseline_call_probability", "must be a non-negative number")
    if not fold_ok:
        add("baseline_fold_probability", "must be a non-negative number")
    if call_ok and fold_ok:
        total = form.baseline_call_probability + form.baseline_fold_probability
        if abs(total - 1.0) > _TOLERANCE:
            add(
                "baseline_call_probability",
                f"baseline call/fold probabilities must sum to 1 (got {total})",
            )

    if not isinstance(form.shift_amounts, (list, tuple)) or len(form.shift_amounts) == 0:
        add("shift_amounts", "shift_amounts must contain at least one positive number")
    else:
        for index, amount in enumerate(form.shift_amounts):
            if not _is_finite_number(amount) or amount <= 0:
                add("shift_amounts", f"shift_amounts[{index}] must be a positive number")

    if not isinstance(form.horizons, (list, tuple)) or len(form.horizons) == 0:
        add("horizons", "horizons must contain at least one positive integer")
    else:
        for index, horizon in enumerate(form.horizons):
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                add("horizons", f"horizons[{index}] must be an integer >= 1")

    if not _is_finite_number(form.discount) or not 0.0 < form.discount <= 1.0:
        add("discount", "discount must satisfy 0 < discount <= 1")

    return messages
