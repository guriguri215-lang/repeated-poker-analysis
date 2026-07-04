"""Form-friendly intermediate representation for scenarios.

This is a deliberately small step towards a future GUI/form input layer (see
``docs/gui_input_design.md``). It is *not* a GUI and adds no new solver,
game-theory model, or analysis logic. It only offers flat dataclasses that are
convenient to bind to form fields, plus helpers to convert those forms to and
from the existing JSON scenario format and to surface field-level validation
messages for display.

Scope so far: single-hand mode (:class:`SingleHandScenarioForm`),
Hero-range-only mode (:class:`HeroRangeScenarioForm`), the discrete
showdown-matrix mode (:class:`ShowdownMatrixScenarioForm`), the equity-matrix
mode (:class:`EquityMatrixScenarioForm`), and the river betting-tree mode
(:class:`BettingTreeScenarioForm`). All five JSON scenario modes now have a form
model; the ``*_from_dict`` helpers each reject the other modes.

The optional top-level ``baseline_villain_strategy`` (an explicit baseline
Villain profile) is intentionally *not* modelled by these forms. Because the
forms would drop any field they do not carry on a round-trip, every
``*_from_dict`` helper rejects a scenario that contains it rather than losing it
silently; such scenarios must be edited directly in the JSON.

JSON stays the source of truth. The ``*_from_dict`` helpers reuse the existing
:func:`repeated_poker.scenario_io.river_scenario_from_dict` parser (so loading a
scenario into a form applies the same structural validation, with no duplicated
parsing), and the ``*_to_dict`` helpers produce a dict that the same parser and
``build_river_steal_game_from_scenario`` accept. The separate ``validate_*``
helpers return GUI-facing field-level messages instead of raising, so a form
being edited can be checked without throwing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


def _reject_baseline_villain_strategy(data: dict) -> None:
    """Reject a scenario dict that carries an explicit ``baseline_villain_strategy``.

    The form models (and their ``*_to_dict`` writers) do not carry the optional
    top-level ``baseline_villain_strategy`` field, so a form round-trip
    (``*_from_dict`` -> edit -> ``*_to_dict``) would silently drop it. Rather than
    lose that data quietly, the form layer rejects the field outright: the GUI is
    frozen and gains no control for it, and an explicit baseline Villain profile
    must be authored and kept in the JSON directly. JSON stays the source of
    truth (see ``docs/scenario_format_reference.md``).
    """

    if isinstance(data, dict) and "baseline_villain_strategy" in data:
        raise ValueError(
            "the scenario form does not support 'baseline_villain_strategy'; edit "
            "it directly in the JSON scenario (the form would otherwise drop it on "
            "save)"
        )


def _reject_multi_shift_generation(data: dict) -> None:
    """Reject a scenario whose ``candidate_generation`` requests multi-shift.

    The form models carry only ``candidate_generation.shift_amounts``, so a form
    round-trip would silently drop ``max_simultaneous_info_sets`` and revert the
    analysis to single-information-set shifts. Rather than lose that setting
    quietly, the form layer rejects it: the GUI is frozen and gains no control for
    it, and a ``max_simultaneous_info_sets = 2`` scenario must be authored and kept
    in the JSON directly.
    """

    if not isinstance(data, dict):
        return
    generation = data.get("candidate_generation")
    if isinstance(generation, dict) and "max_simultaneous_info_sets" in generation:
        raise ValueError(
            "the scenario form does not support "
            "'candidate_generation.max_simultaneous_info_sets'; edit it directly in "
            "the JSON scenario (the form would otherwise drop it on save)"
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
    _reject_baseline_villain_strategy(data)
    _reject_multi_shift_generation(data)
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


def _validate_baseline_probabilities(add, call_field, fold_field, call_value, fold_value):
    """Validate a call/fold distribution, appending messages via ``add``.

    Shared by the single-hand baseline and each Hero-range hand. ``call_field`` /
    ``fold_field`` are the GUI field names so messages point at the right input.
    """

    call_ok = _is_finite_number(call_value) and call_value >= 0
    fold_ok = _is_finite_number(fold_value) and fold_value >= 0
    if not call_ok:
        add(call_field, "must be a non-negative number")
    if not fold_ok:
        add(fold_field, "must be a non-negative number")
    if call_ok and fold_ok and abs((call_value + fold_value) - 1.0) > _TOLERANCE:
        add(
            call_field,
            f"baseline call/fold probabilities must sum to 1 (got {call_value + fold_value})",
        )


def _validate_common_fields(form, add) -> None:
    """Validate the top-level fields shared by the single-hand and range forms."""

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

    _validate_common_fields(form, add)

    if form.showdown not in _SHOWDOWN_RESULTS:
        add("showdown", f"showdown must be one of {list(_SHOWDOWN_RESULTS)}")

    _validate_baseline_probabilities(
        add,
        "baseline_call_probability",
        "baseline_fold_probability",
        form.baseline_call_probability,
        form.baseline_fold_probability,
    )

    return messages


# ---------------------------------------------------------------------------
# Hero-range-only mode
# ---------------------------------------------------------------------------

# Top-level keys that, together with a ``hero_range``, select matrix or
# betting-tree mode rather than Hero-range-only mode.
_NON_HERO_RANGE_KEYS = (
    "villain_range",
    "showdown_matrix",
    "equity_matrix",
    "betting_tree",
)


@dataclass
class HeroRangeHandForm:
    """A flat, form-friendly view of one weighted Hero bucket.

    The bucket's baseline ``call`` / ``fold`` distribution is split into two
    probability fields, mirroring :class:`SingleHandScenarioForm`.
    """

    hand_id: str = ""
    weight: float = 0.0
    showdown: str = CHOP
    baseline_call_probability: float = 0.0
    baseline_fold_probability: float = 1.0


@dataclass
class HeroRangeScenarioForm:
    """A flat, form-friendly view of a Hero-range-only river scenario.

    The fields map one-to-one to the Hero-range-only scenario JSON (see
    ``docs/scenario_format_reference.md``): the weighted Hero buckets become a
    list of :class:`HeroRangeHandForm`, and the rake, initial commitment,
    candidate-generation, and repeated-game values are flattened.
    """

    format_version: str = DEFAULT_FORMAT_VERSION
    scenario_id: str = ""
    description: str = ""
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    initial_commitment_hero: float = 0.0
    initial_commitment_villain: float = 0.0
    bet_size: float = 0.0
    hands: List[HeroRangeHandForm] = field(default_factory=list)
    shift_amounts: List[float] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    discount: float = 1.0


def hero_range_form_from_dict(data: dict) -> HeroRangeScenarioForm:
    """Build a :class:`HeroRangeScenarioForm` from a Hero-range-only scenario dict.

    Raises :class:`ValueError` if ``data`` is not a Hero-range-only scenario (it
    has no ``hero_range``, or it carries ``villain_range`` / ``showdown_matrix`` /
    ``equity_matrix`` / ``betting_tree``, or it mixes in single-hand fields) or
    fails the existing parser's validation (an unsupported baseline action, a bad
    ``showdown``, weights not summing to one, an unsupported ``format_version``,
    and so on).
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    _reject_baseline_villain_strategy(data)
    _reject_multi_shift_generation(data)
    if "hero_range" not in data:
        raise ValueError(
            "hero-range form requires a 'hero_range'; this is not a "
            "Hero-range-only scenario"
        )
    forbidden = [key for key in _NON_HERO_RANGE_KEYS if key in data]
    if forbidden:
        raise ValueError(
            "hero-range form supports Hero-range-only mode; remove "
            f"{sorted(forbidden)} (those select a matrix or betting-tree mode)"
        )

    # Reuse the existing parser for all structural validation and mode handling;
    # with a hero_range and none of the forbidden keys this yields a
    # Hero-range-only scenario (single-hand fields are rejected by the parser).
    scenario = river_scenario_from_dict(data)
    repeated = scenario.repeated
    hands = [
        HeroRangeHandForm(
            hand_id=hand.hand_id,
            weight=hand.weight,
            showdown=hand.showdown,
            baseline_call_probability=hand.baseline_strategy["call"],
            baseline_fold_probability=hand.baseline_strategy["fold"],
        )
        for hand in scenario.hero_range.hands
    ]
    return HeroRangeScenarioForm(
        format_version=scenario.format_version,
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        rake_rate=scenario.rake.rate,
        rake_cap=scenario.rake.cap,
        initial_commitment_hero=scenario.initial_commitment.hero,
        initial_commitment_villain=scenario.initial_commitment.villain,
        bet_size=scenario.bet_size,
        hands=hands,
        shift_amounts=list(scenario.shift_amounts) if scenario.shift_amounts else [],
        horizons=list(repeated.horizons) if (repeated and repeated.horizons) else [],
        discount=repeated.discount if repeated else 1.0,
    )


def hero_range_form_to_dict(form: HeroRangeScenarioForm) -> dict:
    """Return the Hero-range-only scenario dict represented by ``form``.

    The result always includes ``"format_version"`` (emitted as-is, like
    :func:`single_hand_form_to_dict`) and the ``hero_range`` /
    ``candidate_generation`` / ``repeated`` sections. A valid form yields a dict
    accepted by ``river_scenario_from_dict`` and
    ``build_river_steal_game_from_scenario``; an invalid form may not (use
    :func:`validate_hero_range_form` first).
    """

    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake": {"rate": form.rake_rate, "cap": form.rake_cap},
        "initial_commitment": {
            "hero": form.initial_commitment_hero,
            "villain": form.initial_commitment_villain,
        },
        "bet_size": form.bet_size,
        "hero_range": [
            {
                "hand_id": hand.hand_id,
                "weight": hand.weight,
                "showdown": hand.showdown,
                "baseline_strategy": {
                    "call": hand.baseline_call_probability,
                    "fold": hand.baseline_fold_probability,
                },
            }
            for hand in form.hands
        ],
        "candidate_generation": {"shift_amounts": list(form.shift_amounts)},
        "repeated": {"horizons": list(form.horizons), "discount": form.discount},
    }


def validate_hero_range_form(
    form: HeroRangeScenarioForm,
) -> List[FormValidationMessage]:
    """Return field-level validation messages for ``form`` (empty when valid).

    Mirrors the Hero-range-only rules of the JSON scenario format with per-hand,
    GUI-friendly field names (for example ``hands[0].hand_id`` and
    ``hands[1].weight``). The authoritative check remains
    ``river_scenario_from_dict`` + ``build_river_steal_game_from_scenario`` on
    :func:`hero_range_form_to_dict`'s output.
    """

    messages: List[FormValidationMessage] = []

    def add(field_name: str, message: str) -> None:
        messages.append(FormValidationMessage(field_name, message))

    _validate_common_fields(form, add)

    if not isinstance(form.hands, (list, tuple)) or len(form.hands) == 0:
        add("hands", "at least one hero hand is required")
        return messages

    seen_ids: set = set()
    valid_weights: List[float] = []
    for index, hand in enumerate(form.hands):
        prefix = f"hands[{index}]"

        # A form being edited may hold a malformed entry (None, a dict, etc.).
        # Report it as a field-level error and skip its detailed checks rather
        # than raising AttributeError.
        if not isinstance(hand, HeroRangeHandForm):
            add(prefix, "hand entry must be a HeroRangeHandForm")
            continue

        if not isinstance(hand.hand_id, str) or not hand.hand_id:
            add(f"{prefix}.hand_id", "hand_id must be a non-empty string")
        elif hand.hand_id in seen_ids:
            add(f"{prefix}.hand_id", f"duplicate hand_id {hand.hand_id!r}")
        else:
            seen_ids.add(hand.hand_id)

        if not _is_finite_number(hand.weight) or hand.weight <= 0:
            add(f"{prefix}.weight", "weight must be a positive number")
        else:
            valid_weights.append(hand.weight)

        if hand.showdown not in _SHOWDOWN_RESULTS:
            add(f"{prefix}.showdown", f"showdown must be one of {list(_SHOWDOWN_RESULTS)}")

        _validate_baseline_probabilities(
            add,
            f"{prefix}.baseline_call_probability",
            f"{prefix}.baseline_fold_probability",
            hand.baseline_call_probability,
            hand.baseline_fold_probability,
        )

    # Only check the weight sum when every weight is individually valid, so the
    # message is not noise on top of per-hand weight errors.
    if len(valid_weights) == len(form.hands):
        total = sum(valid_weights)
        if abs(total - 1.0) > _TOLERANCE:
            add("hands", f"hand weights must sum to 1 (got {total})")

    return messages


# ---------------------------------------------------------------------------
# Showdown-matrix mode
# ---------------------------------------------------------------------------

# Top-level keys that, alongside a ``villain_range`` + ``showdown_matrix``, select
# a different matrix flavour (equity) or the betting-tree model rather than the
# discrete showdown-matrix mode handled here.
_NON_SHOWDOWN_MATRIX_KEYS = (
    "equity_matrix",
    "betting_tree",
)


@dataclass
class HeroMatrixBucketForm:
    """A flat, form-friendly view of one weighted Hero bucket in matrix mode.

    Unlike :class:`HeroRangeHandForm` there is no per-hand ``showdown``: in matrix
    mode the outcome of every Hero/Villain pairing comes from the showdown matrix,
    not the bucket. The bucket's baseline ``call`` / ``fold`` distribution is split
    into two probability fields.
    """

    hand_id: str = ""
    weight: float = 0.0
    baseline_call_probability: float = 0.0
    baseline_fold_probability: float = 1.0


@dataclass
class VillainMatrixBucketForm:
    """A flat, form-friendly view of one weighted Villain bucket in matrix mode.

    Villain buckets carry no baseline strategy in v1: the baseline Villain policy
    is derived as the best response when the game is built, so a Villain bucket is
    just a weighted id used as a matrix column.
    """

    hand_id: str = ""
    weight: float = 0.0


@dataclass
class ShowdownMatrixScenarioForm:
    """A flat, form-friendly view of a discrete showdown-matrix river scenario.

    The fields map one-to-one to the showdown-matrix scenario JSON (see
    ``docs/scenario_format_reference.md``): the weighted Hero and Villain buckets
    become lists of :class:`HeroMatrixBucketForm` / :class:`VillainMatrixBucketForm`,
    and ``showdown_matrix`` is the Hero x Villain grid keyed by
    ``[hero_id][villain_id]`` with each cell one of ``"hero"`` / ``"villain"`` /
    ``"chop"``. The rake, initial commitment, candidate-generation, and
    repeated-game values are flattened.
    """

    format_version: str = DEFAULT_FORMAT_VERSION
    scenario_id: str = ""
    description: str = ""
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    initial_commitment_hero: float = 0.0
    initial_commitment_villain: float = 0.0
    bet_size: float = 0.0
    hero_buckets: List[HeroMatrixBucketForm] = field(default_factory=list)
    villain_buckets: List[VillainMatrixBucketForm] = field(default_factory=list)
    showdown_matrix: Dict[str, Dict[str, str]] = field(default_factory=dict)
    shift_amounts: List[float] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    discount: float = 1.0


def showdown_matrix_form_from_dict(data: dict) -> ShowdownMatrixScenarioForm:
    """Build a :class:`ShowdownMatrixScenarioForm` from a showdown-matrix scenario.

    Raises :class:`ValueError` if ``data`` is not a discrete showdown-matrix
    scenario (it lacks ``villain_range`` / ``showdown_matrix``, so it is a
    single-hand or Hero-range-only scenario; or it carries an ``equity_matrix`` or
    ``betting_tree``, selecting a different model) or fails the existing parser's
    validation (an unsupported baseline action, weights not summing to one,
    overlapping Hero/Villain ids, an incomplete matrix, an unsupported
    ``format_version``, and so on).
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    _reject_baseline_villain_strategy(data)
    _reject_multi_shift_generation(data)
    if "villain_range" not in data or "showdown_matrix" not in data:
        raise ValueError(
            "showdown-matrix form requires both 'villain_range' and "
            "'showdown_matrix'; this is not a showdown-matrix scenario"
        )
    forbidden = [key for key in _NON_SHOWDOWN_MATRIX_KEYS if key in data]
    if forbidden:
        raise ValueError(
            "showdown-matrix form supports the discrete showdown-matrix mode; "
            f"remove {sorted(forbidden)} (those select the equity-matrix or "
            "betting-tree model)"
        )

    # Reuse the existing parser for all structural validation and mode handling;
    # with a villain_range + showdown_matrix and none of the forbidden keys this
    # yields a discrete showdown-matrix scenario (the parser also requires
    # hero_range, rejects per-hand showdown, and enforces matrix completeness and
    # disjoint Hero/Villain ids).
    scenario = river_scenario_from_dict(data)
    repeated = scenario.repeated
    hero_buckets = [
        HeroMatrixBucketForm(
            hand_id=hand.hand_id,
            weight=hand.weight,
            baseline_call_probability=hand.baseline_strategy["call"],
            baseline_fold_probability=hand.baseline_strategy["fold"],
        )
        for hand in scenario.hero_range.hands
    ]
    villain_buckets = [
        VillainMatrixBucketForm(hand_id=hand.hand_id, weight=hand.weight)
        for hand in scenario.villain_range.hands
    ]
    showdown_matrix = {
        hero_id: dict(row) for hero_id, row in scenario.showdown_matrix.items()
    }
    return ShowdownMatrixScenarioForm(
        format_version=scenario.format_version,
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        rake_rate=scenario.rake.rate,
        rake_cap=scenario.rake.cap,
        initial_commitment_hero=scenario.initial_commitment.hero,
        initial_commitment_villain=scenario.initial_commitment.villain,
        bet_size=scenario.bet_size,
        hero_buckets=hero_buckets,
        villain_buckets=villain_buckets,
        showdown_matrix=showdown_matrix,
        shift_amounts=list(scenario.shift_amounts) if scenario.shift_amounts else [],
        horizons=list(repeated.horizons) if (repeated and repeated.horizons) else [],
        discount=repeated.discount if repeated else 1.0,
    )


def showdown_matrix_form_to_dict(form: ShowdownMatrixScenarioForm) -> dict:
    """Return the showdown-matrix scenario dict represented by ``form``.

    The result always includes ``"format_version"`` (emitted as-is, like
    :func:`single_hand_form_to_dict`) and the ``hero_range`` / ``villain_range`` /
    ``showdown_matrix`` / ``candidate_generation`` / ``repeated`` sections. Hero
    buckets carry no per-hand ``showdown`` (matrix mode forbids it). A valid form
    yields a dict accepted by ``river_scenario_from_dict`` and
    ``build_river_steal_game_from_scenario``; an invalid form may not (use
    :func:`validate_showdown_matrix_form` first).
    """

    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake": {"rate": form.rake_rate, "cap": form.rake_cap},
        "initial_commitment": {
            "hero": form.initial_commitment_hero,
            "villain": form.initial_commitment_villain,
        },
        "bet_size": form.bet_size,
        "hero_range": [
            {
                "hand_id": bucket.hand_id,
                "weight": bucket.weight,
                "baseline_strategy": {
                    "call": bucket.baseline_call_probability,
                    "fold": bucket.baseline_fold_probability,
                },
            }
            for bucket in form.hero_buckets
        ],
        "villain_range": [
            {"hand_id": bucket.hand_id, "weight": bucket.weight}
            for bucket in form.villain_buckets
        ],
        "showdown_matrix": {
            hero_id: dict(row) for hero_id, row in form.showdown_matrix.items()
        },
        "candidate_generation": {"shift_amounts": list(form.shift_amounts)},
        "repeated": {"horizons": list(form.horizons), "discount": form.discount},
    }


def _validate_matrix_bucket_ids_and_weights(buckets, prefix, expected_type, add):
    """Validate the shared id/weight rules for a list of matrix buckets.

    Appends messages via ``add`` for malformed entries, bad / duplicate ids, and
    non-positive weights, and (only when every entry is well typed with an
    individually valid weight) for a weight sum that is not one. Returns
    ``(ids, ids_usable)`` where ``ids`` is the list of valid, unique hand ids
    seen and ``ids_usable`` is true only when every entry is an ``expected_type``
    with a non-empty, unique hand id and an individually valid weight -- i.e. the
    id set can be trusted as the expected Hero/Villain axis for matrix grid
    validation, so a downstream grid check is not run against an unreliable axis.
    """

    ids: List[str] = []
    valid_weights: List[float] = []
    for index, bucket in enumerate(buckets):
        item = f"{prefix}[{index}]"

        # A form being edited may hold a malformed entry (None, a dict, etc.).
        # Report it as a field-level error and skip its detailed checks rather
        # than raising AttributeError.
        if not isinstance(bucket, expected_type):
            add(item, f"bucket entry must be a {expected_type.__name__}")
            continue

        if not isinstance(bucket.hand_id, str) or not bucket.hand_id:
            add(f"{item}.hand_id", "hand_id must be a non-empty string")
        elif bucket.hand_id in ids:
            add(f"{item}.hand_id", f"duplicate hand_id {bucket.hand_id!r}")
        else:
            ids.append(bucket.hand_id)

        if not _is_finite_number(bucket.weight) or bucket.weight <= 0:
            add(f"{item}.weight", "weight must be a positive number")
        else:
            valid_weights.append(bucket.weight)

    # Every well-typed bucket with a valid, unique id appends exactly one id (and
    # one weight); malformed / bad-id / duplicate / bad-weight entries do not. So
    # these equalities hold only when every entry passed those checks.
    ids_usable = len(ids) == len(buckets) and len(valid_weights) == len(buckets)

    # Only check the weight sum when every weight is individually valid, so the
    # message is not noise on top of per-bucket weight errors.
    if len(valid_weights) == len(buckets):
        total = sum(valid_weights)
        if abs(total - 1.0) > _TOLERANCE:
            add(prefix, f"{prefix} weights must sum to 1 (got {total})")

    return ids, ids_usable


def _sorted_for_display(values) -> list:
    """Sort ``values`` for stable display, tolerating non-comparable types.

    A form being edited may hold a matrix dict with mixed key types (for example
    ``1`` and ``None`` alongside strings), which a plain ``sorted`` cannot order
    and would raise ``TypeError`` on. Sorting by ``repr`` keeps the output stable
    without changing the keys themselves -- it is only for message ordering.
    """

    return sorted(values, key=repr)


def _validate_matrix_grid(matrix, hero_ids, villain_ids, matrix_field, cell_error, add) -> None:
    """Validate that ``matrix`` covers exactly the Hero x Villain id grid.

    Shared by the showdown and equity matrix validators. Checks the matrix is a
    mapping, every Hero bucket has a row, every row has a cell for every Villain
    bucket, there are no unknown Hero / Villain ids, and every cell value passes
    ``cell_error`` (which returns an error message for a bad value, or ``None``).
    ``matrix_field`` is the GUI field name (``"showdown_matrix"`` /
    ``"equity_matrix"``) so messages point at the matrix, a row, or a single cell.
    Only valid, unique ids from the bucket lists are used so a bad row/cell is not
    double-reported on top of a bad bucket id.

    Unknown keys come from the (possibly broken) form matrix, so their order is
    computed with :func:`_sorted_for_display` to stay exception-free even when the
    matrix mixes non-comparable key types -- the validator must return messages,
    not raise.
    """

    if not isinstance(matrix, dict):
        add(matrix_field, f"{matrix_field} must be an object keyed by hero hand id")
        return

    hero_set = set(hero_ids)
    villain_set = set(villain_ids)
    matrix_hero_ids = set(matrix)

    for missing in _sorted_for_display(hero_set - matrix_hero_ids):
        add(matrix_field, f"missing row for hero id {missing!r}")
    for unknown in _sorted_for_display(matrix_hero_ids - hero_set):
        add(matrix_field, f"unknown hero id {unknown!r}")

    for hero_id in hero_ids:
        if hero_id not in matrix:
            continue
        row = matrix[hero_id]
        row_field = f"{matrix_field}[{hero_id}]"
        if not isinstance(row, dict):
            add(row_field, "row must be an object keyed by villain hand id")
            continue
        row_ids = set(row)
        for missing in _sorted_for_display(villain_set - row_ids):
            add(row_field, f"missing cell for villain id {missing!r}")
        for unknown in _sorted_for_display(row_ids - villain_set):
            add(row_field, f"unknown villain id {unknown!r}")
        for villain_id in villain_ids:
            if villain_id not in row:
                continue
            error = cell_error(row[villain_id])
            if error is not None:
                add(f"{matrix_field}[{hero_id}][{villain_id}]", error)


def _showdown_cell_error(value) -> Optional[str]:
    """Return an error message for a bad showdown cell, or ``None`` if valid."""

    if value not in _SHOWDOWN_RESULTS:
        return f"cell must be one of {list(_SHOWDOWN_RESULTS)}"
    return None


def _equity_cell_error(value) -> Optional[str]:
    """Return an error message for a bad equity cell, or ``None`` if valid.

    An equity cell is the Hero pot share before rake: a finite number (``bool``
    excluded) within ``[0, 1]``.
    """

    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        return "cell must be a number within [0, 1] (Hero pot share before rake)"
    return None


def _validate_matrix_form(
    form, matrix, matrix_field, cell_error
) -> List[FormValidationMessage]:
    """Return field-level messages shared by the matrix-mode form validators.

    Validates the common top-level fields, the Hero buckets (id / weight /
    baseline call-fold split) and Villain buckets (id / weight), the
    disjoint-id rule, and the matrix grid via :func:`_validate_matrix_grid` with
    ``matrix_field`` / ``cell_error``. The grid is only checked once both id axes
    are trustworthy and disjoint, so a broken bucket id does not produce
    misleading missing / unknown row/cell noise.
    """

    messages: List[FormValidationMessage] = []

    def add(field_name: str, message: str) -> None:
        messages.append(FormValidationMessage(field_name, message))

    _validate_common_fields(form, add)

    hero_ok = isinstance(form.hero_buckets, (list, tuple)) and len(form.hero_buckets) > 0
    villain_ok = (
        isinstance(form.villain_buckets, (list, tuple)) and len(form.villain_buckets) > 0
    )
    if not hero_ok:
        add("hero_buckets", "at least one hero bucket is required")
    if not villain_ok:
        add("villain_buckets", "at least one villain bucket is required")

    hero_ids: List[str] = []
    villain_ids: List[str] = []
    hero_ids_usable = False
    villain_ids_usable = False
    if hero_ok:
        hero_ids, hero_ids_usable = _validate_matrix_bucket_ids_and_weights(
            form.hero_buckets, "hero_buckets", HeroMatrixBucketForm, add
        )
        for index, bucket in enumerate(form.hero_buckets):
            if not isinstance(bucket, HeroMatrixBucketForm):
                continue
            _validate_baseline_probabilities(
                add,
                f"hero_buckets[{index}].baseline_call_probability",
                f"hero_buckets[{index}].baseline_fold_probability",
                bucket.baseline_call_probability,
                bucket.baseline_fold_probability,
            )
    if villain_ok:
        villain_ids, villain_ids_usable = _validate_matrix_bucket_ids_and_weights(
            form.villain_buckets, "villain_buckets", VillainMatrixBucketForm, add
        )

    # The parser keeps Hero/Villain ids disjoint so the matrix keys and any future
    # cross-references stay unambiguous; mirror that here on the valid ids.
    shared_ids = set(hero_ids) & set(villain_ids)
    for shared in sorted(shared_ids):
        add(
            "villain_buckets",
            f"hand_id {shared!r} is used by both a hero and a villain bucket; "
            "ids must be disjoint",
        )

    # Only validate the matrix grid once both id axes are trustworthy: every
    # bucket well typed with a valid, unique id, and the two axes disjoint.
    # Otherwise the expected row/column sets are unreliable, so a grid check would
    # emit misleading missing / unknown row/cell noise on top of the real bucket
    # errors.
    if hero_ids_usable and villain_ids_usable and not shared_ids:
        _validate_matrix_grid(matrix, hero_ids, villain_ids, matrix_field, cell_error, add)

    return messages


def validate_showdown_matrix_form(
    form: ShowdownMatrixScenarioForm,
) -> List[FormValidationMessage]:
    """Return field-level validation messages for ``form`` (empty when valid).

    Mirrors the discrete showdown-matrix rules of the JSON scenario format with
    GUI-friendly field names (for example ``hero_buckets[0].hand_id``,
    ``villain_buckets[1].weight``, and ``showdown_matrix[hero_id][villain_id]``).
    The authoritative check remains ``river_scenario_from_dict`` +
    ``build_river_steal_game_from_scenario`` on
    :func:`showdown_matrix_form_to_dict`'s output.
    """

    return _validate_matrix_form(
        form, form.showdown_matrix, "showdown_matrix", _showdown_cell_error
    )


# ---------------------------------------------------------------------------
# Equity-matrix mode
# ---------------------------------------------------------------------------

# Top-level keys that, alongside a ``villain_range`` + ``equity_matrix``, select
# the discrete showdown-matrix flavour or the betting-tree model rather than the
# equity-matrix mode handled here.
_NON_EQUITY_MATRIX_KEYS = (
    "showdown_matrix",
    "betting_tree",
)


@dataclass
class EquityMatrixScenarioForm:
    """A flat, form-friendly view of an equity-matrix river scenario.

    The fields map one-to-one to the equity-matrix scenario JSON (see
    ``docs/scenario_format_reference.md``): the weighted Hero and Villain buckets
    reuse :class:`HeroMatrixBucketForm` / :class:`VillainMatrixBucketForm`, and
    ``equity_matrix`` is the Hero x Villain grid keyed by ``[hero_id][villain_id]``
    with each cell the Hero pot share before rake -- a finite number in ``[0, 1]``
    (``1.0`` = Hero wins, ``0.5`` = chop, ``0.0`` = Villain wins). The rake,
    initial commitment, candidate-generation, and repeated-game values are
    flattened. This is the equity flavour of :class:`ShowdownMatrixScenarioForm`.
    """

    format_version: str = DEFAULT_FORMAT_VERSION
    scenario_id: str = ""
    description: str = ""
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    initial_commitment_hero: float = 0.0
    initial_commitment_villain: float = 0.0
    bet_size: float = 0.0
    hero_buckets: List[HeroMatrixBucketForm] = field(default_factory=list)
    villain_buckets: List[VillainMatrixBucketForm] = field(default_factory=list)
    equity_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    shift_amounts: List[float] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    discount: float = 1.0


def equity_matrix_form_from_dict(data: dict) -> EquityMatrixScenarioForm:
    """Build an :class:`EquityMatrixScenarioForm` from an equity-matrix scenario.

    Raises :class:`ValueError` if ``data`` is not an equity-matrix scenario (it
    lacks ``villain_range`` / ``equity_matrix``, so it is a single-hand or
    Hero-range-only scenario; or it carries a ``showdown_matrix`` or
    ``betting_tree``, selecting a different model) or fails the existing parser's
    validation (an unsupported baseline action, weights not summing to one,
    overlapping Hero/Villain ids, an incomplete matrix, an equity outside
    ``[0, 1]``, an unsupported ``format_version``, and so on).
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    _reject_baseline_villain_strategy(data)
    _reject_multi_shift_generation(data)
    if "villain_range" not in data or "equity_matrix" not in data:
        raise ValueError(
            "equity-matrix form requires both 'villain_range' and "
            "'equity_matrix'; this is not an equity-matrix scenario"
        )
    forbidden = [key for key in _NON_EQUITY_MATRIX_KEYS if key in data]
    if forbidden:
        raise ValueError(
            "equity-matrix form supports the equity-matrix mode; "
            f"remove {sorted(forbidden)} (those select the showdown-matrix or "
            "betting-tree model)"
        )

    # Reuse the existing parser for all structural validation and mode handling;
    # with a villain_range + equity_matrix and none of the forbidden keys this
    # yields an equity-matrix scenario (the parser also requires hero_range,
    # rejects per-hand showdown, enforces matrix completeness and equities in
    # [0, 1], and keeps Hero/Villain ids disjoint).
    scenario = river_scenario_from_dict(data)
    repeated = scenario.repeated
    hero_buckets = [
        HeroMatrixBucketForm(
            hand_id=hand.hand_id,
            weight=hand.weight,
            baseline_call_probability=hand.baseline_strategy["call"],
            baseline_fold_probability=hand.baseline_strategy["fold"],
        )
        for hand in scenario.hero_range.hands
    ]
    villain_buckets = [
        VillainMatrixBucketForm(hand_id=hand.hand_id, weight=hand.weight)
        for hand in scenario.villain_range.hands
    ]
    equity_matrix = {
        hero_id: dict(row) for hero_id, row in scenario.equity_matrix.items()
    }
    return EquityMatrixScenarioForm(
        format_version=scenario.format_version,
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        rake_rate=scenario.rake.rate,
        rake_cap=scenario.rake.cap,
        initial_commitment_hero=scenario.initial_commitment.hero,
        initial_commitment_villain=scenario.initial_commitment.villain,
        bet_size=scenario.bet_size,
        hero_buckets=hero_buckets,
        villain_buckets=villain_buckets,
        equity_matrix=equity_matrix,
        shift_amounts=list(scenario.shift_amounts) if scenario.shift_amounts else [],
        horizons=list(repeated.horizons) if (repeated and repeated.horizons) else [],
        discount=repeated.discount if repeated else 1.0,
    )


def equity_matrix_form_to_dict(form: EquityMatrixScenarioForm) -> dict:
    """Return the equity-matrix scenario dict represented by ``form``.

    The result always includes ``"format_version"`` (emitted as-is, like
    :func:`single_hand_form_to_dict`) and the ``hero_range`` / ``villain_range`` /
    ``equity_matrix`` / ``candidate_generation`` / ``repeated`` sections. Hero
    buckets carry no per-hand ``showdown`` (matrix mode forbids it). A valid form
    yields a dict accepted by ``river_scenario_from_dict`` and
    ``build_river_steal_game_from_scenario``; an invalid form may not (use
    :func:`validate_equity_matrix_form` first).
    """

    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake": {"rate": form.rake_rate, "cap": form.rake_cap},
        "initial_commitment": {
            "hero": form.initial_commitment_hero,
            "villain": form.initial_commitment_villain,
        },
        "bet_size": form.bet_size,
        "hero_range": [
            {
                "hand_id": bucket.hand_id,
                "weight": bucket.weight,
                "baseline_strategy": {
                    "call": bucket.baseline_call_probability,
                    "fold": bucket.baseline_fold_probability,
                },
            }
            for bucket in form.hero_buckets
        ],
        "villain_range": [
            {"hand_id": bucket.hand_id, "weight": bucket.weight}
            for bucket in form.villain_buckets
        ],
        "equity_matrix": {
            hero_id: dict(row) for hero_id, row in form.equity_matrix.items()
        },
        "candidate_generation": {"shift_amounts": list(form.shift_amounts)},
        "repeated": {"horizons": list(form.horizons), "discount": form.discount},
    }


def validate_equity_matrix_form(
    form: EquityMatrixScenarioForm,
) -> List[FormValidationMessage]:
    """Return field-level validation messages for ``form`` (empty when valid).

    Mirrors the equity-matrix rules of the JSON scenario format with GUI-friendly
    field names (for example ``hero_buckets[0].hand_id``,
    ``villain_buckets[1].weight``, and ``equity_matrix[hero_id][villain_id]``).
    Only the matrix cell rule differs from
    :func:`validate_showdown_matrix_form`: each cell must be a finite number in
    ``[0, 1]``. The authoritative check remains ``river_scenario_from_dict`` +
    ``build_river_steal_game_from_scenario`` on
    :func:`equity_matrix_form_to_dict`'s output.
    """

    return _validate_matrix_form(
        form, form.equity_matrix, "equity_matrix", _equity_cell_error
    )


# ---------------------------------------------------------------------------
# Betting-tree mode
# ---------------------------------------------------------------------------

# Betting-tree mode is the matrix mode (hero_range + villain_range + one matrix)
# plus a ``betting_tree``. It supports either a discrete ``showdown_matrix`` or an
# ``equity_matrix``; the form tracks which via ``matrix_type``.
_BETTING_TREE_MATRIX_TYPES = ("showdown", "equity")
# Hero betting-tree decision points (see river_betting_tree.py / the format
# reference): the two distributions the form splits into flat probability fields.
_AFTER_OOP_CHECK_ACTIONS = ("check", "bet")
_VS_OOP_BET_ACTIONS = ("call", "fold", "raise")


@dataclass
class BettingTreeSizingForm:
    """The three river betting-tree sizes (betting-tree mode only).

    Mirrors the JSON ``betting_tree`` object. ``ip_raise_size`` is the *total*
    chips each player commits once IP's raise is called (not the raise
    increment); it must exceed ``oop_bet_size``.
    """

    oop_bet_size: float = 0.0
    ip_bet_after_check_size: float = 0.0
    ip_raise_size: float = 0.0


@dataclass
class HeroBettingTreeBucketForm:
    """A flat, form-friendly view of one weighted Hero bucket in betting-tree mode.

    Like :class:`HeroMatrixBucketForm` there is no per-hand ``showdown`` (the
    outcome comes from the matrix). Instead of a single call/fold baseline, a
    betting-tree bucket has two Hero decision points, each split into flat
    probability fields:

    * ``after_oop_check`` (IP acts after an OOP check): ``check`` / ``bet``;
    * ``vs_oop_bet`` (IP faces an OOP bet): ``call`` / ``fold`` / ``raise``.
    """

    hand_id: str = ""
    weight: float = 0.0
    after_oop_check_check_probability: float = 0.0
    after_oop_check_bet_probability: float = 1.0
    vs_oop_bet_call_probability: float = 0.0
    vs_oop_bet_fold_probability: float = 1.0
    vs_oop_bet_raise_probability: float = 0.0


@dataclass
class BettingTreeScenarioForm:
    """A flat, form-friendly view of a river betting-tree scenario.

    The fields map to the betting-tree scenario JSON (see
    ``docs/scenario_format_reference.md``): the weighted Hero buckets are
    :class:`HeroBettingTreeBucketForm` (two decision points), the Villain buckets
    reuse :class:`VillainMatrixBucketForm`, ``betting_tree`` carries the three
    sizes, and the matchup outcomes come from ``matrix`` keyed by
    ``[hero_id][villain_id]``. ``matrix_type`` selects how a cell is read:
    ``"showdown"`` (discrete ``hero`` / ``villain`` / ``chop``) or ``"equity"``
    (Hero pot share before rake in ``[0, 1]``).

    ``bet_size`` mirrors the simple-tree top-level field; in betting-tree mode it
    must equal ``betting_tree.oop_bet_size`` (the JSON allows it to be omitted and
    default to that, but the form keeps it explicit and validates the match).
    """

    format_version: str = DEFAULT_FORMAT_VERSION
    scenario_id: str = ""
    description: str = ""
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    initial_commitment_hero: float = 0.0
    initial_commitment_villain: float = 0.0
    bet_size: float = 0.0
    betting_tree: BettingTreeSizingForm = field(default_factory=BettingTreeSizingForm)
    matrix_type: str = "showdown"
    matrix: Dict[str, Dict[str, object]] = field(default_factory=dict)
    hero_buckets: List[HeroBettingTreeBucketForm] = field(default_factory=list)
    villain_buckets: List[VillainMatrixBucketForm] = field(default_factory=list)
    shift_amounts: List[float] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    discount: float = 1.0


def betting_tree_form_from_dict(data: dict) -> BettingTreeScenarioForm:
    """Build a :class:`BettingTreeScenarioForm` from a betting-tree scenario dict.

    Raises :class:`ValueError` if ``data`` is not a betting-tree scenario (it has
    no ``betting_tree``, so it is a single-hand / Hero-range-only / showdown- or
    equity-matrix scenario) or fails the existing parser's validation (a
    ``betting_tree`` outside matrix mode, an unsupported baseline action, bad
    sizes, an incomplete matrix, an unsupported ``format_version``, and so on).
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    _reject_baseline_villain_strategy(data)
    _reject_multi_shift_generation(data)
    if "betting_tree" not in data:
        raise ValueError(
            "betting-tree form requires a 'betting_tree'; this is not a "
            "betting-tree scenario"
        )

    # Reuse the existing parser for all structural validation and mode handling.
    # The parser only accepts a betting_tree in matrix mode (hero_range +
    # villain_range + exactly one matrix, each Hero hand carrying
    # baseline_strategies), so a non-matrix betting_tree raises here.
    scenario = river_scenario_from_dict(data)
    betting = scenario.betting_tree
    use_equity = scenario.equity_matrix is not None
    matrix = scenario.equity_matrix if use_equity else scenario.showdown_matrix
    repeated = scenario.repeated

    hero_buckets = []
    for hand in scenario.hero_range.hands:
        after_check = hand.baseline_strategies["after_oop_check"]
        vs_bet = hand.baseline_strategies["vs_oop_bet"]
        hero_buckets.append(
            HeroBettingTreeBucketForm(
                hand_id=hand.hand_id,
                weight=hand.weight,
                after_oop_check_check_probability=after_check["check"],
                after_oop_check_bet_probability=after_check["bet"],
                vs_oop_bet_call_probability=vs_bet["call"],
                vs_oop_bet_fold_probability=vs_bet["fold"],
                vs_oop_bet_raise_probability=vs_bet["raise"],
            )
        )
    villain_buckets = [
        VillainMatrixBucketForm(hand_id=hand.hand_id, weight=hand.weight)
        for hand in scenario.villain_range.hands
    ]
    return BettingTreeScenarioForm(
        format_version=scenario.format_version,
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        rake_rate=scenario.rake.rate,
        rake_cap=scenario.rake.cap,
        initial_commitment_hero=scenario.initial_commitment.hero,
        initial_commitment_villain=scenario.initial_commitment.villain,
        bet_size=scenario.bet_size,
        betting_tree=BettingTreeSizingForm(
            oop_bet_size=betting.oop_bet_size,
            ip_bet_after_check_size=betting.ip_bet_after_check_size,
            ip_raise_size=betting.ip_raise_size,
        ),
        matrix_type="equity" if use_equity else "showdown",
        matrix={hero_id: dict(row) for hero_id, row in matrix.items()},
        hero_buckets=hero_buckets,
        villain_buckets=villain_buckets,
        shift_amounts=list(scenario.shift_amounts) if scenario.shift_amounts else [],
        horizons=list(repeated.horizons) if (repeated and repeated.horizons) else [],
        discount=repeated.discount if repeated else 1.0,
    )


def betting_tree_form_to_dict(form: BettingTreeScenarioForm) -> dict:
    """Return the betting-tree scenario dict represented by ``form``.

    The result always includes ``"format_version"`` (emitted as-is, like
    :func:`single_hand_form_to_dict`), the ``betting_tree`` sizes, the
    ``hero_range`` with per-hand ``baseline_strategies`` (not the simple
    ``baseline_strategy``), the ``villain_range``, and one matrix under
    ``"showdown_matrix"`` or ``"equity_matrix"`` chosen by ``matrix_type``. A
    valid form yields a dict accepted by ``river_scenario_from_dict`` and
    ``build_river_steal_game_from_scenario``; an invalid form may not (use
    :func:`validate_betting_tree_form` first).

    Unlike ``format_version`` (a passthrough value), ``matrix_type`` selects which
    matrix key is emitted, so an unrecognised value cannot be passed through: it
    raises :class:`ValueError` rather than being silently coerced to
    ``showdown_matrix`` (which would hide the invalid form behind a parseable
    dict).
    """

    if form.matrix_type == "equity":
        matrix_key = "equity_matrix"
    elif form.matrix_type == "showdown":
        matrix_key = "showdown_matrix"
    else:
        raise ValueError(
            f"matrix_type must be one of {list(_BETTING_TREE_MATRIX_TYPES)}, "
            f"got {form.matrix_type!r}"
        )
    sizing = form.betting_tree
    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake": {"rate": form.rake_rate, "cap": form.rake_cap},
        "initial_commitment": {
            "hero": form.initial_commitment_hero,
            "villain": form.initial_commitment_villain,
        },
        "bet_size": form.bet_size,
        "betting_tree": {
            "oop_bet_size": sizing.oop_bet_size,
            "ip_bet_after_check_size": sizing.ip_bet_after_check_size,
            "ip_raise_size": sizing.ip_raise_size,
        },
        "hero_range": [
            {
                "hand_id": bucket.hand_id,
                "weight": bucket.weight,
                "baseline_strategies": {
                    "after_oop_check": {
                        "check": bucket.after_oop_check_check_probability,
                        "bet": bucket.after_oop_check_bet_probability,
                    },
                    "vs_oop_bet": {
                        "call": bucket.vs_oop_bet_call_probability,
                        "fold": bucket.vs_oop_bet_fold_probability,
                        "raise": bucket.vs_oop_bet_raise_probability,
                    },
                },
            }
            for bucket in form.hero_buckets
        ],
        "villain_range": [
            {"hand_id": bucket.hand_id, "weight": bucket.weight}
            for bucket in form.villain_buckets
        ],
        matrix_key: {hero_id: dict(row) for hero_id, row in form.matrix.items()},
        "candidate_generation": {"shift_amounts": list(form.shift_amounts)},
        "repeated": {"horizons": list(form.horizons), "discount": form.discount},
    }


def _validate_probability_group(add, entries, sum_field, label) -> None:
    """Validate one Hero decision distribution, appending messages via ``add``.

    ``entries`` is a list of ``(field_name, value)`` pairs; each value must be a
    finite, non-negative number and the values must sum to one. The sum error is
    reported on ``sum_field`` (the group's first field) so a GUI can anchor it,
    and ``label`` names the decision point in the message.
    """

    all_ok = True
    total = 0.0
    for field_name, value in entries:
        if not _is_finite_number(value) or value < 0:
            add(field_name, "must be a non-negative number")
            all_ok = False
        else:
            total += value
    if all_ok and abs(total - 1.0) > _TOLERANCE:
        add(sum_field, f"{label} probabilities must sum to 1 (got {total})")


def validate_betting_tree_form(
    form: BettingTreeScenarioForm,
) -> List[FormValidationMessage]:
    """Return field-level validation messages for ``form`` (empty when valid).

    Mirrors the betting-tree rules of the JSON scenario format with GUI-friendly
    field names (for example ``betting_tree.ip_raise_size``,
    ``hero_buckets[0].vs_oop_bet_raise_probability``, and the matrix cell field
    ``showdown_matrix[hero_id][villain_id]`` / ``equity_matrix[...]`` chosen by
    ``matrix_type``). The bucket id/weight, disjoint-id, and matrix-grid checks
    are shared with the matrix forms; betting-tree adds the three sizes, the
    ``bet_size`` match, and the two Hero decision distributions. The authoritative
    check remains ``river_scenario_from_dict`` +
    ``build_river_steal_game_from_scenario`` on
    :func:`betting_tree_form_to_dict`'s output.
    """

    messages: List[FormValidationMessage] = []

    def add(field_name: str, message: str) -> None:
        messages.append(FormValidationMessage(field_name, message))

    _validate_common_fields(form, add)

    # Betting-tree sizes.
    sizing = form.betting_tree
    if not isinstance(sizing, BettingTreeSizingForm):
        add("betting_tree", "betting_tree must be a BettingTreeSizingForm")
    else:
        oop_ok = _is_finite_number(sizing.oop_bet_size) and sizing.oop_bet_size > 0
        raise_ok = _is_finite_number(sizing.ip_raise_size) and sizing.ip_raise_size > 0
        if not oop_ok:
            add("betting_tree.oop_bet_size", "must be a positive number")
        if not (
            _is_finite_number(sizing.ip_bet_after_check_size)
            and sizing.ip_bet_after_check_size > 0
        ):
            add("betting_tree.ip_bet_after_check_size", "must be a positive number")
        if not raise_ok:
            add("betting_tree.ip_raise_size", "must be a positive number")
        if oop_ok and raise_ok and sizing.ip_raise_size <= sizing.oop_bet_size:
            add(
                "betting_tree.ip_raise_size",
                "must be greater than betting_tree.oop_bet_size (it is the total "
                "committed after the raise is called, not the raise increment)",
            )
        # bet_size must equal oop_bet_size in betting-tree mode.
        if (
            oop_ok
            and _is_finite_number(form.bet_size)
            and abs(form.bet_size - sizing.oop_bet_size) > _TOLERANCE
        ):
            add("bet_size", "must equal betting_tree.oop_bet_size in betting-tree mode")

    matrix_type_ok = form.matrix_type in _BETTING_TREE_MATRIX_TYPES
    if not matrix_type_ok:
        add(
            "matrix_type",
            f"matrix_type must be one of {list(_BETTING_TREE_MATRIX_TYPES)}",
        )

    hero_ok = isinstance(form.hero_buckets, (list, tuple)) and len(form.hero_buckets) > 0
    villain_ok = (
        isinstance(form.villain_buckets, (list, tuple)) and len(form.villain_buckets) > 0
    )
    if not hero_ok:
        add("hero_buckets", "at least one hero bucket is required")
    if not villain_ok:
        add("villain_buckets", "at least one villain bucket is required")

    hero_ids: List[str] = []
    villain_ids: List[str] = []
    hero_ids_usable = False
    villain_ids_usable = False
    if hero_ok:
        hero_ids, hero_ids_usable = _validate_matrix_bucket_ids_and_weights(
            form.hero_buckets, "hero_buckets", HeroBettingTreeBucketForm, add
        )
        for index, bucket in enumerate(form.hero_buckets):
            if not isinstance(bucket, HeroBettingTreeBucketForm):
                continue
            prefix = f"hero_buckets[{index}]"
            _validate_probability_group(
                add,
                [
                    (
                        f"{prefix}.after_oop_check_check_probability",
                        bucket.after_oop_check_check_probability,
                    ),
                    (
                        f"{prefix}.after_oop_check_bet_probability",
                        bucket.after_oop_check_bet_probability,
                    ),
                ],
                f"{prefix}.after_oop_check_check_probability",
                "after_oop_check",
            )
            _validate_probability_group(
                add,
                [
                    (
                        f"{prefix}.vs_oop_bet_call_probability",
                        bucket.vs_oop_bet_call_probability,
                    ),
                    (
                        f"{prefix}.vs_oop_bet_fold_probability",
                        bucket.vs_oop_bet_fold_probability,
                    ),
                    (
                        f"{prefix}.vs_oop_bet_raise_probability",
                        bucket.vs_oop_bet_raise_probability,
                    ),
                ],
                f"{prefix}.vs_oop_bet_call_probability",
                "vs_oop_bet",
            )
    if villain_ok:
        villain_ids, villain_ids_usable = _validate_matrix_bucket_ids_and_weights(
            form.villain_buckets, "villain_buckets", VillainMatrixBucketForm, add
        )

    # The parser keeps Hero/Villain ids disjoint so the matrix keys stay
    # unambiguous; mirror that here on the valid ids.
    shared_ids = set(hero_ids) & set(villain_ids)
    for shared in sorted(shared_ids):
        add(
            "villain_buckets",
            f"hand_id {shared!r} is used by both a hero and a villain bucket; "
            "ids must be disjoint",
        )

    # Only validate the matrix grid once both id axes are trustworthy and disjoint
    # and the matrix flavour is known, so a broken bucket id or matrix_type does
    # not produce misleading missing / unknown row/cell noise.
    if hero_ids_usable and villain_ids_usable and not shared_ids and matrix_type_ok:
        if form.matrix_type == "equity":
            _validate_matrix_grid(
                form.matrix, hero_ids, villain_ids, "equity_matrix", _equity_cell_error, add
            )
        else:
            _validate_matrix_grid(
                form.matrix, hero_ids, villain_ids, "showdown_matrix", _showdown_cell_error, add
            )

    return messages


# ---------------------------------------------------------------------------
# Mode detection (shared by the inspection CLI and a future GUI loader)
# ---------------------------------------------------------------------------

# The form-model mode labels, in roughly increasing structure. Each maps to one
# ``*ScenarioForm`` dataclass and its ``*_form_from_dict`` / ``validate_*_form`` /
# ``*_form_to_dict`` helpers.
SCENARIO_FORM_MODES = (
    "single-hand",
    "hero-range",
    "showdown-matrix",
    "equity-matrix",
    "betting-tree",
)


def detect_scenario_form_mode(data: dict) -> str:
    """Return the form-model mode label for a scenario dict (one of
    :data:`SCENARIO_FORM_MODES`).

    The label is chosen from the top-level keys, mirroring how each
    ``*_form_from_dict`` decides which mode it accepts:

    * ``betting_tree`` present -> ``"betting-tree"``;
    * otherwise a ``villain_range`` / ``showdown_matrix`` / ``equity_matrix``
      present -> ``"equity-matrix"`` if an ``equity_matrix`` is present, else
      ``"showdown-matrix"``;
    * otherwise a ``hero_range`` present -> ``"hero-range"``;
    * otherwise ``"single-hand"``.

    This only selects the candidate mode from which keys exist; the authoritative
    structural validation is still done by the matching ``*_form_from_dict`` (and
    the underlying :func:`river_scenario_from_dict`), which raises if the scenario
    is inconsistent (for example a ``villain_range`` without any matrix, or both
    matrices at once). Raises :class:`ValueError` if ``data`` is not a dict.
    """

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    if "betting_tree" in data:
        return "betting-tree"
    if "villain_range" in data or "showdown_matrix" in data or "equity_matrix" in data:
        if "equity_matrix" in data:
            return "equity-matrix"
        return "showdown-matrix"
    if "hero_range" in data:
        return "hero-range"
    return "single-hand"
