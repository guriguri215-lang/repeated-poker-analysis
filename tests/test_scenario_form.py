"""Tests for the single-hand scenario form model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from repeated_poker import (
    FormValidationMessage,
    SingleHandScenarioForm,
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
    single_hand_form_from_dict,
    single_hand_form_to_dict,
    validate_single_hand_form,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"


def _sample_dict() -> dict:
    return json.loads(_SINGLE_HAND.read_text(encoding="utf-8"))


def _valid_form() -> SingleHandScenarioForm:
    return single_hand_form_from_dict(_sample_dict())


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


def test_from_dict_reads_bundled_single_hand_sample():
    form = _valid_form()
    assert isinstance(form, SingleHandScenarioForm)
    assert form.scenario_id == "nuts_chop_steal_bet98"
    assert form.showdown == "chop"
    assert form.bet_size == 98.0
    assert form.rake_rate == 0.05
    assert form.rake_cap == 4.0
    assert form.baseline_call_probability == 0.0
    assert form.baseline_fold_probability == 1.0
    assert form.shift_amounts == [1.0]
    assert form.horizons == [10, 20, 50, 100]
    assert form.discount == 1.0
    assert form.format_version == "1"


def test_to_dict_includes_format_version_one():
    assert single_hand_form_to_dict(_valid_form())["format_version"] == "1"


def test_to_dict_round_trips_through_parser_and_build():
    data = single_hand_form_to_dict(_valid_form())
    scenario = river_scenario_from_dict(data)
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["format_version"] == "1"
    assert build.metadata["mode"] == "single_hand"


def test_round_trip_preserves_main_fields():
    original = _sample_dict()
    data = single_hand_form_to_dict(single_hand_form_from_dict(original))
    assert data["scenario_id"] == original["scenario_id"]
    assert data["bet_size"] == original["bet_size"]
    assert data["showdown"] == original["showdown"]
    assert data["rake"] == original["rake"]
    assert data["initial_commitment"] == original["initial_commitment"]
    assert data["baseline_hero_strategy"] == original["baseline_hero_strategy"]
    assert data["repeated"] == original["repeated"]
    assert data["candidate_generation"] == original["candidate_generation"]


# ---------------------------------------------------------------------------
# from_dict rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [_RANGE, _MATRIX, _BETTING_TREE])
def test_non_single_hand_modes_are_rejected(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        single_hand_form_from_dict(data)


def test_unsupported_baseline_action_is_rejected():
    data = _sample_dict()
    data["baseline_hero_strategy"]["IP_vs_bet"] = {"call": 0.5, "raise": 0.5}
    with pytest.raises(ValueError):
        single_hand_form_from_dict(data)


def test_non_dict_input_is_rejected():
    with pytest.raises(ValueError):
        single_hand_form_from_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# validate_single_hand_form
# ---------------------------------------------------------------------------


def test_valid_form_has_no_messages():
    assert validate_single_hand_form(_valid_form()) == []


def _fields_with_errors(form) -> set:
    return {message.field for message in validate_single_hand_form(form)}


def test_validation_returns_multiple_field_errors():
    form = replace(
        _valid_form(),
        scenario_id="",
        rake_rate=2.0,
        bet_size=0.0,
        discount=1.5,
    )
    messages = validate_single_hand_form(form)
    assert len(messages) >= 4
    assert all(isinstance(m, FormValidationMessage) for m in messages)
    assert all(m.severity == "error" for m in messages)
    assert {"scenario_id", "rake_rate", "bet_size", "discount"} <= _fields_with_errors(form)


def test_validation_detects_probabilities_not_summing_to_one():
    form = replace(
        _valid_form(),
        baseline_call_probability=0.4,
        baseline_fold_probability=0.4,
    )
    assert "baseline_call_probability" in _fields_with_errors(form)


def test_validation_detects_invalid_rake_and_bet_size():
    assert "rake_rate" in _fields_with_errors(replace(_valid_form(), rake_rate=1.5))
    assert "rake_cap" in _fields_with_errors(replace(_valid_form(), rake_cap=-1.0))
    assert "bet_size" in _fields_with_errors(replace(_valid_form(), bet_size=-5.0))


def test_validation_detects_invalid_discount():
    assert "discount" in _fields_with_errors(replace(_valid_form(), discount=0.0))
    assert "discount" in _fields_with_errors(replace(_valid_form(), discount=1.5))


def test_validation_detects_invalid_horizons():
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[]))
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[0, 10]))
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[-1]))


def test_validation_detects_empty_or_non_positive_shift_amounts():
    assert "shift_amounts" in _fields_with_errors(replace(_valid_form(), shift_amounts=[]))
    assert "shift_amounts" in _fields_with_errors(
        replace(_valid_form(), shift_amounts=[0.0])
    )
    assert "shift_amounts" in _fields_with_errors(
        replace(_valid_form(), shift_amounts=[-0.5])
    )


def test_validation_detects_invalid_showdown():
    assert "showdown" in _fields_with_errors(replace(_valid_form(), showdown="split"))


def test_validation_detects_invalid_commitments():
    form = replace(
        _valid_form(),
        initial_commitment_hero=-1.0,
        initial_commitment_villain=-2.0,
    )
    fields = _fields_with_errors(form)
    assert "initial_commitment_hero" in fields
    assert "initial_commitment_villain" in fields


# ---------------------------------------------------------------------------
# format_version is preserved (not silently corrected) by to_dict
# ---------------------------------------------------------------------------


def test_empty_format_version_is_a_validation_error():
    assert "format_version" in _fields_with_errors(
        replace(_valid_form(), format_version="")
    )


def test_to_dict_preserves_invalid_format_version():
    data = single_hand_form_to_dict(replace(_valid_form(), format_version=""))
    # The invalid value is kept, not corrected to "1".
    assert data["format_version"] == ""
    # And the parser then rejects it, matching the field-level validation.
    with pytest.raises(ValueError):
        river_scenario_from_dict(data)


def test_to_dict_keeps_format_version_one_for_valid_form():
    assert single_hand_form_to_dict(_valid_form())["format_version"] == "1"


# ---------------------------------------------------------------------------
# description type validation
# ---------------------------------------------------------------------------


def test_non_string_description_is_a_validation_error():
    assert "description" in _fields_with_errors(replace(_valid_form(), description=123))


def test_empty_string_description_is_valid():
    assert "description" not in _fields_with_errors(
        replace(_valid_form(), description="")
    )


# ---------------------------------------------------------------------------
# Edited form round-trips through the parser
# ---------------------------------------------------------------------------


def test_edited_valid_form_still_parses_and_builds():
    form = replace(
        _valid_form(),
        scenario_id="edited_single_hand",
        showdown="hero",
        baseline_call_probability=1.0,
        baseline_fold_probability=0.0,
        bet_size=50.0,
    )
    assert validate_single_hand_form(form) == []
    scenario = river_scenario_from_dict(single_hand_form_to_dict(form))
    build_river_steal_game_from_scenario(scenario)
    assert scenario.scenario_id == "edited_single_hand"
