"""Tests for the Hero-range-only scenario form model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from repeated_poker import (
    FormValidationMessage,
    HeroRangeHandForm,
    HeroRangeScenarioForm,
    build_river_steal_game_from_scenario,
    hero_range_form_from_dict,
    hero_range_form_to_dict,
    river_scenario_from_dict,
    validate_hero_range_form,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"


def _sample_dict() -> dict:
    return json.loads(_HERO_RANGE.read_text(encoding="utf-8"))


def _valid_form() -> HeroRangeScenarioForm:
    return hero_range_form_from_dict(_sample_dict())


def _fields_with_errors(form) -> set:
    return {message.field for message in validate_hero_range_form(form)}


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


def test_from_dict_reads_bundled_hero_range_sample():
    form = _valid_form()
    assert isinstance(form, HeroRangeScenarioForm)
    assert form.scenario_id == "abstract_range_steal_bet98"
    assert form.bet_size == 98.0
    assert [h.hand_id for h in form.hands] == ["chop_fold_candidate", "hero_winner"]
    assert [h.weight for h in form.hands] == [0.8, 0.2]
    assert [h.showdown for h in form.hands] == ["chop", "hero"]
    assert form.hands[0].baseline_call_probability == 0.0
    assert form.hands[0].baseline_fold_probability == 1.0
    assert form.format_version == "1"


def test_to_dict_round_trips_through_parser_and_build():
    data = hero_range_form_to_dict(_valid_form())
    scenario = river_scenario_from_dict(data)
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["format_version"] == "1"
    assert build.metadata["mode"] == "range"


def test_round_trip_preserves_main_and_hand_fields():
    original = _sample_dict()
    data = hero_range_form_to_dict(hero_range_form_from_dict(original))
    assert data["scenario_id"] == original["scenario_id"]
    assert data["bet_size"] == original["bet_size"]
    assert data["rake"] == original["rake"]
    assert data["initial_commitment"] == original["initial_commitment"]
    assert data["repeated"] == original["repeated"]
    assert data["candidate_generation"] == original["candidate_generation"]
    assert data["hero_range"] == original["hero_range"]


def test_to_dict_keeps_format_version_one_for_valid_form():
    assert hero_range_form_to_dict(_valid_form())["format_version"] == "1"


def test_to_dict_preserves_invalid_format_version():
    data = hero_range_form_to_dict(replace(_valid_form(), format_version=""))
    assert data["format_version"] == ""
    with pytest.raises(ValueError):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# from_dict rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [_SINGLE_HAND, _MATRIX, _BETTING_TREE])
def test_non_hero_range_modes_are_rejected(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        hero_range_form_from_dict(data)


def test_unsupported_baseline_action_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategy"] = {"call": 0.5, "raise": 0.5}
    with pytest.raises(ValueError):
        hero_range_form_from_dict(data)


def test_non_dict_input_is_rejected():
    with pytest.raises(ValueError):
        hero_range_form_from_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# validate_hero_range_form
# ---------------------------------------------------------------------------


def test_valid_form_has_no_messages():
    assert validate_hero_range_form(_valid_form()) == []


def test_validation_returns_multiple_field_errors():
    form = replace(
        _valid_form(),
        scenario_id="",
        rake_rate=2.0,
        bet_size=0.0,
        discount=1.5,
    )
    messages = validate_hero_range_form(form)
    assert len(messages) >= 4
    assert all(isinstance(m, FormValidationMessage) for m in messages)
    assert all(m.severity == "error" for m in messages)
    assert {"scenario_id", "rake_rate", "bet_size", "discount"} <= _fields_with_errors(form)


def test_validation_requires_at_least_one_hand():
    assert "hands" in _fields_with_errors(replace(_valid_form(), hands=[]))


def test_validation_detects_empty_hand_id():
    form = _valid_form()
    form.hands[0] = replace(form.hands[0], hand_id="")
    assert "hands[0].hand_id" in _fields_with_errors(form)


def test_validation_detects_duplicate_hand_id():
    form = _valid_form()
    form.hands[1] = replace(form.hands[1], hand_id=form.hands[0].hand_id)
    fields = _fields_with_errors(form)
    assert "hands[1].hand_id" in fields


def test_validation_detects_non_positive_weight():
    form = _valid_form()
    form.hands[0] = replace(form.hands[0], weight=0.0)
    assert "hands[0].weight" in _fields_with_errors(form)


def test_validation_detects_weights_not_summing_to_one():
    form = _valid_form()
    # Both weights individually valid, but they no longer sum to 1.
    form.hands[0] = replace(form.hands[0], weight=0.5)
    form.hands[1] = replace(form.hands[1], weight=0.2)
    assert "hands" in _fields_with_errors(form)


def test_validation_detects_invalid_showdown():
    form = _valid_form()
    form.hands[0] = replace(form.hands[0], showdown="split")
    assert "hands[0].showdown" in _fields_with_errors(form)


def test_validation_detects_probabilities_not_summing_to_one():
    form = _valid_form()
    form.hands[0] = replace(
        form.hands[0],
        baseline_call_probability=0.4,
        baseline_fold_probability=0.4,
    )
    assert "hands[0].baseline_call_probability" in _fields_with_errors(form)


def test_validation_detects_invalid_rake_and_bet_size():
    assert "rake_rate" in _fields_with_errors(replace(_valid_form(), rake_rate=1.5))
    assert "rake_cap" in _fields_with_errors(replace(_valid_form(), rake_cap=-1.0))
    assert "bet_size" in _fields_with_errors(replace(_valid_form(), bet_size=-5.0))


def test_validation_detects_invalid_horizon_and_discount():
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[0, 10]))
    assert "discount" in _fields_with_errors(replace(_valid_form(), discount=1.5))


def test_validation_detects_empty_or_non_positive_shift_amounts():
    assert "shift_amounts" in _fields_with_errors(replace(_valid_form(), shift_amounts=[]))
    assert "shift_amounts" in _fields_with_errors(
        replace(_valid_form(), shift_amounts=[-0.5])
    )


def test_validation_detects_non_string_description():
    assert "description" in _fields_with_errors(replace(_valid_form(), description=123))


# ---------------------------------------------------------------------------
# Edited form round-trips through the parser
# ---------------------------------------------------------------------------


def test_edited_valid_form_still_parses_and_builds():
    form = HeroRangeScenarioForm(
        scenario_id="edited_hero_range",
        description="edited",
        rake_rate=0.0,
        rake_cap=None,
        initial_commitment_hero=1.0,
        initial_commitment_villain=1.0,
        bet_size=50.0,
        hands=[
            HeroRangeHandForm(
                hand_id="a",
                weight=0.5,
                showdown="hero",
                baseline_call_probability=1.0,
                baseline_fold_probability=0.0,
            ),
            HeroRangeHandForm(
                hand_id="b",
                weight=0.5,
                showdown="chop",
                baseline_call_probability=0.0,
                baseline_fold_probability=1.0,
            ),
        ],
        shift_amounts=[0.5, 1.0],
        horizons=[10, 50],
        discount=0.9,
    )
    assert validate_hero_range_form(form) == []
    scenario = river_scenario_from_dict(hero_range_form_to_dict(form))
    build_river_steal_game_from_scenario(scenario)
    assert scenario.scenario_id == "edited_hero_range"
    assert [h.hand_id for h in scenario.hero_range.hands] == ["a", "b"]
