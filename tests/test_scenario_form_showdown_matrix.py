"""Tests for the discrete showdown-matrix scenario form model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from repeated_poker import (
    FormValidationMessage,
    HeroMatrixBucketForm,
    ShowdownMatrixScenarioForm,
    VillainMatrixBucketForm,
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
    showdown_matrix_form_from_dict,
    showdown_matrix_form_to_dict,
    validate_showdown_matrix_form,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SHOWDOWN_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_EQUITY_MATRIX = _SCENARIOS / "range_equity_steal_bet98.json"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"


def _sample_dict() -> dict:
    return json.loads(_SHOWDOWN_MATRIX.read_text(encoding="utf-8"))


def _valid_form() -> ShowdownMatrixScenarioForm:
    return showdown_matrix_form_from_dict(_sample_dict())


def _fields_with_errors(form) -> set:
    return {message.field for message in validate_showdown_matrix_form(form)}


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


def test_from_dict_reads_bundled_showdown_matrix_sample():
    form = _valid_form()
    assert isinstance(form, ShowdownMatrixScenarioForm)
    assert form.scenario_id == "range_matrix_steal_bet98"
    assert form.bet_size == 98.0
    assert [b.hand_id for b in form.hero_buckets] == ["hero_chop", "hero_strong"]
    assert [b.weight for b in form.hero_buckets] == [0.8, 0.2]
    assert form.hero_buckets[0].baseline_call_probability == 0.0
    assert form.hero_buckets[0].baseline_fold_probability == 1.0
    assert [b.hand_id for b in form.villain_buckets] == ["villain_chop", "villain_strong"]
    assert [b.weight for b in form.villain_buckets] == [0.7, 0.3]
    assert form.showdown_matrix["hero_chop"]["villain_strong"] == "villain"
    assert form.showdown_matrix["hero_strong"]["villain_chop"] == "hero"
    assert form.format_version == "1"


def test_to_dict_round_trips_through_parser_and_build():
    data = showdown_matrix_form_to_dict(_valid_form())
    scenario = river_scenario_from_dict(data)
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["format_version"] == "1"
    assert build.metadata["mode"] == "range_matrix"
    assert build.metadata["matrix_type"] == "showdown"


def test_round_trip_preserves_main_buckets_and_matrix():
    original = _sample_dict()
    data = showdown_matrix_form_to_dict(showdown_matrix_form_from_dict(original))
    assert data["scenario_id"] == original["scenario_id"]
    assert data["bet_size"] == original["bet_size"]
    assert data["rake"] == original["rake"]
    assert data["initial_commitment"] == original["initial_commitment"]
    assert data["repeated"] == original["repeated"]
    assert data["candidate_generation"] == original["candidate_generation"]
    assert data["hero_range"] == original["hero_range"]
    assert data["villain_range"] == original["villain_range"]
    assert data["showdown_matrix"] == original["showdown_matrix"]


def test_to_dict_omits_per_hand_showdown_on_hero_buckets():
    data = showdown_matrix_form_to_dict(_valid_form())
    assert all("showdown" not in bucket for bucket in data["hero_range"])


def test_to_dict_keeps_format_version_one_for_valid_form():
    assert showdown_matrix_form_to_dict(_valid_form())["format_version"] == "1"


def test_to_dict_preserves_invalid_format_version():
    data = showdown_matrix_form_to_dict(replace(_valid_form(), format_version="9"))
    assert data["format_version"] == "9"
    with pytest.raises(ValueError):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# from_dict rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", [_SINGLE_HAND, _HERO_RANGE, _EQUITY_MATRIX, _BETTING_TREE]
)
def test_non_showdown_matrix_modes_are_rejected(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        showdown_matrix_form_from_dict(data)


def test_unsupported_hero_baseline_action_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategy"] = {"call": 0.5, "raise": 0.5}
    with pytest.raises(ValueError):
        showdown_matrix_form_from_dict(data)


def test_non_dict_input_is_rejected():
    with pytest.raises(ValueError):
        showdown_matrix_form_from_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# validate_showdown_matrix_form
# ---------------------------------------------------------------------------


def test_valid_form_has_no_messages():
    assert validate_showdown_matrix_form(_valid_form()) == []


def test_validation_returns_multiple_field_errors():
    form = replace(
        _valid_form(),
        scenario_id="",
        rake_rate=2.0,
        bet_size=0.0,
        discount=1.5,
    )
    messages = validate_showdown_matrix_form(form)
    assert len(messages) >= 4
    assert all(isinstance(m, FormValidationMessage) for m in messages)
    assert all(m.severity == "error" for m in messages)
    assert {"scenario_id", "rake_rate", "bet_size", "discount"} <= _fields_with_errors(form)


def test_validation_requires_at_least_one_hero_and_villain_bucket():
    fields = _fields_with_errors(replace(_valid_form(), hero_buckets=[], villain_buckets=[]))
    assert "hero_buckets" in fields
    assert "villain_buckets" in fields


def test_validation_detects_empty_and_duplicate_hero_hand_id():
    form = _valid_form()
    form.hero_buckets[0] = replace(form.hero_buckets[0], hand_id="")
    assert "hero_buckets[0].hand_id" in _fields_with_errors(form)

    form = _valid_form()
    form.hero_buckets[1] = replace(form.hero_buckets[1], hand_id=form.hero_buckets[0].hand_id)
    assert "hero_buckets[1].hand_id" in _fields_with_errors(form)


def test_validation_detects_empty_and_duplicate_villain_hand_id():
    form = _valid_form()
    form.villain_buckets[0] = replace(form.villain_buckets[0], hand_id="")
    assert "villain_buckets[0].hand_id" in _fields_with_errors(form)

    form = _valid_form()
    form.villain_buckets[1] = replace(
        form.villain_buckets[1], hand_id=form.villain_buckets[0].hand_id
    )
    assert "villain_buckets[1].hand_id" in _fields_with_errors(form)


def test_validation_detects_hero_villain_id_overlap():
    form = _valid_form()
    # Reuse a hero id as a villain id and add a matching matrix column so the only
    # error is the id overlap.
    shared = form.hero_buckets[0].hand_id
    form.villain_buckets[0] = replace(form.villain_buckets[0], hand_id=shared)
    for row in form.showdown_matrix.values():
        row[shared] = row.pop("villain_chop")
    assert "villain_buckets" in _fields_with_errors(form)


def test_validation_detects_non_positive_weight_and_bad_sum():
    form = _valid_form()
    form.hero_buckets[0] = replace(form.hero_buckets[0], weight=0.0)
    assert "hero_buckets[0].weight" in _fields_with_errors(form)

    form = _valid_form()
    # Both weights individually valid, but they no longer sum to 1.
    form.villain_buckets[0] = replace(form.villain_buckets[0], weight=0.5)
    form.villain_buckets[1] = replace(form.villain_buckets[1], weight=0.2)
    assert "villain_buckets" in _fields_with_errors(form)


def test_validation_detects_hero_baseline_probabilities_not_summing_to_one():
    form = _valid_form()
    form.hero_buckets[0] = replace(
        form.hero_buckets[0],
        baseline_call_probability=0.4,
        baseline_fold_probability=0.4,
    )
    assert "hero_buckets[0].baseline_call_probability" in _fields_with_errors(form)


def test_validation_detects_missing_matrix_row():
    form = _valid_form()
    del form.showdown_matrix["hero_strong"]
    assert "showdown_matrix" in _fields_with_errors(form)


def test_validation_detects_missing_matrix_cell():
    form = _valid_form()
    del form.showdown_matrix["hero_chop"]["villain_strong"]
    assert "showdown_matrix[hero_chop]" in _fields_with_errors(form)


def test_validation_detects_unknown_matrix_row_and_villain_id():
    form = _valid_form()
    form.showdown_matrix["hero_ghost"] = {"villain_chop": "chop", "villain_strong": "chop"}
    assert "showdown_matrix" in _fields_with_errors(form)

    form = _valid_form()
    form.showdown_matrix["hero_chop"]["villain_ghost"] = "chop"
    assert "showdown_matrix[hero_chop]" in _fields_with_errors(form)


def test_validation_detects_invalid_cell_value():
    form = _valid_form()
    form.showdown_matrix["hero_chop"]["villain_chop"] = "split"
    assert "showdown_matrix[hero_chop][villain_chop]" in _fields_with_errors(form)


@pytest.mark.parametrize("bad_entry", [None, {"hand_id": "x"}, "not-a-bucket"])
def test_validation_handles_malformed_hero_bucket_without_raising(bad_entry):
    form = replace(_valid_form(), hero_buckets=[bad_entry])
    fields = {m.field for m in validate_showdown_matrix_form(form)}
    assert "hero_buckets[0]" in fields
    # No weight-sum noise is added when an entry is malformed.
    assert "hero_buckets" not in fields


@pytest.mark.parametrize("bad_entry", [None, {"hand_id": "x"}, "not-a-bucket"])
def test_validation_handles_malformed_villain_bucket_without_raising(bad_entry):
    form = replace(_valid_form(), villain_buckets=[bad_entry])
    fields = {m.field for m in validate_showdown_matrix_form(form)}
    assert "villain_buckets[0]" in fields
    assert "villain_buckets" not in fields


def test_validation_detects_invalid_rake_horizon_discount_and_bet_size():
    assert "rake_rate" in _fields_with_errors(replace(_valid_form(), rake_rate=1.5))
    assert "rake_cap" in _fields_with_errors(replace(_valid_form(), rake_cap=-1.0))
    assert "bet_size" in _fields_with_errors(replace(_valid_form(), bet_size=-5.0))
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[0, 10]))
    assert "discount" in _fields_with_errors(replace(_valid_form(), discount=1.5))


def test_validation_detects_empty_or_non_positive_shift_amounts():
    assert "shift_amounts" in _fields_with_errors(replace(_valid_form(), shift_amounts=[]))
    assert "shift_amounts" in _fields_with_errors(
        replace(_valid_form(), shift_amounts=[-0.5])
    )


# ---------------------------------------------------------------------------
# Edited form round-trips through the parser
# ---------------------------------------------------------------------------


def test_edited_valid_form_still_parses_and_builds():
    form = ShowdownMatrixScenarioForm(
        scenario_id="edited_showdown_matrix",
        description="edited",
        rake_rate=0.0,
        rake_cap=None,
        initial_commitment_hero=1.0,
        initial_commitment_villain=1.0,
        bet_size=50.0,
        hero_buckets=[
            HeroMatrixBucketForm(
                hand_id="ha",
                weight=0.5,
                baseline_call_probability=1.0,
                baseline_fold_probability=0.0,
            ),
            HeroMatrixBucketForm(
                hand_id="hb",
                weight=0.5,
                baseline_call_probability=0.0,
                baseline_fold_probability=1.0,
            ),
        ],
        villain_buckets=[
            VillainMatrixBucketForm(hand_id="va", weight=0.6),
            VillainMatrixBucketForm(hand_id="vb", weight=0.4),
        ],
        showdown_matrix={
            "ha": {"va": "hero", "vb": "chop"},
            "hb": {"va": "villain", "vb": "hero"},
        },
        shift_amounts=[0.5, 1.0],
        horizons=[10, 50],
        discount=0.9,
    )
    assert validate_showdown_matrix_form(form) == []
    scenario = river_scenario_from_dict(showdown_matrix_form_to_dict(form))
    build_river_steal_game_from_scenario(scenario)
    assert scenario.scenario_id == "edited_showdown_matrix"
    assert [h.hand_id for h in scenario.hero_range.hands] == ["ha", "hb"]
    assert [v.hand_id for v in scenario.villain_range.hands] == ["va", "vb"]
