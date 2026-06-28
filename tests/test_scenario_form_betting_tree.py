"""Tests for the river betting-tree scenario form model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from repeated_poker import (
    BettingTreeScenarioForm,
    BettingTreeSizingForm,
    FormValidationMessage,
    HeroBettingTreeBucketForm,
    VillainMatrixBucketForm,
    betting_tree_form_from_dict,
    betting_tree_form_to_dict,
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
    validate_betting_tree_form,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"
_EQUITY_MATRIX = _SCENARIOS / "range_equity_steal_bet98.json"
_SHOWDOWN_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"


def _sample_dict() -> dict:
    return json.loads(_BETTING_TREE.read_text(encoding="utf-8"))


def _valid_form() -> BettingTreeScenarioForm:
    return betting_tree_form_from_dict(_sample_dict())


def _fields_with_errors(form) -> set:
    return {message.field for message in validate_betting_tree_form(form)}


def _has_matrix_grid_error(form) -> bool:
    return any(
        message.field in ("showdown_matrix", "equity_matrix")
        or message.field.startswith("showdown_matrix[")
        or message.field.startswith("equity_matrix[")
        for message in validate_betting_tree_form(form)
    )


def _showdown_form() -> BettingTreeScenarioForm:
    """A hand-built, valid showdown-matrix betting-tree form."""
    return BettingTreeScenarioForm(
        scenario_id="bt_showdown",
        description="showdown betting tree",
        rake_rate=0.0,
        rake_cap=None,
        initial_commitment_hero=1.0,
        initial_commitment_villain=1.0,
        bet_size=50.0,
        betting_tree=BettingTreeSizingForm(
            oop_bet_size=50.0, ip_bet_after_check_size=50.0, ip_raise_size=120.0
        ),
        matrix_type="showdown",
        matrix={
            "ha": {"va": "hero", "vb": "chop"},
            "hb": {"va": "villain", "vb": "hero"},
        },
        hero_buckets=[
            HeroBettingTreeBucketForm(
                hand_id="ha",
                weight=0.5,
                after_oop_check_check_probability=1.0,
                after_oop_check_bet_probability=0.0,
                vs_oop_bet_call_probability=1.0,
                vs_oop_bet_fold_probability=0.0,
                vs_oop_bet_raise_probability=0.0,
            ),
            HeroBettingTreeBucketForm(
                hand_id="hb",
                weight=0.5,
                after_oop_check_check_probability=0.0,
                after_oop_check_bet_probability=1.0,
                vs_oop_bet_call_probability=0.0,
                vs_oop_bet_fold_probability=0.5,
                vs_oop_bet_raise_probability=0.5,
            ),
        ],
        villain_buckets=[
            VillainMatrixBucketForm(hand_id="va", weight=0.6),
            VillainMatrixBucketForm(hand_id="vb", weight=0.4),
        ],
        shift_amounts=[0.5, 1.0],
        horizons=[10, 50],
        discount=0.9,
    )


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


def test_from_dict_reads_bundled_betting_tree_sample():
    form = _valid_form()
    assert isinstance(form, BettingTreeScenarioForm)
    assert form.scenario_id == "range_equity_betting_tree_bet98"
    assert form.matrix_type == "equity"
    # bet_size defaults to betting_tree.oop_bet_size when absent in the JSON.
    assert form.bet_size == 98.0
    assert form.betting_tree == BettingTreeSizingForm(98.0, 98.0, 196.0)
    assert [b.hand_id for b in form.hero_buckets] == ["hero_medium", "hero_strong"]
    assert form.hero_buckets[0].after_oop_check_check_probability == 1.0
    assert form.hero_buckets[1].after_oop_check_bet_probability == 1.0
    assert form.hero_buckets[1].vs_oop_bet_call_probability == 1.0
    assert [b.hand_id for b in form.villain_buckets] == ["villain_weak", "villain_strong"]
    assert form.matrix["hero_strong"]["villain_weak"] == 0.90
    assert form.format_version == "1"


def test_to_dict_round_trips_through_parser_and_build():
    data = betting_tree_form_to_dict(_valid_form())
    scenario = river_scenario_from_dict(data)
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["format_version"] == "1"
    assert build.metadata["mode"] == "range_matrix"
    assert build.metadata["matrix_type"] == "equity"
    assert build.metadata["betting_tree"]["ip_raise_size"] == 196.0


def test_round_trip_preserves_main_buckets_tree_and_matrix():
    original = _sample_dict()
    data = betting_tree_form_to_dict(betting_tree_form_from_dict(original))
    assert data["scenario_id"] == original["scenario_id"]
    assert data["rake"] == original["rake"]
    assert data["initial_commitment"] == original["initial_commitment"]
    assert data["repeated"] == original["repeated"]
    assert data["candidate_generation"] == original["candidate_generation"]
    assert data["hero_range"] == original["hero_range"]
    assert data["villain_range"] == original["villain_range"]
    assert data["equity_matrix"] == original["equity_matrix"]
    assert data["betting_tree"] == original["betting_tree"]
    # The sample omits top-level bet_size; the form emits it matching oop_bet_size.
    assert "bet_size" not in original
    assert data["bet_size"] == original["betting_tree"]["oop_bet_size"]


def test_showdown_betting_tree_form_round_trips():
    data = betting_tree_form_to_dict(_showdown_form())
    assert "showdown_matrix" in data
    assert "equity_matrix" not in data
    scenario = river_scenario_from_dict(data)
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["matrix_type"] == "showdown"


def test_to_dict_uses_baseline_strategies_not_baseline_strategy():
    data = betting_tree_form_to_dict(_valid_form())
    for bucket in data["hero_range"]:
        assert "baseline_strategies" in bucket
        assert "baseline_strategy" not in bucket
        assert set(bucket["baseline_strategies"]) == {"after_oop_check", "vs_oop_bet"}


def test_to_dict_keeps_format_version_one_for_valid_form():
    assert betting_tree_form_to_dict(_valid_form())["format_version"] == "1"


def test_to_dict_preserves_invalid_format_version():
    data = betting_tree_form_to_dict(replace(_valid_form(), format_version="9"))
    assert data["format_version"] == "9"
    with pytest.raises(ValueError):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# from_dict rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", [_SINGLE_HAND, _HERO_RANGE, _SHOWDOWN_MATRIX, _EQUITY_MATRIX]
)
def test_non_betting_tree_modes_are_rejected(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        betting_tree_form_from_dict(data)


def test_unsupported_hero_baseline_action_is_rejected():
    data = _sample_dict()
    # vs_oop_bet only allows call/fold/raise; an unknown action is rejected.
    data["hero_range"][0]["baseline_strategies"]["vs_oop_bet"] = {
        "call": 0.5,
        "shove": 0.5,
    }
    with pytest.raises(ValueError):
        betting_tree_form_from_dict(data)


def test_old_style_baseline_strategy_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategy"] = {"call": 1.0, "fold": 0.0}
    with pytest.raises(ValueError):
        betting_tree_form_from_dict(data)


def test_non_dict_input_is_rejected():
    with pytest.raises(ValueError):
        betting_tree_form_from_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# validate_betting_tree_form
# ---------------------------------------------------------------------------


def test_valid_form_has_no_messages():
    assert validate_betting_tree_form(_valid_form()) == []
    assert validate_betting_tree_form(_showdown_form()) == []


def test_validation_returns_multiple_field_errors():
    form = replace(
        _valid_form(),
        scenario_id="",
        rake_rate=2.0,
        discount=1.5,
    )
    messages = validate_betting_tree_form(form)
    assert len(messages) >= 3
    assert all(isinstance(m, FormValidationMessage) for m in messages)
    assert all(m.severity == "error" for m in messages)
    assert {"scenario_id", "rake_rate", "discount"} <= _fields_with_errors(form)


def test_validation_requires_at_least_one_hero_and_villain_bucket():
    fields = _fields_with_errors(replace(_valid_form(), hero_buckets=[], villain_buckets=[]))
    assert "hero_buckets" in fields
    assert "villain_buckets" in fields


def test_validation_detects_non_positive_betting_tree_sizes():
    form = _valid_form()
    form.betting_tree = replace(form.betting_tree, oop_bet_size=0.0)
    assert "betting_tree.oop_bet_size" in _fields_with_errors(form)

    form = _valid_form()
    form.betting_tree = replace(form.betting_tree, ip_bet_after_check_size=-1.0)
    assert "betting_tree.ip_bet_after_check_size" in _fields_with_errors(form)


def test_validation_detects_raise_not_greater_than_oop_bet():
    form = _valid_form()
    form.betting_tree = replace(form.betting_tree, ip_raise_size=98.0)  # == oop_bet_size
    assert "betting_tree.ip_raise_size" in _fields_with_errors(form)


def test_validation_detects_bet_size_mismatch():
    form = replace(_valid_form(), bet_size=42.0)  # != oop_bet_size 98.0
    assert "bet_size" in _fields_with_errors(form)


def test_validation_detects_invalid_matrix_type():
    form = replace(_valid_form(), matrix_type="weird")
    fields = _fields_with_errors(form)
    assert "matrix_type" in fields
    # An unknown matrix_type suppresses grid validation (cell rule is unknown).
    assert not _has_matrix_grid_error(form)


def test_to_dict_rejects_invalid_matrix_type():
    # to_dict must not silently coerce an unknown matrix_type to showdown_matrix
    # (which would hide the invalid form behind a parseable dict).
    form = replace(_valid_form(), matrix_type="weird")
    assert "matrix_type" in _fields_with_errors(form)
    with pytest.raises(ValueError):
        betting_tree_form_to_dict(form)


def test_validation_detects_empty_and_duplicate_hero_hand_id():
    form = _valid_form()
    form.hero_buckets[0] = replace(form.hero_buckets[0], hand_id="")
    assert "hero_buckets[0].hand_id" in _fields_with_errors(form)
    assert not _has_matrix_grid_error(form)

    form = _valid_form()
    form.hero_buckets[1] = replace(form.hero_buckets[1], hand_id=form.hero_buckets[0].hand_id)
    assert "hero_buckets[1].hand_id" in _fields_with_errors(form)
    assert not _has_matrix_grid_error(form)


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
    shared = form.hero_buckets[0].hand_id
    form.villain_buckets[0] = replace(form.villain_buckets[0], hand_id=shared)
    assert "villain_buckets" in _fields_with_errors(form)
    assert not _has_matrix_grid_error(form)


def test_validation_detects_non_positive_weight_and_bad_sum():
    form = _valid_form()
    form.hero_buckets[0] = replace(form.hero_buckets[0], weight=0.0)
    assert "hero_buckets[0].weight" in _fields_with_errors(form)

    form = _valid_form()
    form.villain_buckets[0] = replace(form.villain_buckets[0], weight=0.5)
    form.villain_buckets[1] = replace(form.villain_buckets[1], weight=0.2)
    assert "villain_buckets" in _fields_with_errors(form)


def test_validation_detects_after_oop_check_distribution_not_summing_to_one():
    form = _valid_form()
    form.hero_buckets[0] = replace(
        form.hero_buckets[0],
        after_oop_check_check_probability=0.4,
        after_oop_check_bet_probability=0.4,
    )
    assert "hero_buckets[0].after_oop_check_check_probability" in _fields_with_errors(form)


def test_validation_detects_vs_oop_bet_distribution_not_summing_to_one():
    form = _valid_form()
    form.hero_buckets[0] = replace(
        form.hero_buckets[0],
        vs_oop_bet_call_probability=0.3,
        vs_oop_bet_fold_probability=0.3,
        vs_oop_bet_raise_probability=0.3,
    )
    assert "hero_buckets[0].vs_oop_bet_call_probability" in _fields_with_errors(form)


def test_validation_detects_negative_probability():
    form = _valid_form()
    form.hero_buckets[0] = replace(
        form.hero_buckets[0],
        vs_oop_bet_call_probability=-0.5,
        vs_oop_bet_fold_probability=1.5,
        vs_oop_bet_raise_probability=0.0,
    )
    assert "hero_buckets[0].vs_oop_bet_call_probability" in _fields_with_errors(form)


def test_validation_detects_missing_and_unknown_matrix_cells():
    form = _valid_form()
    del form.matrix["hero_strong"]
    assert "equity_matrix" in _fields_with_errors(form)

    form = _valid_form()
    del form.matrix["hero_medium"]["villain_strong"]
    assert "equity_matrix[hero_medium]" in _fields_with_errors(form)

    form = _valid_form()
    form.matrix["hero_medium"]["villain_ghost"] = 0.5
    assert "equity_matrix[hero_medium]" in _fields_with_errors(form)


def test_validation_detects_invalid_equity_cell_value():
    form = _valid_form()
    form.matrix["hero_medium"]["villain_weak"] = 1.5
    assert "equity_matrix[hero_medium][villain_weak]" in _fields_with_errors(form)


def test_validation_detects_invalid_showdown_cell_value():
    form = _showdown_form()
    form.matrix["ha"]["va"] = "split"
    assert "showdown_matrix[ha][va]" in _fields_with_errors(form)


@pytest.mark.parametrize("bad_entry", [None, {"hand_id": "x"}, "not-a-bucket"])
def test_validation_handles_malformed_hero_bucket_without_raising(bad_entry):
    form = replace(_valid_form(), hero_buckets=[bad_entry])
    fields = {m.field for m in validate_betting_tree_form(form)}
    assert "hero_buckets[0]" in fields
    assert "hero_buckets" not in fields
    assert not _has_matrix_grid_error(form)


@pytest.mark.parametrize("bad_entry", [None, {"hand_id": "x"}, "not-a-bucket"])
def test_validation_handles_malformed_villain_bucket_without_raising(bad_entry):
    form = replace(_valid_form(), villain_buckets=[bad_entry])
    fields = {m.field for m in validate_betting_tree_form(form)}
    assert "villain_buckets[0]" in fields
    assert "villain_buckets" not in fields
    assert not _has_matrix_grid_error(form)


def test_validation_handles_malformed_betting_tree_without_raising():
    form = replace(_valid_form(), betting_tree=None)
    fields = {m.field for m in validate_betting_tree_form(form)}
    assert "betting_tree" in fields


def test_validation_tolerates_non_comparable_unknown_matrix_keys():
    form = _valid_form()
    form.matrix[1] = {"villain_weak": 0.5, "villain_strong": 0.5}
    form.matrix[None] = {"villain_weak": 0.5, "villain_strong": 0.5}
    assert "equity_matrix" in _fields_with_errors(form)


def test_validation_detects_invalid_rake_horizon_discount_and_shift_amounts():
    assert "rake_rate" in _fields_with_errors(replace(_valid_form(), rake_rate=1.5))
    assert "rake_cap" in _fields_with_errors(replace(_valid_form(), rake_cap=-1.0))
    assert "horizons" in _fields_with_errors(replace(_valid_form(), horizons=[0, 10]))
    assert "discount" in _fields_with_errors(replace(_valid_form(), discount=1.5))
    assert "shift_amounts" in _fields_with_errors(replace(_valid_form(), shift_amounts=[]))


# ---------------------------------------------------------------------------
# Edited form round-trips through the parser
# ---------------------------------------------------------------------------


def test_edited_valid_form_still_parses_and_builds():
    form = _showdown_form()
    assert validate_betting_tree_form(form) == []
    scenario = river_scenario_from_dict(betting_tree_form_to_dict(form))
    build_river_steal_game_from_scenario(scenario)
    assert scenario.scenario_id == "bt_showdown"
    assert scenario.is_betting_tree_mode
    assert [h.hand_id for h in scenario.hero_range.hands] == ["ha", "hb"]
    assert [v.hand_id for v in scenario.villain_range.hands] == ["va", "vb"]
