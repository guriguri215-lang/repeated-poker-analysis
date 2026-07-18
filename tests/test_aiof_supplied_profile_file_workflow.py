"""Independent contract checks for the M22 supplied-profile file adapter."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import repeated_poker.aiof_cards as cards_module
import repeated_poker.aiof_chip_ev as chip_ev_module
import repeated_poker.aiof_equity as equity_module
import repeated_poker.aiof_supplied_profile_file_workflow as module
from repeated_poker.aiof_cards import (
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
)
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRequest,
    PushFoldRunResult,
    SuppliedProfile,
    analyze_pushfold,
)
from repeated_poker.aiof_equity import EquityAlgorithm


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "aiof_supplied_profile_file_v1.json"
SCRIPT = ROOT / "scripts" / "run_aiof_supplied_profile_file.py"
SOURCE = ROOT / "src" / "repeated_poker" / "aiof_supplied_profile_file_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "aiof_supplied_profile_file_workflow.md"
FORMAT = "aiof-supplied-profile-file-v1"


def example_document() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def encoded(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def inspect_document(
    document: dict[str, object] | None = None,
    limits: module.AiofSuppliedProfileFileLimits = module.AiofSuppliedProfileFileLimits(),
) -> module.AiofSuppliedProfileFileResult:
    return module.inspect_aiof_supplied_profile_file(
        encoded(document or example_document()), limits
    )


def run_document(
    document: dict[str, object],
    limits: module.AiofSuppliedProfileFileLimits = module.AiofSuppliedProfileFileLimits(),
) -> module.AiofSuppliedProfileFileResult:
    return module.run_aiof_supplied_profile_file(encoded(document), limits)


def filled_run_document(
    document: dict[str, object] | None = None,
    *,
    shove: float = 1.0,
    call: float = 0.0,
) -> dict[str, object]:
    source = copy.deepcopy(document or example_document())
    inspected = inspect_document(source)
    assert inspected.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert inspected.output is not None
    source["operation"] = "run"
    source["template_identity"] = copy.deepcopy(inspected.output["identity"])
    profile = copy.deepcopy(inspected.output["profile_template"])
    for row in profile["sb_shove"]:
        row["probability"] = shove
    for row in profile["bb_call"]:
        row["probability"] = call
    source["profile"] = profile
    return source


def run_cli(path: Path = EXAMPLE) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def dead_except(live_cards: tuple[str, ...]) -> list[str]:
    live = {card_id(card) for card in live_cards}
    return [card_from_id(value) for value in range(52) if value not in live]


def five_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Test-owned Hold'em oracle independent of production evaluators."""

    rank_value = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
    ranks = tuple(rank_value[card[0]] for card in cards)
    suits = tuple(card[1] for card in cards)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    straight_high = next(
        (high for high in range(14, 4, -1) if set(range(high - 4, high + 1)) <= unique),
        None,
    )
    flush = len(set(suits)) == 1
    if flush and straight_high is not None:
        return 8, (straight_high,)
    if groups[0][0] == 4:
        quad = groups[0][1]
        return 7, (quad, max(rank for rank in ranks if rank != quad))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return 6, (groups[0][1], groups[1][1])
    if flush:
        return 5, tuple(sorted(ranks, reverse=True))
    if straight_high is not None:
        return 4, (straight_high,)
    if groups[0][0] == 3:
        trip = groups[0][1]
        return 3, (trip,) + tuple(
            sorted((rank for rank in ranks if rank != trip), reverse=True)
        )
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        return 2, (
            pairs[0],
            pairs[1],
            max(rank for rank, count in counts.items() if count == 1),
        )
    if len(pairs) == 1:
        pair = pairs[0]
        return 1, (pair,) + tuple(
            sorted((rank for rank in ranks if rank != pair), reverse=True)
        )
    return 0, tuple(sorted(ranks, reverse=True))


def seven_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return max(five_card_rank(tuple(combo)) for combo in itertools.combinations(cards, 5))


def public_range(items: object) -> RangeSpec:
    assert isinstance(items, list)
    return RangeSpec(
        tuple(
            RangeEntry(
                item["label"],
                float(item["weight"]),
                WeightBasis(item["weight_basis"]),
            )
            for item in items
        )
    )


def direct_public_run(document: dict[str, object]) -> PushFoldRunResult:
    game_doc = document["game"]
    analysis_doc = document["analysis"]
    limits_doc = document["limits"]
    profile_doc = document["profile"]
    assert isinstance(game_doc, dict)
    assert isinstance(analysis_doc, dict)
    assert isinstance(limits_doc, dict)
    assert isinstance(profile_doc, dict)
    profile = SuppliedProfile(
        tuple(
            ComboActionProbability(row["combo"], float(row["probability"]))
            for row in profile_doc["sb_shove"]
        ),
        tuple(
            ComboActionProbability(row["combo"], float(row["probability"]))
            for row in profile_doc["bb_call"]
        ),
    )
    return analyze_pushfold(
        PushFoldRequest(
            public_range(document["sb_range"]),
            public_range(document["bb_range"]),
            tuple(document["dead_cards"]),
            EquityAlgorithm.EXACT_EXHAUSTIVE,
            AiofLimits(**limits_doc),
            0,
            HeadsUpChipEvGame(
                float(game_doc["starting_stack_sb"]),
                float(game_doc["starting_stack_bb"]),
                float(game_doc["small_blind"]),
                float(game_doc["big_blind"]),
                float(game_doc["ante"]),
                float(game_doc["fee"]),
                float(game_doc["third_party_dead_money"]),
                bool(game_doc["side_pot"]),
            ),
            profile,
            ("sb", "bb"),
            float(analysis_doc["deviation_tolerance"]),
            None,
            None,
        )
    )


def assert_failure_is_no_partial(result: module.AiofSuppliedProfileFileResult) -> None:
    assert result.status is not module.AiofSuppliedProfileFileStatus.SUCCESS
    assert result.output is None
    assert result.error is not None
    serialized = module.aiof_supplied_profile_file_json(result)
    payload = json.loads(serialized)
    assert set(payload) == {"status", "output", "error"}
    assert payload["output"] is None
    assert set(payload["error"]) == {"phase", "message", "nested_status"}
    assert "\n" not in result.error.message and "\r" not in result.error.message
    assert len(result.error.message) <= 500


def test_inspect_tiny_fixture_returns_identity_and_null_complete_template(monkeypatch):
    called = False

    def bomb(_request):
        nonlocal called
        called = True
        raise AssertionError("inspect must not run supplied-profile analysis")

    monkeypatch.setattr(module, "analyze_pushfold", bomb)
    result = inspect_document()
    assert called is False
    assert result.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["format_version"] == FORMAT
    assert output["operation"] == "inspect"
    assert output["algorithm"] == "exact_exhaustive-v1"
    assert output["counts"] == {
        "compatible_pairs": 1,
        "sb": {"projected_combos": 1, "removed_combos": 0, "surviving_combos": 1},
        "bb": {"projected_combos": 1, "removed_combos": 0, "surviving_combos": 1},
    }
    assert output["profile_template"] == {
        "sb_shove": [{"combo": "AsAh", "probability": None}],
        "bb_call": [{"combo": "KsKh", "probability": None}],
    }
    assert output["identity"]["template_id"] == module.AIOF_SUPPLIED_PROFILE_TEMPLATE_ID
    for name in ("semantic_sha256", "prepared_ranges_identity", "support_sha256"):
        assert len(output["identity"][name]) == 64


def test_inspect_class_combo_dead_card_fixture_has_expected_canonical_support():
    document = example_document()
    document["request_id"] = "class-combo-dead-card"
    document["sb_range"] = [
        {"label": "AKs", "weight": 2, "weight_basis": "class_total_mass"},
        {"label": "QcQd", "weight": 1, "weight_basis": "exact_combo_mass"},
    ]
    document["bb_range"] = [
        {"label": "JJ", "weight": 1, "weight_basis": "class_total_mass"},
        {"label": "AsKh", "weight": 1, "weight_basis": "exact_combo_mass"},
    ]
    live = (
        "Qc", "Qd", "Ah", "Kh", "As", "Ks", "Jc", "Jd",
        "2c", "3d", "4h", "5s", "7c",
    )
    document["dead_cards"] = dead_except(live)
    document["limits"].update(
        max_range_entries_per_side=2,
        max_exact_combos_per_side=7,
        max_compatible_combo_pairs=4,
        max_dead_cards=39,
        max_exact_board_evaluations=504,
    )
    result = inspect_document(document)
    assert result.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert result.output is not None
    template = result.output["profile_template"]
    assert [row["combo"] for row in template["sb_shove"]] == [
        "QdQc", "AhKh", "AsKs"
    ]
    assert [row["combo"] for row in template["bb_call"]] == ["JdJc", "AsKh"]
    assert result.output["counts"]["compatible_pairs"] == 4
    run = run_document(filled_run_document(document, shove=0.5, call=0.25))
    assert run.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert run.output is not None
    assert run.output["board_evaluations"] == 504
    assert run.output["profile_values"]["sb"] == pytest.approx(-0.2730654761904761)
    assert [response["seat"] for response in run.output["best_responses"]] == [
        "sb", "bb"
    ]


def test_inspect_run_tiny_fixture_matches_hand_chipev_and_action_oracle():
    board = ("2c", "3d", "4h", "5s", "7c")
    assert seven_card_rank(("As", "Ah") + board) > seven_card_rank(("Ks", "Kh") + board)
    document = filled_run_document(shove=1.0, call=0.0)
    result = run_document(document)
    assert result.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["outcome_counts"] == {"wins": 1, "losses": 0, "ties": 0, "trials": 1}
    assert output["outcome_probabilities"] == {"win": 1.0, "loss": 0.0, "tie": 0.0}
    assert output["profile_values"] == {"sb": 1.0, "bb": -1.0, "conservation_sum": 0.0}
    assert output["board_evaluations"] == 1
    assert output["sampling"] == {
        "accepted_samples": None,
        "rejected_hole_draws": 0,
        "profile_sample_variance": None,
        "profile_standard_error": None,
        "seed": None,
        "requested_samples": None,
    }
    sb, bb = output["best_responses"]
    assert (sb["seat"], bb["seat"]) == ("sb", "bb")
    assert sb["raw_gain"] == bb["raw_gain"] == 0.0
    assert sb["rows"][0]["action_values"] == [
        {"action": "shove", "value": 1.0},
        {"action": "fold", "value": -0.5},
    ]
    assert sb["rows"][0]["best_actions"] == ["shove"]
    assert bb["rows"][0]["action_values"] == [
        {"action": "call", "value": -10.0},
        {"action": "fold", "value": -1.0},
    ]
    assert bb["rows"][0]["best_actions"] == ["fold"]


def test_run_full_projection_matches_direct_existing_public_api():
    document = filled_run_document(shove=0.25, call=0.4)
    wrapped = run_document(document)
    direct = direct_public_run(document)
    assert wrapped.output is not None
    assert direct.status is AiofStatus.SUCCESS and direct.analysis is not None
    output = wrapped.output
    analysis = direct.analysis
    assert output["input_identity"] == analysis.input_identity
    assert output["prepared_ranges_identity"] == analysis.prepared_ranges_identity
    assert output["compatible_pair_count"] == analysis.compatible_pair_count
    assert output["board_evaluations"] == analysis.board_evaluations
    assert output["profile_values"]["sb"] == analysis.profile_value_sb
    assert output["profile_values"]["bb"] == analysis.profile_value_bb
    assert output["outcome_counts"] == {
        "wins": analysis.outcome_counts.wins,
        "losses": analysis.outcome_counts.losses,
        "ties": analysis.outcome_counts.ties,
        "trials": analysis.outcome_counts.trials,
    }
    assert output["outcome_probabilities"] == {
        "win": analysis.outcome_probabilities.win,
        "loss": analysis.outcome_probabilities.loss,
        "tie": analysis.outcome_probabilities.tie,
    }
    for projected, direct in zip(output["best_responses"], analysis.best_responses):
        assert projected["seat"] == direct.seat
        assert projected["supplied_profile_value"] == direct.supplied_profile_value
        assert projected["best_response_value"] == direct.best_response_value
        assert projected["raw_gain"] == direct.raw_gain
        assert len(projected["rows"]) == len(direct.rows)
        for projected_row, direct_row in zip(projected["rows"], direct.rows):
            assert projected_row == {
                "combo": direct_row.combo,
                "compatible_probability": direct_row.compatible_probability,
                "information_reach_probability": direct_row.information_reach_probability,
                "action_values": [
                    {"action": action, "value": value}
                    for action, value in direct_row.action_values
                ],
                "best_actions": list(direct_row.best_actions),
                "supplied_action_probability": direct_row.supplied_action_probability,
                "raw_gain": direct_row.raw_gain,
            }


def test_same_input_identity_and_json_are_byte_deterministic(tmp_path: Path):
    raw = EXAMPLE.read_bytes()
    first = module.aiof_supplied_profile_file_json(
        module.process_aiof_supplied_profile_file(raw)
    )
    second = module.aiof_supplied_profile_file_json(
        module.process_aiof_supplied_profile_file(raw)
    )
    assert first == second
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()

    run_doc = filled_run_document()
    path = tmp_path / "run.json"
    path.write_bytes(encoded(run_doc))
    first_cli = run_cli(path)
    second_cli = run_cli(path)
    assert first_cli.returncode == second_cli.returncode == 0
    assert first_cli.stderr == second_cli.stderr == ""
    assert first_cli.stdout == second_cli.stdout
    assert first_cli.stdout.count("\n") == 1


def test_template_identity_changes_with_bound_spec_and_old_identity_is_rejected():
    original = inspect_document()
    changed_doc = example_document()
    changed_doc["game"]["starting_stack_sb"] = 11
    changed = inspect_document(changed_doc)
    assert original.output is not None and changed.output is not None
    assert original.output["identity"] != changed.output["identity"]
    run_doc = filled_run_document(changed_doc)
    run_doc["template_identity"] = original.output["identity"]
    result = run_document(run_doc)
    assert result.status is module.AiofSuppliedProfileFileStatus.IDENTITY_MISMATCH
    assert_failure_is_no_partial(result)


def test_semantically_equivalent_range_order_has_same_canonical_identity():
    document = example_document()
    document["sb_range"] = [
        {"label": "AsAh", "weight": 1, "weight_basis": "exact_combo_mass"},
        {"label": "QcQd", "weight": 1, "weight_basis": "exact_combo_mass"},
    ]
    document["limits"]["max_range_entries_per_side"] = 2
    document["limits"]["max_exact_combos_per_side"] = 2
    document["limits"]["max_compatible_combo_pairs"] = 2
    first = inspect_document(document)
    reversed_document = copy.deepcopy(document)
    reversed_document["sb_range"].reverse()
    second = inspect_document(reversed_document)
    assert first.status is second.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    assert first.output is not None and second.output is not None
    assert first.output["identity"] == second.output["identity"]
    assert first.output["profile_template"] == second.output["profile_template"]


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"\xff", module.AiofSuppliedProfileFileStatus.PARSE_FAILURE),
        (b"\xef\xbb\xbf{}", module.AiofSuppliedProfileFileStatus.PARSE_FAILURE),
        (b"[]", module.AiofSuppliedProfileFileStatus.INVALID_INPUT),
        (b'{"weight":NaN}', module.AiofSuppliedProfileFileStatus.PARSE_FAILURE),
        (
            b'{"operation":"inspect","operation":"run"}',
            module.AiofSuppliedProfileFileStatus.PARSE_FAILURE,
        ),
    ],
)
def test_parse_failures_are_bounded_no_partial(raw: bytes, status):
    result = module.process_aiof_supplied_profile_file(raw)
    assert result.status is status
    assert_failure_is_no_partial(result)


def test_nonbytes_and_invalid_workflow_limit_are_rejected_before_parse():
    result = module.process_aiof_supplied_profile_file("{}")
    assert result.status is module.AiofSuppliedProfileFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)
    bad = replace(module.AiofSuppliedProfileFileLimits(), max_input_bytes=2_000_001)
    result = module.process_aiof_supplied_profile_file(b"{}", bad)
    assert result.status is module.AiofSuppliedProfileFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "bool_weight",
        "infinite_weight",
        "bad_game",
        "fee",
        "monte_carlo",
        "trace",
        "seat_order",
        "seed",
        "bool_limit",
        "limit_above_ceiling",
        "sampling_attempts",
        "empty_range",
    ],
)
def test_strict_schema_fixed_exact_controls_and_limits_reject_invalid(mutation: str):
    document = example_document()
    if mutation == "missing":
        del document["request_id"]
    elif mutation == "unknown":
        document["unknown"] = 1
    elif mutation == "bool_weight":
        document["sb_range"][0]["weight"] = True
    elif mutation == "infinite_weight":
        document["sb_range"][0]["weight"] = 10**400
    elif mutation == "bad_game":
        document["game"]["starting_stack_sb"] = 0
    elif mutation == "fee":
        document["game"]["fee"] = 0.1
    elif mutation == "monte_carlo":
        document["analysis"]["equity_algorithm"] = "deterministic_monte_carlo-v1"
    elif mutation == "trace":
        document["analysis"]["requested_trace_points"] = 1
        document["limits"]["max_trace_points"] = 1
    elif mutation == "seat_order":
        document["analysis"]["best_response_seats"] = ["bb", "sb"]
    elif mutation == "seed":
        document["analysis"]["seed"] = 1
    elif mutation == "bool_limit":
        document["limits"]["max_exact_board_evaluations"] = True
    elif mutation == "limit_above_ceiling":
        document["limits"]["max_dead_cards"] = 44
    elif mutation == "sampling_attempts":
        document["limits"]["max_sampling_attempts"] = 1
    elif mutation == "empty_range":
        document["sb_range"] = []
    result = inspect_document(document)
    assert result.status is module.AiofSuppliedProfileFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "mutation"),
    [
        ("max_input_bytes", 1, None),
        ("max_total_json_values", 1, None),
        ("max_json_depth", 1, None),
        ("max_dead_card_items", 42, None),
        ("max_range_entries_per_side", 1, "two_entries"),
    ],
)
def test_structural_caps_fire_before_prepare(
    monkeypatch, limit_name: str, limit_value: int, mutation: str | None
):
    document = example_document()
    if mutation == "two_entries":
        document["sb_range"].append(
            {"label": "AcAd", "weight": 1, "weight_basis": "exact_combo_mass"}
        )
        document["limits"]["max_range_entries_per_side"] = 2
        document["limits"]["max_exact_combos_per_side"] = 2
    called = False

    def bomb(*_args):
        nonlocal called
        called = True
        raise AssertionError("prepare must not run")

    monkeypatch.setattr(module, "prepare_compatible_ranges", bomb)
    limits = replace(module.AiofSuppliedProfileFileLimits(), **{limit_name: limit_value})
    result = inspect_document(document, limits)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert called is False
    assert_failure_is_no_partial(result)


def test_core_pair_and_board_caps_return_no_partial():
    inspect_doc = example_document()
    inspect_doc["limits"]["max_compatible_combo_pairs"] = 0
    result = inspect_document(inspect_doc)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert result.error is not None and result.error.nested_status == AiofStatus.CAP_EXCEEDED.value
    assert_failure_is_no_partial(result)

    run_doc = filled_run_document()
    run_doc["limits"]["max_exact_board_evaluations"] = 0
    inspected = inspect_document({key: value for key, value in run_doc.items() if key not in ("template_identity", "profile")} | {"operation": "inspect"})
    assert inspected.status is module.AiofSuppliedProfileFileStatus.SUCCESS
    run_doc["template_identity"] = inspected.output["identity"]
    result = run_document(run_doc)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "noncanonical",
        "bool",
        "nan",
        "below",
        "above",
        "unknown_key",
    ],
)
def test_profile_support_probability_and_schema_fail_closed(mutation: str):
    document = filled_run_document()
    rows = document["profile"]["sb_shove"]
    if mutation == "missing":
        rows.clear()
    elif mutation == "extra":
        rows.append({"combo": "AcAd", "probability": 0.5})
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "noncanonical":
        rows[0]["combo"] = "AhAs"
    elif mutation == "bool":
        rows[0]["probability"] = True
    elif mutation == "nan":
        rows[0]["probability"] = math.nan
    elif mutation == "below":
        rows[0]["probability"] = -0.1
    elif mutation == "above":
        rows[0]["probability"] = 1.1
    elif mutation == "unknown_key":
        rows[0]["unknown"] = 1
    raw = (
        json.dumps(document, allow_nan=True).encode()
        if mutation == "nan"
        else encoded(document)
    )
    result = module.run_aiof_supplied_profile_file(raw)
    expected = (
        module.AiofSuppliedProfileFileStatus.PARSE_FAILURE
        if mutation == "nan"
        else module.AiofSuppliedProfileFileStatus.PROFILE_FAILURE
    )
    assert result.status is expected
    assert_failure_is_no_partial(result)


def test_profile_row_cap_fires_before_analysis(monkeypatch):
    document = filled_run_document()
    called = False

    def bomb(_request):
        nonlocal called
        called = True
        raise AssertionError("analysis must not run")

    monkeypatch.setattr(module, "analyze_pushfold", bomb)
    limits = replace(module.AiofSuppliedProfileFileLimits(), max_profile_rows_per_side=1)
    document["profile"]["sb_shove"].append(
        {"combo": "AcAd", "probability": 0.5}
    )
    result = run_document(document, limits)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert called is False
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    ("nested", "outer"),
    [
        (AiofStatus.INVALID_INPUT, module.AiofSuppliedProfileFileStatus.ANALYSIS_FAILURE),
        (AiofStatus.CAP_EXCEEDED, module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED),
        (AiofStatus.INVALID_STRATEGY, module.AiofSuppliedProfileFileStatus.PROFILE_FAILURE),
        (AiofStatus.NON_REPRODUCIBLE, module.AiofSuppliedProfileFileStatus.NON_REPRODUCIBLE),
    ],
)
def test_controlled_analysis_failures_preserve_status_without_partial(
    monkeypatch, nested: AiofStatus, outer
):
    failure = PushFoldRunResult(nested, None, "controlled\nsecret")
    monkeypatch.setattr(module, "analyze_pushfold", lambda _request: failure)
    result = run_document(filled_run_document())
    assert result.status is outer
    assert result.error is not None and result.error.nested_status == nested.value
    assert_failure_is_no_partial(result)


def test_output_record_and_byte_caps_have_no_partial():
    inspect_limits = replace(module.AiofSuppliedProfileFileLimits(), max_output_records=1)
    result = inspect_document(limits=inspect_limits)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert_failure_is_no_partial(result)

    run_limits = replace(module.AiofSuppliedProfileFileLimits(), max_output_bytes=1)
    result = run_document(filled_run_document(), run_limits)
    assert result.status is module.AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    assert_failure_is_no_partial(result)


def test_malformed_success_and_unexpected_exception_fail_closed(monkeypatch):
    malformed = PushFoldRunResult(AiofStatus.SUCCESS, None, None)
    monkeypatch.setattr(module, "analyze_pushfold", lambda _request: malformed)
    result = run_document(filled_run_document())
    assert result.status is module.AiofSuppliedProfileFileStatus.INTERNAL_FAILURE
    assert_failure_is_no_partial(result)

    monkeypatch.setattr(module, "prepare_compatible_ranges", lambda *_args: 1 / 0)
    result = inspect_document()
    assert result.status is module.AiofSuppliedProfileFileStatus.INTERNAL_FAILURE
    assert result.error is not None and result.error.message == "unexpected workflow failure"
    assert_failure_is_no_partial(result)


def test_cli_inspect_run_and_controlled_failure_contract(tmp_path: Path):
    first = run_cli()
    second = run_cli()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert first.stdout.count("\n") == 1
    assert json.loads(first.stdout)["output"]["operation"] == "inspect"

    run_path = tmp_path / "run.json"
    run_path.write_bytes(encoded(filled_run_document()))
    run_result = run_cli(run_path)
    assert run_result.returncode == 0 and run_result.stderr == ""
    assert run_result.stdout.count("\n") == 1
    assert json.loads(run_result.stdout)["output"]["operation"] == "run"

    missing = run_cli(tmp_path / "missing.json")
    assert missing.returncode == 2 and missing.stderr == ""
    assert missing.stdout.count("\n") == 1
    payload = json.loads(missing.stdout)
    assert payload == {
        "status": "INVALID_INPUT",
        "output": None,
        "error": {
            "phase": "input",
            "message": "cannot read input file",
            "nested_status": None,
        },
    }
    assert "Traceback" not in missing.stdout


def test_success_output_excludes_runtime_path_trace_and_strategy_claims():
    serialized = module.aiof_supplied_profile_file_json(
        run_document(filled_run_document())
    )
    for forbidden in (
        "runtime_identity",
        "run_identity",
        "timestamp",
        "platform",
        "absolute_path",
        '"trace"',
        "claim_kind",
        "NASH",
        "profitability",
    ):
        assert forbidden not in serialized


def test_source_imports_only_nonprivate_public_aiof_names():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    public = {
        "aiof_cards": set(cards_module.__all__),
        "aiof_chip_ev": set(chip_ev_module.__all__),
        "aiof_equity": set(equity_module.__all__),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("aiof_"):
            assert node.level == 1
            assert all(not alias.name.startswith("_") for alias in node.names)
            assert {alias.name for alias in node.names} <= public[node.module]
        if isinstance(node, ast.Import):
            assert all("aiof_" not in alias.name for alias in node.names)


def test_cli_is_thin_and_docs_link_exact_command_and_boundaries():
    script_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(script_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "process_aiof_supplied_profile_file"
    ]
    assert len(calls) == 1
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    command = (
        "python scripts/run_aiof_supplied_profile_file.py "
        "examples/aiof_supplied_profile_file_v1.json"
    )
    assert "docs/aiof_supplied_profile_file_workflow.md" in readme
    assert command in readme and command in guide
    for phrase in (
        "two-phase",
        "no-partial",
        "fixed-opponent",
        "not endogenous",
        "real-money",
    ):
        assert phrase in readme.lower() and phrase in guide.lower()


def test_serializer_rejects_wrong_result_type():
    with pytest.raises(TypeError):
        module.aiof_supplied_profile_file_json(object())
