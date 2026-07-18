"""Independent contract checks for the M21 AIoF rational-lift file adapter."""

from __future__ import annotations

import ast
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.aiof_rational_lift_file_workflow as module
from repeated_poker.aiof_cards import RangeEntry, RangeSpec, WeightBasis
from repeated_poker.aiof_chip_ev import HeadsUpChipEvGame
from repeated_poker.aiof_equity import EquityAlgorithm
from repeated_poker.aiof_strategy import (
    AiofStrategyAlgorithm,
    AiofStrategyLimits,
    AiofStrategyStatus,
    RationalStrategyRequest,
    RationalStrategyRunResult,
    StrategyError,
    generate_rational_lift_strategy,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "aiof_rational_lift_file_v1.json"
SCRIPT = ROOT / "scripts" / "run_aiof_rational_lift_file.py"
SOURCE = ROOT / "src" / "repeated_poker" / "aiof_rational_lift_file_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "aiof_rational_lift_file_workflow.md"
FORMAT = "aiof-rational-lift-file-v1"


def example_document() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def encoded(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def run_document(
    document: object,
    limits: module.AiofRationalLiftFileLimits = module.AiofRationalLiftFileLimits(),
) -> module.AiofRationalLiftFileResult:
    return module.run_aiof_rational_lift_file(encoded(document), limits)


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


def five_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Test-owned Hold'em oracle, independent of production evaluators."""

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


def direct_public_run(document: dict[str, object]) -> RationalStrategyRunResult:
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

    game_doc = document["game"]
    strategy_doc = document["strategy"]
    limits_doc = document["limits"]
    assert isinstance(game_doc, dict)
    assert isinstance(strategy_doc, dict)
    assert isinstance(limits_doc, dict)
    game = HeadsUpChipEvGame(
        float(game_doc["starting_stack_sb"]),
        float(game_doc["starting_stack_bb"]),
        float(game_doc["small_blind"]),
        float(game_doc["big_blind"]),
        float(game_doc["ante"]),
        float(game_doc["fee"]),
        float(game_doc["third_party_dead_money"]),
        bool(game_doc["side_pot"]),
    )
    request = RationalStrategyRequest(
        public_range(document["sb_range"]),
        public_range(document["bb_range"]),
        tuple(document["dead_cards"]),
        EquityAlgorithm(strategy_doc["equity_algorithm"]),
        game,
        AiofStrategyLimits(**limits_doc),
        AiofStrategyAlgorithm(strategy_doc["algorithm"]),
        Fraction(strategy_doc["claim_epsilon"]),
        Fraction(strategy_doc["display_tie_tolerance"]),
        int(strategy_doc["requested_trace_points"]),
        bool(strategy_doc["run_reference_oracle"]),
        bool(strategy_doc["run_phase1_float_diagnostic"]),
        strategy_doc["seed"],
        strategy_doc["samples"],
    )
    return generate_rational_lift_strategy(request)


def assert_failure_is_no_partial(result: module.AiofRationalLiftFileResult) -> None:
    assert result.status is not module.AiofRationalLiftFileStatus.SUCCESS
    assert result.output is None
    assert result.error is not None
    serialized = module.aiof_rational_lift_file_json(result)
    payload = json.loads(serialized)
    assert payload["output"] is None
    assert set(payload) == {"status", "output", "error"}
    forbidden = (
        "profile_value",
        "claim_kind",
        "semantic_identity",
        "payoff_cell_count",
        "completed_payoff_cells",
        "completed_pivots",
        "verification_identity",
    )
    assert all(name not in serialized for name in forbidden)
    assert "\n" not in result.error.message and "\r" not in result.error.message
    assert len(result.error.message) <= 500


def test_example_success_matches_hand_oracle_and_exact_output_contract():
    board = ("2c", "3d", "4h", "5s", "7c")
    assert seven_card_rank(("As", "Ah") + board) > seven_card_rank(("Ks", "Kh") + board)
    sb_fold_value = Fraction(-1, 2)
    sb_shove_bb_fold_value = Fraction(1)
    assert sb_shove_bb_fold_value > sb_fold_value

    result = run_document(example_document())
    assert result.status is module.AiofRationalLiftFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["format_version"] == FORMAT
    assert output["request_id"] == "tiny-aa-vs-kk-known-board"
    assert output["algorithm"] == "aiof-compact-zero-sum-rational-lp-v1"
    assert output["game_id"] == "aiof-rational-lift-game-v1"
    assert output["payoff_cell_count"] == 1
    assert output["exact_board_evaluations"] == 1
    assert output["profile"]["sb_shove"] == [{"combo": "AsAh", "probability": "1"}]
    assert output["profile"]["bb_call"] == [{"combo": "KsKh", "probability": "0"}]
    witness = output["witness"]
    assert witness["claim_kind"] == "aiof-rational-lift-game-v1:EXACT_NASH"
    assert witness["claim_epsilon"] == witness["numeric_error_bound"] == "0"
    assert witness["primal_feasible"] is witness["dual_feasible"] is True
    assert witness["zero_objective_gap"] is True
    assert witness["gains"] == {
        "profile_value": "1",
        "sb_best_response_value": "1",
        "bb_best_response_sb_value": "1",
        "g_sb": "0",
        "g_bb": "0",
        "nash_conv": "0",
        "max_unilateral_gain": "0",
        "value_lower": "1",
        "value_upper": "1",
    }
    assert output["oracle_comparison"] is output["phase1_float_diagnostic"] is None
    for name in (
        "prepared_ranges_identity",
        "payoff_identity",
        "semantic_identity",
        "input_identity",
    ):
        assert len(output[name]) == 64 and set(output[name]) <= set("0123456789abcdef")
    serialized = module.aiof_rational_lift_file_json(result)
    assert "runtime_identity" not in serialized
    assert "run_identity" not in serialized
    assert "timestamp" not in serialized


def test_adapter_projection_matches_direct_existing_public_api():
    document = example_document()
    wrapped = run_document(document)
    direct = direct_public_run(document)
    assert wrapped.output is not None
    strategy = direct.strategy_result
    assert direct.status is AiofStrategyStatus.SUCCESS and strategy is not None
    output = wrapped.output
    assert output["semantic_identity"] == strategy.semantic_identity
    assert output["input_identity"] == strategy.input_identity
    assert output["prepared_ranges_identity"] == strategy.prepared_ranges_identity
    assert output["payoff_identity"] == strategy.payoff_identity
    assert output["profile"]["content_identity"] == strategy.profile.content_identity
    assert output["witness"]["verification_identity"] == strategy.witness.verification_identity


def test_same_content_and_cli_are_byte_deterministic():
    raw = EXAMPLE.read_bytes()
    first = module.aiof_rational_lift_file_json(module.run_aiof_rational_lift_file(raw))
    second = module.aiof_rational_lift_file_json(module.run_aiof_rational_lift_file(raw))
    assert first == second
    first_cli = run_cli()
    second_cli = run_cli()
    assert first_cli.returncode == second_cli.returncode == 0
    assert first_cli.stderr == second_cli.stderr == ""
    assert first_cli.stdout == second_cli.stdout == first + "\n"
    assert first_cli.stdout.count("\n") == 1


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"\xff", module.AiofRationalLiftFileStatus.PARSE_FAILURE),
        (b"\xef\xbb\xbf{}", module.AiofRationalLiftFileStatus.PARSE_FAILURE),
        (b"[]", module.AiofRationalLiftFileStatus.INVALID_INPUT),
        (b'{"weight":NaN}', module.AiofRationalLiftFileStatus.PARSE_FAILURE),
        (
            b'{"request_id":"a","request_id":"b"}',
            module.AiofRationalLiftFileStatus.PARSE_FAILURE,
        ),
    ],
)
def test_parse_failures_are_bounded_and_no_partial(raw: bytes, status):
    result = module.run_aiof_rational_lift_file(raw)
    assert result.status is status
    assert_failure_is_no_partial(result)


def test_non_bytes_and_invalid_workflow_limits_fail_before_parse():
    result = module.run_aiof_rational_lift_file("{}")
    assert result.status is module.AiofRationalLiftFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)
    limits = replace(module.AiofRationalLiftFileLimits(), max_input_bytes=1_000_001)
    result = module.run_aiof_rational_lift_file(b"{}", limits)
    assert result.status is module.AiofRationalLiftFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "bool_weight",
        "zero_weight",
        "noncanonical_fraction",
        "reference_oracle",
        "phase1_diagnostic",
        "monte_carlo",
        "seed",
        "side_pot",
        "fee",
        "bool_limit",
        "limit_above_ceiling",
        "empty_range",
    ],
)
def test_strict_schema_and_fixed_v1_controls_reject_invalid_documents(mutation: str):
    document = example_document()
    if mutation == "missing":
        del document["request_id"]
    elif mutation == "unknown":
        document["unknown"] = 1
    elif mutation == "bool_weight":
        document["sb_range"][0]["weight"] = True
    elif mutation == "zero_weight":
        document["sb_range"][0]["weight"] = 0
    elif mutation == "noncanonical_fraction":
        document["strategy"]["claim_epsilon"] = "0/1"
    elif mutation == "reference_oracle":
        document["strategy"]["run_reference_oracle"] = True
    elif mutation == "phase1_diagnostic":
        document["strategy"]["run_phase1_float_diagnostic"] = True
    elif mutation == "monte_carlo":
        document["strategy"]["equity_algorithm"] = "deterministic_monte_carlo-v1"
    elif mutation == "seed":
        document["strategy"]["seed"] = 1
    elif mutation == "side_pot":
        document["game"]["side_pot"] = True
    elif mutation == "fee":
        document["game"]["fee"] = 0.1
    elif mutation == "bool_limit":
        document["limits"]["max_payoff_cells"] = True
    elif mutation == "limit_above_ceiling":
        document["limits"]["max_solver_combos_per_side"] = 65
    elif mutation == "empty_range":
        document["sb_range"] = []
    result = run_document(document)
    assert result.status is module.AiofRationalLiftFileStatus.INVALID_INPUT
    assert_failure_is_no_partial(result)


def test_canonical_nonzero_fraction_is_accepted():
    document = example_document()
    document["strategy"]["claim_epsilon"] = "1/100"
    result = run_document(document)
    assert result.status is module.AiofRationalLiftFileStatus.SUCCESS
    assert result.output is not None
    assert result.output["witness"]["claim_epsilon"] == "1/100"


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "document_mutation"),
    [
        ("max_input_bytes", 1, None),
        ("max_total_json_values", 1, None),
        ("max_json_depth", 1, None),
        ("max_dead_card_items", 42, None),
        ("max_range_entries_per_side", 1, "two_entries"),
    ],
)
def test_structural_caps_fire_before_strategy_call(
    monkeypatch, limit_name: str, limit_value: int, document_mutation: str | None
):
    document = example_document()
    if document_mutation == "two_entries":
        document["sb_range"].append(
            {"label": "AcAd", "weight": 1, "weight_basis": "exact_combo_mass"}
        )
        document["limits"]["max_solver_combos_per_side"] = 2
    called = False

    def bomb(_request):
        nonlocal called
        called = True
        raise AssertionError("strategy must not run")

    monkeypatch.setattr(module, "generate_rational_lift_strategy", bomb)
    limits = replace(module.AiofRationalLiftFileLimits(), **{limit_name: limit_value})
    result = run_document(document, limits)
    assert result.status is module.AiofRationalLiftFileStatus.CAP_EXCEEDED
    assert called is False
    assert_failure_is_no_partial(result)


def test_upstream_invalid_card_preserves_exact_nested_status_without_partial():
    document = example_document()
    document["dead_cards"][0] = "ZZ"
    result = run_document(document)
    assert result.status is module.AiofRationalLiftFileStatus.STRATEGY_FAILURE
    assert result.error is not None
    assert result.error.nested_status == AiofStrategyStatus.INPUT_PREPARATION_FAILED.value
    assert_failure_is_no_partial(result)


@pytest.mark.parametrize(
    ("nested", "outer"),
    [
        (
            AiofStrategyStatus.VERIFICATION_FAILED,
            module.AiofRationalLiftFileStatus.STRATEGY_FAILURE,
        ),
        (
            AiofStrategyStatus.CAP_EXCEEDED,
            module.AiofRationalLiftFileStatus.CAP_EXCEEDED,
        ),
        (
            AiofStrategyStatus.NON_REPRODUCIBLE,
            module.AiofRationalLiftFileStatus.NON_REPRODUCIBLE,
        ),
    ],
)
def test_controlled_strategy_failures_map_status_and_hide_partial_counts(
    monkeypatch, nested: AiofStrategyStatus, outer
):
    failure = RationalStrategyRunResult(
        nested,
        None,
        StrategyError("controlled\nsecret", "verification", None, 99, 88),
    )
    monkeypatch.setattr(module, "generate_rational_lift_strategy", lambda _request: failure)
    result = run_document(example_document())
    assert result.status is outer
    assert result.error is not None and result.error.nested_status == nested.value
    assert_failure_is_no_partial(result)


def test_output_caps_return_no_partial(monkeypatch):
    direct = direct_public_run(example_document())
    assert direct.status is AiofStrategyStatus.SUCCESS
    monkeypatch.setattr(module, "generate_rational_lift_strategy", lambda _request: direct)
    limits = replace(module.AiofRationalLiftFileLimits(), max_output_records=1)
    result = run_document(example_document(), limits)
    assert result.status is module.AiofRationalLiftFileStatus.CAP_EXCEEDED
    assert_failure_is_no_partial(result)


def test_impossible_success_shape_fails_closed(monkeypatch):
    malformed = RationalStrategyRunResult(AiofStrategyStatus.SUCCESS, None, None)
    monkeypatch.setattr(module, "generate_rational_lift_strategy", lambda _request: malformed)
    result = run_document(example_document())
    assert result.status is module.AiofRationalLiftFileStatus.INTERNAL_FAILURE
    assert_failure_is_no_partial(result)


def test_cli_read_failure_is_json_exit_two_and_no_stderr(tmp_path: Path):
    missing = tmp_path / "missing.json"
    result = run_cli(missing)
    assert result.returncode == 2
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "INVALID_INPUT"
    assert payload["output"] is None
    assert payload["error"] == {
        "message": "cannot read input file",
        "nested_status": None,
        "phase": "input",
    }


def test_source_imports_only_nonprivate_public_aiof_names():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("aiof_"):
            assert node.level == 1
            assert node.names
            assert all(not alias.name.startswith("_") for alias in node.names)
        if isinstance(node, ast.Import):
            assert all("aiof_" not in alias.name for alias in node.names)


def test_docs_link_exact_command_and_claim_boundaries():
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    normalized_readme = readme.lower()
    normalized_guide = guide.lower()
    command = (
        "python scripts/run_aiof_rational_lift_file.py "
        "examples/aiof_rational_lift_file_v1.json"
    )
    assert "docs/aiof_rational_lift_file_workflow.md" in readme
    assert command in readme and command in guide
    for phrase in (
        "no-partial",
        "runtime",
        "external-game",
        "real-money",
        "exact rational-lift",
    ):
        assert phrase in normalized_readme and phrase in normalized_guide


def test_json_serializer_rejects_wrong_result_type():
    with pytest.raises(TypeError):
        module.aiof_rational_lift_file_json(object())
