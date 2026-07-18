"""Independent contract checks for the public-only M13 worked workflow."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from repeated_poker.aiof_strategy import (
    AiofStrategyStatus,
    RationalStrategyRunResult,
    StrategyError,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "aiof_real_card_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "aiof_real_card_workflow.md"
EXAMPLES_GUIDE = ROOT / "docs" / "examples_guide.md"
CHECK_MVP = ROOT / "scripts" / "check_mvp.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("aiof_real_card_workflow_example", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_example() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    additions = os.pathsep.join((str(ROOT / "src"), str(ROOT / "examples")))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = additions if not existing else additions + os.pathsep + existing
    return subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def five_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Test-owned Hold'em oracle, intentionally independent of production code."""

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
        return 3, (trip,) + tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        return 2, (pairs[0], pairs[1], max(rank for rank, count in counts.items() if count == 1))
    if len(pairs) == 1:
        pair = pairs[0]
        return 1, (pair,) + tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
    return 0, tuple(sorted(ranks, reverse=True))


def seven_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return max(five_card_rank(tuple(combo)) for combo in itertools.combinations(cards, 5))


def expected_summary() -> dict[str, object]:
    board = ("2c", "3d", "4h", "5s", "7c")
    assert seven_card_rank(("As", "Ah") + board) > seven_card_rank(("Ks", "Kh") + board)

    sb_post = Fraction(1, 2)
    bb_post = Fraction(1)
    sb_shove_bb_fold = bb_post
    sb_value = sb_shove_bb_fold
    bb_value = -sb_value
    assert sb_post == Fraction(1, 2)
    assert sb_value + bb_value == 0

    return {
        "fixture": {
            "sb_combo": "AsAh",
            "bb_combo": "KsKh",
            "known_board": list(board),
            "dead_card_count": 43,
        },
        "equity": {
            "algorithm": "exact_exhaustive-v1",
            "trials": 1,
            "wins": 1,
            "losses": 0,
            "ties": 0,
            "board_evaluations": 1,
        },
        "chip_ev": {
            "profile_value_sb": float(sb_value),
            "profile_value_bb": float(bb_value),
            "conservation_sum": 0.0,
            "fixed_opponent_sb_gain": 0.0,
            "fixed_opponent_bb_gain": 0.0,
        },
        "rational_strategy": {
            "game_id": "aiof-rational-lift-game-v1",
            "claim_kind": "aiof-rational-lift-game-v1:EXACT_NASH",
            "profile_value": "1",
            "g_sb": "0",
            "g_bb": "0",
            "payoff_cell_count": 1,
            "exact_board_evaluations": 1,
        },
    }


def test_worked_example_subprocess_is_bounded_deterministic_and_oracle_exact():
    first = run_example()
    second = run_example()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert len(first.stdout.encode("utf-8")) < 2_000
    assert first.stdout.endswith("\n") and first.stdout.count("\n") == 1
    assert json.loads(first.stdout) == expected_summary()


def test_same_public_input_has_identical_semantic_summary_twice():
    module = load_example_module()
    assert module.run_workflow() == module.run_workflow() == expected_summary()


def test_example_imports_only_public_names_from_m13_modules():
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "repeated_poker.aiof_"
        ):
            assert node.names
            assert all(not alias.name.startswith("_") for alias in node.names)
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("repeated_poker.aiof_") for alias in node.names)


def test_controlled_strategy_failure_is_nonzero_and_prints_no_partial_payload(
    monkeypatch, capsys
):
    module = load_example_module()
    failure = RationalStrategyRunResult(
        AiofStrategyStatus.VERIFICATION_FAILED,
        None,
        StrategyError("controlled failure", "verification", None, 1, 0),
    )
    monkeypatch.setattr(module, "generate_rational_lift_strategy", lambda _request: failure)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "VERIFICATION_FAILED" in captured.err
    assert "profile_value" not in captured.err
    assert "claim_kind" not in captured.err


def test_public_docs_link_command_and_guardrail_contract():
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    examples_guide = EXAMPLES_GUIDE.read_text(encoding="utf-8")
    example = EXAMPLE_PATH.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_guide = " ".join(guide.split())
    command = "python examples/aiof_real_card_workflow.py"

    assert "docs/aiof_real_card_workflow.md" in readme
    assert command in readme and command in guide and command in examples_guide
    assert "exact and deterministic Monte Carlo are non-interchangeable" in guide
    assert "no partial payload" in guide
    assert "fixed-opponent response" in normalized_readme
    assert "fixed-opponent response" in normalized_guide
    assert "aiof-rational-lift-game-v1" in readme and "aiof-rational-lift-game-v1" in guide
    assert "not a range chart" in normalized_readme and "not a range chart" in normalized_guide
    assert "allow_nan=False" in example


def test_mvp_keeps_order_and_adds_worked_example_as_one_command():
    tree = ast.parse(CHECK_MVP.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMANDS" for target in node.targets
        )
    )
    command_text = ast.unparse(assignment.value)
    assert command_text.count("examples/aiof_real_card_workflow.py") == 1
    assert command_text.index("examples/aiof_real_card_workflow.py") > command_text.index(
        "examples/candidate_filters.py"
    )
