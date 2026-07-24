"""Public-workflow checks for the exact M30-M32 worked example."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from repeated_poker.automatic_commitment_selection import (
    NO_BENEFICIAL_COMMITMENT,
)
from repeated_poker.three_player_candidate_repeated import (
    EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
)
from repeated_poker.three_player_response import (
    EXACT_CORRESPONDENCE_COMPLETE,
)
from repeated_poker.three_player_river_rake import (
    EXACT_SCENARIO_RESPONSE_COMPLETE,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "three_player_candidate_repeated_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "three_player_candidate_repeated_workflow.md"
EXAMPLES_GUIDE = ROOT / "docs" / "examples_guide.md"
ASSUMPTIONS = ROOT / "docs" / "assumptions_and_limitations.md"
CHECK_MVP = ROOT / "scripts" / "check_mvp.py"

HALF_CANDIDATE_ID = (
    "7408e047d4e90f758c50da5fb43214cc99a62230c840a7bc04843e95cd7398f1"
)
FULL_CANDIDATE_ID = (
    "29aca66e522717db41f7324f20c763bcf8f7008a876f2255bda534c807c53ab9"
)
FULL_TIE_IDS = [FULL_CANDIDATE_ID, HALF_CANDIDATE_ID]


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "three_player_candidate_repeated_workflow_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_example(hash_seed: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    additions = os.pathsep.join((str(ROOT / "src"), str(ROOT / "examples")))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = additions if not existing else additions + os.pathsep + existing
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def hand_oracle() -> dict[str, object]:
    """Compute the fixture values without calling M30, M31, M32, or the example."""

    initial_contribution = {"H": Fraction(10), "O1": Fraction(10), "O2": Fraction(10)}
    initial_pot = sum(initial_contribution.values())

    check_line = {
        "H": initial_pot - initial_contribution["H"],
        "O1": -initial_contribution["O1"],
        "O2": -initial_contribution["O2"],
        "R": Fraction(0),
    }

    hero_bet = Fraction(20)
    uncalled_return = hero_bet
    hero_total_contribution = initial_contribution["H"] + hero_bet
    fold_line = {
        "H": initial_pot + uncalled_return - hero_total_contribution,
        "O1": -initial_contribution["O1"],
        "O2": -initial_contribution["O2"],
        "R": Fraction(0),
    }
    assert check_line == fold_line == {
        "H": Fraction(20),
        "O1": Fraction(-10),
        "O2": Fraction(-10),
        "R": Fraction(0),
    }

    horizon = 3
    discount = Fraction(1)
    baseline_total = sum(discount**period * check_line["H"] for period in range(horizon))
    deltas = {}
    for adaptation_opportunity in range(1, horizon + 2):
        pre_count = adaptation_opportunity - 1
        post_count = horizon - pre_count
        candidate_total = pre_count * fold_line["H"] + post_count * fold_line["H"]
        deltas[adaptation_opportunity] = candidate_total - baseline_total
    assert baseline_total == 60
    assert deltas == {1: 0, 2: 0, 3: 0, 4: 0}
    return {
        "utility": check_line,
        "baseline_total": baseline_total,
        "deltas": deltas,
    }


def expected_summary() -> dict[str, object]:
    oracle = hand_oracle()
    exact_utility = {
        player: str(value.numerator)
        for player, value in oracle["utility"].items()
    }
    candidates = [
        {
            "candidate_id": HALF_CANDIDATE_ID,
            "edits": [
                {
                    "information_set_id": "H_root",
                    "source_action_id": "check",
                    "target_action_id": "bet",
                    "shift_amount": "1/2",
                }
            ],
            "pre_adaptation_H": "20",
            "post_response_H_hero_worst": "20",
        },
        {
            "candidate_id": FULL_CANDIDATE_ID,
            "edits": [
                {
                    "information_set_id": "H_root",
                    "source_action_id": "check",
                    "target_action_id": "bet",
                    "shift_amount": "1",
                }
            ],
            "pre_adaptation_H": "20",
            "post_response_H_hero_worst": "20",
        },
    ]
    timing_rows = [
        {
            "adaptation_opportunity": opportunity,
            "status": NO_BENEFICIAL_COMMITMENT,
            "best_total_hero_ev_delta": float(oracle["deltas"][opportunity]),
            "selected_candidate_id": None,
            "full_primary_tie_candidate_ids": FULL_TIE_IDS,
            "display_candidate_id": HALF_CANDIDATE_ID,
        }
        for opportunity in range(1, 5)
    ]
    return {
        "status": EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
        "fixture": {
            "version": "m34-three-player-candidate-repeated-public-workflow-v1",
            "street": "river",
            "rake_rate": "0",
            "initial_pot": "30",
            "initial_contribution": {"H": "10", "O1": "10", "O2": "10"},
            "fixed_hero": {"H_root": {"check": "1", "bet": "0"}},
            "complete_initial_profile": {
                "O1": {
                    "O1_after_check": {"check": "1"},
                    "O1_after_bet": {"fold": "1"},
                },
                "O2": {
                    "O2_after_check": {"check": "1"},
                    "O2_after_bet": {"fold": "1"},
                },
            },
            "perfect_recall": {
                "o1_confirmed": True,
                "o2_confirmed": True,
                "evidence_version": (
                    "m34-three-player-candidate-repeated-public-workflow-v1"
                ),
                "human_trace": (
                    "each O1/O2 information set is singleton and no player "
                    "forgets a prior private observation or own action"
                ),
            },
            "search_mode": "robust_all",
            "adaptation_mode": "simultaneous_o1_o2",
            "shift_amounts": ["1/2", "1"],
            "max_simultaneous_info_sets": 1,
            "horizon": 3,
            "discount": 1.0,
        },
        "candidate_count": 2,
        "baseline_initial": exact_utility,
        "candidates": candidates,
        "hero_safety": {
            "source_path": "m31_scenario_response.response.hero_worst",
            "current_cfr_used": False,
            "first_witness_used": False,
            "pure_subset_used": False,
            "coalition_stress_used": False,
            "hero_best_used": False,
        },
        "timing_rows": timing_rows,
        "identities": {
            "scenario": (
                "df2ad726a9964d5b39eae73a5710ae884aeb0cc2d9ccb6196f6d440686618dab"
            ),
            "tree_structure": (
                "3ce0a045e7bba39a0c23afb183305d537242eab6d3cef8595afafb0be573a740"
            ),
            "baseline_fixed_hero": (
                "01a4366712189e5ce5d791828e66814321aea3d50c4edb5fcbe5fc2260840d92"
            ),
            "initial_profile": (
                "75f85a7fa6d0459dfd19f53ed98305e071b7c54974ef119a994d1290f7a59515"
            ),
            "baseline_response_game": (
                "c403f5c3b39f3713f4859628c9289e0a57c7a84022655a197df07a08ad2d6bef"
            ),
            "baseline_response_run": (
                "64cc2d873a14f4559565d25ae6aef4fe313446ae78a4b2b2f242bf699a602962"
            ),
            "candidate_universe": (
                "5678f45cab8ac35afe37d5ee0bdc06d1ec693c9300bb53bb82d47f8efa86f205"
            ),
            "m32_run": (
                "338c7eafe25dcab2737e99bf9511bf4133b3470a44b3926d93df6a088243665d"
            ),
        },
    }


def test_two_processes_are_byte_identical_and_match_exact_allowlist():
    first = run_example("1")
    second = run_example("777")
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert first.stdout.endswith("\n") and first.stdout.count("\n") == 1
    payload = json.loads(first.stdout, parse_constant=lambda value: pytest.fail(value))
    assert payload == expected_summary()
    assert set(payload) == {
        "status",
        "fixture",
        "candidate_count",
        "baseline_initial",
        "candidates",
        "hero_safety",
        "timing_rows",
        "identities",
    }
    assert len(first.stdout.encode("utf-8")) < 6_000


def test_public_result_matches_hand_oracle_and_native_hero_worst_only():
    module = load_example_module()
    arguments = module.worked_input_arguments()
    scenario = arguments["scenario"]
    policy = arguments["baseline_fixed_hero_policy"]
    profile = arguments["initial_profile"]
    generation = arguments["generation"]
    repeated = arguments["repeated"]
    attestation = arguments["attestation"]

    assert scenario.rake_rate == "0"
    assert scenario.initial_pot == "30"
    assert dict(policy.probabilities) == {"H_root": {"check": "1", "bet": "0"}}
    assert set(profile.o1_probabilities) == {"O1_after_check", "O1_after_bet"}
    assert set(profile.o2_probabilities) == {"O2_after_check", "O2_after_bet"}
    assert generation.shift_amounts == ("1/2", "1")
    assert generation.max_simultaneous_info_sets == 1
    assert generation.search_mode == "robust_all"
    assert generation.adaptation_mode == "simultaneous_o1_o2"
    assert repeated.horizon == 3 and repeated.discount == 1.0
    assert attestation.o1_confirmed is attestation.o2_confirmed is True
    assert attestation.verifier == "public worked-example fixture author"
    assert attestation.evidence_version == module.FIXTURE_VERSION

    result = module.run_analysis()
    assert result.status == EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE
    assert result.analysis is not None and result.error is None
    assert result.partial_result is False
    assert result.analysis.baseline_exact_values == {
        "H": "20",
        "O1": "-10",
        "O2": "-10",
        "R": "0",
    }
    for candidate in result.analysis.candidates:
        native = candidate.scenario_response.to_dict()
        assert native["status"] == EXACT_SCENARIO_RESPONSE_COMPLETE
        assert native["response"]["status"] == EXACT_CORRESPONDENCE_COMPLETE
        assert native["response"]["coverage"] == "complete"
        assert native["response"]["partial_response"] is False
        assert candidate.exact_values["initial_profile"]["H"] == "20"
        assert candidate.exact_values["response"] == {
            "hero_worst": native["response"]["hero_worst"],
            "hero_best": native["response"]["hero_best"],
            "safety_scalar_source_path": (
                "m31_scenario_response.response.hero_worst"
            ),
        }
        assert native["response"]["hero_worst"] == "20"
        assert candidate.scalar_projections["hero_worst"].source_exact == "20"

    oracle = hand_oracle()
    assert oracle["utility"]["H"] == Fraction(20)
    assert oracle["baseline_total"] == Fraction(60)
    assert tuple(oracle["deltas"].values()) == (0, 0, 0, 0)


def test_full_ties_remain_no_benefit_and_display_is_not_selection():
    payload = load_example_module().run_workflow()
    assert payload["candidate_count"] == 2
    for row in payload["timing_rows"]:
        assert row["status"] == NO_BENEFICIAL_COMMITMENT
        assert row["best_total_hero_ev_delta"] == 0.0
        assert row["selected_candidate_id"] is None
        assert row["full_primary_tie_candidate_ids"] == FULL_TIE_IDS
        assert row["display_candidate_id"] == HALF_CANDIDATE_ID


def test_controlled_failure_emits_no_partial_stdout(monkeypatch, capsys):
    module = load_example_module()
    monkeypatch.setattr(
        module,
        "run_analysis",
        lambda: SimpleNamespace(
            status="CAP_EXCEEDED",
            analysis=None,
            error=object(),
            partial_result=False,
        ),
    )
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "CAP_EXCEEDED" in captured.err
    assert "baseline_initial" not in captured.err


def test_example_imports_nonprivate_public_submodule_names_only():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_modules = {
        "repeated_poker.automatic_commitment_selection",
        "repeated_poker.three_player_candidate_repeated",
        "repeated_poker.three_player_response",
        "repeated_poker.three_player_river_rake",
    }
    repeated_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("repeated_poker")
            or isinstance(node, ast.Import)
            and any(alias.name.startswith("repeated_poker") for alias in node.names)
        )
    ]
    assert repeated_imports
    for node in repeated_imports:
        assert isinstance(node, ast.ImportFrom)
        assert node.module in allowed_modules
        assert node.names and all(not alias.name.startswith("_") for alias in node.names)
    assert "from repeated_poker import" not in source
    assert "tests." not in source
    assert "three_player_cfr" not in source


def test_stdout_omits_native_strategy_and_diagnostic_details():
    payload = load_example_module().run_workflow()
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "support_cells",
        "hero_worst_witnesses",
        "hero_best_witnesses",
        "pure_profile_unilateral_stability",
        "hero_min_joint_plan_stress",
        "complete_fixed_hero_policy",
    ):
        assert forbidden not in serialized
    assert payload["hero_safety"] == {
        "source_path": "m31_scenario_response.response.hero_worst",
        "current_cfr_used": False,
        "first_witness_used": False,
        "pure_subset_used": False,
        "coalition_stress_used": False,
        "hero_best_used": False,
    }


def test_output_numbers_are_finite_and_identities_are_full_sha256():
    payload = expected_summary()

    def check(value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)

    check(payload)
    for identity in payload["identities"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", identity)
    for candidate in payload["candidates"]:
        assert re.fullmatch(r"[0-9a-f]{64}", candidate["candidate_id"])


def test_public_docs_link_command_modes_safety_and_claim_boundaries():
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    examples_guide = EXAMPLES_GUIDE.read_text(encoding="utf-8")
    assumptions = ASSUMPTIONS.read_text(encoding="utf-8")
    command = "python examples/three_player_candidate_repeated_workflow.py"
    assert "docs/three_player_candidate_repeated_workflow.md" in readme
    assert command in readme and command in guide and command in examples_guide

    combined = " ".join((readme + guide + examples_guide + assumptions).split())
    for phrase in (
        "search_mode=robust_all",
        "adaptation_mode=simultaneous_o1_o2",
        "bounded finite",
        "m31_scenario_response.response.hero_worst",
        "current CFR",
        "first witness",
        "pure subset",
        "coalition stress",
        "hero_best",
        "not the guarded CFR-style diagnostic",
        "full solver",
        "Nash/equilibrium certificate",
        "global optimum",
        "real-card three-player",
        "profitability",
        "real-money advice",
    ):
        assert phrase in combined


def test_mvp_appends_exact_workflow_after_cfr_as_ninth_check():
    source = CHECK_MVP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "COMMANDS"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.List)
    assert len(assignment.value.elts) == 7

    spec = importlib.util.spec_from_file_location("m34_check_mvp", CHECK_MVP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.COMMANDS) == 9
    assert module.COMMANDS[7][1] == "examples/three_player_cfr_diagnostic_workflow.py"
    assert (
        module.COMMANDS[8][1]
        == "examples/three_player_candidate_repeated_workflow.py"
    )
