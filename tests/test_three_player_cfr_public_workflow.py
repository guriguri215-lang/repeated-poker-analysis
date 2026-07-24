"""Independent contract checks for the public-only M12 submodule workflow."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker.three_player_cfr import (
    NUMERIC_FAILURE,
    ORACLE_UNAVAILABLE_CAP,
    UNSUPPORTED_MODEL,
    CfrConfig,
    CfrSafetyLimits,
    DiagnosticContractError,
    ThreePlayerGameTree,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "three_player_cfr_diagnostic_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "three_player_cfr_diagnostic_workflow.md"
EXAMPLES_GUIDE = ROOT / "docs" / "examples_guide.md"
CHECK_MVP = ROOT / "scripts" / "check_mvp.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "three_player_cfr_diagnostic_workflow_example", EXAMPLE_PATH
    )
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


def expected_summary() -> dict[str, object]:
    """Test-owned hand oracle plus deterministic two-iteration regression."""

    utilities = {
        ("A", "L"): (-2.0, 1.0, 1.0, 0.0),
        ("A", "R"): (0.0, 0.0, 0.0, 0.0),
        ("B", "L"): (0.0, 0.0, 0.0, 0.0),
        ("B", "R"): (-4.0, 2.0, 2.0, 0.0),
    }
    assert len(utilities) == 4
    stable = []
    for row, column in utilities:
        current = utilities[(row, column)]
        other_row = "B" if row == "A" else "A"
        other_column = "R" if column == "L" else "L"
        gain_o1 = utilities[(other_row, column)][1] - current[1]
        gain_o2 = utilities[(row, other_column)][2] - current[2]
        if gain_o1 <= 0 and gain_o2 <= 0:
            stable.append((row, column))
    assert stable == [("A", "L"), ("B", "R")]
    assert 2 * len(utilities) + 2 + 2 + 1 == 13
    return {
        "status": {
            "component": "DIAGNOSTIC_COMPLETE",
            "overall": "DIAGNOSTIC_COMPLETE",
        },
        "qualified_diagnostic": (
            "deterministic two-iteration fixed-Hero three-player CFR-style diagnostic snapshot"
        ),
        "iterations": {"requested": 2, "completed": 2},
        "expected_utility": {"H": -2.375, "O1": 1.1875, "O2": 1.1875, "R": 0.0},
        "unilateral_deviation_gain": {"O1": 0.3125, "O2": 0.3125},
        "oracle": {
            "status": "MATCH",
            "coverage": "complete",
            "counts": {
                "pure_plans": {"O1": 2, "O2": 2},
                "joint_profiles": 4,
                "complete_table_rows": 4,
                "predicted_profile_evaluations": 13,
                "actual_profile_evaluations": 13,
                "predicted_output_rows": 0,
                "actual_output_rows": 0,
            },
            "pure_profile_unilateral_stability_rows": 2,
            "warnings": [
                "multiple_pure_profile_unilateral_stability_rows",
                "pure_profile_utility_ties_present",
            ],
        },
    }


def test_subprocess_is_strict_deterministic_and_matches_independent_projection():
    first = run_example()
    second = run_example()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert first.stdout.endswith("\n") and first.stdout.count("\n") == 1
    payload = json.loads(first.stdout, parse_constant=lambda value: pytest.fail(value))
    assert payload == expected_summary()
    assert set(payload) == {
        "status",
        "qualified_diagnostic",
        "iterations",
        "expected_utility",
        "unilateral_deviation_gain",
        "oracle",
    }
    assert len(first.stdout.encode("utf-8")) < 1_500


def test_result_matches_fixture_counts_attestation_and_bounded_regression():
    module = load_example_module()
    arguments = module.worked_input_arguments()
    attestation = arguments["attestation"]
    tree = arguments["tree"]
    assert attestation.tree_content_identity == module.tree_content_identity(tree)
    assert attestation.o1_confirmed is attestation.o2_confirmed is True
    assert attestation.verifier and attestation.verification_date
    assert attestation.evidence_version == module.FIXTURE_VERSION
    result = module.run_diagnostic()
    assert result.to_dict()["expected_utility_vector"] == expected_summary()["expected_utility"]
    assert result.unilateral_deviation_gain_by_player == {"O1": 0.3125, "O2": 0.3125}
    assert result.oracle_attachment["counts"]["actual_profile_evaluations"] == 13
    assert result.oracle_attachment["stable_profile_count"] == 2


def test_oracle_joint_cap_is_no_partial_and_main_emits_no_success(monkeypatch, capsys):
    module = load_example_module()
    config = CfrConfig(
        iterations=2,
        request_oracle=True,
        include_oracle_rows=False,
        limits=CfrSafetyLimits(max_oracle_joint_profiles=3),
    )
    capped = module.run_diagnostic(config=config)
    assert capped.overall_status == ORACLE_UNAVAILABLE_CAP
    assert capped.oracle_attachment["coverage"] == "none"
    assert capped.oracle_attachment["rows"] == []
    assert capped.oracle_attachment["counts"]["joint_profiles"] == 4
    monkeypatch.setattr(module, "run_diagnostic", lambda: capped)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert ORACLE_UNAVAILABLE_CAP in captured.err
    assert "expected_utility" not in captured.err


def test_changed_tree_invalidates_old_attestation_and_emits_no_partial(monkeypatch, capsys):
    module = load_example_module()
    arguments = module.worked_input_arguments()
    changed_tree = ThreePlayerGameTree(
        arguments["tree"].root,
        description="m20-three-player-public-example-v1-changed",
    )
    with pytest.raises(DiagnosticContractError) as mismatch:
        module.run_diagnostic(tree=changed_tree)
    assert mismatch.value.status == UNSUPPORTED_MODEL
    assert "identity mismatch" in str(mismatch.value)

    def raise_mismatch():
        raise mismatch.value

    monkeypatch.setattr(module, "run_diagnostic", raise_mismatch)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert UNSUPPORTED_MODEL in captured.err
    assert "expected_utility" not in captured.err


@pytest.mark.parametrize("status", [NUMERIC_FAILURE, UNSUPPORTED_MODEL])
def test_contract_errors_are_nonzero_short_stderr_and_stdout_empty(status, monkeypatch, capsys):
    module = load_example_module()

    def fail():
        raise DiagnosticContractError(status, "controlled contract failure")

    monkeypatch.setattr(module, "run_diagnostic", fail)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error [{status}]: controlled contract failure\n"


def test_example_imports_nonprivate_public_submodule_names_only():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
        assert node.module == "repeated_poker.three_player_cfr"
        assert node.names and all(not alias.name.startswith("_") for alias in node.names)
    assert "from repeated_poker import" not in source
    assert "tests." not in source


def test_summary_contains_only_finite_json_numbers():
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


def test_public_docs_link_command_status_and_claim_boundaries():
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    examples_guide = EXAMPLES_GUIDE.read_text(encoding="utf-8")
    command = "python examples/three_player_cfr_diagnostic_workflow.py"
    normalized = " ".join(guide.split())
    assert "docs/three_player_cfr_diagnostic_workflow.md" in readme
    assert command in readme and command in guide and command in examples_guide
    for phrase in (
        "fixed Hero",
        "separate O1/O2",
        "DIAGNOSTIC_COMPLETE",
        "ORACLE_UNAVAILABLE_CAP",
        "UNSUPPORTED_MODEL",
        "human-authored evidence",
        "allocation-before-materialization",
        "joint or coalition",
        "not a convergence",
        "real-money advice",
    ):
        assert phrase in normalized or phrase in " ".join(readme.split())


def test_mvp_keeps_cfr_as_eighth_and_appends_exact_workflow_as_ninth():
    source = CHECK_MVP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COMMANDS" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.List)
    assert len(assignment.value.elts) == 7
    cfr_example = "examples/three_player_cfr_diagnostic_workflow.py"
    exact_example = "examples/three_player_candidate_repeated_workflow.py"
    assert source.count(cfr_example) == 1
    assert source.count(exact_example) == 1
    assert source.index(cfr_example) > source.index(
        "examples/stage_plan_diagnostic_workflow.py"
    )
    assert source.index(exact_example) > source.index(cfr_example)

    spec = importlib.util.spec_from_file_location("m20_check_mvp", CHECK_MVP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.COMMANDS) == 9
    assert module.COMMANDS[-2][1] == cfr_example
    assert module.COMMANDS[-1][1] == exact_example
