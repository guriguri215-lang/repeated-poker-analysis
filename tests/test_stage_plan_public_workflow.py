"""Independent contract checks for the public-only M11 worked workflow."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from repeated_poker import (
    COOPERATE,
    STAGE_PLAN_HERO,
    STAGE_PLAN_VILLAIN,
    DiagnosticStatus,
    GameTree,
    HeroNode,
    TerminalNode,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "stage_plan_diagnostic_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "stage_plan_diagnostic_workflow.md"
EXAMPLES_GUIDE = ROOT / "docs" / "examples_guide.md"
CHECK_MVP = ROOT / "scripts" / "check_mvp.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "stage_plan_diagnostic_workflow_example", EXAMPLE_PATH
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
    """Test-owned hand oracle; it does not call a production serializer/helper."""

    prescribed_value = Fraction(0)
    deviate_gain = Fraction(1) - prescribed_value
    assert deviate_gain == 1
    assert Fraction(1, 2) * prescribed_value == 0
    hero_plans = 2
    villain_plans = 1
    deviation_rows = 2 * (hero_plans + villain_plans)
    assert deviation_rows == 6
    return {
        "status": "FAIL",
        "qualified_claim": "bounded exhaustive one-period stage-plan deviation diagnostic",
        "counts": {
            "hero_plans": hero_plans,
            "villain_plans": villain_plans,
            "deviation_rows": deviation_rows,
        },
        "exact_fraction_strings": {
            "maximum_lower": "1",
            "maximum_upper": "1",
        },
    }


def test_worked_example_subprocess_is_strict_deterministic_and_oracle_exact():
    first = run_example()
    second = run_example()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert first.stdout.endswith("\n") and first.stdout.count("\n") == 1
    assert len(first.stdout.encode("utf-8")) < 1_000
    assert json.loads(first.stdout) == expected_summary()


def test_exact_result_matches_manual_value_gain_and_row_oracle():
    module = load_example_module()
    result = module.run_diagnostic()
    assert result.status is DiagnosticStatus.FAIL
    assert result.prescribed_values[(STAGE_PLAN_HERO, COOPERATE)] == 0
    assert result.plan_counts == {STAGE_PLAN_HERO: 2, STAGE_PLAN_VILLAIN: 1}
    assert len(result.deviations) == 6
    deviate = next(
        row
        for row in result.deviations
        if row.player == STAGE_PLAN_HERO
        and row.state == COOPERATE
        and dict(row.plan.actions) == {"H1": "deviate"}
    )
    assert deviate.prescribed_value == 0
    assert deviate.deviation_value == deviate.gain == 1
    assert (result.maximum_lower, result.maximum_upper) == (Fraction(1), Fraction(1))


def test_fixture_authors_complete_manual_evidence_instead_of_generating_it():
    module = load_example_module()
    arguments = module.worked_input_arguments()
    model = arguments["model_attestation"]
    assert all(getattr(model, name) is True for name in model.__dataclass_fields__)

    recall = arguments["perfect_recall_attestation"]
    assert recall.target_version == recall.valid_through_version == module.FIXTURE_VERSION
    assert recall.information_set_members == {
        STAGE_PLAN_HERO: {"H1": ("h",)},
        STAGE_PLAN_VILLAIN: {},
    }
    history = recall.member_histories[STAGE_PLAN_HERO]["H1"]["h"]
    assert history.observations == history.own_actions == history.information_sets == ()
    assert recall.legal_actions[STAGE_PLAN_HERO]["H1"] == ("stay", "deviate")
    assert recall.reviewer and recall.review_method and recall.evidence
    assert recall.known_limitations and recall.invalidation_conditions


def test_cap_is_preallocation_unsupported_and_invalid_cap_is_input_error():
    module = load_example_module()
    capped = module.run_diagnostic(max_plans_per_player=1)
    assert capped.status is DiagnosticStatus.UNSUPPORTED
    assert capped.plan_counts == {STAGE_PLAN_HERO: 2, STAGE_PLAN_VILLAIN: 1}
    assert capped.prescribed_values == {}
    assert capped.deviations == ()
    assert capped.maximum_lower is capped.maximum_upper is None

    with pytest.raises(ValueError, match="must be positive"):
        module.run_diagnostic(max_plans_per_player=0)


def test_tree_content_change_invalidates_attestation_without_partial_result():
    module = load_example_module()
    changed_tree = GameTree(
        HeroNode(
            "h",
            "H1",
            (
                ("stay", TerminalNode("stay_terminal", Fraction(0), Fraction(0), Fraction(0))),
                (
                    "deviate",
                    TerminalNode(
                        "deviate_terminal",
                        Fraction(1, 2),
                        Fraction(-1, 2),
                        Fraction(0),
                    ),
                ),
            ),
        )
    )
    mismatch = module.run_diagnostic(tree=changed_tree)
    assert mismatch.status is DiagnosticStatus.UNSUPPORTED
    assert "identity" in mismatch.message
    assert mismatch.prescribed_values == {}
    assert mismatch.deviations == ()


def test_indeterminate_and_unsupported_main_paths_emit_no_partial_stdout(
    monkeypatch, capsys
):
    module = load_example_module()
    arguments = module.worked_input_arguments()
    no_enclosure = replace(arguments["numeric_error_bound"], enclosure_established=False)
    indeterminate = module.run_diagnostic(numeric_error_bound=no_enclosure)
    unsupported = module.run_diagnostic(max_plans_per_player=1)
    assert indeterminate.status is DiagnosticStatus.INDETERMINATE
    assert indeterminate.deviations == ()

    for result in (indeterminate, unsupported):
        monkeypatch.setattr(module, "run_diagnostic", lambda result=result: result)
        assert module.main() == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert result.status.value in captured.err
        assert "maximum_lower" not in captured.err


def test_unexpected_or_malformed_input_exception_is_nonzero_and_stdout_empty(
    monkeypatch, capsys
):
    module = load_example_module()

    def malformed():
        raise ValueError("malformed manual attestation")

    monkeypatch.setattr(module, "run_diagnostic", malformed)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "malformed manual attestation" in captured.err
    assert "qualified_claim" not in captured.err


def test_example_imports_top_level_repeated_poker_public_names_only():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "repeated_poker.stage_plan_diagnostic" not in source
    assert "tests." not in source
    repeated_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("repeated_poker"))
            or (
                isinstance(node, ast.Import)
                and any(alias.name.startswith("repeated_poker") for alias in node.names)
            )
        )
    ]
    assert repeated_imports
    for node in repeated_imports:
        assert isinstance(node, ast.ImportFrom)
        assert node.module == "repeated_poker"
        assert node.names and all(not alias.name.startswith("_") for alias in node.names)


def test_public_docs_link_command_status_and_claim_boundaries():
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    examples_guide = EXAMPLES_GUIDE.read_text(encoding="utf-8")
    command = "python examples/stage_plan_diagnostic_workflow.py"
    normalized_readme = " ".join(readme.split())
    normalized_guide = " ".join(guide.split())

    assert "docs/stage_plan_diagnostic_workflow.md" in readme
    assert command in readme and command in guide and command in examples_guide
    assert all(status in guide for status in ("`PASS`", "`FAIL`", "`INDETERMINATE`", "`UNSUPPORTED`"))
    assert "`ValueError`" in guide
    assert "allocation-before-materialization refusal boundary" in normalized_guide
    assert "fixture-specific human evidence, not a general perfect-recall proof" in normalized_guide
    assert "successful diagnostic execution, not a process failure" in normalized_readme
    for boundary in (
        "arbitrary multi-period",
        "sequential rationality",
        "zero-reach action quality",
        "finite punishment",
        "known finite horizon",
        "equilibrium",
        "optimality",
        "real-money advice",
    ):
        assert boundary in normalized_guide


def test_mvp_preserves_six_checks_and_adds_stage_plan_as_seventh():
    tree = ast.parse(CHECK_MVP.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COMMANDS" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.List)
    assert len(assignment.value.elts) == 7
    command_text = ast.unparse(assignment.value)
    assert command_text.count("examples/stage_plan_diagnostic_workflow.py") == 1
    assert command_text.index("examples/stage_plan_diagnostic_workflow.py") > command_text.index(
        "examples/aiof_real_card_workflow.py"
    )
