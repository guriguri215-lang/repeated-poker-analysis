"""Independent contract checks for the M23 three-player file adapter."""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import repeated_poker.three_player_cfr as core
import repeated_poker.three_player_cfr_file_workflow as module
from repeated_poker.three_player_cfr import (
    BehaviorStrategy,
    CfrConfig,
    CfrSafetyLimits,
    OpponentDecisionNode,
    PerfectRecallAttestation,
    ThreePlayerGameTree,
    ThreePlayerTerminalNode,
    UtilityVector,
    run_three_player_cfr_diagnostic,
    tree_content_identity,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "three_player_cfr_file_v1.json"
SCRIPT = ROOT / "scripts" / "run_three_player_cfr_file.py"
SOURCE = ROOT / "src" / "repeated_poker" / "three_player_cfr_file_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "three_player_cfr_file_workflow.md"
FORMAT = "three-player-cfr-file-v1"
TREE_IDENTITY = "ae7cde83467e8a7dd156971028706a9e10b9070e33d523b7e6f2d14251463c97"


def example_document() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def encoded(document: object, *, allow_nan: bool = False) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=allow_nan,
    ).encode("utf-8")


def inspect_document(
    document: dict[str, object] | None = None,
    limits: module.ThreePlayerCfrFileLimits = module.ThreePlayerCfrFileLimits(),
) -> module.ThreePlayerCfrFileResult:
    return module.inspect_three_player_cfr_file(
        encoded(document or example_document()), limits
    )


def filled_run_document(
    document: dict[str, object] | None = None,
) -> dict[str, object]:
    source = copy.deepcopy(document or example_document())
    inspected = inspect_document(source)
    assert inspected.status is module.ThreePlayerCfrFileStatus.SUCCESS
    assert inspected.output is not None
    source["operation"] = "run"
    source["inspection_identity"] = copy.deepcopy(
        inspected.output["inspection_identity"]
    )
    source["attestation"] = {
        "tree_content_identity": inspected.output["tree_content_identity"],
        "o1_confirmed": True,
        "o2_confirmed": True,
        "verifier": "test fixture author",
        "verification_date": "2026-07-19",
        "evidence_version": "m23-test-evidence-v1",
    }
    return source


def run_document(
    document: dict[str, object],
    limits: module.ThreePlayerCfrFileLimits = module.ThreePlayerCfrFileLimits(),
) -> module.ThreePlayerCfrFileResult:
    return module.run_three_player_cfr_file(encoded(document), limits)


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


def assert_no_partial(result: module.ThreePlayerCfrFileResult) -> None:
    assert result.status is not module.ThreePlayerCfrFileStatus.SUCCESS
    assert result.output is None
    assert result.error is not None
    payload = json.loads(module.three_player_cfr_file_json(result))
    assert set(payload) == {"status", "output", "error"}
    assert payload["output"] is None
    assert set(payload["error"]) == {"phase", "message", "nested_status"}
    assert "\n" not in payload["error"]["message"]
    assert "\r" not in payload["error"]["message"]
    assert len(payload["error"]["message"]) <= 500
    serialized = json.dumps(payload)
    for forbidden in (
        "expected_utility",
        "strategies",
        "positive_regret",
        "unilateral_deviation_gain",
    ):
        assert forbidden not in serialized


def public_fixture_arguments() -> dict[str, object]:
    def terminal(node_id: str, h: float, o1: float, o2: float):
        return ThreePlayerTerminalNode(node_id, UtilityVector(h, o1, o2, 0.0))

    def o2(action: str, left, right):
        return OpponentDecisionNode(
            f"o2_after_{action}",
            "opponent_2",
            "O2_root",
            (("L", left), ("R", right)),
        )

    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_root",
            "opponent_1",
            "O1_root",
            (
                (
                    "A",
                    o2(
                        "A",
                        terminal("t_A_L", -2, 1, 1),
                        terminal("t_A_R", 0, 0, 0),
                    ),
                ),
                (
                    "B",
                    o2(
                        "B",
                        terminal("t_B_L", 0, 0, 0),
                        terminal("t_B_R", -4, 2, 2),
                    ),
                ),
            ),
        ),
        description="m20-three-player-public-example-v1",
    )
    return {
        "tree": tree,
        "fixed_hero_policy": BehaviorStrategy({}),
        "config": CfrConfig(
            iterations=2,
            request_oracle=True,
            include_oracle_rows=False,
        ),
        "attestation": PerfectRecallAttestation(
            tree_content_identity(tree),
            True,
            True,
            "test fixture author",
            "2026-07-19",
            "m23-test-evidence-v1",
        ),
    }


def test_inspect_returns_exact_identity_counts_templates_without_analysis(monkeypatch):
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("inspect must not run CFR or the oracle")

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", bomb)
    result = inspect_document()
    assert called is False
    assert result.status is module.ThreePlayerCfrFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["tree_content_identity"] == TREE_IDENTITY
    assert output["inspection_identity"]["template_id"] == module.THREE_PLAYER_CFR_INSPECTION_ID
    assert len(output["inspection_identity"]["semantic_sha256"]) == 64
    assert output["counts"] == {
        "nodes": 7,
        "terminals": 4,
        "chance_nodes": 0,
        "branches": 6,
        "fixed_hero_info_sets": 0,
        "opponent_info_sets": {"O1": 1, "O2": 1},
        "decision_actions_total": 6,
        "actions_per_info_set_max": 2,
    }
    assert output["opponent_action_template"] == {
        "O1": [{"info_set": "O1_root", "actions": ["A", "B"]}],
        "O2": [{"info_set": "O2_root", "actions": ["L", "R"]}],
    }
    assert output["fixed_hero_policy"] == []
    assert output["attestation_template"] == {
        "tree_content_identity": TREE_IDENTITY,
        "o1_confirmed": None,
        "o2_confirmed": None,
        "verifier": None,
        "verification_date": None,
        "evidence_version": None,
    }


def test_run_matches_hand_values_and_13_evaluation_oracle():
    result = run_document(filled_run_document())
    assert result.status is module.ThreePlayerCfrFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["status"] == {
        "component": "DIAGNOSTIC_COMPLETE",
        "overall": "DIAGNOSTIC_COMPLETE",
    }
    assert output["iterations"] == {"requested": 2, "completed": 2}
    assert output["expected_utility"] == {
        "H": -2.375,
        "O1": 1.1875,
        "O2": 1.1875,
        "R": 0.0,
    }
    assert output["unilateral_deviation_gain"] == {"O1": 0.3125, "O2": 0.3125}
    assert output["strategies"]["average"] == {
        "O1": {"O1_root": {"A": 0.25, "B": 0.75}},
        "O2": {"O2_root": {"L": 0.25, "R": 0.75}},
    }
    assert output["oracle"]["status"] == "MATCH"
    assert output["oracle"]["coverage"] == "complete"
    assert output["oracle"]["counts"]["pure_plans_by_player"] == {"O1": 2, "O2": 2}
    assert output["oracle"]["counts"]["joint_profiles"] == 4
    assert output["oracle"]["counts"]["actual_profile_evaluations"] == 13
    assert output["oracle"]["counts"]["actual_output_rows"] == 0
    assert output["oracle"]["stable_profile_count"] == 2


def test_run_projection_matches_direct_existing_public_api():
    wrapped = run_document(filled_run_document())
    direct = run_three_player_cfr_diagnostic(**public_fixture_arguments())
    assert wrapped.output is not None
    output = wrapped.output
    assert output["contract_version"] == direct.contract_version
    assert output["algorithm_version"] == direct.algorithm_version
    assert output["counts"] == {
        "nodes": direct.node_count,
        "terminals": direct.terminal_count,
        "opponent_info_sets": direct.info_set_count_by_player,
        "actions_per_info_set_max": direct.action_count_max,
    }
    assert output["strategies"]["current"] == direct.current_strategy_by_player
    assert output["strategies"]["average"] == direct.average_strategy_by_player
    assert output["expected_utility"] == direct.expected_utility_vector
    assert output["positive_regret"]["average_by_player"] == direct.average_positive_regret_by_player
    assert output["positive_regret"]["max_by_player"] == direct.max_positive_regret_by_player
    assert output["unilateral_deviation_gain"] == direct.unilateral_deviation_gain_by_player
    assert output["tolerances"] == direct.tolerances
    assert output["normalization_records"] == list(direct.normalization_records)


def test_core_is_called_exactly_once_after_all_run_gates(monkeypatch):
    original = module.run_three_player_cfr_diagnostic
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", counted)
    result = run_document(filled_run_document())
    assert result.status is module.ThreePlayerCfrFileStatus.SUCCESS
    assert calls == 1


def test_same_input_and_json_are_byte_deterministic(tmp_path: Path):
    document = filled_run_document()
    first = module.three_player_cfr_file_json(run_document(document))
    second = module.three_player_cfr_file_json(run_document(document))
    assert first == second
    path = tmp_path / "run.json"
    path.write_bytes(encoded(document))
    first_cli = run_cli(path)
    second_cli = run_cli(path)
    assert first_cli.returncode == second_cli.returncode == 0
    assert first_cli.stderr == second_cli.stderr == ""
    assert first_cli.stdout == second_cli.stdout
    assert first_cli.stdout.count("\n") == 1


@pytest.mark.parametrize("field", ["tree", "config", "core_limits", "workflow_limits"])
def test_inspection_identity_changes_with_every_bound_spec_class(field: str):
    first = inspect_document()
    changed = example_document()
    if field == "tree":
        changed["tree"]["description"] += "-changed"
    elif field == "config":
        changed["config"]["iterations"] = 3
    elif field == "core_limits":
        changed["core_limits"]["max_iterations"] = 4999
    else:
        changed["workflow_limits"]["max_output_bytes"] = 3999999
    second = inspect_document(changed)
    assert first.output is not None and second.output is not None
    assert first.output["inspection_identity"] != second.output["inspection_identity"]


def test_old_inspection_and_old_attestation_are_rejected_before_analysis(monkeypatch):
    original = inspect_document()
    assert original.output is not None
    changed = example_document()
    changed["tree"]["description"] += "-changed"
    run = filled_run_document(changed)
    run["inspection_identity"] = original.output["inspection_identity"]
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", bomb)
    mismatch = run_document(run)
    assert mismatch.status is module.ThreePlayerCfrFileStatus.IDENTITY_MISMATCH
    assert called is False
    assert_no_partial(mismatch)

    run = filled_run_document(changed)
    run["attestation"]["tree_content_identity"] = TREE_IDENTITY
    mismatch = run_document(run)
    assert mismatch.status is module.ThreePlayerCfrFileStatus.ATTESTATION_FAILURE
    assert called is False
    assert_no_partial(mismatch)


def test_fixed_hero_policy_drift_changes_inspection_identity_and_rejects_old_identity():
    document = example_document()
    document["tree"] = {
        "description": "m23-fixed-hero-policy-identity-v1",
        "root": {
            "type": "fixed_hero",
            "node_id": "hero_root",
            "info_set": "H_root",
            "actions": [
                {
                    "action": "left",
                    "child": {
                        "type": "terminal",
                        "node_id": "left_terminal",
                        "utility": {"H": 0, "O1": 0, "O2": 0, "R": 0},
                    },
                },
                {
                    "action": "right",
                    "child": {
                        "type": "terminal",
                        "node_id": "right_terminal",
                        "utility": {"H": 0, "O1": 0, "O2": 0, "R": 0},
                    },
                },
            ],
        },
    }
    document["fixed_hero_policy"] = {
        "info_sets": [
            {
                "info_set": "H_root",
                "actions": [
                    {"action": "left", "probability": 0.25},
                    {"action": "right", "probability": 0.75},
                ],
            }
        ]
    }
    first = inspect_document(document)
    changed = copy.deepcopy(document)
    changed["fixed_hero_policy"]["info_sets"][0]["actions"][0]["probability"] = 0.5
    changed["fixed_hero_policy"]["info_sets"][0]["actions"][1]["probability"] = 0.5
    second = inspect_document(changed)
    assert first.output is not None and second.output is not None
    assert first.output["inspection_identity"] != second.output["inspection_identity"]
    run = filled_run_document(changed)
    run["inspection_identity"] = first.output["inspection_identity"]
    result = run_document(run)
    assert result.status is module.ThreePlayerCfrFileStatus.IDENTITY_MISMATCH
    assert_no_partial(result)


@pytest.mark.parametrize("mutation", ["missing", "false", "null", "long", "unknown"])
def test_incomplete_or_invalid_attestation_is_rejected_before_analysis(monkeypatch, mutation: str):
    document = filled_run_document()
    if mutation == "missing":
        del document["attestation"]["verifier"]
    elif mutation == "false":
        document["attestation"]["o2_confirmed"] = False
    elif mutation == "null":
        document["attestation"]["verification_date"] = None
    elif mutation == "long":
        document["attestation"]["evidence_version"] = "x" * 201
    else:
        document["attestation"]["unknown"] = 1
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", bomb)
    result = run_document(document)
    assert result.status in {
        module.ThreePlayerCfrFileStatus.INVALID_INPUT,
        module.ThreePlayerCfrFileStatus.ATTESTATION_FAILURE,
    }
    assert called is False
    assert_no_partial(result)


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"\xff", module.ThreePlayerCfrFileStatus.PARSE_FAILURE),
        (b"\xef\xbb\xbf{}", module.ThreePlayerCfrFileStatus.PARSE_FAILURE),
        (b"[]", module.ThreePlayerCfrFileStatus.INVALID_INPUT),
        (b'{"value":NaN}', module.ThreePlayerCfrFileStatus.PARSE_FAILURE),
        (b'{"operation":"inspect","operation":"run"}', module.ThreePlayerCfrFileStatus.PARSE_FAILURE),
    ],
)
def test_parse_failures_are_bounded_no_partial(raw: bytes, status):
    result = module.process_three_player_cfr_file(raw)
    assert result.status is status
    assert_no_partial(result)


def test_nonbytes_and_invalid_outer_limits_fail_before_parse():
    result = module.process_three_player_cfr_file("{}")
    assert result.status is module.ThreePlayerCfrFileStatus.INVALID_INPUT
    assert_no_partial(result)
    bad = replace(module.ThreePlayerCfrFileLimits(), max_input_bytes=1_000_001)
    result = module.process_three_player_cfr_file(b"{}", bad)
    assert result.status is module.ThreePlayerCfrFileStatus.INVALID_INPUT
    assert_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "bool_number",
        "infinite",
        "bad_operation",
        "bad_format",
        "trace",
        "seed",
        "rows",
        "deviation_disabled",
        "bool_core_limit",
        "core_above_ceiling",
        "workflow_above_ceiling",
    ],
)
def test_strict_schema_numeric_and_fixed_v1_controls_reject_invalid(mutation: str):
    document = example_document()
    if mutation == "missing":
        del document["request_id"]
    elif mutation == "unknown":
        document["unknown"] = 1
    elif mutation == "bool_number":
        document["config"]["input_tolerance"] = True
    elif mutation == "infinite":
        document["config"]["epsilon_deviation"] = 10**400
    elif mutation == "bad_operation":
        document["operation"] = "solve"
    elif mutation == "bad_format":
        document["format_version"] = "three-player-cfr-file-v2"
    elif mutation == "trace":
        document["config"]["trace_checkpoint_interval"] = 1
    elif mutation == "seed":
        document["config"]["seed"] = 1
    elif mutation == "rows":
        document["config"]["include_oracle_rows"] = True
    elif mutation == "deviation_disabled":
        document["config"]["compute_deviation_gains"] = False
    elif mutation == "bool_core_limit":
        document["core_limits"]["max_nodes"] = True
    elif mutation == "core_above_ceiling":
        document["core_limits"]["max_nodes"] = 201
    else:
        document["workflow_limits"]["max_tree_nodes"] = 501
    result = inspect_document(document)
    assert result.status is module.ThreePlayerCfrFileStatus.INVALID_INPUT
    assert_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_node",
        "repeated_info_set",
        "action_mismatch",
        "chance_mass",
        "utility_conservation",
        "incomplete_hero_policy",
    ],
)
def test_recursive_tree_policy_and_core_validation_fail_closed(mutation: str):
    document = example_document()
    root = document["tree"]["root"]
    if mutation == "duplicate_node":
        root["actions"][1]["child"]["node_id"] = "o2_after_A"
    elif mutation == "repeated_info_set":
        child = root["actions"][0]["child"]
        child["type"] = "opponent_1"
        child["info_set"] = "O1_root"
    elif mutation == "action_mismatch":
        root["actions"][1]["child"]["actions"][1]["action"] = "X"
    elif mutation == "chance_mass":
        document["tree"]["root"] = {
            "type": "chance",
            "node_id": "chance",
            "children": [
                {"probability": 0.6, "child": root},
                {
                    "probability": 0.5,
                    "child": {
                        "type": "terminal",
                        "node_id": "extra_terminal",
                        "utility": {"H": 0, "O1": 0, "O2": 0, "R": 0},
                    },
                },
            ],
        }
    elif mutation == "utility_conservation":
        root["actions"][0]["child"]["actions"][0]["child"]["utility"]["R"] = 1
    else:
        document["tree"]["root"] = {
            "type": "fixed_hero",
            "node_id": "hero",
            "info_set": "H_root",
            "actions": [{"action": "continue", "child": root}],
        }
    result = inspect_document(document)
    assert result.status is module.ThreePlayerCfrFileStatus.INVALID_INPUT
    assert_no_partial(result)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("max_input_bytes", 1),
        ("max_total_json_values", 1),
        ("max_json_depth", 1),
        ("max_tree_depth", 1),
        ("max_tree_nodes", 1),
        ("max_tree_branches", 1),
    ],
)
def test_input_json_and_tree_caps_fail_closed(target: str, value: int):
    limits = replace(module.ThreePlayerCfrFileLimits(), **{target: value})
    result = inspect_document(limits=limits)
    assert result.status is module.ThreePlayerCfrFileStatus.CAP_EXCEEDED
    assert_no_partial(result)


@pytest.mark.parametrize(
    ("cap", "value"),
    [
        ("max_oracle_pure_profiles", 3),
        ("max_oracle_pure_plans_per_player", 1),
        ("max_oracle_joint_profiles", 3),
        ("max_oracle_profile_evaluations", 12),
    ],
)
def test_deviation_and_oracle_caps_fire_before_core(monkeypatch, cap: str, value: int):
    document = example_document()
    document["core_limits"][cap] = value
    run = filled_run_document(document)
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", bomb)
    result = run_document(run)
    assert result.status is module.ThreePlayerCfrFileStatus.CAP_EXCEEDED
    assert called is False
    assert_no_partial(result)


@pytest.mark.parametrize("cap", ["max_output_records", "max_output_bytes"])
def test_output_caps_fire_before_core(monkeypatch, cap: str):
    document = example_document()
    document["workflow_limits"][cap] = 1
    inspected = inspect_document(document)
    assert inspected.status is module.ThreePlayerCfrFileStatus.CAP_EXCEEDED

    document = filled_run_document()
    document["workflow_limits"][cap] = 1
    fresh = copy.deepcopy(document)
    fresh["operation"] = "inspect"
    fresh.pop("inspection_identity")
    fresh.pop("attestation")
    identity = inspect_document(fresh)
    assert identity.status is module.ThreePlayerCfrFileStatus.CAP_EXCEEDED


def test_core_noncomplete_and_exception_are_no_partial(monkeypatch):
    direct = run_three_player_cfr_diagnostic(**public_fixture_arguments())
    monkeypatch.setattr(
        module,
        "run_three_player_cfr_diagnostic",
        lambda *_args, **_kwargs: replace(direct, overall_status="ORACLE_MISMATCH"),
    )
    result = run_document(filled_run_document())
    assert result.status is module.ThreePlayerCfrFileStatus.DIAGNOSTIC_FAILURE
    assert_no_partial(result)

    monkeypatch.setattr(module, "run_three_player_cfr_diagnostic", lambda *_a, **_k: 1 / 0)
    result = run_document(filled_run_document())
    assert result.status is module.ThreePlayerCfrFileStatus.INTERNAL_FAILURE
    assert result.error is not None and result.error.message == "unexpected workflow failure"
    assert_no_partial(result)


def test_oracle_not_requested_is_complete_bounded_success():
    document = example_document()
    document["config"]["request_oracle"] = False
    result = run_document(filled_run_document(document))
    assert result.status is module.ThreePlayerCfrFileStatus.SUCCESS
    assert result.output is not None
    assert result.output["oracle"] == {
        "requested": False,
        "status": "NOT_REQUESTED",
        "coverage": "none",
        "counts": None,
        "stable_profile_count": None,
        "warnings": [],
    }


def test_success_excludes_runtime_trace_rows_and_solver_claims():
    serialized = module.three_player_cfr_file_json(
        run_document(filled_run_document())
    )
    for forbidden in (
        "execution_metadata",
        "python_version",
        "git_commit",
        "platform",
        "runtime_identity",
        "run_identity",
        '"trace"',
        '"rows"',
        "equilibrium",
        "Nash",
        "convergence",
        "exact-BR",
        "optimality",
        "profitability",
    ):
        assert forbidden not in serialized
    payload = json.loads(serialized, parse_constant=lambda value: pytest.fail(value))

    def finite(value):
        if type(value) in (int, float):
            assert math.isfinite(value)
        elif type(value) is dict:
            for child in value.values():
                finite(child)
        elif type(value) is list:
            for child in value:
                finite(child)

    finite(payload)


def test_cli_inspect_run_and_controlled_failure(tmp_path: Path):
    first = run_cli()
    second = run_cli()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout and first.stdout.count("\n") == 1
    assert json.loads(first.stdout)["output"]["operation"] == "inspect"

    run_path = tmp_path / "run.json"
    run_path.write_bytes(encoded(filled_run_document()))
    run = run_cli(run_path)
    assert run.returncode == 0 and run.stderr == "" and run.stdout.count("\n") == 1
    assert json.loads(run.stdout)["output"]["operation"] == "run"

    missing = run_cli(tmp_path / "missing.json")
    assert missing.returncode == 2 and missing.stderr == ""
    assert missing.stdout.count("\n") == 1
    assert json.loads(missing.stdout) == {
        "status": "INVALID_INPUT",
        "output": None,
        "error": {
            "phase": "input",
            "message": "cannot read input file",
            "nested_status": None,
        },
    }
    assert "Traceback" not in missing.stdout


def test_source_imports_only_nonprivate_public_core_names():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "three_player_cfr"
    ]
    assert len(imports) == 1 and imports[0].level == 1
    assert all(not alias.name.startswith("_") for alias in imports[0].names)
    assert "from .three_player_cfr import _" not in SOURCE.read_text(encoding="utf-8")


def test_cli_is_thin_and_docs_link_exact_command_and_boundaries():
    script_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "process_three_player_cfr_file"
    ]
    assert len(calls) == 1
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    command = "python scripts/run_three_player_cfr_file.py examples/three_player_cfr_file_v1.json"
    assert "docs/three_player_cfr_file_workflow.md" in readme
    assert command in readme and command in guide
    for phrase in ("two-phase", "human-authored", "no-partial", "not", "real-money"):
        assert phrase in readme.lower() and phrase in guide.lower()


def test_serializer_rejects_wrong_result_type():
    with pytest.raises(TypeError):
        module.three_player_cfr_file_json(object())
