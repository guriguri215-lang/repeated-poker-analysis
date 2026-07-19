"""Independent contract checks for the M24 stage-plan file adapter."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.stage_plan_diagnostic_file_workflow as module
from repeated_poker import (
    COOPERATE,
    PUNISH,
    STAGE_PLAN_HERO,
    STAGE_PLAN_VILLAIN,
    DiagnosticStatus,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "stage_plan_diagnostic_file_v1.json"
SCRIPT = ROOT / "scripts" / "run_stage_plan_diagnostic_file.py"
SOURCE = ROOT / "src" / "repeated_poker" / "stage_plan_diagnostic_file_workflow.py"
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "stage_plan_diagnostic_file_workflow.md"
PUBLIC_EXAMPLE = ROOT / "examples" / "stage_plan_diagnostic_workflow.py"
FORMAT = "stage-plan-diagnostic-file-v1"
TREE_IDENTITY = "ee5ece31bba86570f33a5ff6bf069247be6585114a9bf9f0f2421953c17d345d"


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
    limits: module.StagePlanDiagnosticFileLimits = module.StagePlanDiagnosticFileLimits(),
) -> module.StagePlanDiagnosticFileResult:
    return module.inspect_stage_plan_diagnostic_file(
        encoded(document or example_document()), limits
    )


def filled_run_document(
    document: dict[str, object] | None = None,
) -> dict[str, object]:
    source = copy.deepcopy(document or example_document())
    inspected = inspect_document(source)
    assert inspected.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert inspected.output is not None
    output = inspected.output
    source["operation"] = "run"
    source["inspection_identity"] = copy.deepcopy(output["inspection_identity"])
    source["model_attestation"] = {
        key: True for key in output["model_attestation_template"]
    }
    recall = copy.deepcopy(output["perfect_recall_attestation_template"])
    recall.update(
        {
            "fixture_id": "stage-plan-public-example",
            "reviewer": "test fixture author",
            "review_date": "2026-07-19",
            "review_method": "manual root-to-member observation and own-action review",
            "evidence": "the only Hero member h is the public root with empty prior history",
            "result_confirmed": True,
            "known_limitations": [
                "fixture-specific human evidence, not a general perfect-recall proof"
            ],
            "invalidation_conditions": [
                "any bound tree, monitoring, profile, numeric, cap, member, history, or legal-action change"
            ],
            "valid_through_version": source["fixture_version"],
            "invalidated": False,
        }
    )
    for player in (STAGE_PLAN_HERO, STAGE_PLAN_VILLAIN):
        for group in recall["member_histories"][player]:
            for member in group["members"]:
                member["observations"] = []
    source["perfect_recall_attestation"] = recall
    return source


def run_document(
    document: dict[str, object],
    limits: module.StagePlanDiagnosticFileLimits = module.StagePlanDiagnosticFileLimits(),
) -> module.StagePlanDiagnosticFileResult:
    return module.run_stage_plan_diagnostic_file(encoded(document), limits)


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


def assert_no_partial(result: module.StagePlanDiagnosticFileResult) -> None:
    assert result.status is not module.StagePlanDiagnosticFileStatus.SUCCESS
    assert result.output is None
    assert result.error is not None
    payload = json.loads(module.stage_plan_diagnostic_file_json(result))
    assert set(payload) == {"status", "output", "error"}
    assert payload["output"] is None
    assert set(payload["error"]) == {"phase", "message", "nested_status"}
    assert "\n" not in payload["error"]["message"]
    assert "\r" not in payload["error"]["message"]
    assert len(payload["error"]["message"]) <= 500
    serialized = json.dumps(payload)
    for forbidden in (
        "prescribed_values",
        "deviations",
        "maximum_lower",
        "maximum_upper",
        '"counts"',
        TREE_IDENTITY,
        "576805f36f38af154a5836f5e7f08bf778d15ec2d151b02ce2117bb7a48b0d3f",
    ):
        assert forbidden not in serialized


def load_public_example():
    spec = importlib.util.spec_from_file_location(
        "m19_public_stage_plan_example_for_m24_test", PUBLIC_EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_inspect_returns_identity_counts_and_null_templates_without_core(monkeypatch):
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("inspect must not call the analytic diagnostic")

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", bomb)
    result = inspect_document()
    assert called is False
    assert result.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert result.error is None and result.output is not None
    output = result.output
    assert output["tree_content_identity"] == TREE_IDENTITY
    assert output["inspection_identity"] == {
        "template_id": module.STAGE_PLAN_DIAGNOSTIC_INSPECTION_ID,
        "semantic_sha256": "576805f36f38af154a5836f5e7f08bf778d15ec2d151b02ce2117bb7a48b0d3f",
        "tree_content_identity": TREE_IDENTITY,
    }
    assert output["counts"] == {
        "tree": {
            "nodes": 3,
            "terminals": 2,
            "chance_nodes": 0,
            "hero_decision_nodes": 1,
            "villain_decision_nodes": 0,
            "branches": 2,
            "hero_information_sets": 1,
            "villain_information_sets": 0,
        },
        "monitoring": {
            "public_action_nodes": 1,
            "terminal_observables": 2,
            "public_signals": 2,
            "public_signal_actions": 2,
            "transitions": 4,
        },
        "profiles": {"rows": 2, "actions": 4},
        "plans": {"Hero": 2, "Villain": 1},
        "predicted_deviation_rows": 6,
        "model_attestation_fields": 11,
        "perfect_recall_attestation_fields": 15,
    }
    model = output["model_attestation_template"]
    recall = output["perfect_recall_attestation_template"]
    assert len(model) == 11 and set(model.values()) == {None}
    assert len(recall) == 15
    assert recall["tree_content_identity"] == TREE_IDENTITY
    assert recall["target_version"] == "stage-plan-public-example-v1"
    assert recall["information_set_members"] == {
        "Hero": [{"info_set": "H1", "members": ["h"]}],
        "Villain": [],
    }
    history = recall["member_histories"]["Hero"][0]["members"][0]
    assert history == {
        "node_id": "h",
        "observations": None,
        "own_actions": [],
        "information_sets": [],
    }
    assert recall["result_confirmed"] is recall["reviewer"] is None
    assert recall["known_limitations"] is recall["invalidation_conditions"] is None


def test_run_matches_manual_oracle_and_direct_public_projection():
    wrapped = run_document(filled_run_document())
    assert wrapped.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert wrapped.error is None and wrapped.output is not None
    output = wrapped.output
    assert output["status"] == "FAIL"
    assert output["counts"] == {
        "plans": {"Hero": 2, "Villain": 1},
        "deviation_rows": 6,
    }
    assert output["maximum"] == {"lower": "1", "upper": "1"}
    assert [row["value"] for row in output["prescribed_values"]] == ["0"] * 4
    assert len(output["deviations"]) == 6

    public = load_public_example().run_diagnostic()
    assert public.status is DiagnosticStatus.FAIL
    assert output["counts"]["plans"] == dict(public.plan_counts)
    assert output["counts"]["deviation_rows"] == len(public.deviations)
    expected_values = [
        {
            "player": player,
            "state": state,
            "value": str(public.prescribed_values[(player, state)]),
        }
        for player in (STAGE_PLAN_HERO, STAGE_PLAN_VILLAIN)
        for state in (COOPERATE, PUNISH)
    ]
    assert output["prescribed_values"] == expected_values
    for actual, expected in zip(output["deviations"], public.deviations):
        assert actual["player"] == expected.player
        assert actual["state"] == expected.state
        assert actual["plan"] == [
            {"info_set": info_set, "action": action}
            for info_set, action in expected.plan.actions
        ]
        for name in (
            "prescribed_value",
            "deviation_value",
            "gain",
            "lower",
            "upper",
        ):
            assert actual[name] == str(getattr(expected, name))

    prescribed = Fraction(0)
    assert Fraction(1) + Fraction(1, 2) * prescribed - prescribed == 1
    assert 2 * (2 + 1) == 6


def test_pass_and_fail_are_both_complete_file_successes():
    fail = run_document(filled_run_document())
    assert fail.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert fail.output["status"] == "FAIL"

    document = example_document()
    terminal = document["tree"]["root"]["actions"][1]["child"]
    terminal["hero_payoff"] = terminal["villain_payoff"] = "0"
    passed = run_document(filled_run_document(document))
    assert passed.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert passed.output["status"] == "PASS"
    assert passed.output["maximum"] == {"lower": "0", "upper": "0"}


def test_core_is_called_exactly_once_after_every_run_gate(monkeypatch):
    original = module.diagnose_stage_plan_deviations
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", counted)
    result = run_document(filled_run_document())
    assert result.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert calls == 1


@pytest.mark.parametrize(
    "field",
    [
        "tree",
        "monitoring",
        "profiles",
        "fixture_version",
        "numeric",
        "core_limits",
        "workflow_limits",
    ],
)
def test_inspection_identity_changes_with_every_bound_spec_class(field: str):
    original = inspect_document()
    document = example_document()
    if field == "tree":
        terminal = document["tree"]["root"]["actions"][1]["child"]
        terminal["hero_payoff"] = "1/2"
        terminal["villain_payoff"] = "-1/2"
    elif field == "monitoring":
        document["monitoring"]["terminal_observables"][0]["observable"] = "end"
        document["monitoring"]["signal_alphabet"][0]["terminal_observable"] = "end"
    elif field == "profiles":
        actions = document["profiles"]["C"]["Hero"][0]["actions"]
        actions[0]["probability"], actions[1]["probability"] = "0", "1"
    elif field == "fixture_version":
        document["fixture_version"] = "stage-plan-public-example-v2"
    elif field == "numeric":
        document["numeric"]["delta"] = "1/3"
    elif field == "core_limits":
        document["core_limits"]["max_plans_per_player"] = 3
    else:
        document["workflow_limits"]["max_output_bytes"] = 3999999
    changed = inspect_document(document)
    assert original.output is not None and changed.output is not None
    assert original.output["inspection_identity"] != changed.output["inspection_identity"]


def test_old_identity_is_rejected_before_core(monkeypatch):
    old = inspect_document()
    changed = example_document()
    changed["numeric"]["delta"] = "1/3"
    run = filled_run_document(changed)
    run["inspection_identity"] = old.output["inspection_identity"]
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", bomb)
    result = run_document(run)
    assert result.status is module.StagePlanDiagnosticFileStatus.IDENTITY_MISMATCH
    assert called is False
    assert_no_partial(result)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "false", "null"])
def test_model_attestation_failures_precede_core(monkeypatch, mutation: str):
    document = filled_run_document()
    key = next(iter(document["model_attestation"]))
    if mutation == "missing":
        del document["model_attestation"][key]
    elif mutation == "unknown":
        document["model_attestation"]["unknown"] = True
    elif mutation == "false":
        document["model_attestation"][key] = False
    else:
        document["model_attestation"][key] = None
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", bomb)
    result = run_document(document)
    assert result.status in {
        module.StagePlanDiagnosticFileStatus.INVALID_INPUT,
        module.StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
    }
    assert called is False
    assert_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "stale",
        "unconfirmed",
        "invalidated",
        "identity",
        "history",
        "members",
        "legal_actions",
        "empty_evidence",
        "unknown",
    ],
)
def test_recall_attestation_failures_precede_core(monkeypatch, mutation: str):
    document = filled_run_document()
    recall = document["perfect_recall_attestation"]
    if mutation == "missing":
        del recall["reviewer"]
    elif mutation == "stale":
        recall["valid_through_version"] = "old"
    elif mutation == "unconfirmed":
        recall["result_confirmed"] = False
    elif mutation == "invalidated":
        recall["invalidated"] = True
    elif mutation == "identity":
        recall["tree_content_identity"] = "0" * 64
    elif mutation == "history":
        recall["member_histories"]["Hero"][0]["members"][0]["own_actions"] = ["stay"]
    elif mutation == "members":
        recall["information_set_members"]["Hero"][0]["members"] = ["foreign"]
    elif mutation == "legal_actions":
        recall["legal_actions"]["Hero"][0]["actions"].reverse()
    elif mutation == "empty_evidence":
        recall["known_limitations"] = []
    else:
        recall["unknown"] = 1
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", bomb)
    result = run_document(document)
    assert result.status in {
        module.StagePlanDiagnosticFileStatus.INVALID_INPUT,
        module.StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
    }
    assert called is False
    assert_no_partial(result)


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"\xff", module.StagePlanDiagnosticFileStatus.PARSE_FAILURE),
        (b"\xef\xbb\xbf{}", module.StagePlanDiagnosticFileStatus.PARSE_FAILURE),
        (b"[]", module.StagePlanDiagnosticFileStatus.INVALID_INPUT),
        (b'{"value":NaN}', module.StagePlanDiagnosticFileStatus.PARSE_FAILURE),
        (
            b'{"operation":"inspect","operation":"run"}',
            module.StagePlanDiagnosticFileStatus.PARSE_FAILURE,
        ),
    ],
)
def test_parse_failures_are_bounded_no_partial(raw: bytes, status):
    result = module.process_stage_plan_diagnostic_file(raw)
    assert result.status is status
    assert_no_partial(result)


def test_nonbytes_and_invalid_outer_limits_fail_before_parse():
    result = module.process_stage_plan_diagnostic_file("{}")
    assert result.status is module.StagePlanDiagnosticFileStatus.INVALID_INPUT
    assert_no_partial(result)
    bad = replace(module.StagePlanDiagnosticFileLimits(), max_input_bytes=1_000_001)
    result = module.process_stage_plan_diagnostic_file(b"{}", bad)
    assert result.status is module.StagePlanDiagnosticFileStatus.INVALID_INPUT
    assert_no_partial(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "float_fraction",
        "decimal_fraction",
        "bool_fraction",
        "noncanonical_fraction",
        "zero_denominator",
        "negative_probability",
        "bad_probability_sum",
        "bool_limit",
        "limit_above_ceiling",
        "partial_transition",
        "duplicate_signal",
    ],
)
def test_strict_schema_canonical_fractions_and_finite_semantics(mutation: str):
    document = example_document()
    if mutation == "missing":
        del document["request_id"]
    elif mutation == "unknown":
        document["unknown"] = 1
    elif mutation == "float_fraction":
        document["numeric"]["delta"] = 0.5
    elif mutation == "decimal_fraction":
        document["numeric"]["delta"] = "0.5"
    elif mutation == "bool_fraction":
        document["numeric"]["delta"] = True
    elif mutation == "noncanonical_fraction":
        document["numeric"]["delta"] = "2/4"
    elif mutation == "zero_denominator":
        document["numeric"]["delta"] = "1/0"
    elif mutation == "negative_probability":
        document["profiles"]["C"]["Hero"][0]["actions"][1]["probability"] = "-1"
    elif mutation == "bad_probability_sum":
        document["profiles"]["C"]["Hero"][0]["actions"][0]["probability"] = "1/2"
    elif mutation == "bool_limit":
        document["core_limits"]["max_plans_per_player"] = True
    elif mutation == "limit_above_ceiling":
        document["workflow_limits"]["max_tree_nodes"] = 501
    elif mutation == "partial_transition":
        document["monitoring"]["transitions"].pop()
    else:
        document["monitoring"]["signal_alphabet"][1] = copy.deepcopy(
            document["monitoring"]["signal_alphabet"][0]
        )
    result = inspect_document(document)
    assert result.status is module.StagePlanDiagnosticFileStatus.INVALID_INPUT
    assert_no_partial(result)


def test_signed_canonical_fraction_is_accepted_for_terminal_payoff():
    result = inspect_document()
    assert result.status is module.StagePlanDiagnosticFileStatus.SUCCESS
    assert example_document()["tree"]["root"]["actions"][1]["child"]["villain_payoff"] == "-1"


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("max_input_bytes", 1),
        ("max_total_json_values", 1),
        ("max_json_depth", 1),
        ("max_tree_depth", 1),
        ("max_tree_nodes", 1),
        ("max_tree_branches", 1),
        ("max_public_action_nodes", 1),
        ("max_terminal_observables", 1),
        ("max_signals", 1),
        ("max_transitions", 1),
        ("max_profile_rows", 1),
        ("max_profile_actions", 1),
        ("max_plan_rows", 1),
        ("max_output_records", 1),
        ("max_output_bytes", 1),
    ],
)
def test_file_structure_plan_and_output_caps_fail_closed(target: str, value: int):
    limits = replace(module.StagePlanDiagnosticFileLimits(), **{target: value})
    document = example_document()
    document["workflow_limits"][target] = value
    if target == "max_public_action_nodes":
        document["monitoring"]["public_action_node_ids"].append("h")
    result = inspect_document(document, limits)
    assert result.status is module.StagePlanDiagnosticFileStatus.CAP_EXCEEDED
    assert_no_partial(result)


def test_core_plan_cap_and_run_output_cap_fire_before_core(monkeypatch):
    capped = example_document()
    capped["core_limits"]["max_plans_per_player"] = 1
    result = inspect_document(capped)
    assert result.status is module.StagePlanDiagnosticFileStatus.CAP_EXCEEDED
    assert result.error.nested_status == "UNSUPPORTED"
    assert_no_partial(result)

    run = filled_run_document()
    run["workflow_limits"]["max_output_bytes"] = 1
    called = False

    def bomb(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", bomb)
    result = run_document(run)
    assert result.status is module.StagePlanDiagnosticFileStatus.CAP_EXCEEDED
    assert called is False
    assert_no_partial(result)


def test_core_noncomplete_and_exception_are_no_partial(monkeypatch):
    public = load_public_example().run_diagnostic()
    monkeypatch.setattr(
        module,
        "diagnose_stage_plan_deviations",
        lambda **_kwargs: replace(
            public,
            status=DiagnosticStatus.UNSUPPORTED,
            prescribed_values={},
            deviations=(),
            maximum_lower=None,
            maximum_upper=None,
        ),
    )
    result = run_document(filled_run_document())
    assert result.status is module.StagePlanDiagnosticFileStatus.DIAGNOSTIC_FAILURE
    assert result.error.nested_status == "UNSUPPORTED"
    assert_no_partial(result)

    monkeypatch.setattr(module, "diagnose_stage_plan_deviations", lambda **_kwargs: 1 / 0)
    result = run_document(filled_run_document())
    assert result.status is module.StagePlanDiagnosticFileStatus.INTERNAL_FAILURE
    assert result.error.message == "unexpected workflow failure"
    assert_no_partial(result)


def test_same_input_and_json_are_byte_deterministic(tmp_path: Path):
    document = filled_run_document()
    first = module.stage_plan_diagnostic_file_json(run_document(document))
    second = module.stage_plan_diagnostic_file_json(run_document(document))
    assert first == second
    path = tmp_path / "run.json"
    path.write_bytes(encoded(document))
    first_cli = run_cli(path)
    second_cli = run_cli(path)
    assert first_cli.returncode == second_cli.returncode == 0
    assert first_cli.stderr == second_cli.stderr == ""
    assert first_cli.stdout == second_cli.stdout == first + "\n"
    assert first_cli.stdout.count("\n") == 1


def test_success_output_excludes_runtime_trace_and_overclaims():
    serialized = module.stage_plan_diagnostic_file_json(
        run_document(filled_run_document())
    )
    for forbidden in (
        "execution_metadata",
        "python_version",
        "git_commit",
        "platform",
        "runtime_identity",
        "run_identity",
        "traceback",
        "certificate",
        "equilibrium",
        "Nash",
        "proof",
        "optimality",
        "profitability",
    ):
        assert forbidden not in serialized
    payload = json.loads(serialized, parse_constant=lambda value: pytest.fail(value))
    assert payload["output"]["qualified_claim"] == (
        "bounded exhaustive one-period stage-plan deviation diagnostic"
    )
    assert len(payload["output"]["deviations"]) == 6


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
    assert json.loads(run.stdout)["output"]["status"] == "FAIL"

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
        if isinstance(node, ast.ImportFrom)
        and node.module in {"game", "stage_plan_diagnostic"}
    ]
    assert len(imports) == 2
    assert all(node.level == 1 for node in imports)
    assert all(
        not alias.name.startswith("_")
        for node in imports
        for alias in node.names
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "from .stage_plan_diagnostic import _" not in source
    assert "from .game import _" not in source


def test_cli_is_thin_and_docs_link_exact_command_and_boundaries():
    script_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "process_stage_plan_diagnostic_file"
    ]
    assert len(calls) == 1
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    command = (
        "python scripts/run_stage_plan_diagnostic_file.py "
        "examples/stage_plan_diagnostic_file_v1.json"
    )
    assert "docs/stage_plan_diagnostic_file_workflow.md" in readme
    assert command in readme and command in guide
    for phrase in (
        "two-phase",
        "human-authored",
        "canonical exact rational",
        "no-partial",
        "not",
        "real-money",
    ):
        assert phrase in readme.lower() and phrase in guide.lower()


def test_serializer_rejects_wrong_result_type():
    with pytest.raises(TypeError):
        module.stage_plan_diagnostic_file_json(object())
