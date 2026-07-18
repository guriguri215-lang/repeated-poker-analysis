#!/usr/bin/env python3
"""Run one guarded M12 three-player diagnostic through public submodule APIs.

The output is a bounded two-iteration diagnostic snapshot.  It is not a
solution, convergence claim, or stability claim for the returned profile.
"""

from __future__ import annotations

import json
import sys

from repeated_poker.three_player_cfr import (
    DIAGNOSTIC_COMPLETE,
    BehaviorStrategy,
    CfrConfig,
    DiagnosticContractError,
    OpponentDecisionNode,
    PerfectRecallAttestation,
    ThreePlayerGameTree,
    ThreePlayerTerminalNode,
    UtilityVector,
    run_three_player_cfr_diagnostic,
    tree_content_identity,
)


FIXTURE_VERSION = "m20-three-player-public-example-v1"
QUALIFIED_DIAGNOSTIC = (
    "deterministic two-iteration fixed-Hero three-player CFR-style diagnostic snapshot"
)


class WorkflowFailure(RuntimeError):
    """Internal control-flow error used to keep stdout no-partial on failure."""


def _terminal(node_id: str, hero: float, o1: float, o2: float):
    return ThreePlayerTerminalNode(
        node_id,
        UtilityVector(H=hero, O1=o1, O2=o2, R=0.0),
    )


def _o2_node(o1_action: str, left, right):
    return OpponentDecisionNode(
        node_id=f"o2_after_{o1_action}",
        owner="opponent_2",
        info_set="O2_root",
        actions=(("L", left), ("R", right)),
    )


def worked_input_arguments() -> dict[str, object]:
    """Return the fixed tree, manual attestation, and bounded configuration."""

    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            node_id="o1_root",
            owner="opponent_1",
            info_set="O1_root",
            actions=(
                (
                    "A",
                    _o2_node(
                        "A",
                        _terminal("t_A_L", -2.0, 1.0, 1.0),
                        _terminal("t_A_R", 0.0, 0.0, 0.0),
                    ),
                ),
                (
                    "B",
                    _o2_node(
                        "B",
                        _terminal("t_B_L", 0.0, 0.0, 0.0),
                        _terminal("t_B_R", -4.0, 2.0, 2.0),
                    ),
                ),
            ),
        ),
        description=FIXTURE_VERSION,
    )
    attestation = PerfectRecallAttestation(
        tree_content_identity=tree_content_identity(tree),
        o1_confirmed=True,
        o2_confirmed=True,
        verifier="public example fixture author",
        verification_date="2026-07-18",
        evidence_version=FIXTURE_VERSION,
    )
    return {
        "tree": tree,
        "fixed_hero_policy": BehaviorStrategy({}),
        "config": CfrConfig(
            iterations=2,
            request_oracle=True,
            include_oracle_rows=False,
        ),
        "attestation": attestation,
    }


def run_diagnostic(**overrides: object):
    """Run the fixed public example with explicit testable input overrides."""

    arguments = worked_input_arguments()
    arguments.update(overrides)
    return run_three_player_cfr_diagnostic(**arguments)


def run_workflow() -> dict[str, object]:
    """Return the strict bounded projection after all component checks pass."""

    result = run_diagnostic()
    oracle = result.oracle_attachment
    if (
        result.component_status != DIAGNOSTIC_COMPLETE
        or result.overall_status != DIAGNOSTIC_COMPLETE
        or oracle.get("status") != "MATCH"
        or oracle.get("coverage") != "complete"
        or oracle.get("rows") != []
    ):
        raise WorkflowFailure(
            "expected complete diagnostic with matching complete oracle; received "
            f"component={result.component_status}, overall={result.overall_status}, "
            f"oracle={oracle.get('status')}, coverage={oracle.get('coverage')}"
        )
    counts = oracle["counts"]
    return {
        "status": {
            "component": result.component_status,
            "overall": result.overall_status,
        },
        "qualified_diagnostic": QUALIFIED_DIAGNOSTIC,
        "iterations": {
            "requested": result.requested_iterations,
            "completed": result.completed_iterations,
        },
        "expected_utility": result.expected_utility_vector,
        "unilateral_deviation_gain": result.unilateral_deviation_gain_by_player,
        "oracle": {
            "status": oracle["status"],
            "coverage": oracle["coverage"],
            "counts": {
                "pure_plans": counts["pure_plans_by_player"],
                "joint_profiles": counts["joint_profiles"],
                "complete_table_rows": counts["complete_table_rows"],
                "predicted_profile_evaluations": counts[
                    "predicted_profile_evaluations"
                ],
                "actual_profile_evaluations": counts[
                    "actual_profile_evaluations"
                ],
                "predicted_output_rows": counts["predicted_output_rows"],
                "actual_output_rows": counts["actual_output_rows"],
            },
            "pure_profile_unilateral_stability_rows": oracle[
                "stable_profile_count"
            ],
            "warnings": oracle["warnings"],
        },
    }


def main() -> int:
    try:
        summary = run_workflow()
    except DiagnosticContractError as exc:
        print(f"error [{exc.status}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
