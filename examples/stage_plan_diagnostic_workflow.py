#!/usr/bin/env python3
"""Run one bounded M11 stage-plan diagnostic through top-level public APIs.

The fixture is deliberately tiny and exact.  Its analytic FAIL status is a
successful detection of a positive one-period deviation, not a process error.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction

from repeated_poker import (
    COOPERATE,
    PUNISH,
    STAGE_PLAN_HERO,
    STAGE_PLAN_VILLAIN,
    DiagnosticStatus,
    GameTree,
    HeroNode,
    ManualPerfectRecallAttestation,
    ModelClassAttestation,
    PublicAction,
    PublicMonitoring,
    PublicSignal,
    RecallHistory,
    TerminalNode,
    diagnose_stage_plan_deviations,
    exact_zero_error_bound,
    tree_content_identity,
)


FIXTURE_VERSION = "stage-plan-public-example-v1"
QUALIFIED_CLAIM = "bounded exhaustive one-period stage-plan deviation diagnostic"


class WorkflowFailure(RuntimeError):
    """Internal control-flow error used to keep stdout no-partial on failure."""


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def worked_input_arguments() -> dict[str, object]:
    """Return the fixed, explicitly attested public example inputs."""

    tree = GameTree(
        HeroNode(
            "h",
            "H1",
            (
                ("stay", TerminalNode("stay_terminal", Fraction(0), Fraction(0), Fraction(0))),
                (
                    "deviate",
                    TerminalNode(
                        "deviate_terminal", Fraction(1), Fraction(-1), Fraction(0)
                    ),
                ),
            ),
        )
    )
    stay_signal = PublicSignal(
        (PublicAction(STAGE_PLAN_HERO, "stay"),), "terminal"
    )
    deviate_signal = PublicSignal(
        (PublicAction(STAGE_PLAN_HERO, "deviate"),), "terminal"
    )
    monitoring = PublicMonitoring(
        public_action_node_ids=frozenset({"h"}),
        terminal_observables={
            "stay_terminal": "terminal",
            "deviate_terminal": "terminal",
        },
        signal_alphabet=(stay_signal, deviate_signal),
        transitions={
            (COOPERATE, stay_signal): COOPERATE,
            (COOPERATE, deviate_signal): COOPERATE,
            (PUNISH, stay_signal): PUNISH,
            (PUNISH, deviate_signal): PUNISH,
        },
    )
    hero_profile = {"H1": {"stay": Fraction(1), "deviate": Fraction(0)}}
    profile = {
        COOPERATE: {STAGE_PLAN_HERO: hero_profile, STAGE_PLAN_VILLAIN: {}},
        PUNISH: {
            STAGE_PLAN_HERO: {
                "H1": {"stay": Fraction(1), "deviate": Fraction(0)}
            },
            STAGE_PLAN_VILLAIN: {},
        },
    }
    model_attestation = ModelClassAttestation(
        iid_stage_kernel=True,
        no_persistent_private_state=True,
        no_cross_period_correlation=True,
        no_private_payoff_state=True,
        public_state_does_not_change_stage_kernel=True,
        public_state_is_sufficient=True,
        signal_partition_is_public=True,
        signal_excludes_private_information=True,
        signal_excludes_deviator_identity=True,
        no_known_finite_horizon=True,
        absorbing_grim_only=True,
    )
    perfect_recall_attestation = ManualPerfectRecallAttestation(
        fixture_id="stage-plan-public-example",
        tree_content_identity=tree_content_identity(tree),
        target_version=FIXTURE_VERSION,
        information_set_members={
            STAGE_PLAN_HERO: {"H1": ("h",)},
            STAGE_PLAN_VILLAIN: {},
        },
        member_histories={
            STAGE_PLAN_HERO: {
                "H1": {
                    "h": RecallHistory(
                        observations=(), own_actions=(), information_sets=()
                    )
                }
            },
            STAGE_PLAN_VILLAIN: {},
        },
        legal_actions={
            STAGE_PLAN_HERO: {"H1": ("stay", "deviate")},
            STAGE_PLAN_VILLAIN: {},
        },
        reviewer="public example fixture author",
        review_date="2026-07-18",
        review_method="manual root-to-member observation and own-action comparison",
        evidence="the only Hero member h is the public root, with empty prior history",
        result_confirmed=True,
        known_limitations=(
            "fixture-specific human evidence, not a general perfect-recall proof",
        ),
        invalidation_conditions=(
            "any tree, node, information partition, member, history, or legal-action change",
        ),
        valid_through_version=FIXTURE_VERSION,
        invalidated=False,
    )
    return {
        "tree": tree,
        "fixture_version": FIXTURE_VERSION,
        "profile": profile,
        "monitoring": monitoring,
        "model_attestation": model_attestation,
        "perfect_recall_attestation": perfect_recall_attestation,
        "delta": Fraction(1, 2),
        "stage_payoff_bound": Fraction(1),
        "input_tolerance": Fraction(0),
        "epsilon_claim": Fraction(0),
        "numeric_error_bound": exact_zero_error_bound(),
        "max_plans_per_player": 2,
    }


def run_diagnostic(**overrides: object):
    """Run the fixed public example with explicit testable input overrides."""

    arguments = worked_input_arguments()
    arguments.update(overrides)
    return diagnose_stage_plan_deviations(**arguments)


def run_workflow() -> dict[str, object]:
    """Return the bounded JSON-safe summary after the expected analytic FAIL."""

    result = run_diagnostic()
    if (
        result.status is not DiagnosticStatus.FAIL
        or result.maximum_lower is None
        or result.maximum_upper is None
    ):
        raise WorkflowFailure(
            f"expected analytic FAIL, received {result.status.value}: {result.message}"
        )
    return {
        "status": result.status.value,
        "qualified_claim": QUALIFIED_CLAIM,
        "counts": {
            "hero_plans": result.plan_counts[STAGE_PLAN_HERO],
            "villain_plans": result.plan_counts[STAGE_PLAN_VILLAIN],
            "deviation_rows": len(result.deviations),
        },
        "exact_fraction_strings": {
            "maximum_lower": _fraction_text(result.maximum_lower),
            "maximum_upper": _fraction_text(result.maximum_upper),
        },
    }


def main() -> int:
    try:
        summary = run_workflow()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
