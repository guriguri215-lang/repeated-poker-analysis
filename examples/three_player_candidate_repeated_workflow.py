#!/usr/bin/env python3
"""Run one tiny exact M30-M32 workflow through public submodule APIs.

The fixture is a caller-declared abstract river with zero rake.  It demonstrates
bounded ``robust_all`` candidate generation, fresh exact O1/O2 responses, and
simultaneous-adaptation repeated-value sensitivity.  It is not a full solver,
equilibrium certificate, global optimum, real-card analysis, or strategy advice.
"""

from __future__ import annotations

import json
import sys

from repeated_poker.automatic_commitment_selection import (
    NO_BENEFICIAL_COMMITMENT,
)
from repeated_poker.three_player_candidate_repeated import (
    EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
    ThreePlayerCandidateGenerationConfig,
    ThreePlayerRepeatedConfig,
    evaluate_three_player_candidate_repeated,
)
from repeated_poker.three_player_response import (
    EXACT_CORRESPONDENCE_COMPLETE,
)
from repeated_poker.three_player_river_rake import (
    EXACT_SCENARIO_RESPONSE_COMPLETE,
    AwardShare,
    ExactBehaviorPolicy,
    OpponentInitialProfile,
    RiverAction,
    RiverDecisionNode,
    RiverObservation,
    RiverTerminalNode,
    ThreePlayerRiverRakeScenario,
    create_perfect_recall_attestation,
)


FIXTURE_VERSION = "m34-three-player-candidate-repeated-public-workflow-v1"
SAFETY_SOURCE_PATH = "m31_scenario_response.response.hero_worst"


class WorkflowFailure(RuntimeError):
    """Internal control-flow error used to keep stdout no-partial on failure."""


def _observation() -> RiverObservation:
    return RiverObservation(
        public_observation_id="public-river",
        private_observation_id_by_player={
            "H": "hero-private",
            "O1": "o1-private",
            "O2": "o2-private",
        },
    )


def _hero_root() -> RiverDecisionNode:
    showdown = RiverTerminalNode(
        node_id="showdown-after-checks",
        kind="showdown",
        award_shares=(AwardShare("H", "1"),),
    )
    o2_check = RiverDecisionNode(
        node_id="o2-after-check",
        owner="O2",
        information_set_id="O2_after_check",
        actions=(RiverAction("check", "check", None, showdown),),
    )
    o1_check = RiverDecisionNode(
        node_id="o1-after-check",
        owner="O1",
        information_set_id="O1_after_check",
        actions=(RiverAction("check", "check", None, o2_check),),
    )

    fold_terminal = RiverTerminalNode(
        node_id="fold-after-bet",
        kind="fold",
    )
    o2_fold = RiverDecisionNode(
        node_id="o2-after-bet",
        owner="O2",
        information_set_id="O2_after_bet",
        actions=(RiverAction("fold", "fold", None, fold_terminal),),
    )
    o1_fold = RiverDecisionNode(
        node_id="o1-after-bet",
        owner="O1",
        information_set_id="O1_after_bet",
        actions=(RiverAction("fold", "fold", None, o2_fold),),
    )
    return RiverDecisionNode(
        node_id="hero-root",
        owner="H",
        information_set_id="H_root",
        actions=(
            RiverAction("check", "check", None, o1_check),
            RiverAction("bet", "bet", "20", o1_fold),
        ),
    )


def build_scenario() -> ThreePlayerRiverRakeScenario:
    """Return the tiny abstract one-street, zero-rake scenario."""

    return ThreePlayerRiverRakeScenario(
        root=_hero_root(),
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=_observation(),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def build_baseline_fixed_hero_policy() -> ExactBehaviorPolicy:
    """Return the complete fixed-Hero baseline policy."""

    return ExactBehaviorPolicy(
        {"H_root": {"check": "1", "bet": "0"}},
    )


def build_initial_profile() -> OpponentInitialProfile:
    """Return complete initial O1 and O2 behavior at every information set."""

    return OpponentInitialProfile(
        o1_probabilities={
            "O1_after_check": {"check": "1"},
            "O1_after_bet": {"fold": "1"},
        },
        o2_probabilities={
            "O2_after_check": {"check": "1"},
            "O2_after_bet": {"fold": "1"},
        },
    )


def build_attestation(scenario: ThreePlayerRiverRakeScenario):
    """Bind human-traceable perfect-recall evidence to the exact tree."""

    return create_perfect_recall_attestation(
        scenario,
        verifier="public worked-example fixture author",
        verification_date="2026-07-24",
        evidence_version=FIXTURE_VERSION,
        o1_confirmed=True,
        o2_confirmed=True,
    )


def worked_input_arguments() -> dict[str, object]:
    """Return the complete fixed inputs and explicitly bounded v1 configuration."""

    scenario = build_scenario()
    return {
        "scenario": scenario,
        "baseline_fixed_hero_policy": build_baseline_fixed_hero_policy(),
        "initial_profile": build_initial_profile(),
        "attestation": build_attestation(scenario),
        "generation": ThreePlayerCandidateGenerationConfig(
            shift_amounts=("1/2", "1"),
            max_simultaneous_info_sets=1,
            search_mode="robust_all",
            adaptation_mode="simultaneous_o1_o2",
        ),
        "repeated": ThreePlayerRepeatedConfig(
            horizon=3,
            discount=1.0,
        ),
    }


def run_analysis(**overrides: object):
    """Run the fixed public example with explicit testable input overrides."""

    arguments = worked_input_arguments()
    arguments.update(overrides)
    return evaluate_three_player_candidate_repeated(**arguments)


def run_workflow() -> dict[str, object]:
    """Return a deterministic allowlisted projection after strict checks pass."""

    result = run_analysis()
    if (
        result.status != EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE
        or result.analysis is None
        or result.error is not None
        or result.partial_result
    ):
        raise WorkflowFailure(
            "expected complete M32 analysis; received "
            f"status={result.status}, partial={result.partial_result}"
        )

    full = result.analysis.to_dict()
    baseline = full["baseline"]
    candidates = full["candidates"]
    selector = full["selector"]
    rows = selector["rows"]
    if len(candidates) != 2 or len(rows) != 4:
        raise WorkflowFailure(
            f"expected 2 candidates and 4 timing rows; received "
            f"{len(candidates)} and {len(rows)}"
        )

    all_m31_results = [
        baseline["m31_scenario_response"],
        *(candidate["m31_scenario_response"] for candidate in candidates),
    ]
    for m31_result in all_m31_results:
        if (
            m31_result["status"] != EXACT_SCENARIO_RESPONSE_COMPLETE
            or m31_result["response"]["status"] != EXACT_CORRESPONDENCE_COMPLETE
            or m31_result["response"]["coverage"] != "complete"
            or m31_result["response"]["partial_response"]
        ):
            raise WorkflowFailure("expected complete native M31/M30 response")

    if any(
        row["status"] != NO_BENEFICIAL_COMMITMENT
        or row["selected_candidate_id"] is not None
        for row in rows
    ):
        raise WorkflowFailure("fixture must retain all no-benefit timing rows")

    baseline_m31 = baseline["m31_scenario_response"]
    baseline_identities = baseline_m31["scenario_evaluation"]["identities"]
    m32_identities = full["identities"]
    return {
        "status": result.status,
        "fixture": {
            "version": FIXTURE_VERSION,
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
                "evidence_version": FIXTURE_VERSION,
                "human_trace": (
                    "each O1/O2 information set is singleton and no player "
                    "forgets a prior private observation or own action"
                ),
            },
            "search_mode": full["search_mode"],
            "adaptation_mode": full["adaptation_mode"],
            "shift_amounts": ["1/2", "1"],
            "max_simultaneous_info_sets": 1,
            "horizon": 3,
            "discount": 1.0,
        },
        "candidate_count": len(candidates),
        "baseline_initial": baseline["exact_initial_profile_values"],
        "candidates": [
            {
                "candidate_id": candidate["candidate"]["candidate_id"],
                "edits": candidate["candidate"]["edits"],
                "pre_adaptation_H": candidate["exact_values"][
                    "initial_profile"
                ]["H"],
                "post_response_H_hero_worst": candidate["exact_values"][
                    "response"
                ]["hero_worst"],
            }
            for candidate in candidates
        ],
        "hero_safety": {
            "source_path": SAFETY_SOURCE_PATH,
            "current_cfr_used": False,
            "first_witness_used": False,
            "pure_subset_used": False,
            "coalition_stress_used": False,
            "hero_best_used": False,
        },
        "timing_rows": [
            {
                "adaptation_opportunity": row["adaptation_opportunity"],
                "status": row["status"],
                "best_total_hero_ev_delta": row[
                    "best_total_hero_ev_delta"
                ],
                "selected_candidate_id": row["selected_candidate_id"],
                "full_primary_tie_candidate_ids": row[
                    "primary_tie_candidate_ids"
                ],
                "display_candidate_id": row[
                    "primary_tie_display_candidate_id"
                ],
            }
            for row in rows
        ],
        "identities": {
            "scenario": baseline_identities["scenario"],
            "tree_structure": baseline_identities["tree_structure"],
            "baseline_fixed_hero": baseline_identities["fixed_hero"],
            "initial_profile": baseline_identities["initial_profile"],
            "baseline_response_game": baseline_identities["response_game"],
            "baseline_response_run": baseline_identities["response_run"],
            "candidate_universe": m32_identities["candidate_universe"],
            "m32_run": m32_identities["m32_run"],
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
