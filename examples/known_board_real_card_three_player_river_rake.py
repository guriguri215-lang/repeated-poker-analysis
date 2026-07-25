#!/usr/bin/env python3
"""Run one tiny known-board real-card M30-M32 river/rake analysis.

The public example is a bounded conditional comparison, not a continuous/global
optimizer, equilibrium certificate, profitability claim, or strategy advice.
"""

from __future__ import annotations

import json
import sys

from repeated_poker.aiof_cards import (
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
)
from repeated_poker.known_board_real_card_three_player_river import (
    ExactActionProbability,
    GeneratedTreeAttestation,
    KnownBoardRealCardThreePlayerRequest,
    RealCardProfileRow,
    RealCardThreePlayerProfile,
    analyze_known_board_real_card_three_player_river,
)
from repeated_poker.three_player_candidate_repeated import (
    ThreePlayerCandidateGenerationConfig,
    ThreePlayerRepeatedConfig,
)


FIXTURE_VERSION = "m35-known-board-real-card-three-player-example-v1"


def _range(combo: str) -> RangeSpec:
    return RangeSpec(
        (
            RangeEntry(
                combo,
                1.0,
                WeightBasis.EXACT_COMBO_MASS,
            ),
        )
    )


def _actions(
    *rows: tuple[str, str],
) -> tuple[ExactActionProbability, ...]:
    return tuple(ExactActionProbability(*row) for row in rows)


def build_baseline_profile() -> RealCardThreePlayerProfile:
    """Return a complete scenario-native H/O1/O2 behavior profile."""

    return RealCardThreePlayerProfile(
        (
            RealCardProfileRow(
                "H",
                "AsAh",
                "open",
                _actions(("check", "1"), ("bet", "0")),
            ),
            RealCardProfileRow(
                "O1",
                "KsKh",
                "after_hero_check",
                _actions(("check", "1")),
            ),
            RealCardProfileRow(
                "O1",
                "KsKh",
                "vs_hero_bet",
                _actions(("call", "1"), ("fold", "0")),
            ),
            RealCardProfileRow(
                "O2",
                "QsQh",
                "after_hero_o1_check",
                _actions(("check", "1")),
            ),
            RealCardProfileRow(
                "O2",
                "QsQh",
                "vs_hero_bet_o1_call",
                _actions(("call", "1"), ("fold", "0")),
            ),
            RealCardProfileRow(
                "O2",
                "QsQh",
                "vs_hero_bet_o1_fold",
                _actions(("call", "1"), ("fold", "0")),
            ),
        )
    )


def build_request() -> KnownBoardRealCardThreePlayerRequest:
    """Return the complete deterministic public fixture."""

    return KnownBoardRealCardThreePlayerRequest(
        board=("2c", "3d", "4h", "5s", "9c"),
        dead_cards=(),
        hero_range=_range("AsAh"),
        o1_range=_range("KsKh"),
        o2_range=_range("QsQh"),
        baseline_profile=build_baseline_profile(),
        attestation=GeneratedTreeAttestation(
            verifier="public M35 worked-example fixture author",
            verification_date="2026-07-25",
            evidence_version=FIXTURE_VERSION,
        ),
        initial_contribution="10",
        bet_to="20",
        rake_rate="1/10",
        rake_cap=None,
        generation=ThreePlayerCandidateGenerationConfig(
            shift_amounts=("1",),
            max_simultaneous_info_sets=1,
            search_mode="robust_all",
            adaptation_mode="simultaneous_o1_o2",
        ),
        repeated=ThreePlayerRepeatedConfig(horizon=3, discount=1.0),
    )


def run_workflow() -> dict[str, object]:
    """Return a deterministic allowlisted summary after strict checks."""

    result = analyze_known_board_real_card_three_player_river(build_request())
    if (
        result.status is not AiofStatus.SUCCESS
        or result.payload is None
        or result.error_message is not None
    ):
        raise RuntimeError(
            f"expected complete M35 result, received {result.status.value}"
        )
    payload = result.payload
    analysis = payload.native_m32_result.analysis
    if analysis is None or len(analysis.candidates) != 1:
        raise RuntimeError("expected one complete fresh-response candidate")
    candidate = analysis.candidates[0]
    response = candidate.scenario_response.response
    if response is None or response["coverage"] != "complete":
        raise RuntimeError("expected complete M30 response correspondence")
    return {
        "status": result.status.value,
        "fixture_version": FIXTURE_VERSION,
        "board": list(payload.prepared_support.board),
        "dead_cards": list(payload.prepared_support.dead_cards),
        "ranges": {
            "H": "AsAh",
            "O1": "KsKh",
            "O2": "QsQh",
        },
        "compatible_triples": payload.prepared_support.compatible_triple_count,
        "triple_probability_exact": (
            payload.prepared_support.triples[0].probability_exact
        ),
        "three_way_winners": list(payload.showdown_rows[0].three_way_winners),
        "rake": {"rate": "1/10", "cap": None},
        "baseline_initial": analysis.baseline_exact_values,
        "candidate_initial": candidate.exact_values["initial_profile"],
        "candidate_response": {
            "coverage": response["coverage"],
            "hero_worst": response["hero_worst"],
            "hero_best": response["hero_best"],
            "safety_source": candidate.exact_values["response"][
                "safety_scalar_source_path"
            ],
        },
        "timing_rows": [
            {
                "adaptation_opportunity": row.adaptation_opportunity,
                "status": row.status,
                "best_total_hero_ev_delta": row.best_total_hero_ev_delta,
                "selected_candidate_id": row.selected_candidate_id,
            }
            for row in analysis.selector_report.rows
        ],
        "identities": dict(payload.identities),
        "claim_boundary": {
            "continuous_or_global_optimizer": False,
            "equilibrium_certificate": False,
            "profitability_or_strategy_advice": False,
        },
    }


def main() -> int:
    try:
        output = run_workflow()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
