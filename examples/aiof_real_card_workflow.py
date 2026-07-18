#!/usr/bin/env python3
"""Run one bounded real-card AIoF workflow through public module APIs only.

The fixture deliberately leaves one known five-card board live and marks the
other 43 cards dead.  It is a deterministic contract example, not a range
chart, an external-game solution, a profitability claim, or strategy advice.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction

from repeated_poker.aiof_cards import (
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
)
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRequest,
    SuppliedProfile,
    analyze_pushfold,
)
from repeated_poker.aiof_equity import (
    EquityAlgorithm,
    EquityRequest,
    calculate_preflop_equity,
)
from repeated_poker.aiof_strategy import (
    AiofStrategyLimits,
    AiofStrategyStatus,
    RationalStrategyRequest,
    generate_rational_lift_strategy,
)


SB_COMBO = "AsAh"
BB_COMBO = "KsKh"
KNOWN_BOARD = ("2c", "3d", "4h", "5s", "7c")


class WorkflowFailure(RuntimeError):
    """Internal control-flow error used to keep failure output no-partial."""


def exact_range(combo: str) -> RangeSpec:
    """Return a one-combo public range with explicit combo-mass semantics."""

    return RangeSpec((RangeEntry(combo, 1.0, WeightBasis.EXACT_COMBO_MASS),))


def dead_cards_for_known_board() -> tuple[str, ...]:
    """Return the other 43 cards in canonical deck-ID order."""

    live = {
        card_id(SB_COMBO[:2]),
        card_id(SB_COMBO[2:]),
        card_id(BB_COMBO[:2]),
        card_id(BB_COMBO[2:]),
        *(card_id(card) for card in KNOWN_BOARD),
    }
    return tuple(card_from_id(value) for value in range(52) if value not in live)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _require_phase1_success(label: str, result: object, payload_name: str) -> object:
    status = getattr(result, "status", None)
    payload = getattr(result, payload_name, None)
    error = getattr(result, "error_message", None)
    if status is not AiofStatus.SUCCESS or payload is None or error is not None:
        status_text = status.value if isinstance(status, AiofStatus) else str(status)
        raise WorkflowFailure(f"{label} failed with status {status_text}: {error or 'no payload'}")
    return payload


def run_workflow() -> dict[str, object]:
    """Return a small JSON-safe summary after every phase succeeds."""

    sb_range = exact_range(SB_COMBO)
    bb_range = exact_range(BB_COMBO)
    dead_cards = dead_cards_for_known_board()
    limits = AiofLimits(
        max_range_entries_per_side=1,
        max_exact_combos_per_side=1,
        max_compatible_combo_pairs=1,
        max_dead_cards=43,
        max_exact_board_evaluations=1,
        max_trace_points=0,
    )
    game = HeadsUpChipEvGame(
        starting_stack_sb=10.0,
        starting_stack_bb=10.0,
        small_blind=0.5,
        big_blind=1.0,
        ante=0.0,
        fee=0.0,
        third_party_dead_money=0.0,
        side_pot=False,
    )

    equity = _require_phase1_success(
        "equity",
        calculate_preflop_equity(
            EquityRequest(
                sb_range,
                bb_range,
                dead_cards,
                EquityAlgorithm.EXACT_EXHAUSTIVE,
                limits,
                0,
                None,
                None,
            )
        ),
        "estimate",
    )

    profile = SuppliedProfile(
        (ComboActionProbability(SB_COMBO, 1.0),),
        (ComboActionProbability(BB_COMBO, 0.0),),
    )
    chip_ev = _require_phase1_success(
        "chip_ev",
        analyze_pushfold(
            PushFoldRequest(
                sb_range,
                bb_range,
                dead_cards,
                EquityAlgorithm.EXACT_EXHAUSTIVE,
                limits,
                0,
                game,
                profile,
                ("sb", "bb"),
                0.0,
                None,
                None,
            )
        ),
        "analysis",
    )

    strategy_run = generate_rational_lift_strategy(
        RationalStrategyRequest(
            sb_range,
            bb_range,
            dead_cards,
            EquityAlgorithm.EXACT_EXHAUSTIVE,
            game,
            AiofStrategyLimits(
                max_solver_combos_per_side=1,
                max_payoff_cells=1,
                max_exact_board_evaluations=1,
                max_lp_variables_per_problem=2,
                max_lp_constraints_per_problem=2,
            ),
            claim_epsilon=Fraction(0),
            display_tie_tolerance=Fraction(0),
            requested_trace_points=0,
            run_reference_oracle=False,
            run_phase1_float_diagnostic=False,
            seed=None,
            samples=None,
        )
    )
    strategy = strategy_run.strategy_result
    if (
        strategy_run.status is not AiofStrategyStatus.SUCCESS
        or strategy is None
        or strategy_run.error is not None
    ):
        detail = strategy_run.error.message if strategy_run.error is not None else "no payload"
        raise WorkflowFailure(
            f"rational_strategy failed with status {strategy_run.status.value}: {detail}"
        )

    counts = equity.unweighted_counts
    responses = {response.seat: response for response in chip_ev.best_responses}
    gains = strategy.witness.gains
    return {
        "fixture": {
            "sb_combo": SB_COMBO,
            "bb_combo": BB_COMBO,
            "known_board": list(KNOWN_BOARD),
            "dead_card_count": len(dead_cards),
        },
        "equity": {
            "algorithm": EquityAlgorithm.EXACT_EXHAUSTIVE.value,
            "trials": counts.trials,
            "wins": counts.wins,
            "losses": counts.losses,
            "ties": counts.ties,
            "board_evaluations": equity.board_evaluations,
        },
        "chip_ev": {
            "profile_value_sb": chip_ev.profile_value_sb,
            "profile_value_bb": chip_ev.profile_value_bb,
            "conservation_sum": chip_ev.profile_value_sb + chip_ev.profile_value_bb,
            "fixed_opponent_sb_gain": responses["sb"].raw_gain,
            "fixed_opponent_bb_gain": responses["bb"].raw_gain,
        },
        "rational_strategy": {
            "game_id": strategy.game_id,
            "claim_kind": f"{strategy.game_id}:{strategy.witness.claim_kind.value}",
            "profile_value": _fraction_text(gains.profile_value),
            "g_sb": _fraction_text(gains.g_sb),
            "g_bb": _fraction_text(gains.g_bb),
            "payoff_cell_count": strategy.payoff_cell_count,
            "exact_board_evaluations": strategy.exact_board_evaluations,
        },
    }


def main() -> int:
    try:
        summary = run_workflow()
    except WorkflowFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
