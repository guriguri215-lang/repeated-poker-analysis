"""Deterministic M40 unified certified-global public workflow example."""

import sys

from repeated_poker.aiof_cards import (
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
)
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    SuppliedProfile,
)
from repeated_poker.aiof_preflop_certified_global import (
    AiofPreflopCertifiedGlobalRequest,
)
from repeated_poker.unified_certified_global_workflow import (
    REAL_CARD_AIOF_PREFLOP,
    UnifiedCertifiedGlobalRequest,
    analyze_unified_certified_global,
    exact_unified_certified_global_json,
)


def exact_range(combo: str) -> RangeSpec:
    return RangeSpec(
        (RangeEntry(combo, 1.0, WeightBasis.EXACT_COMBO_MASS),)
    )


def dead_except(live_cards: tuple[str, ...]) -> tuple[str, ...]:
    live = {card_id(card) for card in live_cards}
    return tuple(
        card_from_id(value) for value in range(52) if value not in live
    )


def main() -> None:
    sb_combo = "AsAh"
    bb_combo = "KsKh"
    board = ("2c", "3d", "4h", "5s", "7c")
    nested = AiofPreflopCertifiedGlobalRequest(
        game=HeadsUpChipEvGame(
            starting_stack_sb=10.0,
            starting_stack_bb=10.0,
            small_blind=0.5,
            big_blind=1.0,
            ante=0.0,
        ),
        sb_range=exact_range(sb_combo),
        bb_range=exact_range(bb_combo),
        dead_cards=dead_except(
            ("As", "Ah", "Ks", "Kh") + board
        ),
        baseline_profile=SuppliedProfile(
            sb_shove=(ComboActionProbability(sb_combo, 0.0),),
            bb_call=(ComboActionProbability(bb_combo, 0.0),),
        ),
        hero_seat="sb",
        horizon=2,
        adaptation_opportunity=2,
        discount="1",
        absolute_gap_tolerance="0",
        relative_gap_tolerance="0",
    )
    result = analyze_unified_certified_global(
        UnifiedCertifiedGlobalRequest(
            variant=REAL_CARD_AIOF_PREFLOP,
            nested_request=nested,
        )
    )
    encoded = exact_unified_certified_global_json(result).encode("utf-8")
    sys.stdout.buffer.write(encoded + b"\n")


if __name__ == "__main__":
    main()
