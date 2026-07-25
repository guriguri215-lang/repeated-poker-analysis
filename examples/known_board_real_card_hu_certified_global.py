"""Deterministic M38 known-board HU certified-global example."""

import sys

from repeated_poker.aiof_cards import (
    RangeEntry,
    RangeSpec,
    WeightBasis,
)
from repeated_poker.known_board_real_card_hu_certified_global import (
    KnownBoardRealCardHuCertifiedGlobalRequest,
    analyze_known_board_real_card_hu_certified_global,
    exact_known_board_real_card_hu_certified_global_json,
)
from repeated_poker.known_board_real_card_hu_river import (
    ActionProbability,
    ComboBucketAssignment,
    ComboBucketMap,
    RiverActionProfile,
    RiverProfileRow,
)


def distribution(**values: float) -> tuple[ActionProbability, ...]:
    return tuple(
        ActionProbability(action, probability)
        for action, probability in values.items()
    )


hero_profile = RiverActionProfile(
    (
        RiverProfileRow(
            "H",
            "after_oop_check",
            distribution(check=1.0, bet=0.0),
        ),
        RiverProfileRow(
            "H",
            "vs_oop_bet",
            distribution(**{"call": 0.0, "fold": 1.0, "raise": 0.0}),
        ),
    )
)
villain_profile = RiverActionProfile(
    (
        RiverProfileRow(
            "V", "oop_first", distribution(check=0.5, bet=0.5)
        ),
        RiverProfileRow(
            "V", "vs_ip_bet", distribution(call=0.5, fold=0.5)
        ),
        RiverProfileRow(
            "V", "vs_ip_raise", distribution(call=0.5, fold=0.5)
        ),
    )
)

request = KnownBoardRealCardHuCertifiedGlobalRequest(
    board=("Qc", "Jd", "9h", "7s", "4c"),
    hero_range=RangeSpec(
        (
            RangeEntry("AsAh", 1.0, WeightBasis.EXACT_COMBO_MASS),
            RangeEntry("2s2h", 1.0, WeightBasis.EXACT_COMBO_MASS),
        )
    ),
    villain_range=RangeSpec(
        (RangeEntry("KsKh", 1.0, WeightBasis.EXACT_COMBO_MASS),)
    ),
    baseline_hero_profile=hero_profile,
    hero_combo_to_bucket=ComboBucketMap(
        ("H",),
        (
            ComboBucketAssignment("AsAh", "H"),
            ComboBucketAssignment("2s2h", "H"),
        ),
    ),
    villain_combo_to_bucket=ComboBucketMap(
        ("V",), (ComboBucketAssignment("KsKh", "V"),)
    ),
    baseline_villain_profile=villain_profile,
    initial_commitment_hero=1.0,
    initial_commitment_villain=1.0,
    oop_bet_size=2.0,
    ip_bet_after_check_size=2.0,
    ip_raise_to_size=5.0,
    horizon=1,
    adaptation_opportunity=1,
    discount="1",
    absolute_gap_tolerance="1/64",
)

result = analyze_known_board_real_card_hu_certified_global(request)
encoded = exact_known_board_real_card_hu_certified_global_json(result).encode(
    "utf-8"
)
sys.stdout.buffer.write(encoded + b"\n")
