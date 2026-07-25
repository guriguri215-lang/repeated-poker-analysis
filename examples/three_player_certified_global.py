"""Deterministic M39 known-board real-card three-player example."""

import sys

from repeated_poker.aiof_cards import RangeEntry, RangeSpec, WeightBasis
from repeated_poker.known_board_real_card_three_player_river import (
    ExactActionProbability,
    GeneratedTreeAttestation,
    KnownBoardRealCardThreePlayerRequest,
    RealCardProfileRow,
    RealCardThreePlayerProfile,
)
from repeated_poker.three_player_candidate_repeated import (
    ThreePlayerRepeatedConfig,
)
from repeated_poker.three_player_certified_global import (
    KnownBoardRealCardThreePlayerCertifiedGlobalRequest,
    analyze_known_board_real_card_three_player_certified_global,
    exact_three_player_certified_global_json,
)


def distribution(**values: str) -> tuple[ExactActionProbability, ...]:
    return tuple(
        ExactActionProbability(action, probability)
        for action, probability in values.items()
    )


hero = "AsAh"
o1 = "KsKh"
o2 = "QsQh"
profile = RealCardThreePlayerProfile(
    (
        RealCardProfileRow(
            "H", hero, "open", distribution(check="1", bet="0")
        ),
        RealCardProfileRow(
            "O1", o1, "after_hero_check", distribution(check="1")
        ),
        RealCardProfileRow(
            "O1", o1, "vs_hero_bet", distribution(call="1", fold="0")
        ),
        RealCardProfileRow(
            "O2", o2, "after_hero_o1_check", distribution(check="1")
        ),
        RealCardProfileRow(
            "O2",
            o2,
            "vs_hero_bet_o1_call",
            distribution(call="1", fold="0"),
        ),
        RealCardProfileRow(
            "O2",
            o2,
            "vs_hero_bet_o1_fold",
            distribution(call="1", fold="0"),
        ),
    )
)

source = KnownBoardRealCardThreePlayerRequest(
    board=("2c", "3d", "4h", "5s", "9c"),
    dead_cards=(),
    hero_range=RangeSpec(
        (RangeEntry(hero, 1.0, WeightBasis.EXACT_COMBO_MASS),)
    ),
    o1_range=RangeSpec(
        (RangeEntry(o1, 1.0, WeightBasis.EXACT_COMBO_MASS),)
    ),
    o2_range=RangeSpec(
        (RangeEntry(o2, 1.0, WeightBasis.EXACT_COMBO_MASS),)
    ),
    baseline_profile=profile,
    attestation=GeneratedTreeAttestation(
        verifier="M39 deterministic example",
        verification_date="2026-07-25",
        evidence_version="m39-example-v1",
    ),
    rake_rate="1/20",
    rake_cap="1/5",
    repeated=ThreePlayerRepeatedConfig(horizon=1),
)
request = KnownBoardRealCardThreePlayerCertifiedGlobalRequest(
    source=source,
    adaptation_opportunity=1,
    absolute_gap_tolerance="100",
)
result = analyze_known_board_real_card_three_player_certified_global(request)
encoded = exact_three_player_certified_global_json(result).encode("utf-8")
sys.stdout.buffer.write(encoded + b"\n")
