"""Independent contract/oracle tests for the M37 real-card integration."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.aiof_preflop_certified_global as module
import repeated_poker.certified_global_optimizer as core
from repeated_poker.aiof_cards import (
    AiofLimits,
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
from repeated_poker.aiof_equity import EquityAlgorithm
from repeated_poker.aiof_preflop_candidate_repeated import (
    AiofPreflopCandidateRepeatedRequest,
    analyze_aiof_preflop_candidate_repeated,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "repeated_poker" / (
    "aiof_preflop_certified_global.py"
)
EXAMPLE = ROOT / "examples" / "aiof_preflop_certified_global.py"
GUIDE = ROOT / "docs" / "aiof_preflop_certified_global.md"

WIN_BOARD = ("2c", "3d", "4h", "5s", "7c")
INTERIOR_BOARD = ("Qc", "Jd", "9h", "7s", "4c")


def exact_range(*combos: str) -> RangeSpec:
    return RangeSpec(
        tuple(
            RangeEntry(combo, 1.0, WeightBasis.EXACT_COMBO_MASS)
            for combo in combos
        )
    )


def dead_except(live_cards: tuple[str, ...]) -> tuple[str, ...]:
    live = {card_id(card) for card in live_cards}
    return tuple(
        card_from_id(value) for value in range(52) if value not in live
    )


def profile(
    sb_rows: tuple[tuple[str, float], ...],
    bb_rows: tuple[tuple[str, float], ...],
) -> SuppliedProfile:
    return SuppliedProfile(
        tuple(ComboActionProbability(*row) for row in sb_rows),
        tuple(ComboActionProbability(*row) for row in bb_rows),
    )


def game(**changes: object) -> HeadsUpChipEvGame:
    values: dict[str, object] = {
        "starting_stack_sb": 10.0,
        "starting_stack_bb": 10.0,
        "small_blind": 0.5,
        "big_blind": 1.0,
        "ante": 0.0,
    }
    values.update(changes)
    return HeadsUpChipEvGame(**values)


def one_pair_request(
    *,
    hero_seat: str = "sb",
    shove: float = 0.0,
    call: float = 0.0,
    game_value: HeadsUpChipEvGame | None = None,
    horizon: int = 2,
    adaptation_opportunity: int = 2,
    discount: str = "1",
    integration_limits: module.AiofPreflopCertifiedGlobalLimits | None = None,
    optimizer_limits: core.CertifiedGlobalOptimizerLimits | None = None,
    optimizer_pins: core.CertifiedGlobalOptimizerPins | None = None,
) -> module.AiofPreflopCertifiedGlobalRequest:
    sb_combo = "AsAh"
    bb_combo = "KsKh"
    return module.AiofPreflopCertifiedGlobalRequest(
        game=game_value or game(),
        sb_range=exact_range(sb_combo),
        bb_range=exact_range(bb_combo),
        dead_cards=dead_except(
            ("As", "Ah", "Ks", "Kh") + WIN_BOARD
        ),
        baseline_profile=profile(
            ((sb_combo, shove),), ((bb_combo, call),)
        ),
        hero_seat=hero_seat,
        horizon=horizon,
        adaptation_opportunity=adaptation_opportunity,
        discount=discount,
        integration_limits=(
            integration_limits
            or module.AiofPreflopCertifiedGlobalLimits()
        ),
        optimizer_limits=(
            optimizer_limits or core.CertifiedGlobalOptimizerLimits()
        ),
        optimizer_pins=(
            optimizer_pins or core.CertifiedGlobalOptimizerPins()
        ),
    )


def interior_request(
    *,
    reverse: bool = False,
    integration_limits: module.AiofPreflopCertifiedGlobalLimits | None = None,
    optimizer_limits: core.CertifiedGlobalOptimizerLimits | None = None,
) -> module.AiofPreflopCertifiedGlobalRequest:
    sb_entries = (
        RangeEntry("AsAh", 1.0, WeightBasis.EXACT_COMBO_MASS),
        RangeEntry("2s2h", 1.0, WeightBasis.EXACT_COMBO_MASS),
    )
    sb_rows = (("AsAh", 0.0), ("2s2h", 0.0))
    dead = dead_except(
        ("As", "Ah", "2s", "2h", "Ks", "Kh")
        + INTERIOR_BOARD
    )
    if reverse:
        sb_entries = tuple(reversed(sb_entries))
        sb_rows = tuple(reversed(sb_rows))
        dead = tuple(reversed(dead))
    return module.AiofPreflopCertifiedGlobalRequest(
        game=game(),
        sb_range=RangeSpec(sb_entries),
        bb_range=exact_range("KsKh"),
        dead_cards=dead,
        baseline_profile=profile(sb_rows, (("KsKh", 0.0),)),
        hero_seat="sb",
        horizon=1,
        adaptation_opportunity=1,
        discount="1",
        absolute_gap_tolerance="1/64",
        integration_limits=(
            integration_limits
            or module.AiofPreflopCertifiedGlobalLimits()
        ),
        optimizer_limits=(
            optimizer_limits or core.CertifiedGlobalOptimizerLimits()
        ),
    )


def successful(
    request: module.AiofPreflopCertifiedGlobalRequest,
) -> module.AiofPreflopCertifiedGlobalPayload:
    result = module.analyze_aiof_preflop_certified_global(request)
    assert result.status in (
        core.CERTIFIED_GLOBAL,
        core.CERTIFIED_EPSILON_GLOBAL,
    ), result.error
    assert result.payload is not None
    assert result.error is None
    assert not result.partial_result
    return result.payload


def exact_policy(
    preparation: module.AiofPreflopCertifiedGlobalPreparation,
    probabilities: dict[str, Fraction],
) -> core.ExactBehaviorPolicy:
    active = preparation.oracle.hero_active_action
    return core.ExactBehaviorPolicy(
        tuple(
            core.ExactBehaviorRow(
                row.information_set_id,
                tuple(
                    core.ExactActionProbability(
                        action,
                        rational(
                            probabilities[row.information_set_id]
                            if action == active
                            else 1 - probabilities[row.information_set_id]
                        ),
                    )
                    for action in row.legal_action_ids
                ),
            )
            for row in preparation.scenario.information_sets
        )
    )


def rational(value: Fraction | int) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def root_cell(
    preparation: module.AiofPreflopCertifiedGlobalPreparation,
) -> core.ExactBehaviorCell:
    return core.ExactBehaviorCell(
        tuple(
            core.ExactBehaviorCellRow(
                row.information_set_id,
                tuple(
                    core.ExactActionInterval(action, "0", "1")
                    for action in sorted(row.legal_action_ids)
                ),
            )
            for row in sorted(
                preparation.scenario.information_sets,
                key=lambda item: item.information_set_id,
            )
        )
    )


def cell_for_active_intervals(
    preparation: module.AiofPreflopCertifiedGlobalPreparation,
    intervals: dict[str, tuple[Fraction, Fraction]],
) -> core.ExactBehaviorCell:
    active = preparation.oracle.hero_active_action
    rows = []
    for row in sorted(
        preparation.scenario.information_sets,
        key=lambda item: item.information_set_id,
    ):
        lower, upper = intervals[row.information_set_id]
        rows.append(
            core.ExactBehaviorCellRow(
                row.information_set_id,
                tuple(
                    core.ExactActionInterval(
                        action,
                        rational(
                            lower if action == active else 1 - upper
                        ),
                        rational(
                            upper if action == active else 1 - lower
                        ),
                    )
                    for action in sorted(row.legal_action_ids)
                ),
            )
        )
    return core.ExactBehaviorCell(tuple(rows))


def _straight_high(ranks: set[int]) -> int | None:
    ordered = sorted(ranks | ({1} if 14 in ranks else set()))
    run = 1
    best = None
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            run += 1
            if run >= 5:
                best = current
        elif current != previous:
            run = 1
    return best


def _independent_five_rank(cards: tuple[int, ...]) -> tuple[int, ...]:
    ranks = tuple(value // 4 + 2 for value in cards)
    suits = tuple(value % 4 for value in cards)
    counts = Counter(ranks)
    groups = sorted(
        ((count, rank) for rank, count in counts.items()), reverse=True
    )
    flush = len(set(suits)) == 1
    straight = _straight_high(set(ranks))
    if flush and straight is not None:
        return (8, straight)
    if groups[0][0] == 4:
        quad = groups[0][1]
        return (7, quad, max(rank for rank in ranks if rank != quad))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5,) + tuple(sorted(ranks, reverse=True))
    if straight is not None:
        return (4, straight)
    if groups[0][0] == 3:
        trip = groups[0][1]
        return (3, trip) + tuple(
            sorted(
                (rank for rank in ranks if rank != trip), reverse=True
            )
        )
    pairs = sorted(
        (rank for rank, count in counts.items() if count == 2),
        reverse=True,
    )
    if len(pairs) == 2:
        kicker = max(
            rank for rank, count in counts.items() if count == 1
        )
        return (2, pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        return (1, pair) + tuple(
            sorted(
                (rank for rank in ranks if rank != pair), reverse=True
            )
        )
    return (0,) + tuple(sorted(ranks, reverse=True))


def _independent_seven_rank(cards: tuple[int, ...]) -> tuple[int, ...]:
    return max(
        _independent_five_rank(subset)
        for subset in itertools.combinations(cards, 5)
    )


def independent_matchup_counts(
    sb_combo: str,
    bb_combo: str,
    dead_cards: tuple[str, ...],
) -> tuple[int, int, int]:
    sb = (card_id(sb_combo[:2]), card_id(sb_combo[2:]))
    bb = (card_id(bb_combo[:2]), card_id(bb_combo[2:]))
    excluded = set(sb) | set(bb) | {card_id(card) for card in dead_cards}
    remaining = tuple(value for value in range(52) if value not in excluded)
    wins = losses = ties = 0
    for board in itertools.combinations(remaining, 5):
        sb_rank = _independent_seven_rank(sb + board)
        bb_rank = _independent_seven_rank(bb + board)
        if sb_rank > bb_rank:
            wins += 1
        elif sb_rank < bb_rank:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def independent_interior_uplift(
    weak: Fraction, strong: Fraction
) -> Fraction:
    common = Fraction(-1, 2) + Fraction(1, 4) * (weak + strong)
    opponent_fold = Fraction(1, 2) * (weak + strong)
    opponent_call = -5 * weak + 5 * strong
    hero_worst = common + min(opponent_fold, opponent_call)
    return hero_worst + Fraction(1, 2)


def independent_cell_max(
    weak_interval: tuple[Fraction, Fraction],
    strong_interval: tuple[Fraction, Fraction],
) -> Fraction:
    weak_lower, weak_upper = weak_interval
    strong_lower, strong_upper = strong_interval
    points = {
        (weak, strong)
        for weak in (weak_lower, weak_upper)
        for strong in (strong_lower, strong_upper)
    }
    # Exact response kink: weak = 9/11 * strong.
    for strong in (strong_lower, strong_upper):
        weak = Fraction(9, 11) * strong
        if weak_lower <= weak <= weak_upper:
            points.add((weak, strong))
    for weak in (weak_lower, weak_upper):
        strong = Fraction(11, 9) * weak
        if strong_lower <= strong <= strong_upper:
            points.add((weak, strong))
    return max(independent_interior_uplift(*point) for point in points)


def test_sb_full_domain_certificate_uses_no_shift_or_grid_inputs():
    payload = successful(one_pair_request())
    native = payload.native_optimizer_result.payload
    assert native is not None
    assert native.certificate.incumbent_lower_bound == "3"
    assert native.certificate.valid_global_upper_bound == "3"
    assert native.certificate.absolute_gap == "0"
    assert native.selected_commitment is not None
    assert payload.selected_point is not None
    assert payload.selected_point.uplift == "3"
    assert len(payload.preparation.scenario.information_sets) == 1
    assert set(
        payload.preparation.scenario.information_sets[0].legal_action_ids
    ) == {"shove", "fold"}
    serialized = module.exact_aiof_preflop_certified_global_json(
        module.analyze_aiof_preflop_certified_global(one_pair_request())
    )
    for forbidden in (
        '"shift_amounts":',
        '"max_shifted_combos":',
        '"grid_resolution":',
        '"warm_start":',
        '"local_bounds":',
    ):
        assert forbidden not in serialized
    assert '"shift_amounts_used":false' in serialized


def test_bb_hero_domain_and_no_benefit_returns_no_commitment():
    payload = successful(
        one_pair_request(hero_seat="bb", shove=1.0, call=0.0)
    )
    native = payload.native_optimizer_result.payload
    assert native is not None
    assert native.no_beneficial_commitment
    assert native.selected_commitment is None
    assert payload.selected_point is None
    assert set(
        payload.preparation.scenario.information_sets[0].legal_action_ids
    ) == {"call", "fold"}
    assert native.certificate.incumbent_lower_bound == "0"
    assert native.certificate.valid_global_upper_bound == "0"


def test_selected_point_matches_existing_m28_real_card_payoff_and_response():
    m37 = successful(one_pair_request())
    m28_request = AiofPreflopCandidateRepeatedRequest(
        game=game(),
        sb_range=exact_range("AsAh"),
        bb_range=exact_range("KsKh"),
        dead_cards=dead_except(
            ("As", "Ah", "Ks", "Kh") + WIN_BOARD
        ),
        baseline_profile=profile(
            (("AsAh", 0.0),), (("KsKh", 0.0),)
        ),
        hero_seat="sb",
        shift_amounts=(1.0,),
        max_shifted_combos=1,
        horizon=2,
    )
    m28 = analyze_aiof_preflop_candidate_repeated(m28_request)
    assert m28.payload is not None
    candidate = m28.payload.candidates[0]
    assert m37.selected_point is not None
    assert Fraction(m37.selected_point.fixed_opponent_hero_ev) == Fraction(
        candidate.fixed_opponent_hero_ev
    )
    assert Fraction(
        m37.selected_point.post_response_hero_ev_worst
    ) == Fraction(candidate.post_response_hero_ev_worst)
    assert Fraction(m37.baseline_point.fixed_opponent_hero_ev) == Fraction(
        m28.payload.baseline.baseline_hero_ev
    )


def test_class_range_expands_then_blockers_define_exact_combo_domain():
    request = one_pair_request()
    request = replace(
        request,
        sb_range=RangeSpec(
            (RangeEntry("AA", 1.0, WeightBasis.CLASS_TOTAL_MASS),)
        ),
    )
    payload = successful(request)
    support = payload.preparation.prepared_ranges_projection["sb"]
    assert [row["combo"] for row in support] == ["AsAh"]
    assert support[0]["source_label"] == "AA"
    assert support[0]["raw_mass"] == "1/6"
    assert (
        payload.preparation.scenario.information_sets[0].information_set_id
        == "hero:sb:AsAh"
    )


def class_aa_request(
    hero_combo: str,
) -> module.AiofPreflopCertifiedGlobalRequest:
    hero_cards = (hero_combo[:2], hero_combo[2:])
    return module.AiofPreflopCertifiedGlobalRequest(
        game=game(),
        sb_range=RangeSpec(
            (RangeEntry("AA", 1.0, WeightBasis.CLASS_TOTAL_MASS),)
        ),
        bb_range=exact_range("KsKh"),
        dead_cards=dead_except(
            hero_cards + ("Ks", "Kh") + WIN_BOARD
        ),
        baseline_profile=profile(
            ((hero_combo, 0.0),), (("KsKh", 0.0),)
        ),
        hero_seat="sb",
        horizon=2,
        adaptation_opportunity=2,
    )


def test_blocker_change_changes_support_scenario_and_matchup_identity():
    first = module.prepare_aiof_preflop_certified_global_oracle(
        class_aa_request("AsAh")
    )
    second = module.prepare_aiof_preflop_certified_global_oracle(
        class_aa_request("AdAc")
    )
    assert (
        first.prepared_ranges_projection["sb"][0]["combo"] == "AsAh"
    )
    assert (
        second.prepared_ranges_projection["sb"][0]["combo"] == "AdAc"
    )
    assert (
        first.scenario.scenario_identity
        != second.scenario.scenario_identity
    )
    assert first.matchup_identity != second.matchup_identity


def test_interior_piecewise_concave_global_optimum_is_not_vertex_search():
    request = interior_request()
    assert independent_matchup_counts(
        "AsAh", "KsKh", request.dead_cards
    ) == (21, 0, 0)
    assert independent_matchup_counts(
        "2s2h", "KsKh", request.dead_cards
    ) == (0, 21, 0)
    pure_values = {
        (weak, strong): independent_interior_uplift(weak, strong)
        for weak, strong in itertools.product(
            (Fraction(0), Fraction(1)), repeat=2
        )
    }
    exact_optimum = independent_interior_uplift(
        Fraction(9, 11), Fraction(1)
    )
    assert exact_optimum == Fraction(15, 11)
    assert exact_optimum > max(pure_values.values())

    payload = successful(request)
    native = payload.native_optimizer_result.payload
    assert native is not None
    certificate = native.certificate
    lower = Fraction(certificate.incumbent_lower_bound)
    upper = Fraction(certificate.valid_global_upper_bound)
    assert lower <= exact_optimum <= upper
    assert upper - lower <= Fraction(1, 64)
    assert certificate.work_counters.splits > 0
    assert certificate.work_counters.cells_pruned > 0
    assert certificate.work_counters.active_cells_peak > 0
    assert (
        certificate.work_counters.cells_created
        == 2 * certificate.work_counters.splits + 1
    )


def test_root_and_child_whole_cell_bounds_dominate_independent_analytic_maxima():
    preparation = module.prepare_aiof_preflop_certified_global_oracle(
        interior_request()
    )
    root = root_cell(preparation)
    root_bound = Fraction(preparation.oracle.upper_bound(root).upper_bound)
    assert root_bound >= independent_cell_max(
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(1)),
    )

    ids = {
        row.information_set_id: row.information_set_id.rsplit(":", 1)[1]
        for row in preparation.scenario.information_sets
    }
    intervals = {
        information_set: (
            (Fraction(0), Fraction(1, 2))
            if combo == "2s2h"
            else (Fraction(1, 2), Fraction(1))
        )
        for information_set, combo in ids.items()
    }
    child = cell_for_active_intervals(preparation, intervals)
    child_bound = Fraction(
        preparation.oracle.upper_bound(child).upper_bound
    )
    assert child_bound >= independent_cell_max(
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1)),
    )


def test_complete_response_tie_identity_and_witness_multiplicity():
    preparation = module.prepare_aiof_preflop_certified_global_oracle(
        interior_request()
    )
    probabilities = {
        row.information_set_id: (
            Fraction(9, 11)
            if row.information_set_id.endswith("2s2h")
            else Fraction(1)
        )
        for row in preparation.scenario.information_sets
    }
    policy = exact_policy(preparation, probabilities)
    evaluation = preparation.oracle.evaluate(policy)
    record = preparation.oracle.point_record(evaluation.policy_identity)
    assert record.post_response_hero_ev_worst == "19/22"
    assert len(record.response.rows) == 1
    assert record.response.rows[0].best_actions == ("call", "fold")
    assert record.response.complete_response_record_count == 2
    assert record.response.hero_worst_witness_count == 2
    assert (
        evaluation.complete_response_correspondence_identity
        == record.response.correspondence_identity
    )


def test_baseline_is_exact_no_commitment_comparison_point():
    preparation = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request()
    )
    evaluation = preparation.oracle.evaluate(
        preparation.baseline_policy
    )
    assert evaluation.uplift == "0"
    assert (
        evaluation.baseline_total_repeated_hero_ev
        == evaluation.candidate_total_repeated_hero_ev
        == "-1"
    )


def test_discount_horizon_and_adaptation_are_exact_and_identity_bound():
    first = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request(discount="1/2")
    )
    changed_discount = (
        module.prepare_aiof_preflop_certified_global_oracle(
            one_pair_request(discount="1")
        )
    )
    changed_horizon = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request(horizon=3, adaptation_opportunity=2)
    )
    changed_adaptation = (
        module.prepare_aiof_preflop_certified_global_oracle(
            one_pair_request(horizon=2, adaptation_opportunity=1)
        )
    )
    assert len(
        {
            first.objective_identity,
            changed_discount.objective_identity,
            changed_horizon.objective_identity,
            changed_adaptation.objective_identity,
        }
    ) == 4


def test_binary64_inputs_are_lifted_losslessly_not_decimal_reparsed():
    request = one_pair_request()
    request = replace(
        request,
        sb_range=RangeSpec(
            (
                RangeEntry(
                    "AsAh", 0.1, WeightBasis.EXACT_COMBO_MASS
                ),
            )
        ),
    )
    preparation = module.prepare_aiof_preflop_certified_global_oracle(
        request
    )
    assert (
        preparation.prepared_ranges_projection["sb"][0]["raw_mass"]
        == "3602879701896397/36028797018963968"
    )


def test_semantic_input_permutations_keep_identity_and_bytes():
    forward_preparation = (
        module.prepare_aiof_preflop_certified_global_oracle(
            interior_request()
        )
    )
    reverse_preparation = (
        module.prepare_aiof_preflop_certified_global_oracle(
            interior_request(reverse=True)
        )
    )
    assert (
        forward_preparation.scenario.scenario_identity
        == reverse_preparation.scenario.scenario_identity
    )
    assert (
        forward_preparation.response_oracle_identity
        == reverse_preparation.response_oracle_identity
    )
    assert (
        forward_preparation.objective_identity
        == reverse_preparation.objective_identity
    )
    forward = module.exact_aiof_preflop_certified_global_json(
        module.analyze_aiof_preflop_certified_global(
            interior_request()
        )
    )
    reverse = module.exact_aiof_preflop_certified_global_json(
        module.analyze_aiof_preflop_certified_global(
            interior_request(reverse=True)
        )
    )
    assert forward == reverse


def test_semantic_change_changes_scenario_or_objective_identity():
    preparation = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request()
    )
    hero_change = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request(hero_seat="bb")
    )
    horizon_change = (
        module.prepare_aiof_preflop_certified_global_oracle(
            one_pair_request(horizon=3, adaptation_opportunity=2)
        )
    )
    assert (
        preparation.scenario.scenario_identity
        != hero_change.scenario.scenario_identity
    )
    assert (
        preparation.objective_identity
        != horizon_change.objective_identity
    )


@pytest.mark.parametrize(
    ("changes", "expected_status"),
    [
        ({"hero_seat": "button"}, core.INVALID_INPUT),
        ({"discount": "2/2"}, core.INVALID_INPUT),
        ({"discount": "0"}, core.INVALID_INPUT),
        ({"response_tolerance": "1/1000"}, core.UNSUPPORTED_DOMAIN),
        (
            {"algorithm": EquityAlgorithm.DETERMINISTIC_MONTE_CARLO},
            core.UNSUPPORTED_DOMAIN,
        ),
        ({"adaptation_opportunity": 4}, core.INVALID_INPUT),
    ],
)
def test_invalid_or_unsupported_requests_fail_closed(
    changes: dict[str, object], expected_status: str
):
    result = module.analyze_aiof_preflop_certified_global(
        replace(one_pair_request(), **changes)
    )
    assert result.status == expected_status
    assert result.payload is None
    assert result.error is not None
    assert not result.partial_result


def test_impossible_compatible_support_fails_closed():
    request = one_pair_request()
    result = module.analyze_aiof_preflop_certified_global(
        replace(
            request,
            bb_range=exact_range("AsKs"),
            baseline_profile=profile(
                (("AsAh", 0.0),), (("AsKs", 0.0),)
            ),
        )
    )
    assert result.status == core.INVALID_INPUT
    assert result.payload is None
    assert result.error is not None
    assert (
        result.error.cause_status
        == "EMPTY_COMPATIBLE_SUPPORT"
    )


def test_nonfinite_range_weight_fails_without_rational_fabrication():
    request = one_pair_request()
    result = module.analyze_aiof_preflop_certified_global(
        replace(
            request,
            sb_range=RangeSpec(
                (
                    RangeEntry(
                        "AsAh",
                        float("nan"),
                        WeightBasis.EXACT_COMBO_MASS,
                    ),
                )
            ),
        )
    )
    assert result.status == core.INVALID_INPUT
    assert result.payload is None
    assert result.error is not None


def test_information_set_cap_fails_before_equity_enumeration(monkeypatch):
    called = False

    def forbidden(_prepared):
        nonlocal called
        called = True
        raise AssertionError("equity enumeration must not start")

    monkeypatch.setattr(module, "_iter_exact_matchup_outcomes", forbidden)
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_information_sets_per_seat=1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        replace(interior_request(), integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert not called


def test_board_evaluation_cap_fails_before_equity_enumeration(monkeypatch):
    called = False

    def forbidden(_prepared):
        nonlocal called
        called = True
        raise AssertionError("equity enumeration must not start")

    monkeypatch.setattr(module, "_iter_exact_matchup_outcomes", forbidden)
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_exact_board_evaluations=1,
    )
    # Keep the public lower-layer cap permissive so this checks the M37
    # preflight immediately before its exact matchup iterator.
    result = module.analyze_aiof_preflop_certified_global(
        replace(
            interior_request(),
            integration_limits=limits,
            aiof_limits=AiofLimits(max_exact_board_evaluations=10),
        )
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert not called


def test_coefficient_cap_fails_before_matchup_coefficient_materialization(
    monkeypatch,
):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("coefficient construction must not start")

    monkeypatch.setattr(module, "_build_matchups", forbidden)
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_coefficient_records=1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        replace(interior_request(), integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert not called


def two_opponent_request(
    *,
    limits: module.AiofPreflopCertifiedGlobalLimits,
) -> module.AiofPreflopCertifiedGlobalRequest:
    live = (
        "As",
        "Ah",
        "Ks",
        "Kh",
        "Qs",
        "Qh",
    ) + INTERIOR_BOARD
    return module.AiofPreflopCertifiedGlobalRequest(
        game=game(),
        sb_range=exact_range("AsAh"),
        bb_range=exact_range("KsKh", "QsQh"),
        dead_cards=dead_except(live),
        baseline_profile=profile(
            (("AsAh", 0.0),),
            (("KsKh", 0.0), ("QsQh", 0.0)),
        ),
        hero_seat="sb",
        horizon=1,
        adaptation_opportunity=1,
        absolute_gap_tolerance="1/8",
        integration_limits=limits,
    )


def test_response_row_cap_fails_before_any_response_row_materialization():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_total_response_rows=1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        two_opponent_request(limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.oracle_work_counters.response_rows == 0
    assert result.oracle_work_counters.point_evaluations == 0


def test_child_bound_cap_fails_atomically_after_root_without_partial_prefix():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_oracle_bound_records=1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        interior_request(integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.oracle_work_counters.bound_records == 1
    assert not result.partial_result


def test_m36_child_cell_cap_fails_before_children_are_materialized():
    optimizer_limits = replace(
        core.CertifiedGlobalOptimizerLimits(), max_cells=1
    )
    result = module.analyze_aiof_preflop_certified_global(
        interior_request(optimizer_limits=optimizer_limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.optimizer_work_counters.cells_created == 1
    assert result.optimizer_work_counters.splits == 0


def test_output_byte_cap_discards_completed_internal_work_and_payload():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_output_bytes=100,
    )
    result = module.analyze_aiof_preflop_certified_global(
        one_pair_request(integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.error is not None


def test_output_record_cap_discards_payload_without_truncation():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_output_records=1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        one_pair_request(integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.error is not None
    assert not result.partial_result


def test_horizon_cap_fails_before_discount_weight_materialization():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(), max_horizon=1
    )
    result = module.analyze_aiof_preflop_certified_global(
        one_pair_request(integration_limits=limits)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.oracle_work_counters.exact_board_evaluations == 0


def test_cap_above_hard_ceiling_is_rejected_not_clamped():
    limits = replace(
        module.AiofPreflopCertifiedGlobalLimits(),
        max_matchup_rows=module.MAX_MATCHUP_ROWS + 1,
    )
    result = module.analyze_aiof_preflop_certified_global(
        one_pair_request(integration_limits=limits)
    )
    assert result.status == core.INVALID_INPUT
    assert result.payload is None
    assert "hard ceiling" in result.error.message


def test_stale_m37_pin_fails_before_exact_matchup_or_oracle_calls(monkeypatch):
    baseline = module.prepare_aiof_preflop_certified_global_oracle(
        one_pair_request()
    )
    called = False

    def forbidden(_prepared):
        nonlocal called
        called = True
        raise AssertionError("stale prepared pin must stop before equity")

    monkeypatch.setattr(module, "_iter_exact_matchup_outcomes", forbidden)
    request = replace(
        one_pair_request(),
        pins=module.AiofPreflopCertifiedGlobalPins(
            prepared_ranges_identity="0" * 64
        ),
    )
    assert baseline.prepared_ranges.content_identity != "0" * 64
    result = module.analyze_aiof_preflop_certified_global(request)
    assert result.status == core.STALE_INPUT
    assert result.payload is None
    assert not called


def test_stale_m36_pin_fails_before_point_or_bound_oracle_call(monkeypatch):
    point_calls = 0
    bound_calls = 0
    original_point = module.AiofPreflopCertifiedScalarOracle.evaluate
    original_bound = module.AiofPreflopCertifiedScalarOracle.upper_bound

    def counted_point(self, policy):
        nonlocal point_calls
        point_calls += 1
        return original_point(self, policy)

    def counted_bound(self, cell):
        nonlocal bound_calls
        bound_calls += 1
        return original_bound(self, cell)

    monkeypatch.setattr(
        module.AiofPreflopCertifiedScalarOracle,
        "evaluate",
        counted_point,
    )
    monkeypatch.setattr(
        module.AiofPreflopCertifiedScalarOracle,
        "upper_bound",
        counted_bound,
    )
    request = one_pair_request(
        optimizer_pins=core.CertifiedGlobalOptimizerPins(
            response_oracle_identity="0" * 64
        )
    )
    result = module.analyze_aiof_preflop_certified_global(request)
    assert result.status == core.STALE_INPUT
    assert result.payload is None
    assert point_calls == bound_calls == 0


def test_invalid_consumer_bound_is_rejected_by_native_m36_contract(
    monkeypatch,
):
    original = module.AiofPreflopCertifiedScalarOracle.upper_bound

    def invalid(self, cell):
        valid = original(self, cell)
        return replace(
            valid,
            upper_bound="-999",
            bound_identity=core.scalar_oracle_bound_identity(
                cell_identity=valid.cell_identity,
                response_oracle_identity=valid.response_oracle_identity,
                objective_identity=valid.objective_identity,
                upper_bound="-999",
                candidate_policy_identity=(
                    core.behavior_policy_identity(
                        self.scenario, valid.candidate_policy
                    )
                    if valid.candidate_policy is not None
                    else None
                ),
            ),
        )

    monkeypatch.setattr(
        module.AiofPreflopCertifiedScalarOracle,
        "upper_bound",
        invalid,
    )
    result = module.analyze_aiof_preflop_certified_global(
        one_pair_request()
    )
    assert result.status == core.INVALID_ORACLE_CONTRACT
    assert result.payload is None
    assert not result.partial_result


def test_native_result_and_full_identity_chain_are_losslessly_retained():
    payload = successful(one_pair_request())
    serialized = payload.to_dict()
    native = payload.native_optimizer_result.to_dict()
    assert serialized["native_optimizer_result"] == native
    certificate = native["payload"]["certificate"]
    assert (
        certificate["response_oracle_identity"]
        == payload.preparation.response_oracle_identity
    )
    assert (
        certificate["objective_identity"]
        == payload.preparation.objective_identity
    )
    assert (
        native["payload"]["domain"]["scenario_identity"]
        == payload.preparation.scenario.scenario_identity
    )
    assert (
        payload.baseline_point.native_evaluation.to_dict()
        == native["payload"]["baseline_evaluation"]
    )


def test_exact_json_is_stable_in_process_and_strict():
    request = one_pair_request()
    first = module.exact_aiof_preflop_certified_global_json(
        module.analyze_aiof_preflop_certified_global(request)
    )
    second = module.exact_aiof_preflop_certified_global_json(
        module.analyze_aiof_preflop_certified_global(request)
    )
    assert first == second
    parsed = json.loads(first)
    assert parsed["status"] == core.CERTIFIED_GLOBAL
    assert "NaN" not in first
    assert "\n" not in first


def test_public_example_is_byte_identical_across_two_fresh_processes():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    outputs = [
        subprocess.check_output(
            [sys.executable, str(EXAMPLE)],
            cwd=ROOT,
            env=environment,
        )
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert outputs[0].endswith(b"\n")
    assert hashlib.sha256(outputs[0]).hexdigest() == (
        "db42796d5fc2412dced84e33adac953711f2ecd2adf0bb5d9d5f6e981e6b632c"
    )


def test_claim_boundary_and_bound_derivation_are_source_and_guide_owned():
    source = SOURCE.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "W_pre*max_C a",
        "sum_j min_r max_C L[j,r]",
        "specified-tolerance global maximum",
        "does not claim equilibrium",
        "float.as_integer_ratio()",
    ):
        assert required in source
    for required in (
        "full legal domain",
        "complete response correspondence",
        "whole-cell",
        "not an equilibrium",
        "R8",
    ):
        assert required in guide
