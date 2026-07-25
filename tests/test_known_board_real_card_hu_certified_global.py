"""Independent M38 contract, oracle, bound, and output-cap tests."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from dataclasses import fields, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

import repeated_poker.certified_global_optimizer as core
import repeated_poker.known_board_real_card_hu_certified_global as module
from repeated_poker.aiof_cards import (
    AiofLimits,
    RangeEntry,
    RangeSpec,
    WeightBasis,
)
from repeated_poker.certified_global_optimizer import (
    ExactActionInterval,
    ExactActionProbability,
    ExactBehaviorCell,
    ExactBehaviorCellRow,
    ExactBehaviorPolicy,
    ExactBehaviorRow,
)
from repeated_poker.known_board_real_card_hu_river import (
    ActionProbability,
    ComboBucketAssignment,
    ComboBucketMap,
    KnownBoardRealCardHuRiverLimits,
    RiverActionProfile,
    RiverProfileRow,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "repeated_poker" / (
    "known_board_real_card_hu_certified_global.py"
)
EXAMPLE = ROOT / "examples" / "known_board_real_card_hu_certified_global.py"
GUIDE = ROOT / "docs" / "known_board_real_card_hu_certified_global.md"
BOARD = ("Qc", "Jd", "9h", "7s", "4c")


def exact_range(*rows: tuple[str, float]) -> RangeSpec:
    return RangeSpec(
        tuple(
            RangeEntry(combo, weight, WeightBasis.EXACT_COMBO_MASS)
            for combo, weight in rows
        )
    )


def distribution(**values: float) -> tuple[ActionProbability, ...]:
    return tuple(ActionProbability(action, value) for action, value in values.items())


def hero_profile(
    buckets: tuple[str, ...],
    *,
    after: tuple[float, float] = (1.0, 0.0),
    versus: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> RiverActionProfile:
    return RiverActionProfile(
        tuple(
            row
            for bucket in buckets
            for row in (
                RiverProfileRow(
                    bucket,
                    "after_oop_check",
                    distribution(check=after[0], bet=after[1]),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_oop_bet",
                    distribution(
                        **{
                            "call": versus[0],
                            "fold": versus[1],
                            "raise": versus[2],
                        }
                    ),
                ),
            )
        )
    )


def villain_profile(buckets: tuple[str, ...]) -> RiverActionProfile:
    return RiverActionProfile(
        tuple(
            row
            for bucket in buckets
            for row in (
                RiverProfileRow(
                    bucket,
                    "oop_first",
                    distribution(check=0.5, bet=0.5),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_ip_bet",
                    distribution(call=0.5, fold=0.5),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_ip_raise",
                    distribution(call=0.5, fold=0.5),
                ),
            )
        )
    )


HERO_MAP = ComboBucketMap(
    ("S", "W"),
    (
        ComboBucketAssignment("AsAh", "S"),
        ComboBucketAssignment("2s2h", "W"),
    ),
)
VILLAIN_MAP = ComboBucketMap(
    ("V",), (ComboBucketAssignment("KsKh", "V"),)
)


def request(
    *,
    reverse: bool = False,
    rake_rate: float = 0.0,
    rake_cap: float | None = None,
    auto_villain: bool = False,
    absolute_gap: str = "1",
    integration_limits: module.KnownBoardRealCardHuCertifiedGlobalLimits | None = None,
    optimizer_limits: core.CertifiedGlobalOptimizerLimits | None = None,
    **changes: object,
) -> module.KnownBoardRealCardHuCertifiedGlobalRequest:
    hero_entries = (("AsAh", 1.0), ("2s2h", 1.0))
    board = BOARD
    hero_rows = hero_profile(("S", "W")).rows
    villain_rows = villain_profile(("V",)).rows
    hero_mapping = HERO_MAP
    villain_mapping = VILLAIN_MAP
    if reverse:
        hero_entries = tuple(reversed(hero_entries))
        board = tuple(reversed(board))
        hero_rows = tuple(
            replace(row, actions=tuple(reversed(row.actions)))
            for row in reversed(hero_rows)
        )
        villain_rows = tuple(
            replace(row, actions=tuple(reversed(row.actions)))
            for row in reversed(villain_rows)
        )
        hero_mapping = ComboBucketMap(
            HERO_MAP.bucket_ids, tuple(reversed(HERO_MAP.assignments))
        )
    values: dict[str, object] = {
        "board": board,
        "hero_range": exact_range(*hero_entries),
        "villain_range": exact_range(("KsKh", 1.0)),
        "baseline_hero_profile": RiverActionProfile(hero_rows),
        "hero_combo_to_bucket": hero_mapping,
        "villain_combo_to_bucket": villain_mapping,
        "baseline_villain_profile": (
            None if auto_villain else RiverActionProfile(villain_rows)
        ),
        "initial_commitment_hero": 1.0,
        "initial_commitment_villain": 1.0,
        "rake_rate": rake_rate,
        "rake_cap": rake_cap,
        "oop_bet_size": 2.0,
        "ip_bet_after_check_size": 2.0,
        "ip_raise_to_size": 5.0,
        "horizon": 1,
        "adaptation_opportunity": 1,
        "discount": "1",
        "response_tolerance": "0",
        "absolute_gap_tolerance": absolute_gap,
        "integration_limits": (
            integration_limits
            or module.KnownBoardRealCardHuCertifiedGlobalLimits()
        ),
        "optimizer_limits": (
            optimizer_limits or core.CertifiedGlobalOptimizerLimits()
        ),
    }
    values.update(changes)
    return module.KnownBoardRealCardHuCertifiedGlobalRequest(**values)


def one_pair_request(
    *,
    integration_limits: module.KnownBoardRealCardHuCertifiedGlobalLimits | None = None,
    optimizer_limits: core.CertifiedGlobalOptimizerLimits | None = None,
    **changes: object,
) -> module.KnownBoardRealCardHuCertifiedGlobalRequest:
    values: dict[str, object] = {
        "board": ("2c", "3d", "4h", "5s", "9c"),
        "hero_range": exact_range(("AsAh", 1.0)),
        "villain_range": exact_range(("KsKh", 1.0)),
        "baseline_hero_profile": hero_profile(
            ("AsAh",), after=(0.5, 0.5), versus=(0.5, 0.25, 0.25)
        ),
        "baseline_villain_profile": villain_profile(("KsKh",)),
        "rake_rate": 0.05,
        "rake_cap": 3.0,
        "oop_bet_size": 2.0,
        "ip_bet_after_check_size": 2.0,
        "ip_raise_to_size": 5.0,
        "horizon": 1,
        "adaptation_opportunity": 1,
        "absolute_gap_tolerance": "1/2",
        "integration_limits": (
            integration_limits
            or module.KnownBoardRealCardHuCertifiedGlobalLimits()
        ),
        "optimizer_limits": (
            optimizer_limits or core.CertifiedGlobalOptimizerLimits()
        ),
    }
    values.update(changes)
    return module.KnownBoardRealCardHuCertifiedGlobalRequest(**values)


def successful(
    value: module.KnownBoardRealCardHuCertifiedGlobalRequest,
) -> module.KnownBoardRealCardHuCertifiedGlobalResult:
    result = module.analyze_known_board_real_card_hu_certified_global(value)
    assert result.status in (
        core.CERTIFIED_GLOBAL,
        core.CERTIFIED_EPSILON_GLOBAL,
    ), result.error
    assert result.payload is not None and result.error is None
    assert result.partial_result is False
    return result


# ---------------------------------------------------------------------------
# Independent cards, terminal accounting, response, and exact LP oracle
# ---------------------------------------------------------------------------

RANK_VALUE = {rank: index for index, rank in enumerate("23456789TJQKA", 2)}


def independent_five(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    ranks = sorted((RANK_VALUE[card[0]] for card in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(
        ((count, rank) for rank, count in counts.items()), reverse=True
    )
    flush = len({card[1] for card in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    straight_high = next(
        (
            unique[index]
            for index in range(len(unique) - 4)
            if unique[index] - unique[index + 4] == 4
        ),
        None,
    )
    if flush and straight_high:
        return 8, (straight_high,)
    if groups[0][0] == 4:
        return 7, (groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return 6, (groups[0][1], groups[1][1])
    if flush:
        return 5, tuple(ranks)
    if straight_high:
        return 4, (straight_high,)
    if groups[0][0] == 3:
        kickers = sorted(
            (rank for rank in ranks if rank != groups[0][1]), reverse=True
        )
        return 3, (groups[0][1], *kickers)
    pairs = sorted((rank for count, rank in groups if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank in ranks if rank not in pairs)
        return 2, (pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        kickers = sorted((rank for rank in ranks if rank != pairs[0]), reverse=True)
        return 1, (pairs[0], *kickers)
    return 0, tuple(ranks)


def independent_seven(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    assert len(cards) == 7 and len(set(cards)) == 7
    return max(independent_five(tuple(combo)) for combo in itertools.combinations(cards, 5))


def exact_float(value: float) -> Fraction:
    return Fraction(*value.as_integer_ratio())


def independent_showdown(
    hero_invested: Fraction,
    villain_invested: Fraction,
    result: str,
    rate: Fraction,
    cap: Fraction | None,
) -> tuple[Fraction, Fraction, Fraction]:
    pot = hero_invested + villain_invested
    rake = rate * pot
    if cap is not None:
        rake = min(rake, cap)
    awarded = pot - rake
    if result == "hero":
        received = (awarded, Fraction(0))
    elif result == "villain":
        received = (Fraction(0), awarded)
    else:
        received = (awarded / 2, awarded / 2)
    output = (
        received[0] - hero_invested,
        received[1] - villain_invested,
        rake,
    )
    assert sum(output, Fraction(0)) == 0
    return output


def independent_fold(
    winner: str, loser_committed: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    output = (
        (loser_committed, -loser_committed, Fraction(0))
        if winner == "hero"
        else (-loser_committed, loser_committed, Fraction(0))
    )
    assert sum(output, Fraction(0)) == 0 and output[2] == 0
    return output


def terminal_lines(
    hero_combo: str,
    villain_combo: str,
    *,
    rate: Fraction = Fraction(0),
    cap: Fraction | None = None,
) -> dict[str, tuple[Fraction, Fraction, Fraction]]:
    hero_rank = independent_seven(
        (hero_combo[:2], hero_combo[2:], *BOARD)
    )
    villain_rank = independent_seven(
        (villain_combo[:2], villain_combo[2:], *BOARD)
    )
    result = "hero" if hero_rank > villain_rank else (
        "villain" if hero_rank < villain_rank else "chop"
    )
    return {
        "cc": independent_showdown(Fraction(1), Fraction(1), result, rate, cap),
        "cbc": independent_showdown(Fraction(3), Fraction(3), result, rate, cap),
        "cbf": independent_fold("hero", Fraction(1)),
        "bc": independent_showdown(Fraction(3), Fraction(3), result, rate, cap),
        "bf": independent_fold("villain", Fraction(1)),
        "brc": independent_showdown(Fraction(6), Fraction(6), result, rate, cap),
        "brf": independent_fold("hero", Fraction(3)),
    }


Pure = tuple[str, str, str]  # root, after check-bet, after bet-raise
Policy = tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]


def pure_strategies() -> tuple[Pure, ...]:
    return tuple(
        itertools.product(("check", "bet"), ("call", "fold"), ("call", "fold"))
    )


def independent_value(
    policy: Policy,
    pure: Pure,
    component: int,
    *,
    rate: Fraction = Fraction(0),
    cap: Fraction | None = None,
) -> Fraction:
    total = Fraction(0)
    for index, hero_combo in enumerate(("AsAh", "2s2h")):
        x, call, raise_ = policy[index * 3 : index * 3 + 3]
        fold = 1 - call - raise_
        lines = terminal_lines(
            hero_combo,
            "KsKh",
            rate=rate,
            cap=cap,
        )
        if pure[0] == "check":
            value = (1 - x) * lines["cc"][component] + x * lines[
                "cbc" if pure[1] == "call" else "cbf"
            ][component]
        else:
            value = (
                call * lines["bc"][component]
                + fold * lines["bf"][component]
                + raise_
                * lines["brc" if pure[2] == "call" else "brf"][component]
            )
        total += value / 2
    return total


def independent_response(
    policy: Policy,
) -> tuple[Fraction, Fraction, tuple[Pure, ...]]:
    rows = tuple(
        (
            pure,
            independent_value(policy, pure, 0),
            independent_value(policy, pure, 1),
        )
        for pure in pure_strategies()
    )
    villain_max = max(row[2] for row in rows)
    best = tuple(row for row in rows if row[2] == villain_max)
    return min(row[1] for row in best), max(row[1] for row in best), tuple(
        row[0] for row in best
    )


def affine_for_pure(pure: Pure) -> tuple[Fraction, ...]:
    origin = independent_value((Fraction(0),) * 6, pure, 0)
    return (
        origin,
        *(
            independent_value(
                tuple(
                    Fraction(1) if index == active else Fraction(0)
                    for index in range(6)
                ),
                pure,
                0,
            )
            - origin
            for active in range(6)
        ),
    )


def solve_linear(matrix: list[list[Fraction]]) -> tuple[Fraction, ...] | None:
    size = len(matrix)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                left - factor * right
                for left, right in zip(matrix[row], matrix[column])
            ]
    return tuple(row[-1] for row in matrix)


@lru_cache(maxsize=1)
def independent_continuous_optimum() -> tuple[Fraction, tuple[Fraction, ...]]:
    # Variables are strong(x,c,r), weak(x,c,r), z. Every row is lhs <= rhs.
    inequalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for pure in pure_strategies():
        coefficients = affine_for_pure(pure)
        constant = coefficients[0]
        inequalities.append(
            (tuple(-value for value in coefficients[1:]) + (Fraction(1),), constant)
        )
    for offset in (0, 3):
        for local, rhs in (
            ((-1, 0, 0), 0),
            ((1, 0, 0), 1),
            ((0, -1, 0), 0),
            ((0, 0, -1), 0),
            ((0, 1, 1), 1),
        ):
            row = [Fraction(0)] * 7
            row[offset : offset + 3] = map(Fraction, local)
            inequalities.append((tuple(row), Fraction(rhs)))
    candidates: list[tuple[Fraction, ...]] = []
    for active in itertools.combinations(inequalities, 7):
        solution = solve_linear(
            [list(coefficients) + [rhs] for coefficients, rhs in active]
        )
        if solution is None:
            continue
        if all(
            sum(c * v for c, v in zip(coefficients, solution)) <= rhs
            for coefficients, rhs in inequalities
        ):
            candidates.append(solution)
    best_value = max(row[6] for row in candidates)
    optimal = [row for row in candidates if row[6] == best_value]
    best = max(
        optimal,
        key=lambda row: min(
            *(min(row[offset], 1 - row[offset]) for offset in (0, 3)),
            *(row[offset] for offset in (1, 2, 4, 5)),
            1 - row[1] - row[2],
            1 - row[4] - row[5],
        ),
    )
    return best[6], best


def exact_policy(
    preparation: module.KnownBoardRealCardHuCertifiedGlobalPreparation,
    policy: Policy,
) -> ExactBehaviorPolicy:
    values = {}
    for index, bucket in enumerate(("S", "W")):
        x, call, raise_ = policy[index * 3 : index * 3 + 3]
        values[f"IP_after_OOP_check::{bucket}"] = {
            "bet": x,
            "check": 1 - x,
        }
        values[f"IP_vs_OOP_bet::{bucket}"] = {
            "call": call,
            "fold": 1 - call - raise_,
            "raise": raise_,
        }
    return ExactBehaviorPolicy(
        tuple(
            ExactBehaviorRow(
                row.information_set_id,
                tuple(
                    ExactActionProbability(
                        action, module._rational_text(values[row.information_set_id][action])
                    )
                    for action in row.legal_action_ids
                ),
            )
            for row in preparation.scenario.information_sets
        )
    )


def test_01_m29_joint_cards_showdown_rake_and_conservation_are_independent():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request(rake_rate=0.05, rake_cap=0.2)
    )
    assert prepared.board == tuple(sorted(BOARD))
    assert len(prepared.joint_rows) == 2
    assert {
        row.hero_combo: Fraction(*row.raw_joint_mass.as_integer_ratio())
        for row in prepared.joint_rows
    } == {"2s2h": Fraction(1), "AsAh": Fraction(1)}
    assert {
        row.hero_combo: row.showdown_result for row in prepared.joint_rows
    } == {"2s2h": "villain", "AsAh": "hero"}
    assert sum(
        (
            Fraction(*row.raw_joint_mass.as_integer_ratio())
            for row in prepared.joint_rows
        ),
        Fraction(0),
    ) == 2
    for hero in ("AsAh", "2s2h"):
        lines = terminal_lines(
            hero,
            "KsKh",
            rate=exact_float(0.05),
            cap=exact_float(0.2),
        )
        assert lines["cbf"] == (1, -1, 0)
        assert lines["bf"] == (-1, 1, 0)
        assert lines["brf"] == (3, -3, 0)
        assert all(sum(line, Fraction(0)) == 0 for line in lines.values())
        # The 2-chip check/check pot rakes 0.1; larger showdowns hit cap 0.2.
        assert lines["cc"][2] == exact_float(0.05) * 2
        assert {lines[key][2] for key in ("cbc", "bc", "brc")} == {
            exact_float(0.2)
        }
    chop = module.prepare_known_board_real_card_hu_certified_global_oracle(
        one_pair_request(
            hero_range=exact_range(("AsKh", 1.0)),
            villain_range=exact_range(("AdKc", 1.0)),
            baseline_hero_profile=hero_profile(("AsKh",)),
            baseline_villain_profile=villain_profile(("AdKc",)),
        )
    )
    assert chop.joint_rows[0].showdown_result == "chop"
    assert independent_seven(("As", "Kh", "2c", "3d", "4h", "5s", "9c")) == (
        independent_seven(("Ad", "Kc", "2c", "3d", "4h", "5s", "9c"))
    )


def test_01b_nonfactorized_blocker_conditioned_joint_is_recomputed_exactly():
    hero_mapping = ComboBucketMap(
        ("H1", "H2"),
        (
            ComboBucketAssignment("AsKh", "H1"),
            ComboBucketAssignment("9s8h", "H2"),
        ),
    )
    villain_mapping = ComboBucketMap(
        ("V1", "V2"),
        (
            ComboBucketAssignment("AsQh", "V1"),
            ComboBucketAssignment("JcTd", "V2"),
        ),
    )
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request(
            hero_range=exact_range(("AsKh", 1.0), ("9s8h", 2.0)),
            villain_range=exact_range(("AsQh", 3.0), ("JcTd", 5.0)),
            hero_combo_to_bucket=hero_mapping,
            villain_combo_to_bucket=villain_mapping,
            baseline_hero_profile=hero_profile(("H1", "H2")),
            baseline_villain_profile=villain_profile(("V1", "V2")),
        )
    )
    observed = {
        (row.hero_combo, row.villain_combo): Fraction(probability)
        for row, probability in zip(
            prepared.joint_rows, prepared.exact_joint_probabilities
        )
    }
    assert observed == {
        ("9s8h", "AsQh"): Fraction(6, 21),
        ("9s8h", "JcTd"): Fraction(10, 21),
        ("AsKh", "JcTd"): Fraction(5, 21),
    }
    hero_marginal = sum(
        probability
        for (hero, _), probability in observed.items()
        if hero == "AsKh"
    )
    villain_marginal = sum(
        probability
        for (_, villain), probability in observed.items()
        if villain == "JcTd"
    )
    assert observed[("AsKh", "JcTd")] != hero_marginal * villain_marginal
    assert prepared.joint_provenance.private_overlap_excluded_pair_count == 1


def test_02_full_domain_is_automatic_canonical_and_not_a_candidate_lattice():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request()
    )
    assert [
        (row.information_set_id, row.legal_action_ids)
        for row in prepared.scenario.information_sets
    ] == [
        ("IP_after_OOP_check::S", ("check", "bet")),
        ("IP_after_OOP_check::W", ("check", "bet")),
        ("IP_vs_OOP_bet::S", ("call", "fold", "raise")),
        ("IP_vs_OOP_bet::W", ("call", "fold", "raise")),
    ]
    assert sum(
        len(row.legal_action_ids) - 1
        for row in prepared.scenario.information_sets
    ) == 6
    source = SOURCE.read_text(encoding="utf-8")
    assert "_materialize_candidates" not in source
    assert "_feasible_shift_options" not in source
    assert "grid_resolution" not in source
    assert "warm_start" not in source


def test_03_independent_exact_lp_has_strict_interior_optimum_and_certificate_covers_it():
    optimum_l, point = independent_continuous_optimum()
    policy = point[:6]
    assert optimum_l == Fraction(1, 2)
    assert any(
        0 < probability < 1
        for probability in (
            policy[0],
            policy[1],
            policy[2],
            policy[3],
            policy[4],
            policy[5],
            1 - policy[1] - policy[2],
            1 - policy[4] - policy[5],
        )
    )
    vertices = tuple(
        (x_s, c_s, r_s, x_w, c_w, r_w)
        for x_s in (Fraction(0), Fraction(1))
        for c_s, r_s in (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        for x_w in (Fraction(0), Fraction(1))
        for c_w, r_w in (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
    )
    pure_values = [independent_response(vertex)[0] for vertex in vertices]
    assert max(pure_values) < optimum_l

    result = successful(request())
    payload = result.payload
    assert payload is not None
    certificate = payload.native_optimizer_result.payload.certificate
    exact_uplift = optimum_l - Fraction(-1, 2)
    assert exact_uplift == 1
    assert Fraction(certificate.incumbent_lower_bound) <= exact_uplift
    assert exact_uplift <= Fraction(certificate.valid_global_upper_bound)
    assert Fraction(certificate.absolute_gap) <= Fraction(
        result.payload.request.absolute_gap_tolerance
    )
    assert payload.selected_point is not None
    selected = payload.native_optimizer_result.payload.selected_commitment
    assert selected is not None
    selected_values = {
        row.information_set_id: {
            action.action_id: Fraction(action.probability)
            for action in row.actions
        }
        for row in selected.rows
    }
    assert any(
        0 < probability < 1
        for row in selected_values.values()
        for probability in row.values()
    )


def test_04_point_response_matches_all_eight_independent_pure_responses():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request()
    )
    points: tuple[Policy, ...] = (
        (Fraction(0),) * 6,
        (
            Fraction(1, 2),
            Fraction(1, 3),
            Fraction(1, 6),
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(1, 4),
        ),
        tuple(independent_continuous_optimum()[1][:6]),
    )
    for point in points:
        policy = exact_policy(prepared, point)
        native = prepared.oracle.evaluate(policy)
        record = prepared.oracle.point_record(native.policy_identity)
        worst, best, responses = independent_response(point)
        assert Fraction(record.post_response_hero_ev_worst) == worst
        assert Fraction(record.response.hero_best_value) == best
        assert record.response.complete_response_record_count == len(responses)
        worst_count = sum(
            independent_value(point, pure, 0) == worst
            for pure in responses
        )
        assert record.response.hero_worst_witness_count == worst_count
        assert len(record.response.rows) == 3
        assert all(row.villain_history is not None for row in record.response.rows)
        assert record.response.correspondence_identity == (
            native.complete_response_correspondence_identity
        )


def test_05_tie_uses_hero_worst_and_reports_action_variation_provenance():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request()
    )
    tie_point = independent_continuous_optimum()[1]
    point = tuple(tie_point[:6])
    worst, best, responses = independent_response(point)
    assert len(responses) >= 2
    policy = exact_policy(prepared, point)
    evaluation = prepared.oracle.evaluate(policy)
    response = prepared.oracle.point_record(evaluation.policy_identity).response
    assert Fraction(response.hero_worst_value) == worst
    assert Fraction(response.hero_best_value) == best
    assert response.complete_response_record_count == len(responses)
    assert response.action_variation_information_sets
    assert any(
        len(row.globally_appearing_actions) > 1 for row in response.rows
    )


def test_05b_positive_rake_nested_tie_propagates_independent_hero_range():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request(rake_rate=0.5, rake_cap=None)
    )
    point: Policy = (
        Fraction(1, 4),
        Fraction(1),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(0),
    )
    expected_rows = {
        ("check", "call", "call"): (
            Fraction(-1),
            Fraction(-3, 4),
            Fraction(7, 4),
        ),
        ("check", "call", "fold"): (
            Fraction(-1),
            Fraction(-3, 4),
            Fraction(7, 4),
        ),
        ("check", "fold", "call"): (
            Fraction(1, 8),
            Fraction(-3, 4),
            Fraction(5, 8),
        ),
        ("check", "fold", "fold"): (
            Fraction(1, 8),
            Fraction(-3, 4),
            Fraction(5, 8),
        ),
        ("bet", "call", "call"): (
            Fraction(-3, 2),
            Fraction(-3, 2),
            Fraction(3),
        ),
        ("bet", "call", "fold"): (
            Fraction(-3, 2),
            Fraction(-3, 2),
            Fraction(3),
        ),
        ("bet", "fold", "call"): (
            Fraction(-3, 2),
            Fraction(-3, 2),
            Fraction(3),
        ),
        ("bet", "fold", "fold"): (
            Fraction(-3, 2),
            Fraction(-3, 2),
            Fraction(3),
        ),
    }
    independent_rows = {
        pure: tuple(
            independent_value(
                point,
                pure,
                component,
                rate=Fraction(1, 2),
                cap=None,
            )
            for component in range(3)
        )
        for pure in pure_strategies()
    }
    assert len(independent_rows) == 8
    assert independent_rows == expected_rows
    villain_max = max(row[1] for row in independent_rows.values())
    best_responses = tuple(
        pure
        for pure, row in independent_rows.items()
        if row[1] == villain_max
    )
    assert villain_max == Fraction(-3, 4)
    assert best_responses == (
        ("check", "call", "call"),
        ("check", "call", "fold"),
        ("check", "fold", "call"),
        ("check", "fold", "fold"),
    )
    assert min(independent_rows[pure][0] for pure in best_responses) == -1
    assert max(independent_rows[pure][0] for pure in best_responses) == Fraction(
        1, 8
    )

    evaluation = prepared.oracle.evaluate(exact_policy(prepared, point))
    response = prepared.oracle.point_record(evaluation.policy_identity).response
    assert Fraction(response.hero_worst_value) == -1
    assert Fraction(response.hero_best_value) == Fraction(1, 8)
    assert Fraction(response.villain_max_value) == Fraction(-3, 4)
    assert Fraction(response.house_rake_at_hero_worst) == Fraction(7, 4)
    assert response.complete_response_record_count == 4
    assert response.hero_worst_witness_count == 2
    assert response.action_variation_information_sets == (
        "OOP_vs_IP_bet::V",
        "OOP_vs_IP_raise::V",
    )
    assert tuple(
        (
            row.information_set_id,
            row.villain_bucket_id,
            row.villain_history,
            row.conditional_best_actions,
            row.globally_appearing_actions,
        )
        for row in response.rows
    ) == (
        (
            "OOP_first::V",
            "V",
            (),
            ("check",),
            ("check",),
        ),
        (
            "OOP_vs_IP_bet::V",
            "V",
            (("OOP_first::V", "check"),),
            ("call", "fold"),
            ("call", "fold"),
        ),
        (
            "OOP_vs_IP_raise::V",
            "V",
            (("OOP_first::V", "bet"),),
            ("call", "fold"),
            ("call", "fold"),
        ),
    )


def root_cell(
    prepared: module.KnownBoardRealCardHuCertifiedGlobalPreparation,
) -> ExactBehaviorCell:
    return ExactBehaviorCell(
        tuple(
            ExactBehaviorCellRow(
                row.information_set_id,
                tuple(
                    ExactActionInterval(action, "0", "1")
                    for action in row.legal_action_ids
                ),
            )
            for row in prepared.scenario.information_sets
        )
    )


def constrained_cell(
    prepared: module.KnownBoardRealCardHuCertifiedGlobalPreparation,
) -> ExactBehaviorCell:
    bounds = {}
    for bucket in ("S", "W"):
        bounds[f"IP_after_OOP_check::{bucket}"] = {
            "check": ("1/4", "3/4"),
            "bet": ("1/4", "3/4"),
        }
        bounds[f"IP_vs_OOP_bet::{bucket}"] = {
            "call": ("0", "1/2"),
            "fold": ("0", "1"),
            "raise": ("0", "1/2"),
        }
    return ExactBehaviorCell(
        tuple(
            ExactBehaviorCellRow(
                row.information_set_id,
                tuple(
                    ExactActionInterval(action, *bounds[row.information_set_id][action])
                    for action in row.legal_action_ids
                ),
            )
            for row in prepared.scenario.information_sets
        )
    )


@pytest.mark.parametrize("factory", (root_cell, constrained_cell))
def test_06_whole_cell_bound_covers_independent_rational_exhaustive_samples(factory):
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request()
    )
    cell = factory(prepared)
    bound = Fraction(prepared.oracle.upper_bound(cell).upper_bound)
    intervals = module._cell_intervals(prepared.scenario, cell)
    checked = 0
    grid = (Fraction(0), Fraction(1, 2), Fraction(1))
    for point in itertools.product(grid, repeat=6):
        x_s, call_s, raise_s, x_w, call_w, raise_w = point
        if call_s + raise_s > 1 or call_w + raise_w > 1:
            continue
        values = {}
        for bucket, x, call, raise_ in (
            ("S", x_s, call_s, raise_s),
            ("W", x_w, call_w, raise_w),
        ):
            values.update(
                {
                    (f"IP_after_OOP_check::{bucket}", "bet"): x,
                    (f"IP_after_OOP_check::{bucket}", "check"): 1 - x,
                    (f"IP_vs_OOP_bet::{bucket}", "call"): call,
                    (f"IP_vs_OOP_bet::{bucket}", "raise"): raise_,
                    (f"IP_vs_OOP_bet::{bucket}", "fold"): 1 - call - raise_,
                }
            )
        if not all(
            lower <= values[key] <= upper
            for key, (lower, upper) in intervals.items()
        ):
            continue
        independent_uplift = independent_response(tuple(point))[0] + Fraction(
            1, 2
        )
        assert independent_uplift <= bound
        checked += 1
    assert checked >= 10
    if factory is root_cell:
        assert independent_continuous_optimum()[0] + Fraction(1, 2) <= bound


def test_07_semantic_permutation_is_byte_identical_and_meaningful_change_is_not():
    first = successful(request())
    second = successful(request(reverse=True))
    first_bytes = module.exact_known_board_real_card_hu_certified_global_json(
        first
    ).encode()
    second_bytes = module.exact_known_board_real_card_hu_certified_global_json(
        second
    ).encode()
    assert first_bytes == second_bytes
    assert first.payload.preparation.request_identity == (
        second.payload.preparation.request_identity
    )
    changed = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request(oop_bet_size=3.0)
    )
    assert changed.request_identity != first.payload.preparation.request_identity
    assert changed.tree_identity != first.payload.preparation.tree_identity
    assert changed.analysis_identity != first.payload.preparation.analysis_identity
    changed_result = successful(request(oop_bet_size=3.0))
    changed_bytes = module.exact_known_board_real_card_hu_certified_global_json(
        changed_result
    ).encode()
    assert changed_bytes != first_bytes


def test_08_identity_pins_fail_closed_before_m36():
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        request()
    )
    pins = module.KnownBoardRealCardHuCertifiedGlobalPins(
        request_identity=prepared.request_identity,
        prepared_joint_identity=prepared.prepared_joint_identity,
        baseline_identity=prepared.baseline_identity,
        analysis_identity="0" * 64,
    )
    result = module.analyze_known_board_real_card_hu_certified_global(
        request(pins=pins)
    )
    assert result.status == core.STALE_INPUT
    assert result.payload is None and result.error.phase == "pins"
    assert result.optimizer_work_counters == core.CertifiedGlobalWorkCounters()


def test_08b_every_m38_pin_accepts_raw_and_equivalent_prefix_and_rejects_stale():
    base = one_pair_request()
    prepared = module.prepare_known_board_real_card_hu_certified_global_oracle(
        base
    )
    exposed_identities = {
        "request_identity": prepared.request_identity,
        "board_identity": prepared.board_identity,
        "prepared_joint_identity": prepared.prepared_joint_identity,
        "profile_mapping_identity": prepared.profile_mapping_identity,
        "tree_identity": prepared.tree_identity,
        "baseline_identity": prepared.baseline_identity,
        "scenario_identity": prepared.scenario.scenario_identity,
        "response_oracle_identity": prepared.response_oracle_identity,
        "objective_identity": prepared.objective_identity,
        "analysis_identity": prepared.analysis_identity,
    }
    identities = {
        name: value[7:] if value.startswith("sha256:") else value
        for name, value in exposed_identities.items()
    }
    assert tuple(identities) == tuple(
        field.name
        for field in fields(module.KnownBoardRealCardHuCertifiedGlobalPins)
    )
    for name, identity in identities.items():
        raw = successful(
            replace(
                base,
                pins=module.KnownBoardRealCardHuCertifiedGlobalPins(
                    **{name: identity}
                ),
            )
        )
        prefixed = successful(
            replace(
                base,
                pins=module.KnownBoardRealCardHuCertifiedGlobalPins(
                    **{name: f"sha256:{identity}"}
                ),
            )
        )
        assert (
            module.exact_known_board_real_card_hu_certified_global_json(raw)
            == module.exact_known_board_real_card_hu_certified_global_json(
                prefixed
            )
        )
        stale_identity = ("0" if identity[0] != "0" else "1") + identity[1:]
        stale = module.analyze_known_board_real_card_hu_certified_global(
            replace(
                base,
                pins=module.KnownBoardRealCardHuCertifiedGlobalPins(
                    **{name: stale_identity}
                ),
            )
        )
        assert stale.status == core.STALE_INPUT
        assert stale.payload is None and stale.error.phase == "pins"
        assert stale.error.message == f"pins.{name} mismatch"
        assert stale.optimizer_work_counters == core.CertifiedGlobalWorkCounters()
        assert (
            stale.oracle_work_counters
            == module.KnownBoardRealCardHuCertifiedOracleWorkCounters()
        )


def reviewer_count(value: object) -> int:
    count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        count += 1
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return count


def duplicate_keys(encoded: str) -> int:
    duplicates = 0

    def hook(pairs):
        nonlocal duplicates
        keys = [key for key, _ in pairs]
        duplicates += len(keys) - len(set(keys))
        return dict(pairs)

    json.loads(encoded, object_pairs_hook=hook)
    return duplicates


def test_09_reviewer_projection_records_and_final_shape_are_exact():
    result = successful(one_pair_request())
    projection = module._success_output_projection(result)
    expanded = module._materialize_output_projection(projection)
    assert expanded == result.to_dict()
    records, byte_count = module._preflight_success_output(
        result,
        max_records=module.MAX_OUTPUT_RECORDS,
        max_bytes=module.MAX_OUTPUT_BYTES,
    )
    encoded = module.exact_known_board_real_card_hu_certified_global_json(result)
    payload_records = reviewer_count(result.payload.to_dict())
    whole_records = reviewer_count(result.to_dict())
    assert 0 < payload_records < whole_records
    assert records == reviewer_count(result.to_dict())
    assert byte_count == len(encoded.encode("utf-8"))
    assert duplicate_keys(encoded) == 0
    assert encoded.count("\n") == encoded.count("\r") == 0


def test_10_record_cap_exact_n_succeeds_and_n_minus_one_fails_with_counters():
    reference = successful(one_pair_request())
    records = reviewer_count(reference.to_dict())
    base = one_pair_request()
    at_n = module.analyze_known_board_real_card_hu_certified_global(
        replace(
            base,
            integration_limits=replace(
                base.integration_limits, max_output_records=records
            ),
        )
    )
    below = module.analyze_known_board_real_card_hu_certified_global(
        replace(
            base,
            integration_limits=replace(
                base.integration_limits, max_output_records=records - 1
            ),
        )
    )
    assert at_n.status in (core.CERTIFIED_GLOBAL, core.CERTIFIED_EPSILON_GLOBAL)
    assert below.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert below.payload is None and below.error.phase == "output"
    assert below.partial_result is False
    assert below.optimizer_work_counters == at_n.optimizer_work_counters
    assert below.oracle_work_counters == at_n.oracle_work_counters


def self_consistent_byte_cap() -> tuple[int, module.KnownBoardRealCardHuCertifiedGlobalResult]:
    cap = module.MAX_OUTPUT_BYTES
    latest = None
    for _ in range(12):
        base = one_pair_request()
        latest = successful(
            replace(
                base,
                integration_limits=replace(
                    base.integration_limits, max_output_bytes=cap
                ),
            )
        )
        actual = len(
            module.exact_known_board_real_card_hu_certified_global_json(
                latest
            ).encode("utf-8")
        )
        if actual == cap:
            return cap, latest
        cap = actual
    raise AssertionError("byte-cap fixed point did not converge")


def test_11_byte_cap_self_consistent_n_succeeds_and_n_minus_one_fails():
    cap, at_n = self_consistent_byte_cap()
    base = one_pair_request()
    below = module.analyze_known_board_real_card_hu_certified_global(
        replace(
            base,
            integration_limits=replace(
                base.integration_limits, max_output_bytes=cap - 1
            ),
        )
    )
    assert len(
        module.exact_known_board_real_card_hu_certified_global_json(at_n).encode()
    ) == cap
    assert below.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert below.payload is None and below.error.phase == "output"
    assert below.partial_result is False
    assert below.optimizer_work_counters == at_n.optimizer_work_counters
    assert below.oracle_work_counters == at_n.oracle_work_counters


def test_12_very_low_output_caps_do_not_call_aggregate_to_dict_or_serializer(
    monkeypatch,
):
    calls = Counter()
    original_payload = module.KnownBoardRealCardHuCertifiedGlobalPayload.to_dict
    original_result = module.KnownBoardRealCardHuCertifiedGlobalResult.to_dict

    def payload_spy(self):
        calls["payload"] += 1
        return original_payload(self)

    def result_spy(self):
        calls["result"] += 1
        return original_result(self)

    monkeypatch.setattr(
        module.KnownBoardRealCardHuCertifiedGlobalPayload, "to_dict", payload_spy
    )
    monkeypatch.setattr(
        module.KnownBoardRealCardHuCertifiedGlobalResult, "to_dict", result_spy
    )
    base = one_pair_request()
    for changes in (
        {"max_output_records": 1},
        {"max_output_bytes": 1},
    ):
        low = replace(
            base.integration_limits,
            **changes,
        )
        result = module.analyze_known_board_real_card_hu_certified_global(
            replace(base, integration_limits=low)
        )
        assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
        assert result.payload is None and result.error.phase == "output"
        assert result.optimizer_work_counters.oracle_calls_total > 0
        assert result.oracle_work_counters.point_evaluations > 0
    assert calls == Counter()


@pytest.mark.parametrize(
    ("value", "expected_phase"),
    (
        (
            replace(
                one_pair_request().integration_limits,
                max_preparation_records=1,
            ),
            "preparation",
        ),
        (
            replace(
                one_pair_request().integration_limits,
                max_response_rows=1,
            ),
            "preparation",
        ),
        (
            replace(
                one_pair_request().integration_limits,
                max_response_rows=3,
            ),
            "oracle",
        ),
    ),
)
def test_13_preparation_and_scalar_oracle_caps_keep_taxonomy(value, expected_phase):
    result = module.analyze_known_board_real_card_hu_certified_global(
        one_pair_request(integration_limits=value)
    )
    assert result.status == core.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    if expected_phase == "oracle":
        assert result.error.phase.startswith("oracle")
    else:
        assert result.error.phase == expected_phase
    assert result.partial_result is False


def test_14_m36_cell_rational_and_oracle_caps_remain_native_failures():
    cases = (
        replace(core.CertifiedGlobalOptimizerLimits(), max_cells=1),
        replace(core.CertifiedGlobalOptimizerLimits(), max_oracle_calls=2),
        replace(
            core.CertifiedGlobalOptimizerLimits(),
            max_rational_numerator_bits=2,
            max_rational_denominator_bits=2,
        ),
    )
    for limits in cases:
        result = module.analyze_known_board_real_card_hu_certified_global(
            one_pair_request(
                absolute_gap_tolerance="0", optimizer_limits=limits
            )
        )
        assert result.payload is None
        assert result.status in (
            core.LIMIT_REACHED_NO_CERTIFICATE,
            core.NUMERIC_FAILURE,
        )
        assert result.error.phase != "output"


def test_15_auto_baseline_and_complete_response_ignore_pure_enumeration_cap():
    river_limits = replace(
        KnownBoardRealCardHuRiverLimits(), max_br_list_materialization=1
    )
    result = successful(
        request(
            auto_villain=True,
            absolute_gap="1/8",
            river_limits=river_limits,
        )
    )
    assert result.payload.preparation.baseline_identity
    assert (
        result.payload.baseline_point.response.complete_response_record_count
        >= 1
    )
    assert result.payload.oracle_work_counters.response_rows >= 3


def test_16_strict_wrapper_rejects_malformed_success_failure_and_partial():
    counters = module.KnownBoardRealCardHuCertifiedOracleWorkCounters()
    with pytest.raises(ValueError):
        module.KnownBoardRealCardHuCertifiedGlobalResult(
            core.CERTIFIED_GLOBAL,
            None,
            None,
            core.CertifiedGlobalWorkCounters(),
            counters,
        )
    with pytest.raises(ValueError):
        module.KnownBoardRealCardHuCertifiedGlobalResult(
            core.INVALID_INPUT,
            None,
            None,
            core.CertifiedGlobalWorkCounters(),
            counters,
        )
    with pytest.raises(ValueError):
        module.KnownBoardRealCardHuCertifiedGlobalResult(
            core.INVALID_INPUT,
            None,
            module.KnownBoardRealCardHuCertifiedGlobalError("x", "x"),
            core.CertifiedGlobalWorkCounters(),
            counters,
            True,
        )


def test_17_invalid_collisions_normalization_profiles_and_nonzero_tolerance_fail():
    bad = (
        one_pair_request(board=("2c",) * 5),
        one_pair_request(dead_cards=("2c",)),
        one_pair_request(response_tolerance="1/1000"),
        one_pair_request(discount="1.0"),
        one_pair_request(horizon=True),
        one_pair_request(
            baseline_hero_profile=hero_profile(
                ("AsAh",), after=(0.1, 0.8)
            )
        ),
    )
    for value in bad:
        result = module.analyze_known_board_real_card_hu_certified_global(value)
        assert result.payload is None
        assert result.status in (core.INVALID_INPUT, core.UNSUPPORTED_DOMAIN)
        assert result.partial_result is False


def test_18_example_is_deterministic_in_two_fresh_processes():
    environment = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    outputs = [
        subprocess.run(
            [sys.executable, "-X", "utf8", str(EXAMPLE)],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert b"\r" not in outputs[0]
    assert outputs[0].endswith(b"\n")
    parsed = json.loads(outputs[0])
    assert parsed["status"] in (
        core.CERTIFIED_GLOBAL,
        core.CERTIFIED_EPSILON_GLOBAL,
    )
    assert hashlib.sha256(outputs[0]).hexdigest()


def test_19_contract_docs_and_protected_import_surface_are_explicit():
    source = SOURCE.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs" / "assumptions_and_limitations.md").read_text(
        encoding="utf-8"
    )
    for text in (source, guide):
        for token in (
            module.DOMAIN_CONTRACT,
            module.OBJECTIVE_CONTRACT,
            module.RESPONSE_CONTRACT,
            module.BOUND_CONTRACT,
            module.CLAIM_SCOPE,
            "Hero-worst",
            "rake",
        ):
            assert token in text
    assert "known_board_real_card_hu_certified_global.py" in readme
    assert "M38" in limitations
    assert "equilibrium" in guide
    assert "strategy advice" in guide
    assert "solver-grade" in guide
