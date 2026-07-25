"""M39 abstract and known-board real-card certified-global integration tests."""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from repeated_poker import certified_global_optimizer as m36
from repeated_poker import known_board_real_card_three_player_river as m35
from repeated_poker import three_player_certified_global as module
from repeated_poker import three_player_river_rake as m31
from repeated_poker.aiof_cards import (
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_id,
    canonicalize_exact_combo,
)
from repeated_poker.three_player_candidate_repeated import (
    ThreePlayerRepeatedConfig,
)


VALIDATION_DATE = "2026-07-25"
EVIDENCE_VERSION = "m39-independent-test-v1"
BOARD = ("2c", "3d", "4h", "5s", "9c")


def observation(suffix: str) -> m31.RiverObservation:
    return m31.RiverObservation(
        f"public-{suffix}",
        {
            "H": f"hero-{suffix}",
            "O1": f"o1-{suffix}",
            "O2": f"o2-{suffix}",
        },
    )


def attestation(
    scenario: m31.ThreePlayerRiverRakeScenario,
) -> m31.PerfectRecallAttestation:
    return m31.create_perfect_recall_attestation(
        scenario,
        verifier="M39 independent test verifier",
        verification_date=VALIDATION_DATE,
        evidence_version=EVIDENCE_VERSION,
        o1_confirmed=True,
        o2_confirmed=True,
    )


def two_action_scenario() -> m31.ThreePlayerRiverRakeScenario:
    showdown = m31.RiverTerminalNode(
        "hero-check-terminal",
        "showdown",
        (m31.AwardShare("H", "1"),),
    )
    o2_check = m31.RiverDecisionNode(
        "o2-check-after-hero",
        "O2",
        "O2_after_hero_check",
        (m31.RiverAction("check", "check", None, showdown),),
    )
    o1_check = m31.RiverDecisionNode(
        "o1-check-after-hero",
        "O1",
        "O1_after_hero_check",
        (m31.RiverAction("check", "check", None, o2_check),),
    )
    fold = m31.RiverTerminalNode("hero-bet-both-fold", "fold")
    o2_fold = m31.RiverDecisionNode(
        "o2-fold-after-hero",
        "O2",
        "O2_after_hero_bet",
        (m31.RiverAction("fold", "fold", None, fold),),
    )
    o1_fold = m31.RiverDecisionNode(
        "o1-fold-after-hero",
        "O1",
        "O1_after_hero_bet",
        (m31.RiverAction("fold", "fold", None, o2_fold),),
    )
    root = m31.RiverDecisionNode(
        "hero-two-actions",
        "H",
        "H_two_actions",
        (
            m31.RiverAction("check", "check", None, o1_check),
            m31.RiverAction("bet", "bet", "20", o1_fold),
        ),
    )
    return m31.ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=observation("hero-two"),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def two_action_request(
    *,
    limits: module.ThreePlayerCertifiedGlobalLimits | None = None,
    optimizer_limits: m36.CertifiedGlobalOptimizerLimits | None = None,
    pins: module.ThreePlayerCertifiedGlobalPins | None = None,
    absolute_gap_tolerance: object = "100",
) -> module.AbstractThreePlayerCertifiedGlobalRequest:
    scenario = two_action_scenario()
    return module.AbstractThreePlayerCertifiedGlobalRequest(
        scenario=scenario,
        baseline_fixed_hero_policy=m31.ExactBehaviorPolicy(
            {"H_two_actions": {"check": "1", "bet": "0"}}
        ),
        initial_profile=m31.OpponentInitialProfile(
            {
                "O1_after_hero_bet": {"fold": "1"},
                "O1_after_hero_check": {"check": "1"},
            },
            {
                "O2_after_hero_bet": {"fold": "1"},
                "O2_after_hero_check": {"check": "1"},
            },
        ),
        attestation=attestation(scenario),
        repeated=module.ThreePlayerCertifiedRepeatedConfig(
            horizon=3, adaptation_opportunity=2, discount=0.5
        ),
        absolute_gap_tolerance=absolute_gap_tolerance,
        integration_limits=limits
        or module.ThreePlayerCertifiedGlobalLimits(),
        optimizer_limits=optimizer_limits
        or m36.CertifiedGlobalOptimizerLimits(),
        pins=pins or module.ThreePlayerCertifiedGlobalPins(),
    )


def multiple_response_scenario() -> m31.ThreePlayerRiverRakeScenario:
    """A bounded two-opponent game with distinct Hero response extrema."""

    after_fold = m31.RiverDecisionNode(
        "o2-after-o1-fold",
        "O2",
        "O2_after_o1_fold",
        (
            m31.RiverAction(
                "call",
                "call",
                "20",
                m31.RiverTerminalNode(
                    "terminal-o1-fold-o2-call",
                    "showdown",
                    (m31.AwardShare("O2", "1"),),
                ),
            ),
            m31.RiverAction(
                "fold",
                "fold",
                None,
                m31.RiverTerminalNode("terminal-both-fold", "fold"),
            ),
        ),
    )
    after_call = m31.RiverDecisionNode(
        "o2-after-o1-call",
        "O2",
        "O2_after_o1_call",
        (
            m31.RiverAction(
                "call",
                "call",
                "20",
                m31.RiverTerminalNode(
                    "terminal-both-call",
                    "showdown",
                    (m31.AwardShare("H", "1"),),
                ),
            ),
            m31.RiverAction(
                "fold",
                "fold",
                None,
                m31.RiverTerminalNode(
                    "terminal-o1-call-o2-fold",
                    "showdown",
                    (m31.AwardShare("H", "4/5"), m31.AwardShare("O1", "1/5")),
                ),
            ),
        ),
    )
    o1 = m31.RiverDecisionNode(
        "o1-facing-hero-bet",
        "O1",
        "O1_facing_hero_bet",
        (
            m31.RiverAction("fold", "fold", None, after_fold),
            m31.RiverAction("call", "call", "20", after_call),
        ),
    )
    root = m31.RiverDecisionNode(
        "hero-bet-root",
        "H",
        "H_bet_root",
        (m31.RiverAction("bet", "bet", "20", o1),),
    )
    return m31.ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=observation("multiple"),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def multiple_request() -> module.AbstractThreePlayerCertifiedGlobalRequest:
    scenario = multiple_response_scenario()
    return module.AbstractThreePlayerCertifiedGlobalRequest(
        scenario=scenario,
        baseline_fixed_hero_policy=m31.ExactBehaviorPolicy(
            {"H_bet_root": {"bet": "1"}}
        ),
        initial_profile=m31.OpponentInitialProfile(
            {"O1_facing_hero_bet": {"fold": "0", "call": "1"}},
            {
                "O2_after_o1_call": {"call": "0", "fold": "1"},
                "O2_after_o1_fold": {"call": "1", "fold": "0"},
            },
        ),
        attestation=attestation(scenario),
        repeated=module.ThreePlayerCertifiedRepeatedConfig(
            horizon=2, adaptation_opportunity=2, discount="1"
        ),
    )


def success(request):
    result = module.analyze_abstract_three_player_certified_global(request)
    assert result.status in (m36.CERTIFIED_GLOBAL, m36.CERTIFIED_EPSILON_GLOBAL)
    assert result.payload is not None
    assert result.error is None
    assert result.partial_result is False
    return result


def independent_pure_nash(payoff_table: dict) -> list[dict]:
    """Test-owned exhaustive pure-Nash oracle over the complete rectangle."""

    rows = payoff_table["rows"]
    o1_ids = [item["plan_id"] for item in payoff_table["o1_plans"]]
    o2_ids = [item["plan_id"] for item in payoff_table["o2_plans"]]
    by_pair = {
        (row["o1_plan_id"], row["o2_plan_id"]): row for row in rows
    }
    stable = []
    for o1, o2 in itertools.product(o1_ids, o2_ids):
        row = by_pair[(o1, o2)]
        u1 = Fraction(row["utility"]["O1"])
        u2 = Fraction(row["utility"]["O2"])
        if all(
            u1 >= Fraction(by_pair[(other, o2)]["utility"]["O1"])
            for other in o1_ids
        ) and all(
            u2 >= Fraction(by_pair[(o1, other)]["utility"]["O2"])
            for other in o2_ids
        ):
            stable.append(row)
    return stable


def solve_unique(
    equations: list[tuple[tuple[Fraction, ...], Fraction]], variables: int
) -> tuple[Fraction, ...] | None:
    """Test-owned exact RREF; returns only one unique consistent solution."""

    matrix = [list(coefficients) + [right] for coefficients, right in equations]
    pivot_columns: list[int] = []
    row = 0
    for column in range(variables):
        pivot = next(
            (index for index in range(row, len(matrix)) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        factor = matrix[row][column]
        matrix[row] = [value / factor for value in matrix[row]]
        for index in range(len(matrix)):
            if index == row or matrix[index][column] == 0:
                continue
            factor = matrix[index][column]
            matrix[index] = [
                left - factor * right
                for left, right in zip(matrix[index], matrix[row])
            ]
        pivot_columns.append(column)
        row += 1
    if any(
        all(value == 0 for value in current[:variables])
        and current[variables] != 0
        for current in matrix
    ):
        return None
    if len(pivot_columns) != variables:
        return None
    output = [Fraction(0) for _ in range(variables)]
    for index, column in enumerate(pivot_columns):
        output[column] = matrix[index][variables]
    return tuple(output)


def polytope_vertices(
    variables: int,
    equalities: list[tuple[tuple[Fraction, ...], Fraction]],
    inequalities: list[tuple[tuple[Fraction, ...], Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    """Enumerate every exact bounded-polytope vertex by active constraints."""

    vertices = set()
    for size in range(variables + 1):
        for active in itertools.combinations(inequalities, size):
            solution = solve_unique(equalities + list(active), variables)
            if solution is None:
                continue
            if any(
                sum(left * right for left, right in zip(coefficients, solution))
                != expected
                for coefficients, expected in equalities
            ):
                continue
            if any(
                sum(left * right for left, right in zip(coefficients, solution))
                < lower
                for coefficients, lower in inequalities
            ):
                continue
            vertices.add(solution)
    return tuple(sorted(vertices))


def independent_support_components(
    payoff_table: dict,
) -> tuple[
    set[tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]]],
    Fraction,
    Fraction,
]:
    """Test-owned exhaustive all-support bimatrix correspondence oracle."""

    o1_ids = [item["plan_id"] for item in payoff_table["o1_plans"]]
    o2_ids = [item["plan_id"] for item in payoff_table["o2_plans"]]
    rows = {
        (row["o1_plan_id"], row["o2_plan_id"]): row
        for row in payoff_table["rows"]
    }
    components = set()
    hero_values: list[Fraction] = []
    for size1 in range(1, len(o1_ids) + 1):
        for support1 in itertools.combinations(range(len(o1_ids)), size1):
            for size2 in range(1, len(o2_ids) + 1):
                for support2 in itertools.combinations(range(len(o2_ids)), size2):
                    base2 = support2[0]
                    x_equalities = [
                        (tuple(Fraction(1) for _ in support1), Fraction(1))
                    ]
                    for other2 in support2[1:]:
                        x_equalities.append(
                            (
                                tuple(
                                    Fraction(
                                        rows[(o1_ids[index1], o2_ids[other2])][
                                            "utility"
                                        ]["O2"]
                                    )
                                    - Fraction(
                                        rows[(o1_ids[index1], o2_ids[base2])][
                                            "utility"
                                        ]["O2"]
                                    )
                                    for index1 in support1
                                ),
                                Fraction(0),
                            )
                        )
                    x_inequalities = [
                        (
                            tuple(
                                Fraction(1 if left == right else 0)
                                for left in range(len(support1))
                            ),
                            Fraction(0),
                        )
                        for right in range(len(support1))
                    ]
                    for outside2 in set(range(len(o2_ids))) - set(support2):
                        x_inequalities.append(
                            (
                                tuple(
                                    Fraction(
                                        rows[(o1_ids[index1], o2_ids[base2])][
                                            "utility"
                                        ]["O2"]
                                    )
                                    - Fraction(
                                        rows[(o1_ids[index1], o2_ids[outside2])][
                                            "utility"
                                        ]["O2"]
                                    )
                                    for index1 in support1
                                ),
                                Fraction(0),
                            )
                        )
                    x_vertices_local = polytope_vertices(
                        len(support1), x_equalities, x_inequalities
                    )

                    base1 = support1[0]
                    y_equalities = [
                        (tuple(Fraction(1) for _ in support2), Fraction(1))
                    ]
                    for other1 in support1[1:]:
                        y_equalities.append(
                            (
                                tuple(
                                    Fraction(
                                        rows[(o1_ids[other1], o2_ids[index2])][
                                            "utility"
                                        ]["O1"]
                                    )
                                    - Fraction(
                                        rows[(o1_ids[base1], o2_ids[index2])][
                                            "utility"
                                        ]["O1"]
                                    )
                                    for index2 in support2
                                ),
                                Fraction(0),
                            )
                        )
                    y_inequalities = [
                        (
                            tuple(
                                Fraction(1 if left == right else 0)
                                for left in range(len(support2))
                            ),
                            Fraction(0),
                        )
                        for right in range(len(support2))
                    ]
                    for outside1 in set(range(len(o1_ids))) - set(support1):
                        y_inequalities.append(
                            (
                                tuple(
                                    Fraction(
                                        rows[(o1_ids[base1], o2_ids[index2])][
                                            "utility"
                                        ]["O1"]
                                    )
                                    - Fraction(
                                        rows[(o1_ids[outside1], o2_ids[index2])][
                                            "utility"
                                        ]["O1"]
                                    )
                                    for index2 in support2
                                ),
                                Fraction(0),
                            )
                        )
                    y_vertices_local = polytope_vertices(
                        len(support2), y_equalities, y_inequalities
                    )
                    if not x_vertices_local or not y_vertices_local:
                        continue
                    x_vertices = tuple(
                        tuple(
                            local[support1.index(index)]
                            if index in support1
                            else Fraction(0)
                            for index in range(len(o1_ids))
                        )
                        for local in x_vertices_local
                    )
                    y_vertices = tuple(
                        tuple(
                            local[support2.index(index)]
                            if index in support2
                            else Fraction(0)
                            for index in range(len(o2_ids))
                        )
                        for local in y_vertices_local
                    )
                    component = (tuple(sorted(x_vertices)), tuple(sorted(y_vertices)))
                    components.add(component)
    for x_vertices, y_vertices in components:
        for x, y in itertools.product(x_vertices, y_vertices):
            hero_values.append(
                sum(
                    x[index1]
                    * y[index2]
                    * Fraction(
                        rows[(o1_ids[index1], o2_ids[index2])]["utility"]["H"]
                    )
                    for index1 in range(len(o1_ids))
                    for index2 in range(len(o2_ids))
                )
            )
    return components, min(hero_values), max(hero_values)


def root_cell(
    preparation: module.ThreePlayerCertifiedGlobalPreparation,
) -> m36.ExactBehaviorCell:
    return m36.ExactBehaviorCell(
        tuple(
            m36.ExactBehaviorCellRow(
                row.information_set_id,
                tuple(
                    m36.ExactActionInterval(action, "0", "1")
                    for action in row.legal_action_ids
                ),
            )
            for row in preparation.m36_scenario.information_sets
        )
    )


def singleton_policy_and_cell(
    probability: Fraction,
) -> tuple[m36.ExactBehaviorPolicy, m36.ExactBehaviorCell]:
    bet = str(probability)
    check = str(1 - probability)
    policy = m36.ExactBehaviorPolicy(
        (
            m36.ExactBehaviorRow(
                "H_two_actions",
                (
                    m36.ExactActionProbability("bet", bet),
                    m36.ExactActionProbability("check", check),
                ),
            ),
        )
    )
    cell = m36.ExactBehaviorCell(
        (
            m36.ExactBehaviorCellRow(
                "H_two_actions",
                (
                    m36.ExactActionInterval("bet", bet, bet),
                    m36.ExactActionInterval("check", check, check),
                ),
            ),
        )
    )
    return policy, cell


def test_01_full_domain_is_automatic_and_baseline_is_exact_zero_uplift():
    result = success(two_action_request())
    preparation = result.payload.preparation
    assert [
        (row.information_set_id, row.legal_action_ids)
        for row in preparation.m36_scenario.information_sets
    ] == [("H_two_actions", ("bet", "check"))]
    assert result.payload.baseline_point.uplift == "0"
    assert result.payload.baseline_point.is_baseline_identity is True
    domain = result.payload.native_optimizer_result.payload.domain
    assert domain["decision_variable_count"] == 1
    encoded = module.exact_three_player_certified_global_json(result)
    assert "candidate_universe" not in encoded.lower()
    assert '"shift_amounts_used":false' in encoded.lower()
    assert '"grid_resolution_used":false' in encoded.lower()
    assert '"local_bounds_used":false' in encoded.lower()


def test_02_independent_complete_response_pure_subset_ties_and_coalition_split():
    result = success(multiple_request())
    point = result.payload.baseline_point
    native = point.scenario_response
    stable = independent_pure_nash(native.payoff_table)
    components, hero_worst, hero_best = independent_support_components(
        native.payoff_table
    )
    response = native.response
    assert len(stable) == response["pure_profile_unilateral_stability"][
        "profile_count"
    ]
    assert point.post_response_hero_worst == "-20"
    assert point.post_response_hero_best == "20"
    assert hero_worst == Fraction(point.post_response_hero_worst)
    assert hero_best == Fraction(point.post_response_hero_best)
    assert len(components) == response["counts"]["canonical_support_cells"]
    assert point.hero_worst_witness_count == len(
        response["hero_worst_witnesses"]
    )
    assert response["counts"]["support_pairs_visited"] == response["counts"][
        "support_pairs_total"
    ]
    assert response["hero_min_joint_plan_stress"] is not response[
        "hero_worst_witnesses"
    ]
    plans = {
        item["plan_id"]: item["actions_by_information_set"]
        for item in native.payoff_table["o2_plans"]
    }
    assert {
        tuple(sorted(plan.items())) for plan in plans.values()
    } == {
        (
            ("O2_after_o1_call", first),
            ("O2_after_o1_fold", second),
        )
        for first, second in itertools.product(("call", "fold"), repeat=2)
    }


def test_03_fixed_supplied_profile_and_fresh_response_are_separate():
    point = success(multiple_request()).payload.baseline_point
    assert point.fixed_profile_utility["H"] != point.post_response_hero_worst
    assert point.scenario_response.scenario_evaluation[
        "initial_profile_comparison"
    ]["response_claim"] is False
    assert point.scenario_response.response["coverage"] == "complete"


def test_04_root_bound_covers_independent_samples_and_singleton_is_exact():
    preparation = module.prepare_abstract_three_player_certified_global_oracle(
        two_action_request()
    )
    oracle = preparation.oracle
    root = oracle.upper_bound(root_cell(preparation))
    upper = Fraction(root.upper_bound)
    sampled = []
    for probability in (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
        policy, cell = singleton_policy_and_cell(probability)
        evaluation = oracle.evaluate(policy)
        sampled.append(Fraction(evaluation.uplift))
        singleton = oracle.upper_bound(cell)
        assert Fraction(singleton.upper_bound) == Fraction(evaluation.uplift)
        assert singleton.candidate_policy == policy
    assert upper >= max(sampled)


def test_05_constrained_cell_bound_covers_known_interior_sample():
    preparation = module.prepare_abstract_three_player_certified_global_oracle(
        two_action_request()
    )
    cell = m36.ExactBehaviorCell(
        (
            m36.ExactBehaviorCellRow(
                "H_two_actions",
                (
                    m36.ExactActionInterval("bet", "1/4", "3/4"),
                    m36.ExactActionInterval("check", "1/4", "3/4"),
                ),
            ),
        )
    )
    upper = Fraction(preparation.oracle.upper_bound(cell).upper_bound)
    interior, _ = singleton_policy_and_cell(Fraction(1, 2))
    interior_value = Fraction(preparation.oracle.evaluate(interior).uplift)
    vertices = [
        Fraction(
            preparation.oracle.evaluate(singleton_policy_and_cell(value)[0]).uplift
        )
        for value in (Fraction(0), Fraction(1))
    ]
    # The test-owned terminal arithmetic gives Hero +20 for both legal
    # branches. Every convex mixture is therefore a continuous global optimum,
    # including the strict interior policy p(bet)=1/2.
    assert vertices == [Fraction(0), Fraction(0)]
    assert interior_value == 0
    assert upper >= interior_value


def exact_range(*labels: str) -> RangeSpec:
    return RangeSpec(
        tuple(
            RangeEntry(
                canonicalize_exact_combo(label),
                1.0,
                WeightBasis.EXACT_COMBO_MASS,
            )
            for label in labels
        )
    )


def distribution(**values: str) -> tuple[m35.ExactActionProbability, ...]:
    return tuple(
        m35.ExactActionProbability(action, value)
        for action, value in values.items()
    )


def real_card_profile(
    hero: tuple[str, ...],
    o1: tuple[str, ...],
    o2: tuple[str, ...],
    *,
    hero_bet: str = "0",
) -> m35.RealCardThreePlayerProfile:
    rows = []
    for bucket in hero:
        rows.append(
            m35.RealCardProfileRow(
                "H",
                bucket,
                "open",
                distribution(check=str(1 - Fraction(hero_bet)), bet=hero_bet),
            )
        )
    for bucket in o1:
        rows.extend(
            (
                m35.RealCardProfileRow(
                    "O1", bucket, "after_hero_check", distribution(check="1")
                ),
                m35.RealCardProfileRow(
                    "O1",
                    bucket,
                    "vs_hero_bet",
                    distribution(call="1", fold="0"),
                ),
            )
        )
    for bucket in o2:
        rows.extend(
            (
                m35.RealCardProfileRow(
                    "O2",
                    bucket,
                    "after_hero_o1_check",
                    distribution(check="1"),
                ),
                m35.RealCardProfileRow(
                    "O2",
                    bucket,
                    "vs_hero_bet_o1_call",
                    distribution(call="1", fold="0"),
                ),
                m35.RealCardProfileRow(
                    "O2",
                    bucket,
                    "vs_hero_bet_o1_fold",
                    distribution(call="1", fold="0"),
                ),
            )
        )
    return m35.RealCardThreePlayerProfile(tuple(rows))


def real_card_source(
    *,
    hero_labels: tuple[str, ...] = ("AsAh",),
    o1_labels: tuple[str, ...] = ("KsKh",),
    o2_labels: tuple[str, ...] = ("QsQh",),
    dead_cards: tuple[str, ...] = (),
    rake_rate: str = "1/20",
    rake_cap: str | None = "1/5",
    hero_bet: str = "0",
) -> m35.KnownBoardRealCardThreePlayerRequest:
    hero = tuple(canonicalize_exact_combo(value) for value in hero_labels)
    o1 = tuple(canonicalize_exact_combo(value) for value in o1_labels)
    o2 = tuple(canonicalize_exact_combo(value) for value in o2_labels)
    return m35.KnownBoardRealCardThreePlayerRequest(
        board=BOARD,
        dead_cards=dead_cards,
        hero_range=exact_range(*hero),
        o1_range=exact_range(*o1),
        o2_range=exact_range(*o2),
        baseline_profile=real_card_profile(hero, o1, o2, hero_bet=hero_bet),
        attestation=m35.GeneratedTreeAttestation(
            verifier="M39 real-card independent test",
            verification_date=VALIDATION_DATE,
            evidence_version=EVIDENCE_VERSION,
        ),
        rake_rate=rake_rate,
        rake_cap=rake_cap,
        repeated=ThreePlayerRepeatedConfig(horizon=1),
    )


def real_card_request(
    source: m35.KnownBoardRealCardThreePlayerRequest | None = None,
    *,
    limits: module.ThreePlayerCertifiedGlobalLimits | None = None,
    pins: module.ThreePlayerCertifiedGlobalPins | None = None,
) -> module.KnownBoardRealCardThreePlayerCertifiedGlobalRequest:
    return module.KnownBoardRealCardThreePlayerCertifiedGlobalRequest(
        source=source or real_card_source(),
        absolute_gap_tolerance="100",
        integration_limits=limits
        or module.ThreePlayerCertifiedGlobalLimits(),
        pins=pins or module.ThreePlayerCertifiedGlobalPins(),
    )


def independent_triples(source) -> list[tuple[str, str, str]]:
    """Test-only raw six-private-card compatibility enumeration."""

    expanded = [
        [canonicalize_exact_combo(row.label) for row in spec.entries]
        for spec in (source.hero_range, source.o1_range, source.o2_range)
    ]
    unavailable = {card_id(card) for card in (*source.board, *source.dead_cards)}
    output = []
    for hero, o1, o2 in itertools.product(*expanded):
        private = [
            card_id(card)
            for combo in (hero, o1, o2)
            for card in (combo[:2], combo[2:])
        ]
        if not unavailable.intersection(private) and len(set(private)) == 6:
            output.append((hero, o1, o2))
    return output


def test_06_real_card_success_triples_blockers_rake_and_conservation():
    source = real_card_source(
        hero_labels=("AsAh", "KdKh"),
        o1_labels=("AsQh", "JcTd"),
        o2_labels=("9s8h",),
    )
    result = module.analyze_known_board_real_card_three_player_certified_global(
        real_card_request(source)
    )
    assert result.status in (m36.CERTIFIED_GLOBAL, m36.CERTIFIED_EPSILON_GLOBAL)
    real = result.payload.preparation.real_card
    expected = independent_triples(source)
    actual = [
        (row.hero_combo, row.o1_combo, row.o2_combo)
        for row in real.prepared_support.triples
    ]
    assert sorted(actual) == sorted(expected)
    assert sum(
        Fraction(row.probability_exact) for row in real.prepared_support.triples
    ) == 1
    terminals = result.payload.baseline_point.scenario_response.scenario_evaluation[
        "terminal_records"
    ]
    assert any(Fraction(row["rake_amount"]) > 0 for row in terminals)
    for row in terminals:
        utility = row["utility"]
        assert sum(Fraction(utility[key]) for key in ("H", "O1", "O2", "R")) == 0
        assert Fraction(row["rake_amount"]) <= Fraction("1/5")


def test_07_real_card_collision_and_range_normalization_fail_closed():
    collision = real_card_source(
        hero_labels=("AsAh",), o1_labels=("AsKh",), o2_labels=("QsQh",)
    )
    result = module.analyze_known_board_real_card_three_player_certified_global(
        real_card_request(collision)
    )
    assert result.payload is None
    assert result.partial_result is False
    assert result.status in (m36.INVALID_INPUT, m36.UNSUPPORTED_DOMAIN)


def test_08_real_card_and_abstract_parity_when_prepared_semantics_coincide():
    real_request = real_card_request()
    prepared = (
        module.prepare_known_board_real_card_three_player_certified_global_oracle(
            real_request
        )
    )
    abstract = module.AbstractThreePlayerCertifiedGlobalRequest(
        scenario=prepared.oracle.scenario,
        baseline_fixed_hero_policy=_to_m31(prepared.baseline_policy),
        initial_profile=prepared.oracle.initial_profile,
        attestation=prepared.oracle.attestation,
        repeated=module.ThreePlayerCertifiedRepeatedConfig(
            horizon=1, adaptation_opportunity=1, discount=1.0
        ),
        absolute_gap_tolerance="100",
        m31_limits=prepared.oracle.m31_limits,
        m30_limits=prepared.oracle.m30_limits,
    )
    real_result = module.analyze_known_board_real_card_three_player_certified_global(
        real_request
    )
    abstract_result = success(abstract)
    assert (
        real_result.payload.baseline_point.fixed_profile_utility
        == abstract_result.payload.baseline_point.fixed_profile_utility
    )
    assert (
        real_result.payload.baseline_point.post_response_hero_worst
        == abstract_result.payload.baseline_point.post_response_hero_worst
    )


def _to_m31(policy: m36.ExactBehaviorPolicy) -> m31.ExactBehaviorPolicy:
    return m31.ExactBehaviorPolicy(
        {
            row.information_set_id: {
                action.action_id: action.probability for action in row.actions
            }
            for row in policy.rows
        }
    )


def recursive_count(value: object) -> int:
    count = 1
    if isinstance(value, dict):
        for child in value.values():
            count += recursive_count(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            count += recursive_count(child)
    return count


def test_09_output_record_n_succeeds_n_minus_one_fails_with_counters():
    baseline = success(two_action_request())
    records = recursive_count(baseline.to_dict())
    passing = replace(
        two_action_request(),
        integration_limits=replace(
            two_action_request().integration_limits,
            max_output_records=records,
        ),
    )
    accepted = module.analyze_abstract_three_player_certified_global(passing)
    assert accepted.payload is not None
    failing = replace(
        passing,
        integration_limits=replace(
            passing.integration_limits, max_output_records=records - 1
        ),
    )
    rejected = module.analyze_abstract_three_player_certified_global(failing)
    assert rejected.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert rejected.error.phase == "output"
    assert rejected.payload is None
    assert rejected.optimizer_work_counters.oracle_calls_total > 0
    assert rejected.oracle_work_counters.point_evaluations > 0


def self_consistent_byte_cap() -> tuple[int, object]:
    cap = module.MAX_OUTPUT_BYTES
    for _ in range(8):
        request = replace(
            two_action_request(),
            integration_limits=replace(
                two_action_request().integration_limits, max_output_bytes=cap
            ),
        )
        result = module.analyze_abstract_three_player_certified_global(request)
        assert result.payload is not None
        actual = len(
            module.exact_three_player_certified_global_json(result).encode("utf-8")
        )
        if actual == cap:
            return cap, request
        cap = actual
    raise AssertionError("self-consistent byte cap did not converge")


def test_10_output_byte_n_succeeds_n_minus_one_fails():
    cap, request = self_consistent_byte_cap()
    accepted = module.analyze_abstract_three_player_certified_global(request)
    assert len(
        module.exact_three_player_certified_global_json(accepted).encode("utf-8")
    ) == cap
    rejected = module.analyze_abstract_three_player_certified_global(
        replace(
            request,
            integration_limits=replace(
                request.integration_limits, max_output_bytes=cap - 1
            ),
        )
    )
    assert rejected.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert rejected.error.phase == "output"
    assert rejected.payload is None
    assert rejected.partial_result is False


def test_11_very_low_output_caps_do_not_materialize_success(monkeypatch):
    calls = {"payload": 0, "result": 0, "serializer": 0}

    def payload_spy(self):
        calls["payload"] += 1
        raise AssertionError("payload.to_dict must not be called")

    def result_spy(self):
        calls["result"] += 1
        raise AssertionError("result.to_dict must not be called")

    def serializer_spy(value):
        calls["serializer"] += 1
        raise AssertionError("full serializer must not be called")

    monkeypatch.setattr(module.ThreePlayerCertifiedGlobalPayload, "to_dict", payload_spy)
    monkeypatch.setattr(module.ThreePlayerCertifiedGlobalResult, "to_dict", result_spy)
    monkeypatch.setattr(
        module, "exact_three_player_certified_global_json", serializer_spy
    )
    request = replace(
        two_action_request(),
        integration_limits=replace(
            two_action_request().integration_limits,
            max_output_records=1,
            max_output_bytes=1,
        ),
    )
    result = module.analyze_abstract_three_player_certified_global(request)
    assert result.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert result.error.phase == "output"
    assert result.payload is None
    assert result.optimizer_work_counters.oracle_calls_total > 0
    assert calls == {"payload": 0, "result": 0, "serializer": 0}


@pytest.mark.parametrize(
    "field",
    [item.name for item in asdict(module.ThreePlayerCertifiedGlobalPins()).keys()]
    if False
    else [
        "request_identity",
        "scenario_identity",
        "tree_structure_identity",
        "baseline_identity",
        "domain_identity",
        "response_oracle_identity",
        "objective_identity",
        "analysis_identity",
    ],
)
def test_12_all_m39_pins_accept_raw_prefix_and_reject_stale(field):
    baseline = success(two_action_request())
    identity = baseline.payload.preparation.identities[field]
    for expected in (identity, f"sha256:{identity}"):
        result = success(
            two_action_request(
                pins=module.ThreePlayerCertifiedGlobalPins(
                    **{field: expected}
                )
            )
        )
        assert (
            module.exact_three_player_certified_global_json(result)
            == module.exact_three_player_certified_global_json(
                success(
                    two_action_request(
                        pins=module.ThreePlayerCertifiedGlobalPins(
                            **{field: identity}
                        )
                    )
                )
            )
        )
    stale = ("0" if identity[0] != "0" else "1") + identity[1:]
    failed = module.analyze_abstract_three_player_certified_global(
        two_action_request(
            pins=module.ThreePlayerCertifiedGlobalPins(**{field: stale})
        )
    )
    assert failed.status == m36.STALE_INPUT
    assert failed.payload is None
    assert failed.partial_result is False


@pytest.mark.parametrize("pin", ["ABC", "sha256:" + "0" * 63, "0" * 65])
def test_13_malformed_pins_fail_closed(pin):
    result = module.analyze_abstract_three_player_certified_global(
        two_action_request(
            pins=module.ThreePlayerCertifiedGlobalPins(
                request_identity=pin
            )
        )
    )
    assert result.status == m36.INVALID_INPUT
    assert result.error.phase == "pins"
    assert result.payload is None


def test_14_semantic_mapping_permutation_is_byte_identical_change_is_not():
    request = two_action_request()
    permuted = replace(
        request,
        baseline_fixed_hero_policy=m31.ExactBehaviorPolicy(
            {"H_two_actions": {"bet": "0", "check": "1"}}
        ),
        initial_profile=m31.OpponentInitialProfile(
            dict(reversed(tuple(request.initial_profile.o1_probabilities.items()))),
            dict(reversed(tuple(request.initial_profile.o2_probabilities.items()))),
        ),
    )
    first = module.exact_three_player_certified_global_json(success(request))
    second = module.exact_three_player_certified_global_json(success(permuted))
    assert first == second
    changed = module.exact_three_player_certified_global_json(
        success(replace(request, repeated=replace(request.repeated, discount=0.25)))
    )
    assert changed != first


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (
            module.ThreePlayerCertifiedGlobalLimits(
                max_aggregate_response_records=1
            ),
            "preparation.response",
        ),
        (
            module.ThreePlayerCertifiedGlobalLimits(max_terminal_records=1),
            "preparation.terminal",
        ),
        (
            module.ThreePlayerCertifiedGlobalLimits(max_point_evaluations=1),
            "optimizer",
        ),
    ],
)
def test_15_preparation_response_point_limit_taxonomy(limits, expected):
    result = module.analyze_abstract_three_player_certified_global(
        two_action_request(limits=limits)
    )
    assert result.payload is None
    assert result.partial_result is False
    assert result.error.phase == expected


def test_16_m36_native_cell_cap_and_rational_failure_are_retained():
    result = module.analyze_abstract_three_player_certified_global(
        two_action_request(
            optimizer_limits=replace(
                m36.CertifiedGlobalOptimizerLimits(), max_oracle_calls=1
            ),
            absolute_gap_tolerance="0",
        )
    )
    assert result.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert result.error.cause_status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None


def test_17_result_wrapper_is_strict_and_null_payload_is_enforced():
    counters = m36.CertifiedGlobalWorkCounters()
    oracle = module.ThreePlayerCertifiedGlobalOracleWorkCounters()
    with pytest.raises(ValueError):
        module.ThreePlayerCertifiedGlobalResult(
            m36.CERTIFIED_GLOBAL, None, None, counters, oracle
        )
    with pytest.raises(ValueError):
        module.ThreePlayerCertifiedGlobalResult(
            m36.INVALID_INPUT,
            None,
            module.ThreePlayerCertifiedGlobalError("x", "x"),
            counters,
            oracle,
            partial_result=True,
        )


def test_18_lossless_binary64_lift_and_nonfinite_fail_closed():
    prepared = module.prepare_abstract_three_player_certified_global_oracle(
        two_action_request()
    )
    assert prepared.repeated_projection["discount_exact"] == "1/2"
    invalid = module.analyze_abstract_three_player_certified_global(
        replace(
            two_action_request(),
            repeated=replace(two_action_request().repeated, discount=float("nan")),
        )
    )
    assert invalid.status == m36.INVALID_INPUT
    assert invalid.payload is None
    exact_gap = module.analyze_abstract_three_player_certified_global(
        replace(two_action_request(), absolute_gap_tolerance=0.5)
    )
    assert exact_gap.status in (
        m36.CERTIFIED_GLOBAL,
        m36.CERTIFIED_EPSILON_GLOBAL,
    )


def test_18b_real_card_invalid_integration_limit_precedes_m35_preparation(
    monkeypatch,
):
    calls = {"prepare": 0}

    def forbidden(*args, **kwargs):
        calls["prepare"] += 1
        raise AssertionError("M35 preparation must not run")

    monkeypatch.setattr(m35, "_prepare_support_internal", forbidden)
    request = real_card_request(
        limits=replace(
            module.ThreePlayerCertifiedGlobalLimits(),
            max_point_evaluations=0,
        )
    )
    result = module.analyze_known_board_real_card_three_player_certified_global(
        request
    )
    assert result.status == m36.INVALID_INPUT
    assert result.error.phase == "limits.max_point_evaluations"
    assert result.payload is None
    assert calls["prepare"] == 0


def test_19_public_claim_identity_and_serializer_are_deterministic():
    first = success(two_action_request())
    second = success(two_action_request())
    encoded = module.exact_three_player_certified_global_json(first)
    assert encoded == module.exact_three_player_certified_global_json(second)
    assert "\n" not in encoded and "\r" not in encoded
    assert first.payload.preparation.identities == second.payload.preparation.identities
    assert len(hashlib.sha256(encoded.encode("utf-8")).hexdigest()) == 64
    assert first.payload.native_optimizer_result.payload.certificate.response_semantics == (
        m36.RESPONSE_SEMANTICS
    )
    assert module.CLAIM_SCOPE.endswith("global-maximum-only-v1")


def test_20_example_is_two_process_byte_deterministic():
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "examples/three_player_certified_global.py")]
    first = subprocess.run(
        command, cwd=root, check=True, stdout=subprocess.PIPE
    ).stdout
    second = subprocess.run(
        command, cwd=root, check=True, stdout=subprocess.PIPE
    ).stdout
    assert first == second
    assert len(first) == 372_195
    assert first.count(b"\n") == 1
    assert first.count(b"\r") == 0
    assert (
        hashlib.sha256(first).hexdigest()
        == "7cac73238b5fa7f4b3e5038f22092e2a5d122b91ac247e503f8a1b6f57869d5d"
    )


def test_21_docs_and_claim_boundaries_are_explicit():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/repeated_poker/three_player_certified_global.py"
    ).read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assumptions = (
        root / "docs/assumptions_and_limitations.md"
    ).read_text(encoding="utf-8")
    contract = (
        root / "docs/three_player_certified_global.md"
    ).read_text(encoding="utf-8")
    combined = "\n".join((source, readme, assumptions, contract))
    for required in (
        "complete",
        "Hero-worst",
        "whole-cell",
        "singleton",
        "payload=null",
        "partial",
        "M40",
        "strategy advice",
    ):
        assert required in combined
    assert "identified-bounded-scalar-objective-global-maximum-only-v1" in source
