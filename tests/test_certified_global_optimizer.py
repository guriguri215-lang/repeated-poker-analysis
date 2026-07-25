from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from repeated_poker import certified_global_optimizer as module


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "repeated_poker" / "certified_global_optimizer.py"
TEST = ROOT / "tests" / "test_certified_global_optimizer.py"
EXAMPLE = ROOT / "examples" / "certified_global_optimizer_core.py"


def identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rational(value: Fraction | int) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def scenario(*, reverse: bool = False) -> module.HeroBehaviorScenario:
    rows = (
        module.HeroInformationSet("H:river", ("fold", "call")),
        module.HeroInformationSet("H:turn", ("check", "small", "large")),
    )
    if reverse:
        rows = tuple(
            module.HeroInformationSet(
                row.information_set_id,
                tuple(reversed(row.legal_action_ids)),
            )
            for row in reversed(rows)
        )
    return module.HeroBehaviorScenario(
        scenario_identity=identity("scenario:two-simplexes"),
        information_sets=rows,
    )


def policy(
    river: tuple[str, str] = ("1/2", "1/2"),
    turn: tuple[str, str, str] = ("1/3", "1/3", "1/3"),
    *,
    reverse: bool = False,
) -> module.ExactBehaviorPolicy:
    rows = (
        module.ExactBehaviorRow(
            "H:river",
            (
                module.ExactActionProbability("fold", river[0]),
                module.ExactActionProbability("call", river[1]),
            ),
        ),
        module.ExactBehaviorRow(
            "H:turn",
            (
                module.ExactActionProbability("check", turn[0]),
                module.ExactActionProbability("small", turn[1]),
                module.ExactActionProbability("large", turn[2]),
            ),
        ),
    )
    if reverse:
        rows = tuple(
            module.ExactBehaviorRow(row.information_set_id, tuple(reversed(row.actions)))
            for row in reversed(rows)
        )
    return module.ExactBehaviorPolicy(rows=rows)


class LinearOracle:
    """Test-owned exact additive oracle with an independently simple bound."""

    response_semantics = module.RESPONSE_SEMANTICS
    bound_contract_version = module.BOUND_CONTRACT_VERSION

    def __init__(
        self,
        scenario_value: module.HeroBehaviorScenario,
        baseline: module.ExactBehaviorPolicy,
        coefficients: dict[tuple[str, str], Fraction | int],
        *,
        suggest: bool = True,
        correspondence_records: int = 5,
        worst_witnesses: int = 2,
    ) -> None:
        self.scenario = scenario_value
        self.baseline = baseline
        self.coefficients = {
            key: Fraction(value) for key, value in coefficients.items()
        }
        coefficient_record = sorted(
            (information_set, action, rational(value))
            for (information_set, action), value in self.coefficients.items()
        )
        self.response_oracle_identity = identity(
            "linear-response:" + json.dumps(coefficient_record)
        )
        self.objective_identity = identity(
            "baseline-uplift:" + module.behavior_policy_identity(
                scenario_value, baseline
            )
        )
        self.suggest = suggest
        self.correspondence_records = correspondence_records
        self.worst_witnesses = worst_witnesses
        self.bound_cells: list[module.ExactBehaviorCell] = []
        self.point_policies: list[module.ExactBehaviorPolicy] = []
        self._baseline_value = self._raw_value(baseline)

    def _raw_value(self, policy_value: module.ExactBehaviorPolicy) -> Fraction:
        total = Fraction(0)
        for row in policy_value.rows:
            for action in row.actions:
                total += self.coefficients[
                    (row.information_set_id, action.action_id)
                ] * Fraction(action.probability)
        return total

    def evaluate(
        self, policy_value: module.ExactBehaviorPolicy
    ) -> module.ScalarOracleEvaluation:
        self.point_policies.append(policy_value)
        policy_identity = module.behavior_policy_identity(
            self.scenario, policy_value
        )
        candidate = self._raw_value(policy_value)
        uplift = candidate - self._baseline_value
        return module.ScalarOracleEvaluation(
            policy_identity=policy_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            baseline_total_repeated_hero_ev=rational(self._baseline_value),
            candidate_total_repeated_hero_ev=rational(candidate),
            uplift=rational(uplift),
            complete_response_correspondence_identity=identity(
                "complete-response:" + policy_identity
            ),
            complete_response_record_count=self.correspondence_records,
            hero_worst_witness_count=self.worst_witnesses,
        )

    def _maximizing_policy(
        self, cell: module.ExactBehaviorCell
    ) -> module.ExactBehaviorPolicy:
        rows: list[module.ExactBehaviorRow] = []
        for row in cell.rows:
            values = {
                action.action_id: Fraction(action.lower)
                for action in row.actions
            }
            residual = Fraction(1) - sum(values.values(), Fraction(0))
            by_value = sorted(
                row.actions,
                key=lambda action: (
                    -self.coefficients[
                        (row.information_set_id, action.action_id)
                    ],
                    action.action_id,
                ),
            )
            for action in by_value:
                capacity = Fraction(action.upper) - values[action.action_id]
                addition = min(residual, capacity)
                values[action.action_id] += addition
                residual -= addition
            assert residual == 0
            rows.append(
                module.ExactBehaviorRow(
                    row.information_set_id,
                    tuple(
                        module.ExactActionProbability(
                            action.action_id, rational(values[action.action_id])
                        )
                        for action in row.actions
                    ),
                )
            )
        return module.ExactBehaviorPolicy(rows=tuple(rows))

    def upper_bound(
        self, cell: module.ExactBehaviorCell
    ) -> module.ScalarOracleBound:
        self.bound_cells.append(cell)
        cell_identity = module.behavior_cell_identity(self.scenario, cell)
        candidate = self._maximizing_policy(cell)
        upper = self._raw_value(candidate) - self._baseline_value
        candidate_identity = (
            module.behavior_policy_identity(self.scenario, candidate)
            if self.suggest
            else None
        )
        return module.ScalarOracleBound(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=rational(upper),
            bound_identity=module.scalar_oracle_bound_identity(
                cell_identity=cell_identity,
                response_oracle_identity=self.response_oracle_identity,
                objective_identity=self.objective_identity,
                upper_bound=rational(upper),
                candidate_policy_identity=candidate_identity,
            ),
            candidate_policy=candidate if self.suggest else None,
        )


class QuadraticOracle:
    """One-simplex exact concave fixture that forces actual cell splitting."""

    response_semantics = module.RESPONSE_SEMANTICS
    bound_contract_version = module.BOUND_CONTRACT_VERSION

    def __init__(self) -> None:
        self.scenario = module.HeroBehaviorScenario(
            identity("quadratic-scenario"),
            (module.HeroInformationSet("H:q", ("a", "b")),),
        )
        self.baseline = module.ExactBehaviorPolicy(
            (
                module.ExactBehaviorRow(
                    "H:q",
                    (
                        module.ExactActionProbability("a", "0"),
                        module.ExactActionProbability("b", "1"),
                    ),
                ),
            )
        )
        self.response_oracle_identity = identity("quadratic-response")
        self.objective_identity = identity("quadratic-baseline-uplift")
        self.bound_cells: list[module.ExactBehaviorCell] = []

    @staticmethod
    def _value(p: Fraction) -> Fraction:
        return Fraction(1, 9) - (p - Fraction(1, 3)) ** 2

    def evaluate(
        self, policy_value: module.ExactBehaviorPolicy
    ) -> module.ScalarOracleEvaluation:
        p = Fraction(policy_value.rows[0].actions[0].probability)
        uplift = self._value(p) - self._value(Fraction(0))
        policy_identity = module.behavior_policy_identity(
            self.scenario, policy_value
        )
        return module.ScalarOracleEvaluation(
            policy_identity,
            self.response_oracle_identity,
            self.objective_identity,
            "0",
            rational(uplift),
            rational(uplift),
            identity("quadratic-complete:" + policy_identity),
            3,
            2,
        )

    def upper_bound(
        self, cell: module.ExactBehaviorCell
    ) -> module.ScalarOracleBound:
        self.bound_cells.append(cell)
        interval = cell.rows[0].actions[0]
        lower = Fraction(interval.lower)
        upper = Fraction(interval.upper)
        target = min(max(Fraction(1, 3), lower), upper)
        bound = self._value(target) - self._value(Fraction(0))
        cell_identity = module.behavior_cell_identity(self.scenario, cell)
        return module.ScalarOracleBound(
            cell_identity,
            self.response_oracle_identity,
            self.objective_identity,
            rational(bound),
            module.scalar_oracle_bound_identity(
                cell_identity=cell_identity,
                response_oracle_identity=self.response_oracle_identity,
                objective_identity=self.objective_identity,
                upper_bound=rational(bound),
                candidate_policy_identity=None,
            ),
        )


def binary_scenario(label: str) -> module.HeroBehaviorScenario:
    return module.HeroBehaviorScenario(
        identity(f"binary-scenario:{label}"),
        (module.HeroInformationSet("H:binary", ("a", "b")),),
    )


def binary_policy(a: str, b: str) -> module.ExactBehaviorPolicy:
    return module.ExactBehaviorPolicy(
        (
            module.ExactBehaviorRow(
                "H:binary",
                (
                    module.ExactActionProbability("a", a),
                    module.ExactActionProbability("b", b),
                ),
            ),
        )
    )


class RationalCapOracle:
    """Small exact spy for rational-cap boundary and derived-value tests."""

    response_semantics = module.RESPONSE_SEMANTICS
    bound_contract_version = module.BOUND_CONTRACT_VERSION

    def __init__(
        self,
        scenario_value: module.HeroBehaviorScenario,
        baseline: module.ExactBehaviorPolicy,
        *,
        upper_bound: Fraction | int = 0,
        candidate_policy: module.ExactBehaviorPolicy | None = None,
        baseline_total: Fraction | int = 0,
        values_by_a: dict[Fraction, Fraction] | None = None,
        reported_uplifts_by_a: dict[Fraction, Fraction] | None = None,
    ) -> None:
        self.scenario = scenario_value
        self.baseline = baseline
        self.upper = Fraction(upper_bound)
        self.candidate_policy = candidate_policy
        self.baseline_total = Fraction(baseline_total)
        self.values_by_a = values_by_a or {}
        self.reported_uplifts_by_a = reported_uplifts_by_a or {}
        self.response_oracle_identity = identity("rational-cap-response")
        self.objective_identity = identity("rational-cap-objective")
        self.point_policies: list[module.ExactBehaviorPolicy] = []
        self.bound_cells: list[module.ExactBehaviorCell] = []

    def evaluate(
        self, policy_value: module.ExactBehaviorPolicy
    ) -> module.ScalarOracleEvaluation:
        self.point_policies.append(policy_value)
        a_probability = Fraction(policy_value.rows[0].actions[0].probability)
        candidate_total = self.values_by_a.get(
            a_probability, self.baseline_total
        )
        uplift = self.reported_uplifts_by_a.get(
            a_probability, candidate_total - self.baseline_total
        )
        policy_identity = module.behavior_policy_identity(
            self.scenario, policy_value
        )
        return module.ScalarOracleEvaluation(
            policy_identity,
            self.response_oracle_identity,
            self.objective_identity,
            rational(self.baseline_total),
            rational(candidate_total),
            rational(uplift),
            identity("rational-cap-complete:" + policy_identity),
            1,
            1,
        )

    def upper_bound(
        self, cell: module.ExactBehaviorCell
    ) -> module.ScalarOracleBound:
        self.bound_cells.append(cell)
        cell_identity = module.behavior_cell_identity(self.scenario, cell)
        candidate_identity = (
            None
            if self.candidate_policy is None
            else module.behavior_policy_identity(
                self.scenario, self.candidate_policy
            )
        )
        upper_text = rational(self.upper)
        return module.ScalarOracleBound(
            cell_identity,
            self.response_oracle_identity,
            self.objective_identity,
            upper_text,
            module.scalar_oracle_bound_identity(
                cell_identity=cell_identity,
                response_oracle_identity=self.response_oracle_identity,
                objective_identity=self.objective_identity,
                upper_bound=upper_text,
                candidate_policy_identity=candidate_identity,
            ),
            candidate_policy=self.candidate_policy,
        )


COEFFICIENTS = {
    ("H:river", "fold"): Fraction(-2),
    ("H:river", "call"): Fraction(3),
    ("H:turn", "check"): Fraction(1),
    ("H:turn", "small"): Fraction(5),
    ("H:turn", "large"): Fraction(2),
}


def successful(
    *,
    reverse: bool = False,
    suggest: bool = True,
) -> tuple[module.CertifiedGlobalOptimizerResult, LinearOracle]:
    scenario_value = scenario(reverse=reverse)
    baseline = policy(reverse=reverse)
    oracle = LinearOracle(
        scenario_value, baseline, COEFFICIENTS, suggest=suggest
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.CERTIFIED_GLOBAL
    assert result.payload is not None
    assert result.error is None
    return result, oracle


def test_full_domain_is_derived_from_all_legal_scenario_simplexes():
    result, _ = successful()
    payload = result.payload
    assert payload is not None
    domain = payload.domain
    assert domain["information_set_count"] == 2
    assert domain["legal_action_count"] == 5
    assert domain["decision_variable_count"] == 3
    assert domain["domain_source"] == "scenario-only"
    for forbidden in (
        "candidate_list_used",
        "shift_amounts_used",
        "grid_resolution_used",
        "local_bounds_used",
    ):
        assert domain[forbidden] is False
    rows = domain["information_sets"]
    assert [row["information_set_id"] for row in rows] == [
        "H:river",
        "H:turn",
    ]
    assert all(row["continuous"] is True for row in rows)
    assert all("sum=1-exact" in row["constraint"] for row in rows)


def test_independent_exhaustive_vertex_oracle_matches_global_optimum():
    result, _ = successful()
    assert result.payload is not None

    # Reviewer-owned exhaustive vertex enumeration.  It does not call the
    # production optimizer, bound helper, or candidate suggestion routine.
    independent_values = []
    for river_action, turn_action in itertools.product(
        ("fold", "call"), ("check", "small", "large")
    ):
        independent_values.append(
            (
                COEFFICIENTS[("H:river", river_action)]
                + COEFFICIENTS[("H:turn", turn_action)],
                river_action,
                turn_action,
            )
        )
    independent_optimum, river_action, turn_action = max(independent_values)
    baseline_value = (
        sum(COEFFICIENTS[("H:river", action)] for action in ("fold", "call"))
        / 2
        + sum(
            COEFFICIENTS[("H:turn", action)]
            for action in ("check", "small", "large")
        )
        / 3
    )
    expected_uplift = independent_optimum - baseline_value

    certificate = result.payload.certificate
    assert Fraction(certificate.incumbent_lower_bound) == expected_uplift
    assert Fraction(certificate.valid_global_upper_bound) == expected_uplift
    assert certificate.absolute_gap == "0"
    assert river_action == "call"
    assert turn_action == "small"
    selected = result.payload.selected_commitment
    assert selected is not None
    selected_map = {
        row.information_set_id: {
            action.action_id: action.probability for action in row.actions
        }
        for row in selected.rows
    }
    assert selected_map == {
        "H:river": {"call": "1", "fold": "0"},
        "H:turn": {"check": "0", "large": "0", "small": "1"},
    }


def test_complete_response_ties_are_not_collapsed_to_one_witness():
    result, oracle = successful()
    assert result.payload is not None
    selected = result.payload.selected_evaluation
    assert selected is not None
    assert selected.response_semantics == module.RESPONSE_SEMANTICS
    assert selected.complete_response_record_count == 5
    assert selected.hero_worst_witness_count == 2
    assert selected.complete_response_correspondence_identity
    assert (
        result.payload.certificate.response_oracle_identity
        == oracle.response_oracle_identity
    )


def test_baseline_is_only_comparison_point_and_does_not_restrict_domain():
    baseline = policy(
        river=("1", "0"),
        turn=("1", "0", "0"),
    )
    scenario_value = scenario()
    oracle = LinearOracle(scenario_value, baseline, COEFFICIENTS)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.CERTIFIED_GLOBAL
    assert result.payload is not None
    assert result.payload.selected_commitment != baseline
    assert Fraction(
        result.payload.certificate.incumbent_lower_bound
    ) == Fraction(9)


def test_quadratic_analytic_bound_splits_exact_cells_and_certifies_global():
    oracle = QuadraticOracle()
    result = module.optimize_certified_global_hero_commitment(
        oracle.scenario, oracle.baseline, oracle
    )
    assert result.status == module.CERTIFIED_GLOBAL
    assert result.payload is not None
    certificate = result.payload.certificate
    assert certificate.incumbent_lower_bound == "1/9"
    assert certificate.valid_global_upper_bound == "1/9"
    assert certificate.absolute_gap == "0"
    assert certificate.work_counters.splits == 1
    assert certificate.work_counters.cells_created == 3
    assert len(oracle.bound_cells) == 3

    root, left, right = oracle.bound_cells
    assert root.rows[0].actions[0].lower == "0"
    assert root.rows[0].actions[0].upper == "1"
    assert left.rows[0].actions[0].upper == "1/2"
    assert right.rows[0].actions[0].lower == "1/2"


def test_epsilon_certificate_reports_valid_gap_and_requested_tolerance():
    oracle = QuadraticOracle()
    result = module.optimize_certified_global_hero_commitment(
        oracle.scenario,
        oracle.baseline,
        oracle,
        absolute_gap_tolerance="1/32",
    )
    assert result.status == module.CERTIFIED_EPSILON_GLOBAL
    assert result.payload is not None
    certificate = result.payload.certificate
    assert Fraction(certificate.incumbent_lower_bound) == Fraction(1, 12)
    assert Fraction(certificate.valid_global_upper_bound) == Fraction(1, 9)
    assert Fraction(certificate.absolute_gap) == Fraction(1, 36)
    assert certificate.requested_absolute_tolerance == "1/32"
    assert Fraction(certificate.absolute_gap) <= Fraction(
        certificate.requested_absolute_tolerance
    )
    assert certificate.work_counters.splits == 0


def test_no_benefit_has_certificate_but_no_selected_commitment():
    zero_coefficients = {key: Fraction(0) for key in COEFFICIENTS}
    scenario_value = scenario()
    baseline = policy()
    oracle = LinearOracle(
        scenario_value, baseline, zero_coefficients, correspondence_records=4
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.CERTIFIED_GLOBAL
    assert result.payload is not None
    assert result.payload.no_beneficial_commitment is True
    assert result.payload.selected_commitment is None
    assert result.payload.selected_evaluation is None
    assert result.payload.selected_policy_identity is None
    assert result.payload.certificate.incumbent_lower_bound == "0"
    assert result.payload.certificate.valid_global_upper_bound == "0"


def test_limit_reached_is_fail_closed_before_child_allocation():
    oracle = QuadraticOracle()
    result = module.optimize_certified_global_hero_commitment(
        oracle.scenario,
        oracle.baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(max_cells=1),
    )
    assert result.status == module.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "search.cells"
    assert result.work_counters.cells_created == 1
    assert len(oracle.bound_cells) == 1


@pytest.mark.parametrize(
    ("limits", "phase"),
    (
        (
            module.CertifiedGlobalOptimizerLimits(max_oracle_calls=2),
            "oracle.root_preflight",
        ),
        (
            module.CertifiedGlobalOptimizerLimits(max_bound_records=1, max_cells=1),
            "search.cells",
        ),
        (
            module.CertifiedGlobalOptimizerLimits(max_output_bytes=128),
            "output.bytes",
        ),
    ),
)
def test_caps_fail_without_success_payload(limits, phase):
    if limits.max_output_bytes == 128:
        scenario_value = scenario()
        baseline = policy()
        oracle = LinearOracle(scenario_value, baseline, COEFFICIENTS)
    else:
        oracle = QuadraticOracle()
        scenario_value = oracle.scenario
        baseline = oracle.baseline
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle, limits=limits
    )
    assert result.status == module.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == phase


def test_generated_center_cap_fails_before_oracle_receives_half():
    scenario_value = binary_scenario("center-cap")
    baseline = binary_policy("0", "1")
    oracle = RationalCapOracle(scenario_value, baseline)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(
            max_rational_denominator_bits=1
        ),
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase.startswith("cell.center.")
    assert [
        tuple(action.probability for action in row.actions)
        for policy_value in oracle.point_policies
        for row in policy_value.rows
    ] == [("0", "1")]


@pytest.mark.parametrize(
    ("probabilities", "limit_name", "cap", "expected_status"),
    (
        (("4/5", "1/5"), "max_rational_numerator_bits", 3, module.CERTIFIED_GLOBAL),
        (("4/5", "1/5"), "max_rational_numerator_bits", 2, module.NUMERIC_FAILURE),
        (("1/4", "3/4"), "max_rational_denominator_bits", 3, module.CERTIFIED_GLOBAL),
        (("1/4", "3/4"), "max_rational_denominator_bits", 2, module.NUMERIC_FAILURE),
    ),
)
def test_rational_numerator_and_denominator_exact_boundary_and_cap_plus_one(
    probabilities, limit_name, cap, expected_status
):
    scenario_value = binary_scenario(
        f"boundary:{limit_name}:{cap}:{probabilities[0]}"
    )
    baseline = binary_policy(*probabilities)
    oracle = RationalCapOracle(
        scenario_value, baseline, candidate_policy=baseline
    )
    limits = replace(
        module.CertifiedGlobalOptimizerLimits(), **{limit_name: cap}
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle, limits=limits
    )
    assert result.status == expected_status
    assert result.partial_result is False
    if expected_status == module.CERTIFIED_GLOBAL:
        assert result.payload is not None
    else:
        assert result.payload is None
        assert result.error is not None
        assert result.error.phase.startswith("baseline_policy.")


def test_split_midpoint_cap_fails_before_either_child_is_materialized():
    scenario_value = binary_scenario("split-cap")
    baseline = binary_policy("0", "1")
    oracle = RationalCapOracle(
        scenario_value,
        baseline,
        upper_bound=1,
        candidate_policy=baseline,
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(
            max_rational_denominator_bits=1
        ),
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "cell.split.midpoint"
    assert result.work_counters.cells_created == 1
    assert result.work_counters.splits == 0
    assert len(oracle.bound_cells) == 1


def test_candidate_baseline_derived_arithmetic_obeys_rational_cap():
    scenario_value = binary_scenario("derived-uplift-cap")
    baseline = binary_policy("0", "1")
    candidate = binary_policy("1", "0")
    oracle = RationalCapOracle(
        scenario_value,
        baseline,
        upper_bound=1,
        candidate_policy=candidate,
        baseline_total=Fraction(1, 2),
        values_by_a={Fraction(1): Fraction(2, 3)},
        reported_uplifts_by_a={Fraction(1): Fraction(0)},
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(
            max_rational_denominator_bits=2
        ),
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "oracle.point.root.derived_uplift"


def test_absolute_gap_derived_arithmetic_obeys_rational_cap():
    scenario_value = binary_scenario("absolute-gap-cap")
    baseline = binary_policy("0", "1")
    candidate = binary_policy("1", "0")
    oracle = RationalCapOracle(
        scenario_value,
        baseline,
        upper_bound=Fraction(2, 3),
        candidate_policy=candidate,
        values_by_a={Fraction(1): Fraction(1, 2)},
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(
            max_rational_denominator_bits=2
        ),
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "certificate.absolute_gap"


def test_relative_gap_derived_arithmetic_obeys_rational_cap():
    scenario_value = binary_scenario("relative-gap-cap")
    baseline = binary_policy("0", "1")
    candidate = binary_policy("1", "0")
    oracle = RationalCapOracle(
        scenario_value,
        baseline,
        upper_bound=3,
        candidate_policy=candidate,
        values_by_a={Fraction(1): Fraction(5, 2)},
    )
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        absolute_gap_tolerance="1",
        limits=module.CertifiedGlobalOptimizerLimits(
            max_rational_numerator_bits=3,
            max_rational_denominator_bits=2,
        ),
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "certificate.relative_gap"


def test_oversized_rational_token_fails_before_integer_conversion():
    scenario_value = binary_scenario("oversized-token")
    baseline = binary_policy("0", "1")
    oracle = RationalCapOracle(scenario_value, baseline)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        absolute_gap_tolerance="9" * 5000,
    )
    assert result.status == module.NUMERIC_FAILURE
    assert result.payload is None
    assert result.partial_result is False
    assert result.error is not None
    assert result.error.phase == "tolerance.absolute"
    assert oracle.point_policies == []
    assert oracle.bound_cells == []


def test_hard_ceiling_is_rejected_not_clamped():
    scenario_value = scenario()
    baseline = policy()
    oracle = LinearOracle(scenario_value, baseline, COEFFICIENTS)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        oracle,
        limits=module.CertifiedGlobalOptimizerLimits(
            max_cells=module.HARD_MAX_CELLS + 1
        ),
    )
    assert result.status == module.INVALID_INPUT
    assert result.payload is None
    assert result.error is not None
    assert result.error.phase == "limits.max_cells"


def test_unsupported_bound_is_explicit_and_has_no_fallback():
    scenario_value = scenario()
    baseline = policy()

    class Unsupported(LinearOracle):
        def upper_bound(self, cell):
            raise module.UnsupportedScalarOracleDomain(
                "fixture has no sound whole-cell upper bound"
            )

    oracle = Unsupported(scenario_value, baseline, COEFFICIENTS)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.UNSUPPORTED_DOMAIN
    assert result.payload is None
    assert result.error is not None
    assert "sound whole-cell" in result.error.message
    assert result.work_counters.oracle_point_calls == 1
    assert result.work_counters.oracle_bound_calls == 1


def test_invalid_upper_bound_below_its_candidate_is_rejected():
    scenario_value = scenario()
    baseline = policy()

    class InvalidBound(LinearOracle):
        def upper_bound(self, cell):
            valid = super().upper_bound(cell)
            invalid_upper = rational(Fraction(valid.upper_bound) - 1)
            candidate_identity = module.behavior_policy_identity(
                self.scenario, valid.candidate_policy
            )
            return replace(
                valid,
                upper_bound=invalid_upper,
                bound_identity=module.scalar_oracle_bound_identity(
                    cell_identity=valid.cell_identity,
                    response_oracle_identity=self.response_oracle_identity,
                    objective_identity=self.objective_identity,
                    upper_bound=invalid_upper,
                    candidate_policy_identity=candidate_identity,
                ),
            )

    oracle = InvalidBound(scenario_value, baseline, COEFFICIENTS)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.INVALID_ORACLE_CONTRACT
    assert result.payload is None
    assert result.error is not None
    assert "below an evaluated point" in result.error.message


def test_wrong_response_semantics_and_nonzero_baseline_uplift_fail_contract():
    scenario_value = scenario()
    baseline = policy()
    wrong_semantics = LinearOracle(scenario_value, baseline, COEFFICIENTS)
    wrong_semantics.response_semantics = "first-witness"
    first = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, wrong_semantics
    )
    assert first.status == module.INVALID_ORACLE_CONTRACT
    assert first.payload is None

    class WrongBaseline(LinearOracle):
        def evaluate(self, policy_value):
            value = super().evaluate(policy_value)
            if module.behavior_policy_identity(
                self.scenario, policy_value
            ) == module.behavior_policy_identity(self.scenario, self.baseline):
                return replace(
                    value,
                    candidate_total_repeated_hero_ev=rational(
                        Fraction(value.candidate_total_repeated_hero_ev) + 1
                    ),
                    uplift="1",
                )
            return value

    second = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        WrongBaseline(scenario_value, baseline, COEFFICIENTS),
    )
    assert second.status == module.INVALID_ORACLE_CONTRACT
    assert second.payload is None
    assert second.error is not None
    assert "baseline policy uplift" in second.error.message


@pytest.mark.parametrize(
    "bad_probability",
    ("0.5", "2/4", "-1/2", "3/2"),
)
def test_probability_validation_is_exact_canonical_and_fail_closed(
    bad_probability,
):
    scenario_value = scenario()
    baseline = policy(river=(bad_probability, "1/2"))
    oracle = LinearOracle(scenario_value, policy(), COEFFICIENTS)
    result = module.optimize_certified_global_hero_commitment(
        scenario_value, baseline, oracle
    )
    assert result.status == module.INVALID_INPUT
    assert result.payload is None


def test_scenario_and_policy_permutations_are_byte_identical():
    first, _ = successful()
    second, _ = successful(reverse=True)
    first_bytes = module.exact_certified_global_optimizer_json(first).encode()
    second_bytes = module.exact_certified_global_optimizer_json(second).encode()
    assert first_bytes == second_bytes
    decoded = json.loads(first_bytes)
    assert decoded["payload"]["certificate"]["domain_identity"]
    assert b"NaN" not in first_bytes
    assert b"Infinity" not in first_bytes


def test_identity_pins_reject_stale_input():
    result, oracle = successful()
    assert result.payload is not None
    certificate = result.payload.certificate
    scenario_value = scenario()
    baseline = policy()
    stale = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        LinearOracle(scenario_value, baseline, COEFFICIENTS),
        pins=module.CertifiedGlobalOptimizerPins(
            domain_identity=certificate.domain_identity,
            baseline_policy_identity=certificate.baseline_policy_identity,
            response_oracle_identity=oracle.response_oracle_identity,
            objective_identity=oracle.objective_identity,
            run_identity=identity("stale-run"),
        ),
    )
    assert stale.status == module.STALE_INPUT
    assert stale.payload is None
    assert stale.error is not None
    assert stale.error.phase == "pins.run_identity"


def test_oracle_exception_and_nested_limit_have_stable_failures():
    scenario_value = scenario()
    baseline = policy()

    class Broken(LinearOracle):
        def evaluate(self, policy_value):
            raise RuntimeError("secret implementation detail")

    broken = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        Broken(scenario_value, baseline, COEFFICIENTS),
    )
    assert broken.status == module.ORACLE_FAILURE
    assert broken.payload is None
    assert "secret" not in broken.error.message

    class Limited(LinearOracle):
        def upper_bound(self, cell):
            raise module.ScalarOracleResourceLimit("nested exact response cap")

    limited = module.optimize_certified_global_hero_commitment(
        scenario_value,
        baseline,
        Limited(scenario_value, baseline, COEFFICIENTS),
    )
    assert limited.status == module.LIMIT_REACHED_NO_CERTIFICATE
    assert limited.payload is None
    assert limited.error.cause_status is None


def test_source_has_no_heuristic_global_certificate_or_consumer_integration():
    text = SOURCE.read_text(encoding="utf-8")
    assert "import random" not in text
    assert "random." not in text
    assert "import numpy" not in text
    assert "import scipy" not in text
    assert "grid_resolution_used" in text
    assert "shift_amounts_used" in text
    assert "three_player_candidate_repeated" not in text
    assert "known_board_real_card_hu_river" not in text
    assert "known_board_real_card_three_player_river" not in text
    assert "aiof_preflop_candidate_repeated" not in text
    assert "CERTIFIED_GLOBAL" in text
    assert "LIMIT_REACHED_NO_CERTIFICATE" in text


def test_two_process_public_example_is_byte_identical():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    first = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    second = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    decoded = json.loads(first.stdout)
    assert decoded["status"] == module.CERTIFIED_GLOBAL
    assert decoded["payload"]["certificate"]["absolute_gap"] == "0"
    assert first.stdout.endswith(b"\n")


def test_exact_changed_file_boundary_and_no_tracked_cache():
    assert SOURCE.exists()
    assert TEST.exists()
    assert EXAMPLE.exists()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    assert not [
        path
        for path in tracked
        if b"__pycache__" in path
        or path.endswith((b".pyc", b".pyo"))
        or b".pytest_cache" in path
    ]
