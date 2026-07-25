"""Small deterministic M36 certified-global optimizer core example.

The scalar oracle below is intentionally analytic: its objective is the
probability of ``commit`` minus the supplied baseline probability, and its
whole-cell upper bound is the exact upper endpoint of that probability.  It is
only a core-contract fixture, not a poker consumer or profitability example.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

from repeated_poker.certified_global_optimizer import (
    BOUND_CONTRACT_VERSION,
    RESPONSE_SEMANTICS,
    ExactActionProbability,
    ExactBehaviorPolicy,
    ExactBehaviorRow,
    HeroBehaviorScenario,
    HeroInformationSet,
    ScalarOracleBound,
    ScalarOracleEvaluation,
    behavior_cell_identity,
    behavior_policy_identity,
    exact_certified_global_optimizer_json,
    optimize_certified_global_hero_commitment,
    scalar_oracle_bound_identity,
)


def sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rational(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


SCENARIO = HeroBehaviorScenario(
    scenario_identity=sha256("m36-example-scenario"),
    information_sets=(
        HeroInformationSet(
            information_set_id="H:commitment",
            legal_action_ids=("baseline", "commit"),
        ),
    ),
)
BASELINE = ExactBehaviorPolicy(
    rows=(
        ExactBehaviorRow(
            information_set_id="H:commitment",
            actions=(
                ExactActionProbability("baseline", "1/2"),
                ExactActionProbability("commit", "1/2"),
            ),
        ),
    )
)


class AnalyticOracle:
    """Exact linear fixture satisfying the public scalar bound contract."""

    response_oracle_identity = sha256("m36-example-response-oracle")
    objective_identity = sha256("m36-example-baseline-uplift-objective")
    response_semantics = RESPONSE_SEMANTICS
    bound_contract_version = BOUND_CONTRACT_VERSION

    def evaluate(self, policy: ExactBehaviorPolicy) -> ScalarOracleEvaluation:
        probabilities = {
            action.action_id: Fraction(action.probability)
            for action in policy.rows[0].actions
        }
        candidate = probabilities["commit"]
        baseline = Fraction(1, 2)
        uplift = candidate - baseline
        policy_identity = behavior_policy_identity(SCENARIO, policy)
        return ScalarOracleEvaluation(
            policy_identity=policy_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            baseline_total_repeated_hero_ev=rational(baseline),
            candidate_total_repeated_hero_ev=rational(candidate),
            uplift=rational(uplift),
            complete_response_correspondence_identity=sha256(
                "m36-example-complete-response:" + policy_identity
            ),
            complete_response_record_count=2,
            hero_worst_witness_count=2,
        )

    def upper_bound(self, cell) -> ScalarOracleBound:
        intervals = {
            action.action_id: action for action in cell.rows[0].actions
        }
        commit = Fraction(intervals["commit"].upper)
        baseline_probability = Fraction(1) - commit
        candidate = ExactBehaviorPolicy(
            rows=(
                ExactBehaviorRow(
                    information_set_id="H:commitment",
                    actions=(
                        ExactActionProbability(
                            "baseline", rational(baseline_probability)
                        ),
                        ExactActionProbability("commit", rational(commit)),
                    ),
                ),
            )
        )
        upper = commit - Fraction(1, 2)
        cell_identity = behavior_cell_identity(SCENARIO, cell)
        candidate_identity = behavior_policy_identity(SCENARIO, candidate)
        return ScalarOracleBound(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=rational(upper),
            bound_identity=scalar_oracle_bound_identity(
                cell_identity=cell_identity,
                response_oracle_identity=self.response_oracle_identity,
                objective_identity=self.objective_identity,
                upper_bound=rational(upper),
                candidate_policy_identity=candidate_identity,
            ),
            candidate_policy=candidate,
        )


def main() -> None:
    result = optimize_certified_global_hero_commitment(
        SCENARIO,
        BASELINE,
        AnalyticOracle(),
    )
    print(exact_certified_global_optimizer_json(result))


if __name__ == "__main__":
    main()
