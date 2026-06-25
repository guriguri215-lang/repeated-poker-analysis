"""Detection time ``T_detect`` v0: a small observable-distribution model.

``T_detect`` estimates how many observations Villain needs before it can
statistically tell a candidate Hero policy apart from the baseline, using only
*observable event distributions* (for example, action-frequency maps).  It is a
sensitivity analysis, not a psychological model, not a real learning-speed
estimate, and not a full opponent-adaptation model.  It is entirely separate
from ``T_deadline`` (the economic adaptation deadline).

The model compares two distributions over the same event set with the total
variation distance and the Kullback-Leibler divergence ``D(candidate ||
baseline)`` in nats, then turns the divergence into a required number of
observations via a log-likelihood threshold.

Strategy-space L1 distance and observable-distribution distance are different
concepts: the L1 distance reported elsewhere measures how far two strategy
vectors are, while the distances here measure how distinguishable two observed
event distributions are.  This module does not use tree reach probabilities,
CFR, or any learning simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .candidates import HeroStrategyCandidate
from .game import HeroStrategy, require_finite, require_valid_tolerance

# A distribution over observable events (e.g. action name -> probability).
EventDistribution = Dict[str, float]


@dataclass(frozen=True)
class DetectionResult:
    """The observable-distribution detection estimate for one comparison."""

    event_count: int
    total_variation_distance: float
    kl_divergence_nats: float
    log_likelihood_threshold: float
    required_observations: Optional[int]
    occurrence_probability_per_opportunity: Optional[float]
    estimated_opportunities: Optional[int]

    def to_dict(self) -> dict:
        """Return a summary dict with English keys.

        ``kl_divergence_nats`` may be ``inf`` when the candidate places mass on
        an event the baseline never produces.
        """

        return {
            "event_count": self.event_count,
            "total_variation_distance": self.total_variation_distance,
            "kl_divergence_nats": self.kl_divergence_nats,
            "log_likelihood_threshold": self.log_likelihood_threshold,
            "required_observations": self.required_observations,
            "occurrence_probability_per_opportunity": (
                self.occurrence_probability_per_opportunity
            ),
            "estimated_opportunities": self.estimated_opportunities,
        }


def _validate_distribution(
    distribution: EventDistribution, name: str, tolerance: float
) -> None:
    if not distribution:
        raise ValueError(f"{name} must be a non-empty distribution")
    for event, probability in distribution.items():
        require_finite(probability, f"{name}[{event!r}]")
        if probability < 0:
            raise ValueError(
                f"{name}[{event!r}] must be non-negative, got {probability!r}"
            )
    total = math.fsum(distribution.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} sums to {total}, expected 1")


def calculate_detection_time(
    baseline: EventDistribution,
    candidate: EventDistribution,
    log_likelihood_threshold: float,
    occurrence_probability_per_opportunity: Optional[float] = None,
    tolerance: float = 1e-9,
) -> DetectionResult:
    """Estimate detection time from two observable event distributions.

    ``baseline`` and ``candidate`` map each observable event (such as an action
    name) to a probability.  They must cover exactly the same event set; a
    missing key is rejected rather than treated as zero.

    The total variation distance is ``0.5 * sum(abs(p_i - q_i))``.  The KL
    divergence ``D(candidate || baseline)`` is in nats (natural log): terms with
    ``candidate_i == 0`` contribute nothing, and a term with
    ``candidate_i > 0`` while ``baseline_i == 0`` makes the divergence ``inf``.

    ``required_observations`` is ``None`` when the divergence is zero,
    ``ceil(log_likelihood_threshold / kl)`` when it is finite and positive, and
    ``1`` when it is ``inf``.  When ``occurrence_probability_per_opportunity`` is
    given, ``estimated_opportunities`` is
    ``ceil(required_observations / occurrence_probability_per_opportunity)`` (or
    ``None`` when ``required_observations`` is ``None``).
    """

    require_valid_tolerance(tolerance)
    require_finite(log_likelihood_threshold, "log_likelihood_threshold")
    if log_likelihood_threshold <= 0:
        raise ValueError(
            "log_likelihood_threshold must be positive, got "
            f"{log_likelihood_threshold!r}"
        )
    if occurrence_probability_per_opportunity is not None:
        require_finite(
            occurrence_probability_per_opportunity,
            "occurrence_probability_per_opportunity",
        )
        if not 0.0 < occurrence_probability_per_opportunity <= 1.0:
            raise ValueError(
                "occurrence_probability_per_opportunity must satisfy 0 < p <= 1, "
                f"got {occurrence_probability_per_opportunity!r}"
            )

    if set(baseline) != set(candidate):
        raise ValueError(
            "baseline and candidate must cover the same event set; "
            f"baseline events {sorted(baseline)} != candidate events "
            f"{sorted(candidate)}"
        )
    _validate_distribution(baseline, "baseline", tolerance)
    _validate_distribution(candidate, "candidate", tolerance)

    events = sorted(baseline)
    total_variation = 0.5 * math.fsum(
        abs(candidate[event] - baseline[event]) for event in events
    )

    kl_divergence = 0.0
    for event in events:
        candidate_p = candidate[event]
        baseline_p = baseline[event]
        if candidate_p == 0.0:
            continue  # 0 * log(...) contributes nothing
        if baseline_p == 0.0:
            kl_divergence = math.inf
            break
        kl_divergence += candidate_p * math.log(candidate_p / baseline_p)
    if math.isfinite(kl_divergence) and -tolerance <= kl_divergence < 0.0:
        kl_divergence = 0.0  # clamp tiny negative rounding to exact zero

    if kl_divergence == math.inf:
        required_observations: Optional[int] = 1
    elif kl_divergence == 0.0:
        required_observations = None
    else:
        required_observations = math.ceil(log_likelihood_threshold / kl_divergence)

    estimated_opportunities: Optional[int] = None
    if (
        occurrence_probability_per_opportunity is not None
        and required_observations is not None
    ):
        estimated_opportunities = math.ceil(
            required_observations / occurrence_probability_per_opportunity
        )

    return DetectionResult(
        event_count=len(events),
        total_variation_distance=total_variation,
        kl_divergence_nats=kl_divergence,
        log_likelihood_threshold=log_likelihood_threshold,
        required_observations=required_observations,
        occurrence_probability_per_opportunity=occurrence_probability_per_opportunity,
        estimated_opportunities=estimated_opportunities,
    )


def calculate_candidate_local_detection(
    baseline_hero_strategy: HeroStrategy,
    candidate: HeroStrategyCandidate,
    log_likelihood_threshold: float,
    occurrence_probability_per_opportunity: Optional[float] = None,
    tolerance: float = 1e-9,
) -> DetectionResult:
    """Estimate local detection at the candidate's own information set.

    The baseline and candidate Hero action distributions at
    ``candidate.info_set`` are treated as the observable event distributions,
    and :func:`calculate_detection_time` is applied to them.

    This is a *local* model: it is conditional on reaching that information set
    and observing an action there.  It deliberately ignores tree reach
    probabilities (how often the information set is actually reached).  The
    distribution distances computed here are observable-distribution distances,
    which are a different concept from the strategy-space L1 distance carried by
    the candidate.
    """

    info_set = candidate.info_set
    if info_set not in baseline_hero_strategy.probabilities:
        raise ValueError(
            f"baseline Hero strategy is missing information set {info_set!r}"
        )
    if info_set not in candidate.hero_strategy.probabilities:
        raise ValueError(
            f"candidate Hero strategy is missing information set {info_set!r}"
        )

    baseline_distribution = baseline_hero_strategy.probabilities[info_set]
    candidate_distribution = candidate.hero_strategy.probabilities[info_set]

    return calculate_detection_time(
        baseline=baseline_distribution,
        candidate=candidate_distribution,
        log_likelihood_threshold=log_likelihood_threshold,
        occurrence_probability_per_opportunity=occurrence_probability_per_opportunity,
        tolerance=tolerance,
    )
