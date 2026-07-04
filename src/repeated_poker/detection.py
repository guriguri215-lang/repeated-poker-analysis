"""Detection-time diagnostics ``T_detect`` v0 and v1.

``local_v0`` compares local observable event distributions at the candidate's
changed information set. It is conditional on reaching that information set and
does not use tree reach probabilities.

``reach_weighted_v1`` builds per-hand public observation distributions from
root-to-terminal path probabilities under the baseline Villain profile and
baseline or candidate Hero. It therefore includes within-spot reach in the
one-hand distribution before applying the same KL direction,
``D(candidate || baseline)``.

Both methods are sensitivity analyses, not psychological models, not real
learning-speed estimates, and not full opponent-adaptation models. They are
entirely separate from ``T_deadline`` (the economic adaptation deadline).
``reach_weighted_v1`` can be slower than a real observer with more information
because it does not use private buckets, but it can also be faster because it
assumes the candidate distribution ``P1`` is known exactly. It is therefore
neither an upper nor a lower bound on real detection time.

Strategy-space L1 distance and observable-distribution distance are different
concepts: the L1 distance reported elsewhere measures how far two strategy
vectors are, while the distances here measure how distinguishable two observed
event distributions are. This module does not use CFR or learning simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .candidates import HeroStrategyCandidate
from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    iter_terminals,
    require_finite,
    require_valid_tolerance,
    validate_hero_strategy,
    validate_villain_strategy,
)

# A distribution over observable events (e.g. action name -> probability).
EventDistribution = Dict[str, float]
ObservationKey = Tuple[object, ...]
ObservationDistribution = Dict[ObservationKey, float]
TerminalReveals = Mapping[str, Optional[Tuple[str, ...]]]

DETECTION_METHOD_LOCAL_V0 = "local_v0"
DETECTION_METHOD_REACH_WEIGHTED_V1 = "reach_weighted_v1"
DETECTION_METHODS = (DETECTION_METHOD_LOCAL_V0, DETECTION_METHOD_REACH_WEIGHTED_V1)

OBSERVATION_MODEL_ACTIONS_ONLY = "actions_only"
OBSERVATION_MODEL_SHOWDOWN_REVEAL = "showdown_reveal"
OBSERVATION_MODELS = (OBSERVATION_MODEL_ACTIONS_ONLY, OBSERVATION_MODEL_SHOWDOWN_REVEAL)

DETECTION_TIME_BASIS_SPRT_KL = "sprt_kl"
DETECTION_TIME_BASIS_BASELINE_IMPOSSIBLE = "baseline_impossible_event"
DEFAULT_MAX_DETECTION_TERMINALS = 100_000

_OBSERVATION_MODEL_TERMINAL_PATH = "_terminal_path"


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


@dataclass(frozen=True)
class ReachWeightedDetectionResult:
    """The opt-in reach-weighted per-hand ``T_detect`` v1 estimate.

    One observation is one hand: the full root-to-terminal public observation
    distribution is built under the baseline Villain profile and either the
    baseline or candidate Hero profile. The KL value is therefore per hand and
    already includes reach probabilities. ``t_detect_hands`` is a rough
    diagnostic of an expected detection-time scale under the stated idealized
    sequential likelihood-ratio assumptions; it is not a real opponent-learning
    model or a claim about actual detection. Since the public observer does not
    use private buckets, v1 can be slower than a real observer with more
    information; since it assumes ``P1`` is known exactly, it can also be faster.
    It is neither an upper nor a lower bound on real detection time.
    """

    event_count: int
    total_variation_per_hand: float
    kl_per_hand_nats: float
    log_likelihood_threshold: float
    baseline_impossible_mass_per_hand: float
    t_detect_hands: Optional[int]
    detection_time_basis: Optional[str]
    observation_model: str

    def to_dict(self) -> dict:
        """Return a summary dict with English keys."""

        return {
            "event_count": self.event_count,
            "total_variation_per_hand": self.total_variation_per_hand,
            "kl_per_hand_nats": self.kl_per_hand_nats,
            "log_likelihood_threshold": self.log_likelihood_threshold,
            "baseline_impossible_mass_per_hand": (
                self.baseline_impossible_mass_per_hand
            ),
            "t_detect_hands": self.t_detect_hands,
            "detection_time_basis": self.detection_time_basis,
            "observation_model": self.observation_model,
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


def validate_detection_method(method: str) -> None:
    """Validate the detection method selector."""

    if not isinstance(method, str):
        raise ValueError(f"detection_method must be a string, got {method!r}")
    if method not in DETECTION_METHODS:
        raise ValueError(
            "detection_method must be one of "
            f"{DETECTION_METHODS}, got {method!r}"
        )


def resolve_detection_observation_model(
    method: str, observation_model: Optional[str]
) -> Optional[str]:
    """Validate and resolve the observation-model selector.

    ``local_v0`` does not use an observation-model switch, so any non-``None``
    value is rejected. ``reach_weighted_v1`` defaults ``None`` to
    ``actions_only`` and otherwise accepts ``actions_only`` or
    ``showdown_reveal``.
    """

    validate_detection_method(method)
    if method == DETECTION_METHOD_LOCAL_V0:
        if observation_model is not None:
            raise ValueError(
                "detection_observation_model is only valid with "
                "detection_method='reach_weighted_v1'"
            )
        return None

    if observation_model is None:
        return OBSERVATION_MODEL_ACTIONS_ONLY
    if not isinstance(observation_model, str):
        raise ValueError(
            "detection_observation_model must be a string or None, got "
            f"{observation_model!r}"
        )
    if observation_model not in OBSERVATION_MODELS:
        raise ValueError(
            "detection_observation_model must be one of "
            f"{OBSERVATION_MODELS}, got {observation_model!r}"
        )
    return observation_model


def validate_max_detection_terminals(max_detection_terminals: int) -> None:
    """Validate the terminal-count guard for reach-weighted detection."""

    if isinstance(max_detection_terminals, bool) or not isinstance(
        max_detection_terminals, int
    ):
        raise ValueError(
            "max_detection_terminals must be an int, got "
            f"{max_detection_terminals!r}"
        )
    if max_detection_terminals < 1:
        raise ValueError(
            "max_detection_terminals must be at least 1, got "
            f"{max_detection_terminals}"
        )


def validate_detection_parameters(
    log_likelihood_threshold: float,
    occurrence_probability_per_opportunity: Optional[float] = None,
    tolerance: float = 1e-9,
) -> None:
    """Validate the scalar detection parameters (no distributions required).

    Shared by :func:`calculate_detection_time` and any caller (such as the
    analysis-report builder) that must reject invalid parameters even when there
    are no candidate distributions to evaluate.
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

    validate_detection_parameters(
        log_likelihood_threshold, occurrence_probability_per_opportunity, tolerance
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


def _candidate_info_set_distributions(
    baseline_hero_strategy: HeroStrategy,
    candidate: HeroStrategyCandidate,
    info_set: str,
) -> tuple:
    """Return the ``(baseline, candidate)`` action distributions at ``info_set``."""

    if info_set not in baseline_hero_strategy.probabilities:
        raise ValueError(
            f"baseline Hero strategy is missing information set {info_set!r}"
        )
    if info_set not in candidate.hero_strategy.probabilities:
        raise ValueError(
            f"candidate Hero strategy is missing information set {info_set!r}"
        )
    return (
        baseline_hero_strategy.probabilities[info_set],
        candidate.hero_strategy.probabilities[info_set],
    )


def _earliest_detection(results: "list[DetectionResult]") -> DetectionResult:
    """Return the result for the information set detected earliest.

    "Earliest" is the fewest ``required_observations`` (an information set whose
    distribution is unchanged, ``required_observations is None``, sorts last as
    never distinguished); ties break to the larger total-variation distance so
    the choice is deterministic.
    """

    def sort_key(result: DetectionResult):
        required = result.required_observations
        required_rank = required if required is not None else math.inf
        return (required_rank, -result.total_variation_distance)

    return min(results, key=sort_key)


def calculate_candidate_local_detection(
    baseline_hero_strategy: HeroStrategy,
    candidate: HeroStrategyCandidate,
    log_likelihood_threshold: float,
    occurrence_probability_per_opportunity: Optional[float] = None,
    tolerance: float = 1e-9,
) -> DetectionResult:
    """Estimate local detection at the candidate's changed information set(s).

    The baseline and candidate Hero action distributions at each information set
    the candidate changes are treated as observable event distributions, and
    :func:`calculate_detection_time` is applied to them.  A single-shift candidate
    has one changed information set, so this reduces to the local estimate at that
    set.  A multi-shift candidate (M2-T2) changes several information sets; this
    reports the one detected *earliest* (fewest required observations), i.e. the
    first information set at which the deviation becomes observable under this v0
    local model.

    This is a *local* model: it is conditional on reaching an information set and
    observing an action there.  It deliberately ignores tree reach probabilities
    (how often each information set is actually reached) and does not combine
    evidence across information sets -- a reach-weighted / sequential model is
    deferred to ``T_detect`` v1.  The distribution distances computed here are
    observable-distribution distances, which are a different concept from the
    strategy-space L1 distance carried by the candidate.
    """

    results = []
    for info_set in candidate.info_sets:
        baseline_distribution, candidate_distribution = (
            _candidate_info_set_distributions(baseline_hero_strategy, candidate, info_set)
        )
        results.append(
            calculate_detection_time(
                baseline=baseline_distribution,
                candidate=candidate_distribution,
                log_likelihood_threshold=log_likelihood_threshold,
                occurrence_probability_per_opportunity=(
                    occurrence_probability_per_opportunity
                ),
                tolerance=tolerance,
            )
        )
    return _earliest_detection(results)


def calculate_candidate_reach_weighted_detection(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    candidate: HeroStrategyCandidate,
    baseline_villain_strategy: VillainStrategy,
    log_likelihood_threshold: float,
    observation_model: Optional[str] = None,
    terminal_reveals: Optional[TerminalReveals] = None,
    max_detection_terminals: int = DEFAULT_MAX_DETECTION_TERMINALS,
    tolerance: float = 1e-9,
) -> ReachWeightedDetectionResult:
    """Estimate opt-in reach-weighted per-hand detection for one candidate.

    The model builds two one-hand observation distributions over terminal paths:
    ``P0`` under the baseline Hero strategy and ``P1`` under the candidate Hero
    strategy, with the same fixed baseline Villain profile in both. Public
    observations are selected by ``observation_model``:

    * ``actions_only`` (the default) groups paths by public action sequence.
    * ``showdown_reveal`` additionally refines showdown terminal classes by the
      builder-supplied ``terminal_reveals`` annotation. Fold terminals must have
      ``None`` and reveal tuples are used exactly as supplied by the builder.

    ``occurrence_probability_per_opportunity`` is intentionally absent: one v1
    observation is already one hand/opportunity, so the returned
    ``t_detect_hands`` flows directly into the downstream timing comparison.
    The estimate is neither an upper nor a lower bound on real detection time:
    it omits private buckets unless revealed at showdown but assumes the
    candidate distribution ``P1`` is known exactly.
    """

    resolved_model = resolve_detection_observation_model(
        DETECTION_METHOD_REACH_WEIGHTED_V1, observation_model
    )
    validate_detection_parameters(log_likelihood_threshold, None, tolerance)
    validate_max_detection_terminals(max_detection_terminals)
    _require_terminal_count_within_limit(tree, max_detection_terminals)
    validate_hero_strategy(tree, baseline_hero_strategy, tolerance=tolerance)
    validate_hero_strategy(tree, candidate.hero_strategy, tolerance=tolerance)
    validate_villain_strategy(tree, baseline_villain_strategy, tolerance=tolerance)

    baseline, shifted = _candidate_observation_distributions(
        tree,
        baseline_hero_strategy,
        candidate.hero_strategy,
        baseline_villain_strategy,
        observation_model=resolved_model,
        terminal_reveals=terminal_reveals,
        tolerance=tolerance,
    )
    return calculate_reach_weighted_detection_time_from_distributions(
        baseline,
        shifted,
        log_likelihood_threshold=log_likelihood_threshold,
        observation_model=resolved_model,
        tolerance=tolerance,
    )


def calculate_reach_weighted_detection_time_from_distributions(
    baseline: ObservationDistribution,
    candidate: ObservationDistribution,
    log_likelihood_threshold: float,
    observation_model: str = OBSERVATION_MODEL_ACTIONS_ONLY,
    tolerance: float = 1e-9,
) -> ReachWeightedDetectionResult:
    """Compute per-hand TV/KL and the v1 ``T_detect`` estimate.

    Unlike the v0 helper, the two distributions may be sparse over different
    observation keys. Missing keys are treated as exact zero probability, which
    is required for the baseline-impossible-event branch.
    """

    if observation_model not in OBSERVATION_MODELS and (
        observation_model != _OBSERVATION_MODEL_TERMINAL_PATH
    ):
        raise ValueError(f"unknown observation_model {observation_model!r}")
    validate_detection_parameters(log_likelihood_threshold, None, tolerance)
    _validate_observation_distribution(baseline, "baseline", tolerance)
    _validate_observation_distribution(candidate, "candidate", tolerance)

    events = sorted(set(baseline) | set(candidate), key=repr)
    total_variation = 0.5 * math.fsum(
        abs(candidate.get(event, 0.0) - baseline.get(event, 0.0))
        for event in events
    )

    baseline_impossible_mass = math.fsum(
        candidate.get(event, 0.0)
        for event in events
        if baseline.get(event, 0.0) == 0.0 and candidate.get(event, 0.0) > 0.0
    )

    if baseline_impossible_mass > 0.0:
        kl_per_hand = math.inf
        t_detect_hands = math.ceil(1.0 / baseline_impossible_mass)
        detection_time_basis: Optional[str] = DETECTION_TIME_BASIS_BASELINE_IMPOSSIBLE
    else:
        kl_per_hand = 0.0
        for event in events:
            candidate_p = candidate.get(event, 0.0)
            if candidate_p == 0.0:
                continue
            baseline_p = baseline.get(event, 0.0)
            kl_per_hand += candidate_p * math.log(candidate_p / baseline_p)
        if -tolerance <= kl_per_hand < 0.0:
            kl_per_hand = 0.0
        if kl_per_hand < 0.0:
            raise ValueError(
                f"per-hand KL was negative beyond tolerance: {kl_per_hand}"
            )
        if kl_per_hand == 0.0:
            t_detect_hands = None
            detection_time_basis = None
        else:
            t_detect_hands = math.ceil(log_likelihood_threshold / kl_per_hand)
            detection_time_basis = DETECTION_TIME_BASIS_SPRT_KL

    return ReachWeightedDetectionResult(
        event_count=len(events),
        total_variation_per_hand=total_variation,
        kl_per_hand_nats=kl_per_hand,
        log_likelihood_threshold=log_likelihood_threshold,
        baseline_impossible_mass_per_hand=baseline_impossible_mass,
        t_detect_hands=t_detect_hands,
        detection_time_basis=detection_time_basis,
        observation_model=observation_model,
    )


def _validate_observation_distribution(
    distribution: ObservationDistribution, name: str, tolerance: float
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


def _require_terminal_count_within_limit(
    tree: GameTree, max_detection_terminals: int
) -> None:
    terminal_count = sum(1 for _ in iter_terminals(tree.root))
    if terminal_count > max_detection_terminals:
        raise ValueError(
            f"tree has {terminal_count} terminals, exceeding "
            f"max_detection_terminals={max_detection_terminals}"
        )


def _candidate_observation_distributions(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    candidate_hero_strategy: HeroStrategy,
    baseline_villain_strategy: VillainStrategy,
    observation_model: str,
    terminal_reveals: Optional[TerminalReveals] = None,
    tolerance: float = 1e-9,
) -> Tuple[ObservationDistribution, ObservationDistribution]:
    """Return ``(P0, P1)`` observation distributions for tests and v1."""

    if observation_model == OBSERVATION_MODEL_SHOWDOWN_REVEAL:
        _validate_terminal_reveals(tree, terminal_reveals)
    elif observation_model not in (
        OBSERVATION_MODEL_ACTIONS_ONLY,
        _OBSERVATION_MODEL_TERMINAL_PATH,
    ):
        raise ValueError(f"unknown observation_model {observation_model!r}")

    baseline = _observation_distribution_for_strategy(
        tree,
        baseline_hero_strategy,
        baseline_villain_strategy,
        observation_model=observation_model,
        terminal_reveals=terminal_reveals,
    )
    candidate = _observation_distribution_for_strategy(
        tree,
        candidate_hero_strategy,
        baseline_villain_strategy,
        observation_model=observation_model,
        terminal_reveals=terminal_reveals,
    )
    _validate_observation_distribution(baseline, "baseline", tolerance)
    _validate_observation_distribution(candidate, "candidate", tolerance)
    return baseline, candidate


def _validate_terminal_reveals(
    tree: GameTree, terminal_reveals: Optional[TerminalReveals]
) -> None:
    if terminal_reveals is None:
        raise ValueError(
            "terminal_reveals are required when "
            "detection_observation_model='showdown_reveal'"
        )
    terminal_ids = {terminal.node_id for terminal in iter_terminals(tree.root)}
    reveal_ids = set(terminal_reveals)
    missing = sorted(terminal_ids - reveal_ids)
    extra = sorted(reveal_ids - terminal_ids)
    if missing:
        raise ValueError(f"terminal_reveals missing terminal ids {missing}")
    if extra:
        raise ValueError(f"terminal_reveals has unknown terminal ids {extra}")
    for terminal_id, reveal in terminal_reveals.items():
        if reveal is None:
            continue
        if not isinstance(reveal, tuple) or any(
            not isinstance(item, str) for item in reveal
        ):
            raise ValueError(
                "terminal_reveals entries must be None or tuple[str, ...]; "
                f"{terminal_id!r} has {reveal!r}"
            )


def _observation_distribution_for_strategy(
    tree: GameTree,
    hero_strategy: HeroStrategy,
    villain_strategy: VillainStrategy,
    observation_model: str,
    terminal_reveals: Optional[TerminalReveals],
) -> ObservationDistribution:
    distribution: ObservationDistribution = {}

    def visit(node, probability: float, actions: Tuple[Tuple[str, str], ...]) -> None:
        if probability == 0.0:
            return
        if isinstance(node, TerminalNode):
            key = _observation_key(node, actions, observation_model, terminal_reveals)
            distribution[key] = distribution.get(key, 0.0) + probability
            return
        if isinstance(node, ChanceNode):
            for chance_probability, child in node.children:
                visit(child, probability * chance_probability, actions)
            return
        if isinstance(node, HeroNode):
            for action, child in node.actions:
                action_probability = hero_strategy.action_probability(
                    node.info_set, action
                )
                visit(
                    child,
                    probability * action_probability,
                    actions + (("H", action),),
                )
            return
        if isinstance(node, VillainNode):
            for action, child in node.actions:
                action_probability = villain_strategy.action_probability(
                    node.info_set, action
                )
                visit(
                    child,
                    probability * action_probability,
                    actions + (("V", action),),
                )
            return
        raise TypeError(f"unknown node type: {type(node)!r}")

    visit(tree.root, 1.0, ())
    return distribution


def _observation_key(
    terminal: TerminalNode,
    actions: Tuple[Tuple[str, str], ...],
    observation_model: str,
    terminal_reveals: Optional[TerminalReveals],
) -> ObservationKey:
    if observation_model == _OBSERVATION_MODEL_TERMINAL_PATH:
        return (terminal.node_id,)
    if observation_model == OBSERVATION_MODEL_ACTIONS_ONLY:
        return (actions, None)
    if observation_model == OBSERVATION_MODEL_SHOWDOWN_REVEAL:
        reveal = terminal_reveals[terminal.node_id]  # validated earlier
        normalised_reveal: Optional[Tuple[str, ...]] = reveal if reveal else None
        return (actions, normalised_reveal)
    raise ValueError(f"unknown observation_model {observation_model!r}")


def candidate_observation_distance(
    baseline_hero_strategy: HeroStrategy,
    candidate: HeroStrategyCandidate,
    tolerance: float = 1e-9,
) -> float:
    """Return an always-available observable distance for a candidate.

    This is the total-variation distance between the baseline and candidate Hero
    action distributions at the candidate's changed information set(s), and unlike
    :func:`calculate_candidate_local_detection` it needs no detection threshold,
    so it is defined for every candidate regardless of whether the (optional)
    detection-time analysis is enabled.  It is used as the "observation distance"
    axis of the M2-T2 trade-off Pareto frontier.

    For a single-shift candidate it is the total-variation distance at the changed
    information set.  For a multi-shift candidate it is the **maximum** over the
    changed information sets -- the largest single-information-set observable
    change.  This is an observable-distribution distance (a different concept from
    the strategy-space L1 distance), and it uses no tree reach probabilities.
    """

    require_valid_tolerance(tolerance)
    distances = []
    for info_set in candidate.info_sets:
        baseline_distribution, candidate_distribution = (
            _candidate_info_set_distributions(baseline_hero_strategy, candidate, info_set)
        )
        if set(baseline_distribution) != set(candidate_distribution):
            raise ValueError(
                "baseline and candidate must cover the same event set at "
                f"{info_set!r}; baseline events {sorted(baseline_distribution)} != "
                f"candidate events {sorted(candidate_distribution)}"
            )
        _validate_distribution(baseline_distribution, "baseline", tolerance)
        _validate_distribution(candidate_distribution, "candidate", tolerance)
        distances.append(
            0.5
            * math.fsum(
                abs(candidate_distribution[event] - baseline_distribution[event])
                for event in baseline_distribution
            )
        )
    return max(distances) if distances else 0.0
