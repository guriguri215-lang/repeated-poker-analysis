"""Tests for the observable-distribution detection-time (T_detect) model."""

import math

import pytest

from repeated_poker import (
    ChanceNode,
    GameTree,
    HeroStrategy,
    HeroStrategyCandidate,
    HeroNode,
    ShiftComponent,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    calculate_candidate_local_detection,
    calculate_candidate_reach_weighted_detection,
    calculate_detection_time,
    calculate_reach_weighted_detection_time_from_distributions,
)
from repeated_poker.detection import (
    OBSERVATION_MODEL_ACTIONS_ONLY,
    OBSERVATION_MODEL_SHOWDOWN_REVEAL,
    _OBSERVATION_MODEL_TERMINAL_PATH,
    _candidate_observation_distributions,
)


def test_binary_distribution_hand_computed():
    baseline = {"check": 0.8, "bet": 0.2}
    candidate = {"check": 0.5, "bet": 0.5}

    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=3.0
    )

    assert result.event_count == 2
    assert result.total_variation_distance == pytest.approx(0.3)
    # D(candidate||baseline) = 0.5*ln(0.5/0.8) + 0.5*ln(0.5/0.2).
    expected_kl = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
    assert result.kl_divergence_nats == pytest.approx(expected_kl)
    # ceil(3.0 / 0.22314355) = 14.
    assert result.required_observations == 14
    assert result.occurrence_probability_per_opportunity is None
    assert result.estimated_opportunities is None


def test_identical_distribution_has_zero_kl_and_no_required_observations():
    distribution = {"check": 0.7, "bet": 0.3}
    result = calculate_detection_time(
        dict(distribution),
        dict(distribution),
        log_likelihood_threshold=2.0,
        occurrence_probability_per_opportunity=0.5,
    )
    assert result.kl_divergence_nats == 0.0
    assert result.total_variation_distance == pytest.approx(0.0)
    assert result.required_observations is None
    # estimated_opportunities is None when required_observations is None.
    assert result.estimated_opportunities is None


def test_zero_baseline_positive_candidate_gives_infinite_kl():
    baseline = {"check": 1.0, "bet": 0.0}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=3.0
    )
    assert result.kl_divergence_nats == math.inf
    assert result.required_observations == 1
    assert result.total_variation_distance == pytest.approx(0.5)


def test_zero_candidate_term_is_skipped():
    baseline = {"check": 0.5, "bet": 0.5}
    candidate = {"check": 1.0, "bet": 0.0}
    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=1.0
    )
    # Only the "check" term contributes: 1.0 * ln(1.0 / 0.5) = ln 2.
    assert result.kl_divergence_nats == pytest.approx(math.log(2.0))
    assert result.required_observations == math.ceil(1.0 / math.log(2.0))


def test_estimated_opportunities_uses_occurrence_probability():
    baseline = {"check": 0.8, "bet": 0.2}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline,
        candidate,
        log_likelihood_threshold=3.0,
        occurrence_probability_per_opportunity=0.5,
    )
    assert result.required_observations == 14
    # ceil(14 / 0.5) = 28.
    assert result.estimated_opportunities == 28


def test_infinite_kl_estimated_opportunities():
    baseline = {"check": 1.0, "bet": 0.0}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline,
        candidate,
        log_likelihood_threshold=3.0,
        occurrence_probability_per_opportunity=0.25,
    )
    assert result.required_observations == 1
    assert result.estimated_opportunities == 4  # ceil(1 / 0.25)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_VALID_BASELINE = {"check": 0.8, "bet": 0.2}
_VALID_CANDIDATE = {"check": 0.5, "bet": 0.5}


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_log_likelihood_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="log_likelihood_threshold"):
        calculate_detection_time(_VALID_BASELINE, _VALID_CANDIDATE, bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_occurrence_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="occurrence_probability_per_opportunity"):
        calculate_detection_time(
            _VALID_BASELINE,
            _VALID_CANDIDATE,
            log_likelihood_threshold=3.0,
            occurrence_probability_per_opportunity=bad,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        calculate_detection_time(
            _VALID_BASELINE, _VALID_CANDIDATE, log_likelihood_threshold=3.0, tolerance=bad
        )


def test_distribution_sum_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sums to"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.3}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


def test_negative_probability_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        calculate_detection_time(
            {"check": 1.2, "bet": -0.2}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        calculate_detection_time(
            {"check": bad, "bet": 0.2}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


def test_event_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same event set"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.2},
            {"check": 0.5, "raise": 0.5},
            log_likelihood_threshold=3.0,
        )


def test_missing_key_is_not_treated_as_zero():
    # The candidate omits "bet"; it must be rejected, not zero-filled.
    with pytest.raises(ValueError, match="same event set"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.2},
            {"check": 1.0},
            log_likelihood_threshold=3.0,
        )


# ---------------------------------------------------------------------------
# Candidate local detection helper
# ---------------------------------------------------------------------------


def _candidate(info_set, hero_probabilities):
    return HeroStrategyCandidate(
        candidate_id="cand-1",
        info_set=info_set,
        source_action="check",
        target_action="bet",
        shift_amount=0.3,
        hero_strategy=HeroStrategy(hero_probabilities),
        l1_distance=0.6,
    )


def test_candidate_local_detection_uses_info_set_distributions():
    baseline_hero = HeroStrategy(
        {"H1": {"check": 0.8, "bet": 0.2}, "Other": {"x": 1.0}}
    )
    candidate = _candidate(
        "H1", {"H1": {"check": 0.5, "bet": 0.5}, "Other": {"x": 1.0}}
    )

    result = calculate_candidate_local_detection(
        baseline_hero, candidate, log_likelihood_threshold=3.0
    )
    expected = calculate_detection_time(
        {"check": 0.8, "bet": 0.2},
        {"check": 0.5, "bet": 0.5},
        log_likelihood_threshold=3.0,
    )
    assert result == expected


def test_candidate_local_detection_missing_info_set_is_rejected():
    baseline_hero = HeroStrategy({"Other": {"x": 1.0}})
    candidate = _candidate("H1", {"H1": {"check": 0.5, "bet": 0.5}})
    with pytest.raises(ValueError, match="missing information set"):
        calculate_candidate_local_detection(
            baseline_hero, candidate, log_likelihood_threshold=3.0
        )


# ---------------------------------------------------------------------------
# Reach-weighted T_detect v1 acceptance tests
# ---------------------------------------------------------------------------

_THETA = 3.0
_KL_LOCAL_07_03 = 0.7 * math.log(1.4) + 0.3 * math.log(0.6)
_KL_LOCAL_04_06_FROM_06_04 = 0.4 * math.log(0.4 / 0.6) + 0.6 * math.log(0.6 / 0.4)


def _zero_terminal(node_id: str) -> TerminalNode:
    return TerminalNode(node_id=node_id, hero_ev=0.0, villain_ev=0.0, house_rake=0.0)


def _single_hand_detection_tree(hero_actions=("call", "fold")) -> GameTree:
    hero_node = HeroNode(
        node_id="ip",
        info_set="IP_vs_bet",
        actions=tuple((action, _zero_terminal(f"T_bet_{action}")) for action in hero_actions),
    )
    root = VillainNode(
        node_id="oop",
        info_set="OOP",
        actions=(("check", _zero_terminal("T_check")), ("bet", hero_node)),
    )
    return GameTree(root=root)


def _single_shift_candidate(distribution) -> HeroStrategyCandidate:
    return HeroStrategyCandidate(
        candidate_id="single",
        info_set="IP_vs_bet",
        source_action="fold",
        target_action="call",
        shift_amount=0.2,
        hero_strategy=HeroStrategy({"IP_vs_bet": dict(distribution)}),
        l1_distance=0.4,
    )


def _key(*actions, reveal=None):
    return (tuple(actions), reveal)


def test_reach_weighted_v1_test_a_reach_weighting_basic():
    tree = _single_hand_detection_tree()
    baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5}})
    candidate = _single_shift_candidate({"call": 0.7, "fold": 0.3})
    villain = VillainStrategy({"OOP": {"check": 0.5, "bet": 0.5}})

    p0, p1 = _candidate_observation_distributions(
        tree,
        baseline,
        candidate.hero_strategy,
        villain,
        observation_model=OBSERVATION_MODEL_ACTIONS_ONLY,
    )

    assert p0 == pytest.approx(
        {
            _key(("V", "check")): 0.5,
            _key(("V", "bet"), ("H", "call")): 0.25,
            _key(("V", "bet"), ("H", "fold")): 0.25,
        }
    )
    assert p1 == pytest.approx(
        {
            _key(("V", "check")): 0.5,
            _key(("V", "bet"), ("H", "call")): 0.35,
            _key(("V", "bet"), ("H", "fold")): 0.15,
        }
    )

    result = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
    )
    assert _KL_LOCAL_07_03 == pytest.approx(0.08228287850505178, abs=1e-12)
    assert result.kl_per_hand_nats == pytest.approx(0.04114143925252589, abs=1e-12)
    assert result.kl_per_hand_nats == pytest.approx(0.5 * _KL_LOCAL_07_03, abs=1e-12)
    assert result.t_detect_hands == 73

    local = calculate_candidate_local_detection(
        baseline, candidate, log_likelihood_threshold=_THETA
    )
    assert local.required_observations == 37


def test_reach_weighted_v1_test_b_zero_reach_is_not_detectable_in_model():
    tree = _single_hand_detection_tree()
    baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5}})
    candidate = _single_shift_candidate({"call": 0.7, "fold": 0.3})
    villain = VillainStrategy({"OOP": {"check": 1.0, "bet": 0.0}})

    result = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
    )

    assert result.kl_per_hand_nats == pytest.approx(0.0)
    assert result.total_variation_per_hand == pytest.approx(0.0)
    assert result.t_detect_hands is None
    local = calculate_candidate_local_detection(
        baseline, candidate, log_likelihood_threshold=_THETA
    )
    assert local.required_observations == 37


def _matrix_detection_fixture():
    def branch(hero_id):
        hero_node = HeroNode(
            node_id=f"ip_{hero_id}",
            info_set=f"IP::{hero_id}",
            actions=(
                ("call", _zero_terminal(f"T_call_{hero_id}")),
                ("fold", _zero_terminal(f"T_fold_{hero_id}")),
            ),
        )
        return VillainNode(
            node_id=f"oop_{hero_id}",
            info_set="OOP::v",
            actions=(("bet", hero_node),),
        )

    tree = GameTree(
        root=ChanceNode(
            node_id="bucket",
            children=((0.5, branch("A")), (0.5, branch("B"))),
        )
    )
    baseline = HeroStrategy(
        {
            "IP::A": {"call": 0.5, "fold": 0.5},
            "IP::B": {"call": 0.5, "fold": 0.5},
        }
    )
    shifted = HeroStrategy(
        {
            "IP::A": {"call": 0.7, "fold": 0.3},
            "IP::B": {"call": 0.3, "fold": 0.7},
        }
    )
    candidate = HeroStrategyCandidate(
        candidate_id="matrix",
        info_set=None,
        source_action=None,
        target_action=None,
        shift_amount=None,
        hero_strategy=shifted,
        l1_distance=0.8,
        shifts=(
            ShiftComponent("IP::A", "fold", "call", 0.2),
            ShiftComponent("IP::B", "call", "fold", 0.2),
        ),
    )
    villain = VillainStrategy({"OOP::v": {"bet": 1.0}})
    terminal_reveals = {
        "T_call_A": ("A", "v"),
        "T_fold_A": None,
        "T_call_B": ("B", "v"),
        "T_fold_B": None,
    }
    return tree, baseline, candidate, villain, terminal_reveals


def test_reach_weighted_v1_test_c_observation_model_distinguishes_reveals():
    tree, baseline, candidate, villain, terminal_reveals = _matrix_detection_fixture()

    actions = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="actions_only",
    )
    assert actions.kl_per_hand_nats == pytest.approx(0.0)
    assert actions.t_detect_hands is None

    p0, p1 = _candidate_observation_distributions(
        tree,
        baseline,
        candidate.hero_strategy,
        villain,
        observation_model=OBSERVATION_MODEL_SHOWDOWN_REVEAL,
        terminal_reveals=terminal_reveals,
    )
    assert p0 == pytest.approx(
        {
            _key(("V", "bet"), ("H", "call"), reveal=("A", "v")): 0.25,
            _key(("V", "bet"), ("H", "call"), reveal=("B", "v")): 0.25,
            _key(("V", "bet"), ("H", "fold")): 0.5,
        }
    )
    assert p1 == pytest.approx(
        {
            _key(("V", "bet"), ("H", "call"), reveal=("A", "v")): 0.35,
            _key(("V", "bet"), ("H", "call"), reveal=("B", "v")): 0.15,
            _key(("V", "bet"), ("H", "fold")): 0.5,
        }
    )

    showdown = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="showdown_reveal",
        terminal_reveals=terminal_reveals,
    )
    assert showdown.kl_per_hand_nats == pytest.approx(0.04114143925252589, abs=1e-12)
    assert showdown.t_detect_hands == 73
    local = calculate_candidate_local_detection(
        baseline, candidate, log_likelihood_threshold=_THETA
    )
    assert local.required_observations == 37
    assert showdown.kl_per_hand_nats >= actions.kl_per_hand_nats


def test_reach_weighted_v1_test_d_multi_shift_joint_distribution_chain_rule():
    c = 0.25
    after_check = HeroNode(
        node_id="ip_after_check",
        info_set="IP_after_OOP_check",
        actions=(
            ("check", _zero_terminal("T_check_check")),
            ("bet", _zero_terminal("T_check_bet")),
        ),
    )
    vs_bet = HeroNode(
        node_id="ip_vs_bet",
        info_set="IP_vs_OOP_bet",
        actions=(
            ("call", _zero_terminal("T_bet_call")),
            ("fold", _zero_terminal("T_bet_fold")),
            ("raise", _zero_terminal("T_bet_raise")),
        ),
    )
    tree = GameTree(
        root=VillainNode(
            node_id="oop",
            info_set="OOP_first",
            actions=(("check", after_check), ("bet", vs_bet)),
        )
    )
    baseline = HeroStrategy(
        {
            "IP_after_OOP_check": {"check": 0.5, "bet": 0.5},
            "IP_vs_OOP_bet": {"call": 0.6, "fold": 0.4, "raise": 0.0},
        }
    )
    shifted = HeroStrategy(
        {
            "IP_after_OOP_check": {"check": 0.7, "bet": 0.3},
            "IP_vs_OOP_bet": {"call": 0.4, "fold": 0.6, "raise": 0.0},
        }
    )
    candidate = HeroStrategyCandidate(
        candidate_id="multi",
        info_set=None,
        source_action=None,
        target_action=None,
        shift_amount=None,
        hero_strategy=shifted,
        l1_distance=0.8,
        shifts=(
            ShiftComponent("IP_after_OOP_check", "bet", "check", 0.2),
            ShiftComponent("IP_vs_OOP_bet", "call", "fold", 0.2),
        ),
    )
    villain = VillainStrategy({"OOP_first": {"check": c, "bet": 1.0 - c}})

    result = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
    )
    # This is a single joint P1 distribution over one hand, not v0's earliest
    # local-component composition.
    expected = c * _KL_LOCAL_07_03 + (1.0 - c) * _KL_LOCAL_04_06_FROM_06_04
    assert _KL_LOCAL_04_06_FROM_06_04 == pytest.approx(0.08109302162163289, abs=1e-12)
    assert result.kl_per_hand_nats == pytest.approx(expected, abs=1e-12)
    assert result.kl_per_hand_nats == pytest.approx(0.08139048584248761, abs=1e-12)


def test_reach_weighted_v1_test_e_baseline_impossible_event_uses_q():
    tree = _single_hand_detection_tree(hero_actions=("call", "fold", "raise"))
    baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5, "raise": 0.0}})
    candidate = HeroStrategyCandidate(
        candidate_id="raise",
        info_set="IP_vs_bet",
        source_action="fold",
        target_action="raise",
        shift_amount=0.125,
        hero_strategy=HeroStrategy(
            {"IP_vs_bet": {"call": 0.5, "fold": 0.375, "raise": 0.125}}
        ),
        l1_distance=0.25,
    )
    villain = VillainStrategy({"OOP": {"check": 0.0, "bet": 1.0}})

    result = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
    )
    assert result.baseline_impossible_mass_per_hand == pytest.approx(0.125)
    assert result.kl_per_hand_nats == math.inf
    assert result.t_detect_hands == 8
    assert result.detection_time_basis == "baseline_impossible_event"


def test_reach_weighted_v1_test_f_chain_rule_oracle_private_helper():
    tree, baseline, candidate, villain, terminal_reveals = _matrix_detection_fixture()
    actions = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="actions_only",
    )
    showdown = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="showdown_reveal",
        terminal_reveals=terminal_reveals,
    )
    full_p0, full_p1 = _candidate_observation_distributions(
        tree,
        baseline,
        candidate.hero_strategy,
        villain,
        observation_model=_OBSERVATION_MODEL_TERMINAL_PATH,
    )
    full = calculate_reach_weighted_detection_time_from_distributions(
        full_p0,
        full_p1,
        log_likelihood_threshold=_THETA,
        observation_model=_OBSERVATION_MODEL_TERMINAL_PATH,
    )

    assert full.kl_per_hand_nats == pytest.approx(_KL_LOCAL_07_03, abs=1e-12)
    assert full.kl_per_hand_nats == pytest.approx(
        0.5 * _KL_LOCAL_07_03 + 0.5 * _KL_LOCAL_07_03, abs=1e-12
    )
    assert actions.kl_per_hand_nats <= showdown.kl_per_hand_nats <= full.kl_per_hand_nats


def test_reach_weighted_v1_test_g_single_hand_showdown_degenerates_and_identical():
    tree = _single_hand_detection_tree()
    baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5}})
    candidate = _single_shift_candidate({"call": 0.7, "fold": 0.3})
    villain = VillainStrategy({"OOP": {"check": 0.5, "bet": 0.5}})
    terminal_reveals = {"T_check": (), "T_bet_call": (), "T_bet_fold": None}

    actions = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="actions_only",
    )
    showdown = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        candidate,
        villain,
        log_likelihood_threshold=_THETA,
        observation_model="showdown_reveal",
        terminal_reveals=terminal_reveals,
    )
    assert showdown.to_dict() | {"observation_model": "actions_only"} == actions.to_dict()

    identical = _single_shift_candidate({"call": 0.5, "fold": 0.5})
    same = calculate_candidate_reach_weighted_detection(
        tree,
        baseline,
        identical,
        villain,
        log_likelihood_threshold=_THETA,
    )
    assert same.total_variation_per_hand == pytest.approx(0.0)
    assert same.kl_per_hand_nats == pytest.approx(0.0)
    assert same.t_detect_hands is None


def test_reach_weighted_v1_test_h_p0_p1_distributions_sum_to_one():
    single_tree = _single_hand_detection_tree()
    single_baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5}})
    single_candidate = _single_shift_candidate({"call": 0.7, "fold": 0.3})
    single_villain = VillainStrategy({"OOP": {"check": 0.5, "bet": 0.5}})
    single_reveals = {"T_check": (), "T_bet_call": (), "T_bet_fold": None}

    matrix_tree, matrix_baseline, matrix_candidate, matrix_villain, matrix_reveals = (
        _matrix_detection_fixture()
    )

    cases = (
        (single_tree, single_baseline, single_candidate, single_villain, "actions_only", None),
        (
            single_tree,
            single_baseline,
            single_candidate,
            single_villain,
            "showdown_reveal",
            single_reveals,
        ),
        (matrix_tree, matrix_baseline, matrix_candidate, matrix_villain, "actions_only", None),
        (
            matrix_tree,
            matrix_baseline,
            matrix_candidate,
            matrix_villain,
            "showdown_reveal",
            matrix_reveals,
        ),
    )
    for tree, baseline, candidate, villain, observation_model, terminal_reveals in cases:
        p0, p1 = _candidate_observation_distributions(
            tree,
            baseline,
            candidate.hero_strategy,
            villain,
            observation_model=observation_model,
            terminal_reveals=terminal_reveals,
        )
        assert math.fsum(p0.values()) == pytest.approx(1.0, abs=1e-12)
        assert math.fsum(p1.values()) == pytest.approx(1.0, abs=1e-12)


def test_reach_weighted_v1_requires_showdown_reveals_and_checks_terminal_limit():
    tree = _single_hand_detection_tree()
    baseline = HeroStrategy({"IP_vs_bet": {"call": 0.5, "fold": 0.5}})
    candidate = _single_shift_candidate({"call": 0.7, "fold": 0.3})
    villain = VillainStrategy({"OOP": {"check": 0.5, "bet": 0.5}})

    with pytest.raises(ValueError, match="terminal_reveals"):
        calculate_candidate_reach_weighted_detection(
            tree,
            baseline,
            candidate,
            villain,
            log_likelihood_threshold=_THETA,
            observation_model="showdown_reveal",
        )

    with pytest.raises(ValueError, match="max_detection_terminals"):
        calculate_candidate_reach_weighted_detection(
            tree,
            baseline,
            candidate,
            villain,
            log_likelihood_threshold=_THETA,
            max_detection_terminals=2,
        )
