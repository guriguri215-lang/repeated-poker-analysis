"""DP-vs-enumeration equivalence, property, and performance tests.

The backward-induction (``method="dp"``) engine must reproduce the v0
enumerator (``method="enumerate"``, kept as the oracle) on every tree the
enumerator can handle: same Villain max EV, same Hero-EV interval, same
action variation, same off-path classification, and the same materialised
best-response correspondence.  Random trees are generated with dyadic
(exactly representable) probabilities and payoffs so that Villain-EV ties are
exact and both tolerance conventions agree.
"""

import random
import time
from pathlib import Path

import pytest

from nuts_chop_river import build_nuts_chop_river
from nuts_chop_river import default_hero_strategy as nuts_chop_hero_strategy
from value_bluff_river import (
    build_value_bluff_river,
    default_hero_strategy as value_bluff_hero_strategy,
    indifference_hero_strategy,
)
from repeated_poker import (
    build_river_steal_game_from_scenario,
    count_villain_pure_strategies,
    generate_shift_candidates,
    load_river_scenario_json,
    river_scenario_from_dict,
    run_river_scenario_analysis,
    solve_exact_response,
)
from repeated_poker import exact_response as exact_response_module
from repeated_poker.exact_response import (
    DEFAULT_MAX_PURE_STRATEGIES,
    _expected_payoff,
)
from repeated_poker.game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    collect_villain_info_sets,
    validate_tree,
)
from repeated_poker.scenario_pipeline import RiverScenarioAnalysisConfig

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO_PATHS = sorted((_ROOT / "examples" / "scenarios").glob("*.json"))

TOL = 1e-9


# ---------------------------------------------------------------------------
# Shared comparison helper
# ---------------------------------------------------------------------------


def _assert_methods_agree(tree, hero_strategy):
    """Solve with both methods and require the full report to match."""

    enum = solve_exact_response(tree, hero_strategy, method="enumerate")
    dp = solve_exact_response(tree, hero_strategy, method="dp")

    assert dp.villain_max_ev == pytest.approx(enum.villain_max_ev, abs=1e-9)
    assert dp.ev_h_worst == pytest.approx(enum.ev_h_worst, abs=1e-9)
    assert dp.ev_h_best == pytest.approx(enum.ev_h_best, abs=1e-9)
    assert dp.expected_house_rake_worst == pytest.approx(
        enum.expected_house_rake_worst, abs=1e-9
    )
    assert dp.expected_house_rake_best == pytest.approx(
        enum.expected_house_rake_best, abs=1e-9
    )
    assert dp.best_response_action_variation == enum.best_response_action_variation
    assert dp.off_path_info_sets == enum.off_path_info_sets
    assert dp.num_villain_pure_strategies == enum.num_villain_pure_strategies
    assert dp.num_best_response_strategies == enum.num_best_response_strategies
    # Both materialise the correspondence here (small trees); the DP must
    # reproduce the enumerator's list, including its order.
    assert dp.best_response_strategies == enum.best_response_strategies
    assert len(dp.best_response_strategies) == dp.num_best_response_strategies
    return enum, dp


# ---------------------------------------------------------------------------
# Random-tree generation (dyadic probabilities and payoffs)
# ---------------------------------------------------------------------------

# Dyadic values are exactly representable in binary floating point, so ties
# between Villain EVs are exact and shared between both methods.
_HERO_PAYOFFS = (-1.0, -0.5, 0.0, 0.5, 1.0)
_RAKES = (0.0, 0.0, 0.0, 0.5)

_DISTRIBUTIONS = {
    2: ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.75, 0.25), (0.25, 0.75)),
    3: (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.25, 0.25),
        (0.25, 0.25, 0.5),
    ),
}

# Chance outcomes always carry positive probability so that off-path Villain
# information sets arise from Hero's strategy, as in the scenario builders.
_CHANCE_PROBS = {
    2: ((0.5, 0.5), (0.75, 0.25), (0.25, 0.75)),
    3: ((0.5, 0.25, 0.25), (0.25, 0.5, 0.25)),
}


def _random_skeleton(rng, depth):
    """A shared decision skeleton instantiated under each top chance branch."""

    if depth == 0 or rng.random() < 0.3:
        return ("leaf",)
    kind = rng.choice(("hero", "villain", "chance"))
    width = 2 if kind == "villain" else rng.choice((2, 2, 3))
    return (kind, tuple(_random_skeleton(rng, depth - 1) for _ in range(width)))


def _instantiate(rng, skeleton, branch, path):
    """Build one game subtree for a top-level chance branch.

    Information sets are keyed by the skeleton position (``path``) and shared
    across the top-level chance branches, which preserves perfect recall (a
    player pools only nodes with an identical own action history) while
    exercising information sets spanning several nodes.
    """

    kind = skeleton[0]
    node_id = f"{branch}:{path}"
    if kind == "leaf":
        hero_ev = rng.choice(_HERO_PAYOFFS)
        rake = rng.choice(_RAKES)
        return TerminalNode(node_id, hero_ev, -(hero_ev + rake), rake)
    children = skeleton[1]
    if kind == "chance":
        probs = rng.choice(_CHANCE_PROBS[len(children)])
        return ChanceNode(
            node_id,
            tuple(
                (prob, _instantiate(rng, child, branch, f"{path}c{i}"))
                for i, (prob, child) in enumerate(zip(probs, children))
            ),
        )
    actions = tuple(
        (
            f"a{i}",
            _instantiate(rng, child, branch, f"{path}{kind[0]}{i}"),
        )
        for i, child in enumerate(children)
    )
    if kind == "hero":
        return HeroNode(node_id, f"H:{path}", actions)
    return VillainNode(node_id, f"V:{path}", actions)


def _collect_hero_skeleton_paths(skeleton, path, found):
    kind = skeleton[0]
    if kind == "leaf":
        return
    for i, child in enumerate(skeleton[1]):
        prefix = "c" if kind == "chance" else kind[0]
        _collect_hero_skeleton_paths(child, f"{path}{prefix}{i}", found)
    if kind == "hero":
        found[f"H:{path}"] = len(skeleton[1])


def _random_case(seed):
    """A random small game (tree + Hero strategy) with dyadic numbers."""

    rng = random.Random(seed)
    for _ in range(30):
        skeleton = _random_skeleton(rng, depth=3)
        num_branches = rng.choice((1, 2, 3))
        if num_branches == 1:
            root = _instantiate(rng, skeleton, "b0", "r")
        else:
            probs = rng.choice(_CHANCE_PROBS[num_branches])
            root = ChanceNode(
                "root",
                tuple(
                    (prob, _instantiate(rng, skeleton, f"b{i}", "r"))
                    for i, prob in enumerate(probs)
                ),
            )
        tree = GameTree(root=root)
        if count_villain_pure_strategies(tree) > 512:
            continue  # keep the enumeration oracle cheap; try a new skeleton
        hero_info_sets = {}
        _collect_hero_skeleton_paths(skeleton, "r", hero_info_sets)
        hero_strategy = HeroStrategy(
            probabilities={
                info_set: dict(
                    zip(
                        (f"a{i}" for i in range(width)),
                        rng.choice(_DISTRIBUTIONS[width]),
                    )
                )
                for info_set, width in hero_info_sets.items()
            }
        )
        validate_tree(tree)
        return tree, hero_strategy
    raise AssertionError(f"seed {seed} never produced a small enough tree")


@pytest.mark.parametrize("seed", range(150))
def test_dp_matches_enumeration_on_random_trees(seed):
    tree, hero_strategy = _random_case(seed)
    _, dp = _assert_methods_agree(tree, hero_strategy)

    # Property check against the oracle payoff walker: no Villain pure
    # strategy beats the reported maximum.
    rng = random.Random(seed + 10_000)
    villain_info_sets = collect_villain_info_sets(tree)
    for _ in range(5):
        pure = {
            info_set: rng.choice(actions)
            for info_set, actions in villain_info_sets.items()
        }
        _, villain_ev, _ = _expected_payoff(tree.root, hero_strategy, pure)
        assert villain_ev <= dp.villain_max_ev + TOL

    # Terminal payoffs satisfy hero + villain + rake == 0 by construction, so
    # the same identity holds at the reported Hero-EV extremes.
    assert dp.ev_h_worst + dp.villain_max_ev + dp.expected_house_rake_worst == (
        pytest.approx(0.0, abs=1e-9)
    )
    assert dp.ev_h_best + dp.villain_max_ev + dp.expected_house_rake_best == (
        pytest.approx(0.0, abs=1e-9)
    )


# ---------------------------------------------------------------------------
# Existing worked examples and scenario files
# ---------------------------------------------------------------------------


def test_dp_matches_enumeration_on_nuts_chop():
    tree = build_nuts_chop_river(rate=0.05, cap=3.0)
    _assert_methods_agree(tree, nuts_chop_hero_strategy())


def test_dp_matches_enumeration_on_value_bluff():
    tree = build_value_bluff_river(rate=0.0, cap=None)
    _assert_methods_agree(tree, value_bluff_hero_strategy())
    _assert_methods_agree(tree, indifference_hero_strategy())


@pytest.mark.parametrize(
    "path", _SCENARIO_PATHS, ids=lambda path: path.stem
)
def test_dp_matches_enumeration_on_example_scenarios(path):
    build = build_river_steal_game_from_scenario(load_river_scenario_json(path))
    _assert_methods_agree(build.tree, build.baseline_hero_strategy)

    # Also agree against locked candidates, which move Hero probability around
    # and can push Villain information sets on and off the path.
    candidates = generate_shift_candidates(
        build.tree, build.baseline_hero_strategy, build.shift_amounts
    )
    for candidate in candidates[:5]:
        _assert_methods_agree(build.tree, candidate.hero_strategy)


# ---------------------------------------------------------------------------
# Tolerance semantics: per-information-set (dp) vs global (enumerate)
# ---------------------------------------------------------------------------


def _chained_near_tie_tree():
    """Two stacked Villain decisions whose near-ties accumulate past tolerance.

    With the default tolerance 1e-9: at ``V2``, ``d`` is 0.6e-9 worse than
    ``c`` (a local near-tie); at ``V1``, ``b`` is 0.6e-9 worse than ``a``
    (another local near-tie).  The pure strategy ``(b, d)`` is 1.2e-9 below
    the global optimum -- outside the global tolerance band -- while every
    individual step stays inside the local band.
    """

    def leaf(node_id, villain_ev):
        return TerminalNode(node_id, -villain_ev, villain_ev, 0.0)

    v2 = VillainNode(
        "v2",
        "V2",
        (("c", leaf("t_bc", 1.0 - 0.6e-9)), ("d", leaf("t_bd", 1.0 - 1.2e-9))),
    )
    root = VillainNode("v1", "V1", (("a", leaf("t_a", 1.0)), ("b", v2)))
    return GameTree(root=root), HeroStrategy(probabilities={})


def test_chained_near_ties_pin_the_documented_method_difference():
    tree, hero_strategy = _chained_near_tie_tree()

    enum = solve_exact_response(tree, hero_strategy, method="enumerate")
    dp = solve_exact_response(tree, hero_strategy, method="dp")

    # Both methods agree on the optimum itself.
    assert enum.villain_max_ev == pytest.approx(1.0, abs=1e-12)
    assert dp.villain_max_ev == pytest.approx(1.0, abs=1e-12)

    # The enumerator's tolerance is global: (b, d) sits 1.2e-9 below the
    # optimum and is excluded. The DP's tolerance is per information set:
    # each step of (b, d) is a local near-tie, so the strategy stays in the
    # correspondence. This asymmetry is the intended, documented semantics
    # difference; it requires distinct Villain EVs within ``tolerance`` of
    # each other, which the exact ties of real payoff trees do not produce.
    assert {"V1": "b", "V2": "d"} not in enum.best_response_strategies
    assert {"V1": "b", "V2": "d"} in dp.best_response_strategies
    assert enum.num_best_response_strategies == 3
    assert dp.num_best_response_strategies == 4

    # The Hero-EV interval follows the same semantics: the DP's best end
    # includes the chained near-tie strategy, the enumerator's does not.
    assert enum.ev_h_best == pytest.approx(-1.0 + 0.6e-9, abs=1e-10)
    assert dp.ev_h_best == pytest.approx(-1.0 + 1.2e-9, abs=1e-10)
    assert enum.ev_h_worst == pytest.approx(-1.0, abs=1e-12)
    assert dp.ev_h_worst == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Perfect-recall guard
# ---------------------------------------------------------------------------


def _imperfect_recall_tree():
    """Villain reaches one information set with two different histories."""

    def leaf(node_id, hero_ev):
        return TerminalNode(node_id, hero_ev, -hero_ev, 0.0)

    inner_a = VillainNode(
        "va", "V_shared", (("x", leaf("ta_x", 1.0)), ("y", leaf("ta_y", -1.0)))
    )
    inner_b = VillainNode(
        "vb", "V_shared", (("x", leaf("tb_x", -1.0)), ("y", leaf("tb_y", 1.0)))
    )
    root = VillainNode("v_first", "V_first", (("a", inner_a), ("b", inner_b)))
    return GameTree(root=root), HeroStrategy(probabilities={})


def test_dp_rejects_imperfect_recall_but_enumeration_accepts():
    tree, hero_strategy = _imperfect_recall_tree()

    # The structural path guard passes (no info set repeats on one path), but
    # the shared info set carries two different Villain action histories.
    validate_tree(tree)
    with pytest.raises(ValueError, match="perfect-recall"):
        solve_exact_response(tree, hero_strategy, method="dp")

    result = solve_exact_response(tree, hero_strategy, method="enumerate")
    assert result.villain_max_ev == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Performance: betting-tree mode, 10 Villain buckets, 100+ candidates
# ---------------------------------------------------------------------------


def _perf_scenario_dict(num_hero=5, num_villain=10):
    hero_range = [
        {
            "hand_id": f"hero_{i}",
            "weight": 1.0 / num_hero,
            "baseline_strategies": {
                "after_oop_check": {"check": 0.6, "bet": 0.4},
                "vs_oop_bet": {"call": 0.4, "fold": 0.4, "raise": 0.2},
            },
        }
        for i in range(num_hero)
    ]
    villain_range = [
        {"hand_id": f"villain_{j}", "weight": 1.0 / num_villain}
        for j in range(num_villain)
    ]
    equity_matrix = {
        f"hero_{i}": {
            f"villain_{j}": 0.15 + 0.7 * ((5 * i + 3 * j) % 17) / 16.0
            for j in range(num_villain)
        }
        for i in range(num_hero)
    }
    return {
        "format_version": "1",
        "scenario_id": "perf_betting_tree_ten_villain_buckets",
        "description": "DP benchmark: 10 Villain buckets, 100+ candidates.",
        "rake": {"rate": 0.05, "cap": 4.0},
        "initial_commitment": {"hero": 1.0, "villain": 1.0},
        "hero_range": hero_range,
        "villain_range": villain_range,
        "equity_matrix": equity_matrix,
        "betting_tree": {
            "oop_bet_size": 98.0,
            "ip_bet_after_check_size": 98.0,
            "ip_raise_size": 196.0,
        },
        "candidate_generation": {"shift_amounts": [0.05, 0.1, 0.15]},
        "repeated": {"horizons": [10, 50], "discount": 1.0},
    }


def test_betting_tree_ten_villain_buckets_uses_dp_and_finishes_fast(monkeypatch):
    # Fail loudly if anything falls back to the enumeration path.
    def _no_enumeration(*args, **kwargs):
        raise AssertionError(
            "enumerate_villain_pure_strategies must not be called: the "
            "analysis is expected to run on the DP path"
        )

    monkeypatch.setattr(
        exact_response_module,
        "enumerate_villain_pure_strategies",
        _no_enumeration,
    )

    scenario = river_scenario_from_dict(_perf_scenario_dict())

    start = time.perf_counter()
    build = build_river_steal_game_from_scenario(scenario)
    # 3 Villain information sets per bucket, 2 actions each: the space is far
    # beyond the default enumeration limit, so completing at default settings
    # is itself evidence that the DP path was used.
    assert count_villain_pure_strategies(build.tree) == 8**10
    assert 8**10 > DEFAULT_MAX_PURE_STRATEGIES

    result = run_river_scenario_analysis(
        scenario, RiverScenarioAnalysisConfig(markdown=False)
    )
    elapsed = time.perf_counter() - start

    rows = result.pipeline_result.analysis_report.rows
    assert len(rows) >= 100
    # "A few seconds" acceptance with headroom for slow CI machines.
    assert elapsed < 15.0, f"analysis took {elapsed:.1f}s, expected a few seconds"

    baseline_response = solve_exact_response(
        build.tree, build.baseline_hero_strategy
    )
    assert baseline_response.num_villain_pure_strategies == 8**10
    # The representative best response assigns an action to every info set.
    assert set(baseline_response.best_response_strategies[0]) == set(
        collect_villain_info_sets(build.tree)
    )
