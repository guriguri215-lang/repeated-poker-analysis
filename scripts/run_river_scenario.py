#!/usr/bin/env python3
"""Build and report a river steal game from a JSON scenario file.

Usage:

    python scripts/run_river_scenario.py examples/scenarios/nuts_chop_steal_bet98.json

It prints the scenario id, the terminal EVs, the single-hand baseline profile
(OOP/IP pure actions and Hero/Villain EVs), the locked-call response, the
generated candidate count, and (when repeated horizons are given) the
``T_deadline`` table for the locked-call commitment. It uses only the package
and the standard library, and it does not write any file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    HeroStrategy,
    build_river_steal_game_from_scenario,
    calculate_adaptation_deadline,
    evaluate_fixed_profile,
    generate_shift_candidates,
    iter_terminals,
    load_river_scenario_json,
    solve_exact_response,
)

_LOCKED_CALL = HeroStrategy({"IP_vs_bet": {"call": 1.0, "fold": 0.0}})
_OOP_INFO_SET = "OOP_river"
_IP_INFO_SET = "IP_vs_bet"


def _pure_action(distribution) -> str:
    """Return the single action with probability 1 in a pure distribution."""

    for action, probability in distribution.items():
        if probability == 1.0:
            return action
    # Fall back to the highest-probability action for non-pure inputs.
    return max(distribution, key=distribution.get)


def _print_terminal_evs(tree) -> None:
    print("Terminal EVs (Hero / Villain):")
    for terminal in iter_terminals(tree.root):
        print(
            f"  {terminal.node_id}: "
            f"{terminal.hero_ev:+.4f} / {terminal.villain_ev:+.4f}"
        )


def _print_one_shot_baseline(build, baseline_value) -> None:
    oop_action = _pure_action(
        build.baseline_villain_strategy.probabilities[_OOP_INFO_SET]
    )
    ip_action = _pure_action(
        build.baseline_hero_strategy.probabilities[_IP_INFO_SET]
    )
    print(
        f"One-shot baseline: OOP {oop_action} / IP {ip_action} "
        f"(Hero {baseline_value.hero_ev:+.4f} / Villain {baseline_value.villain_ev:+.4f})"
    )


def _print_locked_call_response(build) -> None:
    response = solve_exact_response(build.tree, _LOCKED_CALL)
    oop_action = response.best_response_strategies[0][_OOP_INFO_SET]
    print(f"Locked-call response: OOP {oop_action}")


def _print_deadlines(build, baseline_hero_ev: float) -> None:
    repeated = build.repeated
    if repeated is None or not repeated.horizons:
        return
    pre_adaptation = evaluate_fixed_profile(
        build.tree, _LOCKED_CALL, build.baseline_villain_strategy
    ).hero_ev
    post_adaptation = solve_exact_response(build.tree, _LOCKED_CALL).ev_h_worst
    print(
        "T_deadline for the locked-call commitment "
        f"(baseline Hero EV {baseline_hero_ev:+.4f}, "
        f"pre-adaptation Hero EV {pre_adaptation:+.4f}, "
        f"post-adaptation Hero EV {post_adaptation:+.4f}, "
        f"discount {repeated.discount}):"
    )
    for horizon in repeated.horizons:
        result = calculate_adaptation_deadline(
            baseline_hero_ev=baseline_hero_ev,
            pre_adaptation_hero_ev=pre_adaptation,
            post_adaptation_hero_ev=post_adaptation,
            horizon=horizon,
            discount=repeated.discount,
        )
        print(f"  N={horizon}: T_deadline={result.t_deadline}")
    print(
        "Note: T_deadline is the break-even horizon for the commitment, "
        "not T_detect; it does not predict real human learning."
    )


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: python scripts/run_river_scenario.py <scenario.json>")
        return 2

    scenario = load_river_scenario_json(argv[1])
    build = build_river_steal_game_from_scenario(scenario)

    print(f"scenario_id: {build.scenario_id}")
    if build.description:
        print(f"description: {build.description}")
    _print_terminal_evs(build.tree)

    baseline_value = evaluate_fixed_profile(
        build.tree, build.baseline_hero_strategy, build.baseline_villain_strategy
    )
    print(
        f"Baseline profile EV (Hero / Villain): "
        f"{baseline_value.hero_ev:+.4f} / {baseline_value.villain_ev:+.4f}"
    )
    _print_one_shot_baseline(build, baseline_value)
    _print_locked_call_response(build)

    if build.shift_amounts:
        candidates = generate_shift_candidates(
            build.tree, build.baseline_hero_strategy, build.shift_amounts
        )
        print(f"Generated candidates: {len(candidates)}")
    else:
        print("Generated candidates: 0 (no shift amounts in scenario)")

    _print_deadlines(build, baseline_value.hero_ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
