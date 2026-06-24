"""Worked example: a consolidated per-candidate analysis report.

Reuses the nuts-chop river tree and the baseline mixed Villain strategy.  It
generates Hero candidates, compares them against the baseline profile, then
builds a single consolidated analysis report that bundles, per candidate, the
fixed-baseline EVs, the selection labels (eligible / excluded / minimum-Villain-
EV / Pareto frontier), and the adaptation deadline ``T_deadline``.

The report is JSON-serialisable; this example prints its summary rows.  No
candidate is auto-selected.
"""

from __future__ import annotations

import json

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    build_candidate_analysis_report,
    compare_candidates,
    generate_shift_candidates,
)


def main() -> None:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()

    candidates = generate_shift_candidates(tree, baseline_hero, shift_amounts=[0.1, 0.2])
    comparison_report = compare_candidates(
        tree, baseline_hero, baseline_villain, candidates
    )

    report = build_candidate_analysis_report(
        comparison_report,
        horizon=5,
        discount=1.0,
        response_mode="worst",
        profit_tolerance=-2.0,
        max_l1_distance=0.3,
    )

    print("# Baseline profile value:")
    print(json.dumps(report.baseline_value.to_dict(), indent=2))

    print("\n# Summary counts:")
    print(json.dumps(report.summary_counts.to_dict(), indent=2))

    print(f"\n# {len(report.rows)} candidate summary row(s):")
    print(json.dumps(report.summary_rows(), indent=2))


if __name__ == "__main__":
    main()
