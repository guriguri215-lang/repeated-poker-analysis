"""Tests for M2-T2: multi-information-set shift candidates + Pareto frontier.

Covers, in order:

* generation of the simultaneous two-information-set shift candidates and the
  ``generate_candidate_library`` breadth switch (backward compatible at k=1);
* the multi-shift-aware observation distance and local detection;
* the ``allowed_info_sets`` filter over every changed information set;
* the ``(post_response_hero_ev_worst, observation distance, T_deadline)``
  trade-off Pareto frontier; and
* the scenario-input / report / export wiring.
"""

import json
from pathlib import Path

import pytest

from repeated_poker import (
    build_candidate_analysis_report,
    candidate_observation_distance,
    calculate_candidate_local_detection,
    compare_candidates,
    ev_observation_deadline_pareto_ids,
    filter_candidates,
    generate_candidate_library,
    generate_multi_shift_candidates,
    generate_shift_candidates,
)
from repeated_poker.scenario_io import (
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)
from repeated_poker.scenario_pipeline import (
    RiverScenarioAnalysisConfig,
    run_river_scenario_analysis,
)
from repeated_poker.report_export import analysis_result_to_dict

_ROOT = Path(__file__).resolve().parents[1]
_BETTING_TREE_SAMPLE = (
    _ROOT / "examples" / "scenarios" / "range_equity_betting_tree_bet98.json"
)


# ---------------------------------------------------------------------------
# Fixtures: a small two-Hero-information-set tree
# ---------------------------------------------------------------------------


def _two_info_set_scenario(**overrides) -> dict:
    """A Hero-range scenario with two Hero information sets (IP_vs_bet::a / ::b)."""

    data = {
        "scenario_id": "two_info_set",
        "rake": {"rate": 0.0, "cap": None},
        "initial_commitment": {"hero": 1.0, "villain": 1.0},
        "bet_size": 2.0,
        "hero_range": [
            {
                "hand_id": "a",
                "weight": 0.5,
                "showdown": "hero",
                "baseline_strategy": {"call": 0.5, "fold": 0.5},
            },
            {
                "hand_id": "b",
                "weight": 0.5,
                "showdown": "villain",
                "baseline_strategy": {"call": 0.5, "fold": 0.5},
            },
        ],
        "candidate_generation": {"shift_amounts": [0.25]},
        "repeated": {"horizons": [10], "discount": 1.0},
    }
    data.update(overrides)
    return data


def _two_info_set_build():
    return build_river_steal_game_from_scenario(
        river_scenario_from_dict(_two_info_set_scenario())
    )


# ---------------------------------------------------------------------------
# 1. Generation
# ---------------------------------------------------------------------------


def test_multi_shift_changes_exactly_two_info_sets():
    build = _two_info_set_build()
    baseline = build.baseline_hero_strategy
    multis = generate_multi_shift_candidates(build.tree, baseline, [0.25], num_info_sets=2)

    assert multis, "expected at least one multi-shift candidate"
    for candidate in multis:
        assert candidate.is_multi_shift
        assert candidate.info_set is None
        assert candidate.source_action is None
        assert candidate.shift_amount is None
        assert len(candidate.shifts) == 2
        # Exactly the two information sets differ from the baseline.
        changed = [
            info_set
            for info_set, dist in candidate.hero_strategy.probabilities.items()
            if dist != baseline.probabilities[info_set]
        ]
        assert set(changed) == set(candidate.info_sets)
        assert len(set(candidate.info_sets)) == 2


def test_multi_shift_candidate_id_and_l1_compose_components():
    build = _two_info_set_build()
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    candidate = multis[0]
    # candidate_id is the two component ids joined by " + " in info-set order.
    expected_id = " + ".join(component.component_id for component in candidate.shifts)
    assert candidate.candidate_id == expected_id
    assert candidate.info_sets[0] < candidate.info_sets[1]
    # Disjoint information sets, so the L1 distance is additive: two 0.25 shifts,
    # each moving 0.25 out and 0.25 in => component L1 0.5 each => total 1.0.
    assert candidate.l1_distance == pytest.approx(1.0)


def test_multi_shift_ids_unique():
    build = _two_info_set_build()
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    ids = [candidate.candidate_id for candidate in multis]
    assert len(ids) == len(set(ids))


def test_generate_candidate_library_k1_is_exactly_single_shift():
    build = _two_info_set_build()
    singles = generate_shift_candidates(build.tree, build.baseline_hero_strategy, [0.25])
    library = generate_candidate_library(
        build.tree, build.baseline_hero_strategy, [0.25], max_simultaneous_info_sets=1
    )
    assert [c.candidate_id for c in library] == [c.candidate_id for c in singles]
    assert all(not c.is_multi_shift for c in library)


def test_generate_candidate_library_k2_appends_multi_shifts():
    build = _two_info_set_build()
    singles = generate_shift_candidates(build.tree, build.baseline_hero_strategy, [0.25])
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    library = generate_candidate_library(
        build.tree, build.baseline_hero_strategy, [0.25], max_simultaneous_info_sets=2
    )
    assert len(library) == len(singles) + len(multis)
    # Single shifts keep their original order at the front.
    assert [c.candidate_id for c in library[: len(singles)]] == [
        c.candidate_id for c in singles
    ]
    ids = [c.candidate_id for c in library]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("bad", [0, 3, -1, True, 1.0, "2"])
def test_generate_candidate_library_rejects_bad_breadth(bad):
    build = _two_info_set_build()
    with pytest.raises(ValueError):
        generate_candidate_library(
            build.tree, build.baseline_hero_strategy, [0.25], max_simultaneous_info_sets=bad
        )


@pytest.mark.parametrize("bad", [1, 3, True])
def test_generate_multi_shift_only_supports_two(bad):
    build = _two_info_set_build()
    with pytest.raises(ValueError):
        generate_multi_shift_candidates(
            build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=bad
        )


def test_multi_shift_respects_max_candidates_cap():
    build = _two_info_set_build()
    with pytest.raises(ValueError, match="max_candidates"):
        generate_multi_shift_candidates(
            build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2, max_candidates=1
        )


# ---------------------------------------------------------------------------
# 2. Observation distance and detection
# ---------------------------------------------------------------------------


def test_observation_distance_single_shift_is_local_tv():
    build = _two_info_set_build()
    singles = generate_shift_candidates(build.tree, build.baseline_hero_strategy, [0.25])
    single = singles[0]
    distance = candidate_observation_distance(build.baseline_hero_strategy, single)
    # baseline {call 0.5, fold 0.5} vs a 0.25 shift => TV = 0.25.
    assert distance == pytest.approx(0.25)


def test_observation_distance_multi_shift_is_max_over_info_sets():
    build = _two_info_set_build()
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    distance = candidate_observation_distance(build.baseline_hero_strategy, multis[0])
    # Both changed information sets have TV 0.25, so the max is 0.25.
    assert distance == pytest.approx(0.25)


def test_local_detection_multi_shift_picks_earliest():
    build = _two_info_set_build()
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    result = calculate_candidate_local_detection(
        build.baseline_hero_strategy, multis[0], log_likelihood_threshold=2.0
    )
    # Both changed sets are equally distinguishable here, so a valid finite
    # estimate is returned (the earliest-detected set).
    assert result.required_observations is not None
    assert result.required_observations >= 1


# ---------------------------------------------------------------------------
# 3. Filtering over every changed information set
# ---------------------------------------------------------------------------


def test_allowed_info_sets_filter_requires_all_changed_sets():
    build = _two_info_set_build()
    multis = generate_multi_shift_candidates(
        build.tree, build.baseline_hero_strategy, [0.25], num_info_sets=2
    )
    candidate = multis[0]
    # Allow only one of the two changed information sets -> excluded.
    result = filter_candidates(
        [candidate], allowed_info_sets={candidate.info_sets[0]}
    )
    assert result.summary_counts.kept == 0
    # Allow both -> kept.
    result = filter_candidates(
        [candidate], allowed_info_sets=set(candidate.info_sets)
    )
    assert result.summary_counts.kept == 1


# ---------------------------------------------------------------------------
# 4. Trade-off Pareto frontier
# ---------------------------------------------------------------------------


def test_pareto_frontier_directions():
    # (id, post_response_hero_ev_worst[higher better], observation[lower better],
    #  t_deadline[higher better]).
    objectives = [
        ("dominant", 1.0, 0.1, 10),  # best on every axis
        ("dominated", 0.0, 0.5, 2),  # worse on every axis -> off frontier
        ("tradeoff", 0.5, 0.05, 3),  # lowest observation -> non-dominated
    ]
    frontier = ev_observation_deadline_pareto_ids(objectives, tolerance=1e-9)
    assert frontier == {"dominant", "tradeoff"}


def test_pareto_frontier_none_deadline_is_worst_on_that_axis():
    objectives = [
        ("finite", 0.5, 0.2, 5),
        ("no_deadline", 0.5, 0.2, None),  # equal EV/obs, worse deadline
    ]
    frontier = ev_observation_deadline_pareto_ids(objectives)
    assert frontier == {"finite"}


def test_pareto_frontier_ties_are_all_retained():
    objectives = [
        ("x", 0.5, 0.2, 5),
        ("y", 0.5, 0.2, 5),
    ]
    assert ev_observation_deadline_pareto_ids(objectives) == {"x", "y"}


def test_pareto_frontier_is_none_when_observation_missing():
    objectives = [
        ("x", 0.5, None, 5),
        ("y", 0.4, 0.2, 3),
    ]
    assert ev_observation_deadline_pareto_ids(objectives) is None


# ---------------------------------------------------------------------------
# 5. Report + scenario-input wiring
# ---------------------------------------------------------------------------


def test_scenario_parses_max_simultaneous_info_sets():
    scenario = river_scenario_from_dict(
        _two_info_set_scenario(
            candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": 2}
        )
    )
    assert scenario.max_simultaneous_info_sets == 2
    build = build_river_steal_game_from_scenario(scenario)
    assert build.max_simultaneous_info_sets == 2


def test_scenario_defaults_to_single_info_set():
    scenario = river_scenario_from_dict(_two_info_set_scenario())
    assert scenario.max_simultaneous_info_sets == 1


@pytest.mark.parametrize("bad", [0, 3, True, 1.5, "2"])
def test_scenario_rejects_bad_max_simultaneous_info_sets(bad):
    with pytest.raises(ValueError, match="max_simultaneous_info_sets"):
        river_scenario_from_dict(
            _two_info_set_scenario(
                candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": bad}
            )
        )


def test_report_has_multi_shift_rows_and_new_fields():
    scenario = river_scenario_from_dict(
        _two_info_set_scenario(
            candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": 2}
        )
    )
    result = run_river_scenario_analysis(scenario, RiverScenarioAnalysisConfig())
    report = result.pipeline_result.analysis_report

    multi_rows = [row for row in report.rows if row.info_set is None]
    single_rows = [row for row in report.rows if row.info_set is not None]
    assert multi_rows and single_rows

    for row in report.rows:
        assert row.observation_distance is not None
        assert row.is_ev_observation_deadline_pareto_candidate is not None
        assert row.shifts  # always populated
    for row in multi_rows:
        assert len(row.shifts) == 2
        assert row.source_action is None
    # The new frontier count is an int (baseline available) and is reflected.
    assert report.summary_counts.ev_observation_deadline_pareto_frontier is not None


def test_single_info_set_scenario_reports_no_multi_shift_rows():
    scenario = river_scenario_from_dict(_two_info_set_scenario())  # default k=1
    result = run_river_scenario_analysis(scenario, RiverScenarioAnalysisConfig())
    report = result.pipeline_result.analysis_report
    assert all(row.info_set is not None for row in report.rows)
    assert all(not row.shifts or len(row.shifts) == 1 for row in report.rows)


def test_json_export_exposes_new_fields():
    scenario = river_scenario_from_dict(
        _two_info_set_scenario(
            candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": 2}
        )
    )
    result = run_river_scenario_analysis(scenario, RiverScenarioAnalysisConfig())
    payload = analysis_result_to_dict(result)
    rows = payload["analysis_report"]["candidate_rows"]
    assert any(len(row["shifts"]) == 2 for row in rows)
    for row in rows:
        assert "observation_distance" in row
        assert "shifts" in row
        assert "is_ev_observation_deadline_pareto_candidate" in row
    counts = payload["analysis_report"]["summary_counts"]
    assert "ev_observation_deadline_pareto_frontier" in counts


def test_markdown_summary_renders_multi_shift_and_new_count():
    scenario = river_scenario_from_dict(
        _two_info_set_scenario(
            candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": 2}
        )
    )
    result = run_river_scenario_analysis(scenario, RiverScenarioAnalysisConfig())
    markdown = result.pipeline_result.markdown_summary
    assert "ev_observation_deadline_pareto_frontier" in markdown
    assert "observation_distance" in markdown
    # A multi-shift row shows both changed information sets joined with " + ".
    assert " + " in markdown


@pytest.mark.parametrize(
    "from_dict_name",
    [
        "single_hand_form_from_dict",
        "hero_range_form_from_dict",
        "showdown_matrix_form_from_dict",
        "equity_matrix_form_from_dict",
        "betting_tree_form_from_dict",
    ],
)
def test_scenario_forms_reject_multi_shift_generation(from_dict_name):
    from repeated_poker import scenario_form as sf

    data = _two_info_set_scenario(
        candidate_generation={"shift_amounts": [0.25], "max_simultaneous_info_sets": 2}
    )
    from_dict = getattr(sf, from_dict_name)
    with pytest.raises(ValueError, match="max_simultaneous_info_sets"):
        from_dict(data)


def test_betting_tree_scenario_supports_multi_shift():
    data = json.loads(_BETTING_TREE_SAMPLE.read_text(encoding="utf-8"))
    data["candidate_generation"]["max_simultaneous_info_sets"] = 2
    result = run_river_scenario_analysis(
        river_scenario_from_dict(data), RiverScenarioAnalysisConfig()
    )
    report = result.pipeline_result.analysis_report
    assert any(row.info_set is None for row in report.rows)
