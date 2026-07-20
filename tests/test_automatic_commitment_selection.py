"""Focused tests for bounded automatic conditional commitment selection."""

import json
import math
from dataclasses import FrozenInstanceError, replace

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

import repeated_poker.automatic_commitment_selection as automatic_module
import repeated_poker.pipeline as pipeline_module
from repeated_poker import (
    BestResponseResult,
    CandidateComparison,
    CandidateComparisonReport,
    CandidateFilterConfig,
    CandidateGenerationConfig,
    FixedProfileValue,
    HeroStrategy,
    HeroStrategyCandidate,
    run_candidate_analysis_pipeline,
)
from repeated_poker.automatic_commitment_selection import (
    AUTOMATIC_COMMITMENT_CLAIM_SCOPE,
    AUTOMATIC_COMMITMENT_RESPONSE_SEMANTICS,
    AUTOMATIC_COMMITMENT_SELECTION_CONTRACT_VERSION,
    AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED,
    NO_BENEFICIAL_COMMITMENT,
    AutomaticCommitmentSearchCoverage,
    AutomaticCommitmentSelectionConfig,
    select_automatic_commitments,
)


def _best_response(worst: float, best: float | None = None) -> BestResponseResult:
    if best is None:
        best = worst
    return BestResponseResult(
        villain_max_ev=0.0,
        best_response_strategies=[],
        ev_h_worst=worst,
        ev_h_best=best,
        expected_house_rake_worst=0.0,
        expected_house_rake_best=0.0,
        best_response_action_variation={},
        off_path_info_sets=[],
        num_villain_pure_strategies=0,
    )


def _comparison(
    candidate_id: str,
    *,
    pre: float,
    worst: float,
    best: float | None = None,
    post_worst_diff: float | None = None,
    l1_distance: float = 0.2,
    robustly_profitable: bool | None = None,
) -> CandidateComparison:
    if best is None:
        best = worst
    if post_worst_diff is None:
        post_worst_diff = worst
    if robustly_profitable is None:
        robustly_profitable = worst > 0.0
    candidate = HeroStrategyCandidate(
        candidate_id=candidate_id,
        info_set="H",
        source_action="a",
        target_action="b",
        shift_amount=0.1,
        hero_strategy=HeroStrategy({}),
        l1_distance=l1_distance,
    )
    return CandidateComparison(
        candidate=candidate,
        fixed_profile_value=FixedProfileValue(pre, 0.0, 0.0),
        villain_ev_diff_from_baseline=0.0,
        hero_ev_diff_from_baseline=pre,
        best_response=_best_response(worst, best),
        post_response_hero_ev_worst_diff=post_worst_diff,
        post_response_hero_ev_best_diff=best,
        robustly_profitable=robustly_profitable,
    )


def _report(comparisons, baseline: float = 0.0) -> CandidateComparisonReport:
    return CandidateComparisonReport(
        baseline_value=FixedProfileValue(baseline, 0.0, 0.0),
        comparisons=list(comparisons),
    )


def _run_pipeline(**overrides):
    kwargs = {
        "generation": CandidateGenerationConfig(shift_amounts=[0.1, 0.2]),
        "horizon": 5,
        "render_markdown": False,
    }
    kwargs.update(overrides)
    return run_candidate_analysis_pipeline(
        build_nuts_chop_river(),
        default_hero_strategy(),
        baseline_villain_strategy(),
        **kwargs,
    )


def test_crossing_winner_fixture_switches_and_includes_m_n_plus_one():
    # b=0, N=3. A: a=4,l=-1 -> deltas [-3,2,7,12].
    # B: a=1,l=1 -> deltas [3,3,3,3].
    report = _report(
        [
            _comparison("A", pre=4.0, worst=-1.0),
            _comparison("B", pre=1.0, worst=1.0),
        ]
    )

    result = select_automatic_commitments(report, horizon=3)

    assert [row.adaptation_opportunity for row in result.rows] == [1, 2, 3, 4]
    assert [row.selected_candidate_id for row in result.rows] == ["B", "B", "A", "A"]
    assert [row.best_total_hero_ev_delta for row in result.rows] == pytest.approx(
        [3.0, 3.0, 7.0, 12.0]
    )


def test_discounted_oracle_matches_existing_repeated_formula():
    # b=1, a=3, l=0, N=3, delta=.5: baseline total=1.75 and locked
    # totals are [0,3,4.5,5.25].
    result = select_automatic_commitments(
        _report([_comparison("c", pre=3.0, worst=0.0)], baseline=1.0),
        horizon=3,
        discount=0.5,
    )

    assert [row.best_total_hero_ev_delta for row in result.rows] == pytest.approx(
        [-1.75, 1.25, 2.75, 3.5]
    )
    assert result.rows[0].status == NO_BENEFICIAL_COMMITMENT
    assert result.rows[0].selected_candidate_id is None
    assert all(
        row.status == AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED
        for row in result.rows[1:]
    )


def test_exact_response_hero_worst_is_used_not_hero_best():
    report = _report(
        [
            _comparison("tempting-best", pre=0.0, worst=-1.0, best=100.0),
            _comparison("safe-worst", pre=0.0, worst=2.0, best=2.0),
        ]
    )

    result = select_automatic_commitments(report, horizon=1)

    assert result.rows[0].selected_candidate_id == "safe-worst"
    assert result.to_dict()["response_semantics"] == (
        AUTOMATIC_COMMITMENT_RESPONSE_SEMANTICS
    )


def test_search_uses_all_comparisons_not_eligible_or_pareto_labels():
    # This candidate is explicitly labelled non-robust and would be excluded by
    # the existing selection screen, but its large pre-adaptation value wins the
    # m=N+1 conditional row and must remain in the M27 search universe.
    excluded_by_old_screen = _comparison(
        "pre-adaptation-winner",
        pre=5.0,
        worst=-1.0,
        robustly_profitable=False,
    )
    conventional = _comparison("conventional", pre=1.0, worst=1.0)

    result = select_automatic_commitments(
        _report([excluded_by_old_screen, conventional]), horizon=2
    )

    assert result.rows[-1].selected_candidate_id == "pre-adaptation-winner"
    assert result.search_coverage.kept_candidate_ids == (
        "conventional",
        "pre-adaptation-winner",
    )


def test_primary_tie_set_is_full_and_secondary_order_is_deterministic():
    report = _report(
        [
            _comparison(
                "z", pre=1.0, worst=1.0, post_worst_diff=2.0, l1_distance=0.4
            ),
            _comparison(
                "b", pre=1.0, worst=1.0, post_worst_diff=2.0, l1_distance=0.2
            ),
            _comparison(
                "a", pre=1.0, worst=1.0, post_worst_diff=2.0, l1_distance=0.2
            ),
        ]
    )

    row = select_automatic_commitments(report, horizon=2).rows[0]

    assert row.primary_tie_candidate_ids == ("a", "b", "z")
    assert row.primary_tie_display_candidate_id == "a"
    assert row.selected_candidate_id == "a"
    evidence = {item.candidate_id: item for item in row.tie_break_evidence}
    assert evidence["z"].within_best_post_response_tolerance is True
    assert evidence["z"].within_best_l1_tolerance is False
    assert evidence["a"].is_primary_tie_display_candidate is True


def test_post_response_diff_precedes_l1_in_secondary_tie_break():
    report = _report(
        [
            _comparison(
                "far", pre=1.0, worst=1.0, post_worst_diff=2.0, l1_distance=1.0
            ),
            _comparison(
                "near", pre=1.0, worst=1.0, post_worst_diff=1.0, l1_distance=0.1
            ),
        ]
    )

    row = select_automatic_commitments(report, horizon=1, tolerance=0.0).rows[0]
    assert row.selected_candidate_id == "far"


def test_secondary_float_comparisons_use_the_primary_tolerance():
    report = _report(
        [
            _comparison(
                "a", pre=1.0, worst=1.0, post_worst_diff=1.00, l1_distance=0.5
            ),
            _comparison(
                "b", pre=1.0, worst=1.0, post_worst_diff=1.05, l1_distance=0.1
            ),
        ]
    )

    row = select_automatic_commitments(report, horizon=1, tolerance=0.1).rows[0]
    assert row.selected_candidate_id == "b"
    assert all(
        evidence.within_best_post_response_tolerance
        for evidence in row.tie_break_evidence
    )


def test_input_order_permutation_and_json_bytes_are_deterministic():
    comparisons = [
        _comparison("c", pre=2.0, worst=0.0),
        _comparison("a", pre=1.0, worst=1.0),
        _comparison("b", pre=1.5, worst=0.5),
    ]
    first = select_automatic_commitments(_report(comparisons), horizon=3)
    second = select_automatic_commitments(_report(reversed(comparisons)), horizon=3)

    assert first.to_dict() == second.to_dict()
    first_bytes = json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    second_bytes = json.dumps(
        second.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert first_bytes == second_bytes


def test_positive_threshold_is_strict_beyond_tolerance():
    report = _report([_comparison("c", pre=1.25, worst=1.25)])
    boundary = select_automatic_commitments(
        report,
        horizon=1,
        tolerance=0.25,
        configuration=AutomaticCommitmentSelectionConfig(
            minimum_total_uplift=1.0
        ),
    )
    above = select_automatic_commitments(
        _report([_comparison("c", pre=1.26, worst=1.26)]),
        horizon=1,
        tolerance=0.25,
        configuration=AutomaticCommitmentSelectionConfig(
            minimum_total_uplift=1.0
        ),
    )

    assert boundary.rows[0].status == NO_BENEFICIAL_COMMITMENT
    assert boundary.rows[0].selected_candidate_id is None
    assert boundary.rows[0].primary_tie_display_candidate_id == "c"
    assert above.rows[0].status == AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED


def test_all_non_positive_candidates_return_no_beneficial_commitment():
    result = select_automatic_commitments(
        _report(
            [
                _comparison("negative", pre=-1.0, worst=-1.0),
                _comparison("zero", pre=0.0, worst=0.0),
            ]
        ),
        horizon=2,
    )

    assert all(row.status == NO_BENEFICIAL_COMMITMENT for row in result.rows)
    assert all(row.selected_candidate_id is None for row in result.rows)


def test_empty_report_returns_n_plus_one_no_selection_rows_after_validation():
    result = select_automatic_commitments(_report([]), horizon=3, discount=0.9)

    assert len(result.rows) == 4
    assert result.timing_row_evaluation_count == 0
    assert result.search_coverage.input_candidate_ids == ()
    for opportunity, row in enumerate(result.rows, start=1):
        assert row.adaptation_opportunity == opportunity
        assert row.status == NO_BENEFICIAL_COMMITMENT
        assert row.best_total_hero_ev_delta is None
        assert row.primary_tie_candidate_ids == ()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"horizon": 0}, "horizon"),
        ({"horizon": 1, "discount": math.nan}, "discount"),
        ({"horizon": 1, "tolerance": math.inf}, "tolerance"),
        (
            {
                "horizon": 1,
                "configuration": AutomaticCommitmentSelectionConfig(
                    minimum_total_uplift=-0.1
                ),
            },
            "minimum_total_uplift",
        ),
    ],
)
def test_empty_report_still_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_automatic_commitments(_report([]), **kwargs)


def test_duplicate_candidate_ids_fail_closed():
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        select_automatic_commitments(
            _report(
                [
                    _comparison("dup", pre=1.0, worst=1.0),
                    _comparison("dup", pre=2.0, worst=2.0),
                ]
            ),
            horizon=1,
        )


def test_candidate_cap_is_checked_before_timing_materialisation(monkeypatch):
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("timing rows must not be materialised")

    monkeypatch.setattr(
        automatic_module, "calculate_candidate_adaptation_deadlines", unexpected
    )
    with pytest.raises(ValueError, match="max_candidates"):
        select_automatic_commitments(
            _report(
                [
                    _comparison("a", pre=1.0, worst=1.0),
                    _comparison("b", pre=1.0, worst=1.0),
                ]
            ),
            horizon=1,
            configuration=AutomaticCommitmentSelectionConfig(max_candidates=1),
        )
    assert called is False


def test_combined_timing_cap_is_checked_before_materialisation(monkeypatch):
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("timing rows must not be materialised")

    monkeypatch.setattr(
        automatic_module, "calculate_candidate_adaptation_deadlines", unexpected
    )
    with pytest.raises(ValueError, match="max_timing_rows"):
        select_automatic_commitments(
            _report([_comparison("a", pre=1.0, worst=1.0)]),
            horizon=2,
            configuration=AutomaticCommitmentSelectionConfig(max_timing_rows=2),
        )
    assert called is False


@pytest.mark.parametrize("field", ["baseline", "fixed", "worst", "post", "l1"])
def test_non_finite_report_scalars_fail_closed(field):
    comparison = _comparison("c", pre=1.0, worst=1.0)
    report = _report([comparison])
    if field == "baseline":
        report = replace(
            report,
            baseline_value=replace(report.baseline_value, hero_ev=math.inf),
        )
    elif field == "fixed":
        comparison = replace(
            comparison,
            fixed_profile_value=replace(
                comparison.fixed_profile_value, hero_ev=math.nan
            ),
        )
        report = _report([comparison])
    elif field == "worst":
        comparison = replace(
            comparison,
            best_response=replace(comparison.best_response, ev_h_worst=-math.inf),
        )
        report = _report([comparison])
    elif field == "post":
        report = _report(
            [replace(comparison, post_response_hero_ev_worst_diff=math.inf)]
        )
    else:
        report = _report(
            [
                replace(
                    comparison,
                    candidate=replace(comparison.candidate, l1_distance=math.nan),
                )
            ]
        )

    with pytest.raises(ValueError, match="finite"):
        select_automatic_commitments(report, horizon=1)


def test_finite_inputs_with_non_finite_derived_total_fail_without_partial_report():
    report = _report([_comparison("huge", pre=1e308, worst=1e308)])
    with pytest.raises(ValueError, match="finite"):
        select_automatic_commitments(report, horizon=2)


def test_search_coverage_must_bind_exact_kept_ids():
    coverage = AutomaticCommitmentSearchCoverage(
        input_candidate_ids=("a", "b"), kept_candidate_ids=("a",)
    )
    with pytest.raises(ValueError, match="exactly match"):
        select_automatic_commitments(
            _report([_comparison("b", pre=1.0, worst=1.0)]),
            horizon=1,
            search_coverage=coverage,
        )


def test_contract_identity_scope_and_config_are_serialised():
    result = select_automatic_commitments(
        _report([_comparison("a", pre=1.0, worst=1.0)]), horizon=1
    )
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))

    assert payload["contract_version"] == (
        AUTOMATIC_COMMITMENT_SELECTION_CONTRACT_VERSION
    )
    assert payload["claim_scope"] == AUTOMATIC_COMMITMENT_CLAIM_SCOPE
    assert payload["baseline_identity"].startswith("sha256:")
    assert payload["selection_configuration"]["max_timing_rows"] == 1_000_000


def test_baseline_identity_is_stable_and_changes_with_baseline_value():
    comparison = _comparison("a", pre=1.0, worst=1.0)
    first = select_automatic_commitments(_report([comparison], baseline=0.0), horizon=1)
    repeated = select_automatic_commitments(
        _report([comparison], baseline=0.0), horizon=1
    )
    changed = select_automatic_commitments(
        _report([comparison], baseline=0.5), horizon=1
    )

    assert first.baseline_identity == repeated.baseline_identity
    assert first.baseline_identity != changed.baseline_identity


def test_public_result_types_are_frozen():
    configuration = AutomaticCommitmentSelectionConfig()
    report = select_automatic_commitments(
        _report([_comparison("a", pre=1.0, worst=1.0)]),
        horizon=1,
        configuration=configuration,
    )
    with pytest.raises(FrozenInstanceError):
        configuration.max_candidates = 1
    with pytest.raises(FrozenInstanceError):
        report.rows[0].status = "changed"


def test_pipeline_default_and_explicit_opt_out_preserve_existing_behaviour():
    default = _run_pipeline()
    explicit = _run_pipeline(automatic_selection=None)

    assert default.automatic_selection_report is None
    assert explicit.automatic_selection_report is None
    assert default.analysis_report.to_dict() == explicit.analysis_report.to_dict()
    assert default.comparison_report == explicit.comparison_report


def test_pipeline_opt_in_returns_independent_report_and_full_coverage():
    result = _run_pipeline(
        automatic_selection=AutomaticCommitmentSelectionConfig(),
        filtering=CandidateFilterConfig(max_l1_distance=0.3),
    )
    automatic = result.automatic_selection_report

    assert automatic is not None
    assert len(automatic.rows) == 6
    coverage = automatic.search_coverage
    assert coverage.source == "pipeline"
    assert len(coverage.input_candidate_ids) == len(result.generated_candidates)
    assert len(coverage.kept_candidate_ids) == len(
        result.comparison_report.comparisons
    )
    assert coverage.kept_candidate_ids == tuple(
        sorted(c.candidate.candidate_id for c in result.comparison_report.comparisons)
    )
    assert coverage.shift_amounts == (0.1, 0.2)
    assert coverage.filtering_applied is True
    assert coverage.filter_max_l1_distance == 0.3


def test_pipeline_opt_in_empty_kept_set_returns_no_selection_rows():
    result = _run_pipeline(
        automatic_selection=AutomaticCommitmentSelectionConfig(),
        filtering=CandidateFilterConfig(allowed_info_sets=set()),
    )
    automatic = result.automatic_selection_report

    assert automatic is not None
    assert automatic.search_coverage.input_candidate_ids
    assert automatic.search_coverage.kept_candidate_ids == ()
    assert all(row.status == NO_BENEFICIAL_COMMITMENT for row in automatic.rows)


def test_pipeline_response_mode_mismatch_fails_before_generation(monkeypatch):
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("candidate generation must not run")

    monkeypatch.setattr(pipeline_module, "generate_candidate_library", unexpected)
    with pytest.raises(ValueError, match="response_mode='worst'"):
        _run_pipeline(
            response_mode="best",
            automatic_selection=AutomaticCommitmentSelectionConfig(),
        )
    assert called is False
