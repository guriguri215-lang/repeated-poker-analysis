"""Frozen-contract tests for the known-board real-card HU river adapter."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import repeated_poker.known_board_real_card_hu_river as adapter
from repeated_poker.aiof_cards import (
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    canonicalize_exact_combo,
)
from repeated_poker.aiof_evaluator import evaluate_seven_card_hand
from repeated_poker.comparison import CandidateComparison, CandidateComparisonReport
from repeated_poker.exact_response import BestResponseResult, solve_exact_response
from repeated_poker.fixed_profile import FixedProfileValue, evaluate_fixed_profile
from repeated_poker.game import (
    ChanceNode,
    TerminalNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    validate_tree,
)
from repeated_poker.known_board_real_card_hu_river import (
    ActionProbability,
    ComboBucketAssignment,
    ComboBucketMap,
    KnownBoardRealCardHuRiverLimits,
    KnownBoardRealCardHuRiverRequest,
    RiverActionProfile,
    RiverProfileRow,
    analyze_known_board_real_card_hu_river,
)


def _exact_range(*rows: tuple[str, float]) -> RangeSpec:
    return RangeSpec(
        tuple(
            RangeEntry(combo, weight, WeightBasis.EXACT_COMBO_MASS)
            for combo, weight in rows
        )
    )


def _mixed_range(*rows) -> RangeSpec:
    return RangeSpec(tuple(RangeEntry(*row) for row in rows))


def _distribution(**probabilities: float) -> tuple[ActionProbability, ...]:
    return tuple(
        ActionProbability(action, probability)
        for action, probability in probabilities.items()
    )


def _hero_profile(
    buckets: tuple[str, ...],
    *,
    after=(0.5, 0.5),
    versus=(0.5, 0.25, 0.25),
) -> RiverActionProfile:
    rows = []
    for bucket in buckets:
        rows.extend(
            (
                RiverProfileRow(
                    bucket,
                    "after_oop_check",
                    _distribution(check=after[0], bet=after[1]),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_oop_bet",
                    _distribution(
                        **{
                            "call": versus[0],
                            "fold": versus[1],
                            "raise": versus[2],
                        }
                    ),
                ),
            )
        )
    return RiverActionProfile(tuple(rows))


def _villain_profile(
    buckets: tuple[str, ...],
    *,
    first=(0.5, 0.5),
    versus_bet=(0.5, 0.5),
    versus_raise=(0.5, 0.5),
) -> RiverActionProfile:
    rows = []
    for bucket in buckets:
        rows.extend(
            (
                RiverProfileRow(
                    bucket,
                    "oop_first",
                    _distribution(check=first[0], bet=first[1]),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_ip_bet",
                    _distribution(call=versus_bet[0], fold=versus_bet[1]),
                ),
                RiverProfileRow(
                    bucket,
                    "vs_ip_raise",
                    _distribution(call=versus_raise[0], fold=versus_raise[1]),
                ),
            )
        )
    return RiverActionProfile(tuple(rows))


def _mapping(assignments: dict[str, str], *bucket_ids: str) -> ComboBucketMap:
    return ComboBucketMap(
        tuple(bucket_ids),
        tuple(
            ComboBucketAssignment(combo, bucket)
            for combo, bucket in assignments.items()
        ),
    )


def _request(
    *,
    board=("2c", "3d", "4h", "5s", "9c"),
    hero_range=None,
    villain_range=None,
    dead_cards=(),
    hero_mapping=None,
    villain_mapping=None,
    hero_profile=None,
    villain_profile="default",
    shift_amounts=(0.25,),
    max_simultaneous_info_sets=1,
    horizon=3,
    discount=0.9,
    tolerance=1e-9,
    minimum_total_uplift=0.0,
    limits=None,
    **overrides,
) -> KnownBoardRealCardHuRiverRequest:
    hero_range = hero_range or _exact_range(("AsAh", 1.0))
    villain_range = villain_range or _exact_range(("KsKh", 1.0))
    hero_buckets = (
        tuple(hero_mapping.bucket_ids)
        if hero_mapping is not None
        else tuple(
            canonicalize_exact_combo(row.label) for row in hero_range.entries
        )
    )
    villain_buckets = (
        tuple(villain_mapping.bucket_ids)
        if villain_mapping is not None
        else tuple(
            canonicalize_exact_combo(row.label) for row in villain_range.entries
        )
    )
    if hero_profile is None:
        hero_profile = _hero_profile(hero_buckets)
    if villain_profile == "default":
        villain_profile = _villain_profile(villain_buckets)
    values = dict(
        board=tuple(board),
        hero_range=hero_range,
        villain_range=villain_range,
        baseline_hero_profile=hero_profile,
        dead_cards=tuple(dead_cards),
        hero_combo_to_bucket=hero_mapping,
        villain_combo_to_bucket=villain_mapping,
        baseline_villain_profile=villain_profile,
        initial_commitment_hero=1.0,
        initial_commitment_villain=1.0,
        rake_rate=0.05,
        rake_cap=3.0,
        oop_bet_size=2.0,
        ip_bet_after_check_size=2.0,
        ip_raise_to_size=5.0,
        shift_amounts=tuple(shift_amounts),
        max_simultaneous_info_sets=max_simultaneous_info_sets,
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        minimum_total_uplift=minimum_total_uplift,
        limits=limits or KnownBoardRealCardHuRiverLimits(),
    )
    values.update(overrides)
    return KnownBoardRealCardHuRiverRequest(**values)


def _success(request=None):
    result = analyze_known_board_real_card_hu_river(request or _request())
    assert result.status is AiofStatus.SUCCESS, result.to_dict()
    assert result.payload is not None
    assert result.error_message is None
    return result.payload


def _terminal_map(payload):
    return {
        node.node_id: node
        for node in iter_nodes(payload.tree.root)
        if isinstance(node, TerminalNode)
    }


def _json_bytes(result) -> bytes:
    return json.dumps(
        result.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_01_board_is_exactly_five_canonical_and_collisions_fail_closed():
    first = _success(_request(board=("9c", "2c", "5s", "4h", "3d")))
    second = _success(_request(board=("2c", "3d", "4h", "5s", "9c")))
    assert first.board == second.board == ("2c", "3d", "4h", "5s", "9c")
    assert first.board_identity == second.board_identity

    invalid = (
        _request(board=("2c",) * 5),
        _request(board=("2c", "3d", "4h", "5s")),
        _request(board=("2c", "3d", "4h", "5s", "10c")),
        _request(dead_cards=("2c",)),
    )
    for request in invalid:
        result = analyze_known_board_real_card_hu_river(request)
        assert result.status is AiofStatus.INVALID_CARD_INPUT
        assert result.payload is None


def test_02_class_mass_is_not_redistributed_and_board_dead_are_separate():
    hero = _mixed_range(
        ("AKs", 4.0, WeightBasis.CLASS_TOTAL_MASS),
        ("QhQd", 2.0, WeightBasis.EXACT_COMBO_MASS),
    )
    mapping = _mapping(
        {"AcKc": "H", "AdKd": "H", "QhQd": "H"}, "H"
    )
    payload = _success(
        _request(
            board=("As", "2c", "3d", "4h", "5s"),
            dead_cards=("Kh",),
            hero_range=hero,
            villain_range=_exact_range(("JcTd", 1.0)),
            hero_mapping=mapping,
            villain_mapping=_mapping({"JcTd": "V"}, "V"),
            hero_profile=_hero_profile(("H",)),
            villain_profile=_villain_profile(("V",)),
            shift_amounts=(),
        )
    )
    provenance = payload.provenance.hero
    assert provenance.pre_removal_combo_count == 5
    assert provenance.pre_removal_raw_mass == pytest.approx(6.0)
    assert provenance.board_collision_combo_count == 1
    assert provenance.board_collision_raw_mass == pytest.approx(1.0)
    assert provenance.extra_dead_collision_combo_count == 1
    assert provenance.extra_dead_collision_raw_mass == pytest.approx(1.0)
    assert provenance.surviving_combo_count == 3
    assert provenance.surviving_raw_mass == pytest.approx(4.0)
    masses = {row.hero_combo: row.raw_joint_mass for row in payload.joint_rows}
    assert masses == pytest.approx({"AcKc": 1.0, "AdKd": 1.0, "QhQd": 2.0})


def test_03_joint_conditioning_matches_three_row_hand_calculation():
    payload = _success(
        _request(
            hero_range=_exact_range(("AsKh", 1.0), ("9s8h", 2.0)),
            villain_range=_exact_range(("AsQh", 3.0), ("JcTd", 5.0)),
            hero_mapping=_mapping({"AsKh": "H1", "9s8h": "H2"}, "H1", "H2"),
            villain_mapping=_mapping({"AsQh": "V1", "JcTd": "V2"}, "V1", "V2"),
            hero_profile=_hero_profile(("H1", "H2")),
            villain_profile=_villain_profile(("V1", "V2")),
            shift_amounts=(),
        )
    )
    assert payload.provenance.cross_product_pair_count == 4
    assert payload.provenance.private_overlap_excluded_pair_count == 1
    assert payload.provenance.compatible_pair_count == 3
    assert payload.provenance.compatible_raw_joint_mass == pytest.approx(21.0)
    observed = {
        (row.hero_combo, row.villain_combo): (row.raw_joint_mass, row.probability)
        for row in payload.joint_rows
    }
    assert observed == pytest.approx(
        {
            ("9s8h", "JcTd"): (10.0, 10.0 / 21.0),
            ("9s8h", "AsQh"): (6.0, 6.0 / 21.0),
            ("AsKh", "JcTd"): (5.0, 5.0 / 21.0),
        }
    )


def test_04_blocker_joint_is_not_post_conditioned_marginal_product():
    payload = _success(
        _request(
            hero_range=_exact_range(("AsKh", 1.0), ("9s8h", 2.0)),
            villain_range=_exact_range(("AsQh", 3.0), ("JcTd", 5.0)),
            hero_mapping=_mapping({"AsKh": "H1", "9s8h": "H2"}, "H1", "H2"),
            villain_mapping=_mapping({"AsQh": "V1", "JcTd": "V2"}, "V1", "V2"),
            hero_profile=_hero_profile(("H1", "H2")),
            villain_profile=_villain_profile(("V1", "V2")),
            shift_amounts=(),
        )
    )
    target = next(
        row
        for row in payload.joint_rows
        if row.hero_combo == "AsKh" and row.villain_combo == "JcTd"
    )
    hero_marginal = 5.0 / 21.0
    villain_marginal = 15.0 / 21.0
    assert target.probability == pytest.approx(5.0 / 21.0)
    assert target.probability != pytest.approx(hero_marginal * villain_marginal)


@pytest.mark.parametrize(
    ("board", "hero", "villain", "expected"),
    (
        (("2c", "3d", "4h", "5s", "9c"), "AsAh", "KsKh", "hero"),
        (("2c", "3d", "4h", "5s", "9c"), "KsKh", "AsAh", "villain"),
        (("Ts", "Js", "Qs", "Ks", "As"), "2c3d", "4c5d", "chop"),
        (("2c", "2d", "5h", "7s", "9c"), "AsKh", "QsJh", "hero"),
    ),
)
def test_05_fixed_board_showdown_matches_public_evaluator(
    board, hero, villain, expected
):
    payload = _success(
        _request(
            board=board,
            hero_range=_exact_range((hero, 1.0)),
            villain_range=_exact_range((villain, 1.0)),
            hero_profile=_hero_profile((canonicalize_exact_combo(hero),)),
            villain_profile=_villain_profile((canonicalize_exact_combo(villain),)),
            shift_amounts=(),
        )
    )
    row = payload.joint_rows[0]
    assert row.hero_rank == evaluate_seven_card_hand(
        (row.hero_combo[:2], row.hero_combo[2:]) + payload.board
    )
    assert row.villain_rank == evaluate_seven_card_hand(
        (row.villain_combo[:2], row.villain_combo[2:]) + payload.board
    )
    assert row.showdown_result == expected


def test_06_default_and_shared_bucket_mapping_are_complete_and_strict():
    hero_range = _exact_range(("AsAh", 1.0), ("QsQh", 1.0))
    default = _success(
        _request(
            hero_range=hero_range,
            hero_profile=_hero_profile(("AsAh", "QsQh")),
            shift_amounts=(),
        )
    )
    shared_map = _mapping({"AsAh": "H", "QsQh": "H"}, "H")
    shared = _success(
        _request(
            hero_range=hero_range,
            hero_mapping=shared_map,
            hero_profile=_hero_profile(("H",)),
            shift_amounts=(),
        )
    )
    assert default.workload.hero_buckets == 2
    assert shared.workload.hero_buckets == 1
    assert len(collect_hero_info_sets(shared.tree)) == 2
    assert default.profile_mapping_identity != shared.profile_mapping_identity

    invalid_maps = (
        _mapping({"AsAh": "H"}, "H"),
        _mapping({"AsAh": "H", "QsQh": "H", "JsJh": "H"}, "H"),
        _mapping({"AsAh": "H", "QsQh": "H"}, "H", "EMPTY"),
        ComboBucketMap(("H",), (ComboBucketAssignment("AsAh", "UNKNOWN"),)),
    )
    for invalid in invalid_maps:
        result = analyze_known_board_real_card_hu_river(
            _request(
                hero_range=hero_range,
                hero_mapping=invalid,
                hero_profile=_hero_profile(("H",)),
                shift_amounts=(),
            )
        )
        assert result.status is AiofStatus.INVALID_STRATEGY
        assert result.payload is None


def test_07_profiles_require_all_buckets_decisions_actions_and_finite_mass():
    payload = _success()
    assert payload.baseline.source == "supplied_profile"
    assert len(payload.baseline.hero_profile.rows) == 2
    assert len(payload.baseline.villain_profile.rows) == 3

    base = _hero_profile(("AsAh",))
    cases = (
        RiverActionProfile(base.rows[:-1]),
        RiverActionProfile(
            base.rows
            + (
                RiverProfileRow(
                    "UNKNOWN", "after_oop_check", _distribution(check=1.0, bet=0.0)
                ),
            )
        ),
        RiverActionProfile(
            (
                RiverProfileRow(
                    "AsAh", "after_oop_check", _distribution(check=1.0)
                ),
                base.rows[1],
            )
        ),
        RiverActionProfile(
            (
                RiverProfileRow(
                    "AsAh",
                    "after_oop_check",
                    _distribution(check=float("nan"), bet=0.0),
                ),
                base.rows[1],
            )
        ),
    )
    for profile in cases:
        result = analyze_known_board_real_card_hu_river(
            _request(hero_profile=profile)
        )
        assert result.status is AiofStatus.INVALID_STRATEGY
        assert result.payload is None


def test_08_joint_root_shared_information_sets_and_perfect_recall():
    payload = _success(
        _request(
            hero_range=_exact_range(("AsAh", 1.0), ("QsQh", 1.0)),
            villain_range=_exact_range(("KsKh", 1.0), ("JsJh", 1.0)),
            hero_mapping=_mapping({"AsAh": "H", "QsQh": "H"}, "H"),
            villain_mapping=_mapping({"KsKh": "V", "JsJh": "V"}, "V"),
            hero_profile=_hero_profile(("H",)),
            villain_profile=_villain_profile(("V",)),
            shift_amounts=(),
        )
    )
    assert isinstance(payload.tree.root, ChanceNode)
    assert sum(probability for probability, _ in payload.tree.root.children) == pytest.approx(1.0)
    assert len(collect_hero_info_sets(payload.tree)) == 2
    assert len(collect_villain_info_sets(payload.tree)) == 3
    validate_tree(payload.tree)


def test_09_seven_lines_rake_uncalled_excess_and_conservation():
    payload = _success(
        _request(
            villain_profile=_villain_profile(("KsKh",)),
            shift_amounts=(),
            initial_commitment_hero=1.0,
            initial_commitment_villain=2.0,
            rake_rate=0.10,
            rake_cap=1.0,
            oop_bet_size=3.0,
            ip_bet_after_check_size=4.0,
            ip_raise_to_size=7.0,
        )
    )
    terminals = _terminal_map(payload)
    by_line = {
        line: next(value for key, value in terminals.items() if key.startswith(line))
        for line in (
            "T_check_check",
            "T_check_bet_call",
            "T_check_bet_fold",
            "T_bet_call",
            "T_bet_fold",
            "T_bet_raise_call",
            "T_bet_raise_fold",
        )
    }
    assert by_line["T_check_check"].hero_ev == pytest.approx(1.7)
    assert by_line["T_check_check"].house_rake == pytest.approx(0.3)
    assert by_line["T_check_bet_call"].hero_ev == pytest.approx(5.0)
    assert by_line["T_check_bet_fold"].hero_ev == pytest.approx(2.0)
    assert by_line["T_bet_call"].hero_ev == pytest.approx(4.1)
    assert by_line["T_bet_fold"].hero_ev == pytest.approx(-1.0)
    assert by_line["T_bet_raise_call"].hero_ev == pytest.approx(8.0)
    assert by_line["T_bet_raise_fold"].hero_ev == pytest.approx(5.0)
    for terminal in terminals.values():
        assert terminal.hero_ev + terminal.villain_ev + terminal.house_rake == pytest.approx(0.0)


def test_10_rake_creates_hero_and_house_interval_across_villain_ties():
    board = ("Ts", "Js", "Qs", "Ks", "As")
    hero_bucket = canonicalize_exact_combo("2c3d")
    hero_profile = _hero_profile((hero_bucket,), after=(0.0, 1.0), versus=(1.0, 0.0, 0.0))
    positive = _success(
        _request(
            board=board,
            hero_range=_exact_range(("2c3d", 1.0)),
            villain_range=_exact_range(("4c5d", 1.0)),
            hero_profile=hero_profile,
            villain_profile=None,
            shift_amounts=(),
            rake_rate=0.5,
            rake_cap=None,
            oop_bet_size=1.0,
            ip_bet_after_check_size=1.0,
            ip_raise_to_size=2.0,
        )
    )
    response = positive.baseline.auto_best_response
    assert response is not None
    assert response.ev_h_worst < response.ev_h_best
    assert response.expected_house_rake_worst > response.expected_house_rake_best
    assert response.best_response_action_variation

    zero = _success(
        replace(positive.request, rake_rate=0.0)
    )
    zero_response = zero.baseline.auto_best_response
    assert zero_response is not None
    assert zero_response.expected_house_rake_worst == 0.0
    assert zero_response.expected_house_rake_best == 0.0


def _assert_responses_equal(left, right):
    assert left.villain_max_ev == pytest.approx(right.villain_max_ev)
    assert left.ev_h_worst == pytest.approx(right.ev_h_worst)
    assert left.ev_h_best == pytest.approx(right.ev_h_best)
    assert left.expected_house_rake_worst == pytest.approx(
        right.expected_house_rake_worst
    )
    assert left.expected_house_rake_best == pytest.approx(
        right.expected_house_rake_best
    )
    assert left.best_response_action_variation == right.best_response_action_variation
    assert left.off_path_info_sets == right.off_path_info_sets
    assert left.num_villain_pure_strategies == right.num_villain_pure_strategies
    assert left.num_best_response_strategies == right.num_best_response_strategies
    assert left.best_response_strategies == right.best_response_strategies


def test_11_dp_matches_enumerator_on_native_tiny_tree_including_counts():
    payload = _success(_request(shift_amounts=(0.25,)))
    candidate = payload.comparison_report.comparisons[0].candidate
    dp = solve_exact_response(
        payload.tree,
        candidate.hero_strategy,
        method="dp",
        max_pure_strategies=100_000,
    )
    enumerated = solve_exact_response(
        payload.tree,
        candidate.hero_strategy,
        method="enumerate",
        max_pure_strategies=100_000,
    )
    _assert_responses_equal(dp, enumerated)
    assert dp.best_response_action_sets is not None
    assert enumerated.all_evaluations


def test_12_compact_correspondence_keeps_exact_counts_variation_and_interval():
    board = ("Ts", "Js", "Qs", "Ks", "As")
    profile = _hero_profile(
        (canonicalize_exact_combo("2c3d"),),
        after=(0.0, 1.0),
        versus=(1.0, 0.0, 0.0),
    )
    full = _success(
        _request(
            board=board,
            hero_range=_exact_range(("2c3d", 1.0)),
            villain_range=_exact_range(("4c5d", 1.0)),
            hero_profile=profile,
            villain_profile=None,
            shift_amounts=(),
            rake_rate=0.5,
            rake_cap=None,
            oop_bet_size=1.0,
            ip_bet_after_check_size=1.0,
            ip_raise_to_size=2.0,
        )
    )
    compact = _success(
        replace(
            full.request,
            limits=replace(full.request.limits, max_br_list_materialization=1),
        )
    )
    full_response = full.baseline.auto_best_response
    compact_response = compact.baseline.auto_best_response
    assert full_response is not None and compact_response is not None
    assert compact_response.num_best_response_strategies == full_response.num_best_response_strategies
    assert compact_response.best_response_action_variation == full_response.best_response_action_variation
    assert compact_response.ev_h_worst == pytest.approx(full_response.ev_h_worst)
    assert compact_response.ev_h_best == pytest.approx(full_response.ev_h_best)
    projection = compact.baseline.to_dict()["auto_best_response"]
    assert projection["correspondence_materialization_complete"] is False
    assert projection["materialized_best_response_count"] == 1
    assert projection["exact_best_response_count"] > 1


def test_13_candidate_formula_two_info_shifts_shared_bucket_l1_and_coverage():
    payload = _success(
        _request(
            hero_range=_exact_range(("AsAh", 1.0), ("QsQh", 1.0)),
            hero_mapping=_mapping({"AsAh": "H", "QsQh": "H"}, "H"),
            hero_profile=_hero_profile(("H",)),
            shift_amounts=(0.25,),
            max_simultaneous_info_sets=2,
        )
    )
    counts = dict(payload.workload.feasible_shift_counts_by_info_set)
    assert sorted(counts.values()) == [2, 6]
    assert payload.workload.feasible_single_shift_count == 8
    assert payload.workload.candidate_count == 8 + 2 * 6 == 20
    assert len(payload.candidate_records) == 20
    assert payload.coverage.generated_candidate_ids == payload.coverage.kept_candidate_ids
    assert len(set(payload.coverage.generated_candidate_ids)) == 20
    assert payload.coverage.filtering_applied is False
    assert {
        round(row.comparison.candidate.l1_distance, 9)
        for row in payload.candidate_records
    } == {0.5, 1.0}
    multi = next(
        row for row in payload.candidate_records if len(row.shifts) == 2
    )
    assert len({shift.info_set for shift in multi.shifts}) == 2
    changed = multi.comparison.candidate.hero_strategy.probabilities
    for shift in multi.shifts:
        assert changed[shift.info_set][shift.source_action] >= 0.0
        assert changed[shift.info_set][shift.target_action] <= 1.0


def test_14_native_comparison_a_and_exact_response_l_match_direct_calls():
    payload = _success(_request(shift_amounts=(0.25,)))
    comparison = payload.comparison_report.comparisons[0]
    direct_fixed = evaluate_fixed_profile(
        payload.tree,
        comparison.candidate.hero_strategy,
        payload.baseline.villain_strategy,
    )
    direct_response = solve_exact_response(
        payload.tree,
        comparison.candidate.hero_strategy,
        method="dp",
        max_pure_strategies=payload.request.limits.max_br_list_materialization,
    )
    assert comparison.fixed_profile_value == direct_fixed
    _assert_responses_equal(comparison.best_response, direct_response)
    record = next(
        row
        for row in payload.candidate_records
        if row.candidate_identity == comparison.candidate.candidate_id
    )
    assert comparison.fixed_profile_value.hero_ev == pytest.approx(
        record.to_dict()["values"]["a"]
    )


def _response(hero_worst: float, hero_best: float | None = None) -> BestResponseResult:
    hero_best = hero_worst if hero_best is None else hero_best
    return BestResponseResult(
        villain_max_ev=-hero_worst,
        best_response_strategies=[{}],
        ev_h_worst=hero_worst,
        ev_h_best=hero_best,
        expected_house_rake_worst=0.0,
        expected_house_rake_best=0.0,
        best_response_action_variation={},
        off_path_info_sets=[],
        num_villain_pure_strategies=1,
        num_best_response_strategies=1,
        best_response_action_sets={},
    )


def test_15_m27_receives_native_b_a_l_crossing_ties_threshold_and_secondary(monkeypatch):
    captured = {}

    def crafted_report(tree, baseline_hero, baseline_villain, candidates, **kwargs):
        baseline = FixedProfileValue(0.0, 0.0, 0.0)
        values = [(2.0, -2.0), (0.5, 0.5), (0.5, 0.5)]
        values.extend([(-10.0, -10.0)] * (len(candidates) - len(values)))
        rows = []
        for candidate, (a_value, l_value) in zip(candidates, values):
            response = _response(l_value)
            fixed = FixedProfileValue(a_value, -a_value, 0.0)
            rows.append(
                CandidateComparison(
                    candidate,
                    fixed,
                    -a_value,
                    a_value,
                    response,
                    l_value,
                    l_value,
                    l_value > 0.0,
                )
            )
        report = CandidateComparisonReport(baseline, rows)
        captured["report"] = report
        return report

    monkeypatch.setattr(adapter, "compare_candidates", crafted_report)
    payload = _success(
        _request(
            shift_amounts=(0.25,),
            horizon=3,
            discount=1.0,
        )
    )
    report = captured["report"]
    assert payload.comparison_report is report
    assert payload.comparison_report.baseline_value.hero_ev == 0.0
    rows = payload.automatic_selection.rows
    # At m=1 candidate 2/3 (l=.5) tie; by m=3 candidate 1's a=2 dominates.
    assert len(rows[0].primary_tie_candidate_ids) == 2
    assert rows[0].selected_candidate_id in rows[0].primary_tie_candidate_ids
    assert rows[2].selected_candidate_id == report.comparisons[0].candidate.candidate_id
    assert rows[-1].adaptation_opportunity == 4

    no_benefit = _success(
        _request(
            shift_amounts=(0.25,),
            minimum_total_uplift=10_000.0,
        )
    )
    assert all(row.selected_candidate_id is None for row in no_benefit.automatic_selection.rows)
    assert all(row.status == "NO_BENEFICIAL_COMMITMENT" for row in no_benefit.automatic_selection.rows)


def test_16_empty_candidate_universe_is_complete_success_with_n_plus_one_rows():
    payload = _success(_request(shift_amounts=(), horizon=5))
    assert payload.workload.candidate_count == 0
    assert payload.candidate_records == ()
    assert len(payload.automatic_selection.rows) == 6
    assert payload.automatic_selection.timing_row_evaluation_count == 0
    assert all(row.status == "NO_BENEFICIAL_COMMITMENT" for row in payload.automatic_selection.rows)


def _two_pair_request(**overrides):
    values = dict(
        hero_range=_exact_range(("AsAh", 1.0), ("QsQh", 1.0)),
        hero_profile=_hero_profile(("AsAh", "QsQh")),
        shift_amounts=(),
    )
    values.update(overrides)
    return _request(**values)


@pytest.mark.parametrize(
    ("request_factory", "guard_name"),
    (
        (
            lambda: _two_pair_request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_joint_matchups=1
                )
            ),
            "evaluate_seven_card_hand",
        ),
        (
            lambda: _two_pair_request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(),
                    max_fixed_board_evaluations=1,
                )
            ),
            "evaluate_seven_card_hand",
        ),
        (
            lambda: _request(
                limits=replace(KnownBoardRealCardHuRiverLimits(), max_tree_nodes=12)
            ),
            "evaluate_seven_card_hand",
        ),
        (
            lambda: _two_pair_request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_buckets_per_seat=1
                )
            ),
            "_canonical_profile",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_hero_info_sets=1
                )
            ),
            "evaluate_seven_card_hand",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_villain_info_sets=2
                )
            ),
            "evaluate_seven_card_hand",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_candidates=1
                )
            ),
            "_materialize_candidates",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(),
                    max_candidate_probability_cells=39,
                )
            ),
            "_materialize_candidates",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(),
                    max_fixed_profile_node_visits=116,
                )
            ),
            "compare_candidates",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_response_node_visits=103
                )
            ),
            "compare_candidates",
        ),
        (
            lambda: _request(
                limits=replace(
                    KnownBoardRealCardHuRiverLimits(), max_timing_rows=31
                )
            ),
            "select_automatic_commitments",
        ),
    ),
)
def test_17_every_workload_cap_fails_before_guarded_materialization(
    monkeypatch, request_factory, guard_name
):
    def forbidden(*args, **kwargs):
        raise AssertionError(f"{guard_name} must not be called")

    monkeypatch.setattr(adapter, guard_name, forbidden)
    result = analyze_known_board_real_card_hu_river(request_factory())
    assert result.status is AiofStatus.CAP_EXCEEDED
    assert result.payload is None


def test_17b_br_materialization_ceiling_override_fails_before_range_expansion(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "expand_range",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("range expansion must not start")
        ),
    )
    result = analyze_known_board_real_card_hu_river(
        _request(
            limits=replace(
                KnownBoardRealCardHuRiverLimits(),
                max_br_list_materialization=100_001,
            )
        )
    )
    assert result.status is AiofStatus.INVALID_INPUT
    assert result.payload is None


def test_18_mid_candidate_failure_returns_no_prefix_payload(monkeypatch):
    import repeated_poker.comparison as comparison_module

    original = comparison_module.solve_exact_response
    calls = {"count": 0}

    def fail_second(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("controlled mid-candidate failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "solve_exact_response", fail_second)
    result = analyze_known_board_real_card_hu_river(_request(shift_amounts=(0.25,)))
    assert calls["count"] == 2
    assert result.status is AiofStatus.NUMERIC_FAILURE
    assert result.payload is None
    assert result.error_message == "unexpected adapter failure"


def _permuted_profile(profile: RiverActionProfile) -> RiverActionProfile:
    return RiverActionProfile(
        tuple(
            RiverProfileRow(row.bucket_id, row.decision, tuple(reversed(row.actions)))
            for row in reversed(profile.rows)
        )
    )


def test_19_semantic_permutations_are_byte_identical_and_identity_sensitivity_is_scoped():
    hero_range = _exact_range(("AsAh", 1.0), ("QsQh", 2.0))
    hero_map = _mapping({"AsAh": "Z", "QsQh": "A"}, "Z", "A")
    villain_map = _mapping({"KsKh": "V"}, "V")
    original = _request(
        board=("2c", "3d", "4h", "5s", "9c"),
        dead_cards=("6c", "7d"),
        hero_range=hero_range,
        hero_mapping=hero_map,
        villain_mapping=villain_map,
        hero_profile=_hero_profile(("Z", "A")),
        villain_profile=_villain_profile(("V",)),
        shift_amounts=(0.25, 0.1),
    )
    permuted = replace(
        original,
        board=tuple(reversed(original.board)),
        dead_cards=tuple(reversed(original.dead_cards)),
        hero_range=RangeSpec(tuple(reversed(original.hero_range.entries))),
        hero_combo_to_bucket=ComboBucketMap(
            tuple(reversed(hero_map.bucket_ids)),
            tuple(reversed(hero_map.assignments)),
        ),
        villain_combo_to_bucket=ComboBucketMap(
            tuple(reversed(villain_map.bucket_ids)),
            tuple(reversed(villain_map.assignments)),
        ),
        baseline_hero_profile=_permuted_profile(original.baseline_hero_profile),
        baseline_villain_profile=_permuted_profile(original.baseline_villain_profile),
        shift_amounts=tuple(reversed(original.shift_amounts)),
    )
    first = analyze_known_board_real_card_hu_river(original)
    second = analyze_known_board_real_card_hu_river(permuted)
    assert first.status is second.status is AiofStatus.SUCCESS
    assert _json_bytes(first) == _json_bytes(second)

    base = first.payload
    assert base is not None
    horizon = _success(replace(original, horizon=original.horizon + 1))
    assert horizon.baseline_identity == base.baseline_identity
    assert [row.candidate_identity for row in horizon.candidate_records] == [
        row.candidate_identity for row in base.candidate_records
    ]
    assert horizon.analysis_identity != base.analysis_identity

    changes = (
        replace(original, board=("2c", "3d", "4h", "5s", "Tc")),
        replace(original, dead_cards=("6c", "8d")),
        replace(original, rake_rate=0.1),
        replace(original, oop_bet_size=3.0),
        replace(
            original,
            baseline_hero_profile=_hero_profile(("Z", "A"), after=(0.6, 0.4)),
        ),
        replace(
            original,
            hero_combo_to_bucket=_mapping({"AsAh": "H", "QsQh": "H"}, "H"),
            baseline_hero_profile=_hero_profile(("H",)),
        ),
    )
    for changed in changes:
        changed_payload = _success(changed)
        assert changed_payload.baseline_identity != base.baseline_identity


def test_20_expected_pin_collision_and_nonfinite_derived_values_fail_closed(monkeypatch):
    mismatch = analyze_known_board_real_card_hu_river(
        _request(expected_baseline_identity="sha256:" + "0" * 64)
    )
    assert mismatch.status is AiofStatus.INVALID_INPUT
    assert mismatch.payload is None

    monkeypatch.setattr(adapter, "_candidate_identity", lambda *args: "collision")
    collision = analyze_known_board_real_card_hu_river(_request(shift_amounts=(0.25,)))
    assert collision.status is AiofStatus.INVALID_INPUT
    assert collision.payload is None

    monkeypatch.undo()
    original_compare = adapter.compare_candidates

    def nonfinite(*args, **kwargs):
        report = original_compare(*args, **kwargs)
        first = report.comparisons[0]
        bad = replace(
            first,
            fixed_profile_value=FixedProfileValue(float("inf"), 0.0, 0.0),
        )
        return CandidateComparisonReport(report.baseline_value, [bad] + report.comparisons[1:])

    monkeypatch.setattr(adapter, "compare_candidates", nonfinite)
    numeric = analyze_known_board_real_card_hu_river(_request(shift_amounts=(0.25,)))
    assert numeric.status is AiofStatus.NUMERIC_FAILURE
    assert numeric.payload is None
