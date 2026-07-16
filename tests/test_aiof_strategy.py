from dataclasses import fields
from fractions import Fraction
import math

import pytest

import repeated_poker.aiof_strategy as strategy
from repeated_poker.aiof_cards import (
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
    prepare_compatible_ranges,
)
from repeated_poker.aiof_chip_ev import HeadsUpChipEvGame
from repeated_poker.aiof_equity import EquityAlgorithm, OutcomeCounts
from repeated_poker.aiof_strategy import (
    AiofStrategyAlgorithm,
    AiofStrategyLimits,
    AiofStrategyStatus,
    AlternatingBrDiagnosticRequest,
    HeuristicDiagnosticStatus,
    HeuristicUpdateMode,
    RationalStrategyRequest,
    StrategyClaimKind,
    generate_rational_lift_strategy,
    run_alternating_br_diagnostic,
)


WIN_BOARD = ("2c", "3d", "4h", "5s", "7c")
TIE_BOARD = ("Ts", "Js", "Qs", "Ks", "As")


def exact_range(*labels, weights=None):
    weights = weights or (1.0,) * len(labels)
    return RangeSpec(
        tuple(
            RangeEntry(label, weight, WeightBasis.EXACT_COMBO_MASS)
            for label, weight in zip(labels, weights)
        )
    )


def dead_except(live_cards):
    live = {card_id(card) for card in live_cards}
    return tuple(card_from_id(value) for value in range(52) if value not in live)


def game(sb=10.0, bb=10.0, small=0.5, big=1.0, ante=0.0, **kwargs):
    return HeadsUpChipEvGame(sb, bb, small, big, ante, **kwargs)


def request(
    sb="AsAh",
    bb="KsKh",
    board=WIN_BOARD,
    *,
    game_value=None,
    limits=None,
    oracle=False,
    phase1=False,
    epsilon=Fraction(0),
    display=Fraction(0),
    trace=0,
    equity=EquityAlgorithm.EXACT_EXHAUSTIVE,
    seed=None,
    samples=None,
):
    live = (sb[:2], sb[2:], bb[:2], bb[2:]) + tuple(board)
    return RationalStrategyRequest(
        exact_range(sb),
        exact_range(bb),
        dead_except(live),
        equity,
        game_value or game(),
        limits or AiofStrategyLimits(),
        claim_epsilon=epsilon,
        display_tie_tolerance=display,
        requested_trace_points=trace,
        run_reference_oracle=oracle,
        run_phase1_float_diagnostic=phase1,
        seed=seed,
        samples=samples,
    )


def diagnostic_request(
    *,
    mode=HeuristicUpdateMode.SIMULTANEOUS,
    damping=Fraction(1),
    iterations=4,
    trace=4,
):
    base = request()
    return AlternatingBrDiagnosticRequest(
        base.sb_range,
        base.bb_range,
        base.dead_cards,
        base.equity_algorithm,
        base.game,
        base.limits,
        mode,
        damping,
        iterations,
        Fraction(0),
        trace,
        None,
        None,
    )


def synthetic_game(c_values, *, f=Fraction(-1, 2), g=Fraction(1)):
    h_count = len(c_values)
    v_count = len(c_values[0])
    probability = Fraction(1, h_count * v_count)
    probabilities = tuple(tuple(probability for _ in range(v_count)) for _ in range(h_count))
    counts = tuple(
        tuple(OutcomeCounts(1, 0, 0, 1) for _ in range(v_count))
        for _ in range(h_count)
    )
    return strategy._RationalGame(
        tuple(f"H{i}" for i in range(h_count)),
        tuple(f"V{j}" for j in range(v_count)),
        tuple((2 * i, 2 * i + 1) for i in range(h_count)),
        tuple((20 + 2 * j, 21 + 2 * j) for j in range(v_count)),
        probabilities,
        tuple(tuple(Fraction(value) for value in row) for row in c_values),
        counts,
        (Fraction(1, h_count),) * h_count,
        (Fraction(1, v_count),) * v_count,
        f,
        g,
        Fraction(1),
        "prepared",
        strategy._identity((c_values, f, g)),
        h_count * v_count,
        h_count * v_count,
    )


def solve_synthetic(game_value, *, limits=None):
    limits = limits or AiofStrategyLimits()
    sb_lp = strategy._make_lp(game_value, limits, "SB")
    bb_lp = strategy._make_lp(game_value, limits, "BB")
    sb = strategy._solve_lp(sb_lp, limits, 20)
    bb = strategy._solve_lp(bb_lp, limits, 20)
    x = tuple(sb.values[f"x[{combo}]"] for combo in game_value.sb_combos)
    y = tuple(bb.values[f"y[{combo}]"] for combo in game_value.bb_combos)
    profile = strategy._profile(game_value, x, y)
    witness, sb_summary, bb_summary = strategy._verified_witness(
        game_value,
        profile,
        x,
        y,
        sb_lp,
        bb_lp,
        sb,
        bb,
        limits,
        Fraction(0),
        Fraction(0),
    )
    return x, y, witness, sb_summary, bb_summary


def test_public_surface_and_dataclass_fields_are_contract_fixed():
    assert strategy.__all__ == [
        "AiofStrategyStatus", "AiofStrategyAlgorithm", "StrategyClaimKind",
        "HeuristicUpdateMode", "HeuristicDiagnosticStatus", "AiofStrategyLimits",
        "StrategyError", "ExactComboActionProbability", "ExactBehaviourProfile",
        "ExactActionValue", "ExactStrategyRow", "ExactGainSnapshot", "SimplexTracePoint",
        "SimplexRunSummary", "RationalVerificationWitness", "ReferenceOracleComparison",
        "Phase1FloatDiagnostic", "RationalStrategyRequest", "RationalStrategyResult",
        "RationalStrategyRunResult", "AlternatingBrDiagnosticRequest", "HeuristicTracePoint",
        "AlternatingBrDiagnostic", "AlternatingBrDiagnosticRunResult",
        "generate_rational_lift_strategy", "run_alternating_br_diagnostic",
    ]
    assert [item.name for item in fields(strategy.RationalStrategyRunResult)] == [
        "status", "strategy_result", "error"
    ]


def test_one_cell_exact_lift_matches_phase1_support_counts_and_hand_oracle():
    result = generate_rational_lift_strategy(request())
    assert result.status is AiofStrategyStatus.SUCCESS and result.error is None
    payload = result.strategy_result
    assert payload is not None
    prepared = prepare_compatible_ranges(
        request().sb_range,
        request().bb_range,
        request().dead_cards,
        AiofLimits(max_range_entries_per_side=1, max_exact_combos_per_side=1, max_compatible_combo_pairs=1),
    )
    assert prepared.sb_range.combos[0].raw_mass.as_integer_ratio() == (1, 1)
    assert payload.payoff_cell_count == payload.exact_board_evaluations == 1
    assert payload.witness.gains.profile_value == Fraction(1)
    assert payload.witness.lower_objective == payload.witness.upper_objective


def test_one_time_exact_product_normalization_uses_raw_binary64_ratios():
    game_value = synthetic_game(((2, -2), (-2, 2)))
    assert sum(
        (p for row in game_value.probabilities for p in row if p is not None), Fraction(0)
    ) == 1
    assert game_value.p_h == game_value.p_v == (Fraction(1, 2), Fraction(1, 2))


def test_actual_phase1_raw_mass_lift_and_one_time_normalization_are_exact():
    sb = exact_range("AsAh", "KsKh", weights=(0.1, 0.2))
    bb = exact_range("QsQh", weights=(0.3,))
    live = ("As", "Ah", "Ks", "Kh", "Qs", "Qh", "2c", "3d", "4h", "5s", "7c")
    req = RationalStrategyRequest(
        sb, bb, dead_except(live), EquityAlgorithm.EXACT_EXHAUSTIVE, game(), AiofStrategyLimits()
    )
    built = strategy._build_rational_game(req, req.limits, extra_float_pass=False)
    assert dict(zip(built.sb_combos, built.p_h)) == {
        "KsKh": Fraction(2, 3),
        "AsAh": Fraction(1, 3),
    }
    assert built.p_v == (Fraction(1),)
    assert built.f == Fraction(-1, 2)
    assert built.g == Fraction(1)
    assert built.effective == Fraction(10)
    for i in range(2):
        counts = built.counts[i][0]
        assert built.showdown[i][0] == Fraction(10 * (counts.wins - counts.losses), counts.trials)


def test_run_local_cache_evaluates_each_pair_once(monkeypatch):
    calls = 0
    original = strategy.calculate_preflop_equity

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(strategy, "calculate_preflop_equity", counted)
    assert generate_rational_lift_strategy(request()).status is AiofStrategyStatus.SUCCESS
    assert calls == 1


@pytest.mark.parametrize(
    "unsupported",
    [game(fee=0.1), game(third_party_dead_money=1.0), game(side_pot=True)],
)
def test_unsupported_game_is_rejected_before_equity(monkeypatch, unsupported):
    monkeypatch.setattr(strategy, "calculate_preflop_equity", lambda _: pytest.fail("evaluator called"))
    result = generate_rational_lift_strategy(request(game_value=unsupported))
    assert result.status is AiofStrategyStatus.UNSUPPORTED_MODEL
    assert result.strategy_result is None and result.error


def test_dominant_sb_shove_and_dominant_bb_fold():
    payload = generate_rational_lift_strategy(request()).strategy_result
    assert payload.profile.sb_shove[0].probability == 1
    assert payload.profile.bb_call[0].probability == 0


def test_dominant_sb_fold_and_off_path_deterrence_semantics():
    payload = generate_rational_lift_strategy(request("KsKh", "AsAh")).strategy_result
    assert payload.profile.sb_shove[0].probability == 0
    assert 0 <= payload.profile.bb_call[0].probability <= 1
    row = payload.witness.bb_rows[0]
    assert row.information_reach_probability == 0
    assert tuple(item.value for item in row.action_values) == (None, None)
    assert row.exact_best_actions == ("call", "fold")
    assert row.unilateral_gain == 0


def test_dominant_bb_call_with_tie_showdown_and_sb_shove():
    payload = generate_rational_lift_strategy(
        request("2c3d", "4c5d", TIE_BOARD)
    ).strategy_result
    assert payload.profile.sb_shove[0].probability == 1
    assert payload.profile.bb_call[0].probability == 1


def test_exact_tie_keeps_raw_and_display_full_correspondence():
    payload = generate_rational_lift_strategy(
        request(game_value=game(sb=10, bb=1), display=Fraction(1, 10))
    ).strategy_result
    row = payload.witness.bb_rows[0]
    assert row.action_values[0].value == row.action_values[1].value
    assert row.exact_best_actions == row.display_best_actions == ("call", "fold")


def test_matching_pennies_type_synthetic_game_has_exact_interior_mix():
    # This security-game analogue has exact nontrivial mixing on both sides.
    game_value = synthetic_game(((-6, -5), (2, 3)))
    x, y, witness, _, _ = solve_synthetic(game_value)
    assert x == (Fraction(1, 7), Fraction(1))
    assert y == (Fraction(3, 7), Fraction(0))
    assert witness.gains.g_sb == witness.gains.g_bb == 0


def test_multiple_optimum_selection_and_basis_are_repeatable():
    game_value = synthetic_game(((1, 1), (1, 1)), f=Fraction(0), g=Fraction(1))
    first = solve_synthetic(game_value)
    second = solve_synthetic(game_value)
    assert first[0:2] == second[0:2]
    assert first[3].selected_basis == second[3].selected_basis
    assert first[4].selected_basis == second[4].selected_basis


def test_standard_form_primal_dual_and_strong_dual_hand_oracle():
    lp = strategy._LinearProgram(
        "HAND",
        ("x",),
        (Fraction(1),),
        Fraction(0),
        (strategy._Constraint((Fraction(1),), "<=", Fraction(1)),),
    )
    limits = AiofStrategyLimits()
    solved = strategy._solve_lp(lp, limits, 10)
    objective, primal, dual = strategy._verify_lp(lp, solved, limits)
    assert objective == 1 and primal and dual


def test_zero_trace_request_stores_no_pivots_and_marks_truncation():
    payload = generate_rational_lift_strategy(request(trace=0)).strategy_result
    assert payload.sb_lp.trace == payload.bb_lp.trace == ()
    assert payload.sb_lp.trace_truncated or payload.bb_lp.trace_truncated


def test_full_unilateral_snapshot_and_exact_claim_gate():
    payload = generate_rational_lift_strategy(request()).strategy_result
    gains = payload.witness.gains
    assert gains == strategy.ExactGainSnapshot(
        Fraction(1), Fraction(1), Fraction(1), Fraction(0), Fraction(0),
        Fraction(0), Fraction(0), Fraction(1), Fraction(1)
    )
    assert payload.witness.claim_kind is StrategyClaimKind.EXACT_NASH
    assert payload.witness.numeric_error_bound == 0


def test_epsilon_is_exact_but_cannot_hide_solver_mismatch(monkeypatch):
    original = strategy._verify_lp
    calls = 0

    def mismatched(*args, **kwargs):
        nonlocal calls
        calls += 1
        value, primal, dual = original(*args, **kwargs)
        return (value if calls == 1 else value + 1), primal, dual

    monkeypatch.setattr(strategy, "_verify_lp", mismatched)
    result = generate_rational_lift_strategy(request(epsilon=Fraction(100)))
    assert result.status is AiofStrategyStatus.VERIFICATION_FAILED
    assert result.strategy_result is None


@pytest.mark.parametrize("kind", ["primal", "dual", "gain"])
def test_injected_verifier_failures_have_no_payload(monkeypatch, kind):
    if kind in ("primal", "dual"):
        def failed(*_args, **_kwargs):
            raise strategy._StrategyFailure(
                AiofStrategyStatus.VERIFICATION_FAILED, kind, "verification"
            )
        monkeypatch.setattr(strategy, "_verify_lp", failed)
    else:
        def failed(*_args, **_kwargs):
            raise strategy._StrategyFailure(
                AiofStrategyStatus.VERIFICATION_FAILED, "gain", "verification"
            )
        monkeypatch.setattr(strategy, "_audit_profile", failed)
    result = generate_rational_lift_strategy(request())
    assert result.status is AiofStrategyStatus.VERIFICATION_FAILED
    assert result.strategy_result is None and result.error


def test_negative_gain_and_inverted_enclosure_are_not_clamped():
    game_value = synthetic_game(((1,),))
    with pytest.raises(strategy._StrategyFailure) as error:
        strategy._audit_profile(game_value, (Fraction(2),), (Fraction(0),), Fraction(0), AiofStrategyLimits())
    assert error.value.status is AiofStrategyStatus.INVALID_STRATEGY


@pytest.mark.parametrize(
    "fixture",
    [
        ((3, -1), (-1, 3)),
        ((1, 1), (1, 1)),
        ((2,),),
        ((0,),),
    ],
)
def test_tiny_support_oracle_matches_compact_lp_reference_fixtures(fixture):
    game_value = synthetic_game(fixture, f=Fraction(0), g=Fraction(1))
    x, y, witness, _, _ = solve_synthetic(game_value)
    comparison = strategy._reference_oracle(
        game_value, x, y, witness, AiofStrategyLimits(), Fraction(0)
    )
    assert comparison.value_matches
    assert comparison.selected_profile_value_matches
    assert comparison.selected_profile_gains_match
    assert comparison.tie_classification_matches
    assert comparison.off_path_classification_matches


def test_public_opt_in_oracle_and_injected_mismatch_no_payload(monkeypatch):
    success = generate_rational_lift_strategy(request(oracle=True))
    assert success.status is AiofStrategyStatus.SUCCESS
    assert success.strategy_result.oracle_comparison is not None

    def mismatch(*_args, **_kwargs):
        raise strategy._StrategyFailure(AiofStrategyStatus.ORACLE_MISMATCH, "injected", "oracle")

    monkeypatch.setattr(strategy, "_reference_oracle", mismatch)
    failed = generate_rational_lift_strategy(request(oracle=True))
    assert failed.status is AiofStrategyStatus.ORACLE_MISMATCH
    assert failed.strategy_result is None


def test_oracle_caps_before_plan_matrix_or_support_materialization():
    limits = AiofStrategyLimits(max_oracle_pure_plans_per_side=1)
    result = generate_rational_lift_strategy(request(oracle=True, limits=limits))
    assert result.status is AiofStrategyStatus.CAP_EXCEEDED
    assert result.error.phase == "oracle_preflight"


def test_oracle_support_above_six_has_no_large_fallback():
    game_value = synthetic_game(tuple((Fraction(1),) for _ in range(7)))
    dummy = strategy.RationalVerificationWitness(
        strategy.VERIFIER_ID, strategy.GAME_ID, game_value.payoff_identity, "p", "v",
        Fraction(0), Fraction(0), StrategyClaimKind.EXACT_NASH, Fraction(0), Fraction(0),
        True, True, True,
        strategy.ExactGainSnapshot(*(Fraction(0) for _ in range(9))), (), (),
    )
    with pytest.raises(strategy._StrategyFailure) as error:
        strategy._reference_oracle(
            game_value, (Fraction(0),) * 7, (Fraction(0),), dummy,
            AiofStrategyLimits(), Fraction(0)
        )
    assert error.value.status is AiofStrategyStatus.CAP_EXCEEDED


def test_complete_canonical_exact_profile_and_fold_complements():
    payload = generate_rational_lift_strategy(request()).strategy_result
    assert tuple(item.combo for item in payload.profile.sb_shove) == ("AsAh",)
    assert tuple(item.combo for item in payload.profile.bb_call) == ("KsKh",)
    for row in payload.witness.sb_rows + payload.witness.bb_rows:
        assert isinstance(row.active_probability, Fraction)
        assert 0 <= row.active_probability <= 1
        assert row.active_probability + row.fold_probability == 1


@pytest.mark.parametrize(
    "values",
    [(), (Fraction(0), Fraction(1)), (0.5,), (Fraction(-1),), (Fraction(2),)],
)
def test_invalid_exact_profile_missing_extra_float_and_ratio_rejected(values):
    game_value = synthetic_game(((1,),))
    with pytest.raises(strategy._StrategyFailure) as error:
        strategy._audit_profile(game_value, values, (Fraction(0),), Fraction(0), AiofStrategyLimits())
    assert error.value.status is AiofStrategyStatus.INVALID_STRATEGY


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "noncanonical", "float"])
def test_exact_profile_support_and_ratio_hook_rejects_invalid_forms(case):
    game_value = synthetic_game(((1,),))
    sb = (strategy.ExactComboActionProbability("H0", Fraction(0)),)
    bb = (strategy.ExactComboActionProbability("V0", Fraction(0)),)
    if case == "missing":
        sb = ()
    elif case == "extra":
        sb += (strategy.ExactComboActionProbability("HX", Fraction(0)),)
    elif case == "duplicate":
        sb += sb
    elif case == "noncanonical":
        sb = (strategy.ExactComboActionProbability("HX", Fraction(0)),)
    else:
        sb = (strategy.ExactComboActionProbability("H0", 0.5),)
    profile = strategy.ExactBehaviourProfile(sb, bb, "test")
    with pytest.raises(strategy._StrategyFailure) as error:
        strategy._validate_exact_profile(profile, game_value)
    assert error.value.status is AiofStrategyStatus.INVALID_STRATEGY


@pytest.mark.parametrize(
    ("equity", "seed", "samples"),
    [
        (EquityAlgorithm.DETERMINISTIC_MONTE_CARLO, None, None),
        (EquityAlgorithm.EXACT_EXHAUSTIVE, 1, None),
        (EquityAlgorithm.EXACT_EXHAUSTIVE, None, 10),
    ],
)
def test_exact_only_mc_seed_and_samples_rejected(equity, seed, samples):
    result = generate_rational_lift_strategy(
        request(equity=equity, seed=seed, samples=samples)
    )
    assert result.status is AiofStrategyStatus.EXACT_PAYOFF_REQUIRED
    assert result.strategy_result is None


@pytest.mark.parametrize(
    "limits",
    [
        AiofStrategyLimits(max_solver_combos_per_side=1),
        AiofStrategyLimits(max_payoff_cells=1),
        AiofStrategyLimits(max_exact_board_evaluations=1),
        AiofStrategyLimits(max_lp_variables_per_problem=1),
        AiofStrategyLimits(max_lp_constraints_per_problem=1),
        AiofStrategyLimits(max_tableau_cells=1),
    ],
)
def test_caps_return_no_partial_primary_payload(limits):
    # Two exact combos per side are chosen so every structural cap can be lowered.
    sb = exact_range("AsAh", "KsKh")
    bb = exact_range("QsQh", "JsJh")
    live = ("As", "Ah", "Ks", "Kh", "Qs", "Qh", "Js", "Jh", "2c", "3d", "4h", "5s", "7c")
    req = RationalStrategyRequest(
        sb, bb, dead_except(live), EquityAlgorithm.EXACT_EXHAUSTIVE, game(), limits
    )
    result = generate_rational_lift_strategy(req)
    assert result.status is AiofStrategyStatus.CAP_EXCEEDED
    assert result.strategy_result is None and result.error


@pytest.mark.parametrize(
    "limits",
    [
        AiofStrategyLimits(max_solver_combos_per_side=1),
        AiofStrategyLimits(max_payoff_cells=1),
        AiofStrategyLimits(max_exact_board_evaluations=1),
        AiofStrategyLimits(max_lp_variables_per_problem=1),
        AiofStrategyLimits(max_lp_constraints_per_problem=1),
        AiofStrategyLimits(max_tableau_cells=1),
    ],
)
def test_structural_caps_fail_before_singleton_evaluator(monkeypatch, limits):
    monkeypatch.setattr(strategy, "calculate_preflop_equity", lambda _: pytest.fail("evaluator called"))
    sb = exact_range("AsAh", "KsKh")
    bb = exact_range("QsQh", "JsJh")
    live = ("As", "Ah", "Ks", "Kh", "Qs", "Qh", "Js", "Jh", "2c", "3d", "4h", "5s", "7c")
    result = generate_rational_lift_strategy(
        RationalStrategyRequest(
            sb, bb, dead_except(live), EquityAlgorithm.EXACT_EXHAUSTIVE, game(), limits
        )
    )
    assert result.status is AiofStrategyStatus.CAP_EXCEEDED


def test_trace_cap_fails_before_evaluator(monkeypatch):
    monkeypatch.setattr(strategy, "calculate_preflop_equity", lambda _: pytest.fail("evaluator called"))
    result = generate_rational_lift_strategy(
        request(limits=AiofStrategyLimits(max_trace_points=1), trace=2)
    )
    assert result.status is AiofStrategyStatus.CAP_EXCEEDED


def test_pivot_and_rational_bit_caps_have_distinct_statuses():
    pivot = generate_rational_lift_strategy(
        request(limits=AiofStrategyLimits(max_simplex_pivots=1))
    )
    assert pivot.status is AiofStrategyStatus.SOLVER_LIMIT_REACHED
    bits = generate_rational_lift_strategy(
        request(limits=AiofStrategyLimits(max_exact_rational_bits=1))
    )
    assert bits.status is AiofStrategyStatus.EXACT_ARITHMETIC_CAP_EXCEEDED


@pytest.mark.parametrize("unbounded", [False, True])
def test_internal_infeasible_and_unbounded_lp_are_contract_failures(unbounded):
    if unbounded:
        lp = strategy._LinearProgram("BAD", ("x",), (Fraction(1),), Fraction(0), ())
    else:
        lp = strategy._LinearProgram(
            "BAD", ("x",), (Fraction(0),), Fraction(0),
            (
                strategy._Constraint((Fraction(1),), "<=", Fraction(0)),
                strategy._Constraint((Fraction(1),), ">=", Fraction(1)),
            ),
        )
    with pytest.raises(strategy._StrategyFailure) as error:
        strategy._solve_lp(lp, AiofStrategyLimits(), 0)
    assert error.value.status is AiofStrategyStatus.SOLVER_CONTRACT_FAILURE


def test_same_request_is_exactly_repeatable_including_basis_pivots_and_witness():
    first = generate_rational_lift_strategy(request(trace=10)).strategy_result
    second = generate_rational_lift_strategy(request(trace=10)).strategy_result
    assert first == second
    assert first.semantic_identity == second.semantic_identity
    assert first.input_identity == second.input_identity
    assert first.run_identity == second.run_identity


def test_cross_python_runtime_identity_may_change_but_semantics_do_not(monkeypatch):
    first = generate_rational_lift_strategy(request()).strategy_result
    monkeypatch.setattr(strategy.platform, "python_version", lambda: "3.10.synthetic")
    second = generate_rational_lift_strategy(request()).strategy_result
    assert first.runtime_identity != second.runtime_identity
    assert first.run_identity != second.run_identity
    assert first.semantic_identity == second.semantic_identity
    assert first.profile == second.profile
    assert first.witness.gains == second.witness.gains
    assert first.witness.claim_kind == second.witness.claim_kind


def test_exact_fraction_input_bit_cap_is_checked_before_payoff():
    result = generate_rational_lift_strategy(
        request(
            epsilon=Fraction(2**100, 1),
            limits=AiofStrategyLimits(max_exact_rational_bits=64),
        )
    )
    assert result.status is AiofStrategyStatus.EXACT_ARITHMETIC_CAP_EXCEEDED


def test_phase1_float_diagnostic_compares_profile_gains_ties_and_off_path():
    payload = generate_rational_lift_strategy(request(phase1=True)).strategy_result
    diagnostic = payload.phase1_float_diagnostic
    assert diagnostic.phase1_status is AiofStatus.SUCCESS
    assert diagnostic.profile_value_difference == pytest.approx(0.0)
    assert diagnostic.g_sb_difference == pytest.approx(0.0)
    assert diagnostic.g_bb_difference == pytest.approx(0.0)
    assert diagnostic.within_display_bound is True
    assert payload.witness.numeric_error_bound == 0
    assert payload.exact_board_evaluations == 2


def test_heuristic_half_initialization_damping_post_update_average_and_hash():
    result = run_alternating_br_diagnostic(
        diagnostic_request(damping=Fraction(1, 2), iterations=1)
    )
    assert result.status is AiofStrategyStatus.SUCCESS
    diagnostic = result.diagnostic
    # Initial SB/BB are 1/2. Dominant BR is shove/fold, so post-update is 3/4,1/4.
    assert diagnostic.current_profile.sb_shove[0].probability == Fraction(3, 4)
    assert diagnostic.current_profile.bb_call[0].probability == Fraction(1, 4)
    assert diagnostic.arithmetic_average_profile == diagnostic.current_profile
    assert diagnostic.trace[0].state_identity == diagnostic.current_state_identity


def test_heuristic_simultaneous_and_sequential_sb_then_bb_are_distinct():
    simultaneous = run_alternating_br_diagnostic(
        diagnostic_request(mode=HeuristicUpdateMode.SIMULTANEOUS, iterations=1)
    ).diagnostic
    sequential = run_alternating_br_diagnostic(
        diagnostic_request(mode=HeuristicUpdateMode.SEQUENTIAL_SB_THEN_BB, iterations=1)
    ).diagnostic
    assert simultaneous.update_mode is not sequential.update_mode
    assert simultaneous.input_identity != sequential.input_identity


def test_heuristic_cycle_or_fixed_point_and_bounded_trace():
    diagnostic = run_alternating_br_diagnostic(
        diagnostic_request(iterations=10, trace=1)
    ).diagnostic
    assert diagnostic.status in (
        HeuristicDiagnosticStatus.CYCLE_DETECTED,
        HeuristicDiagnosticStatus.DIAGNOSTIC_COMPLETE,
    )
    assert len(diagnostic.trace) == 1
    assert diagnostic.trace_truncated


def test_heuristic_true_cycle_fixture_reports_cycle_metadata(monkeypatch):
    game_value = synthetic_game(((-4, -4), (-4, 3)))
    monkeypatch.setattr(strategy, "_build_rational_game", lambda *_args, **_kwargs: game_value)
    diagnostic = run_alternating_br_diagnostic(
        diagnostic_request(iterations=10, trace=2)
    ).diagnostic
    assert diagnostic.status is HeuristicDiagnosticStatus.CYCLE_DETECTED
    assert diagnostic.repeated_state_first_seen_iteration is not None
    assert diagnostic.cycle_length > 1
    assert len(diagnostic.trace) == 2


def test_heuristic_nonrepeat_iteration_cap_and_current_average_gain_oracles():
    diagnostic = run_alternating_br_diagnostic(
        diagnostic_request(damping=Fraction(1, 3), iterations=1, trace=1)
    ).diagnostic
    assert diagnostic.status is HeuristicDiagnosticStatus.ITERATION_CAP_REACHED
    point = diagnostic.trace[0]
    assert point.current_nash_conv == point.current_g_sb + point.current_g_bb
    assert point.current_max_unilateral_gain == max(point.current_g_sb, point.current_g_bb)
    assert point.average_nash_conv == point.average_g_sb + point.average_g_bb
    assert point.average_max_unilateral_gain == max(point.average_g_sb, point.average_g_bb)


def test_fixed_point_diagnostic_has_no_solver_or_baseline_fields():
    diagnostic = run_alternating_br_diagnostic(
        diagnostic_request(iterations=10)
    ).diagnostic
    names = {item.name for item in fields(diagnostic)}
    assert "solution" not in names
    assert "equilibrium" not in names
    assert "baseline" not in names
    assert "certificate" not in names


def test_all_outer_failures_obey_no_partial_invariant():
    requests = (
        request(seed=1),
        request(game_value=game(fee=1)),
        request(limits=AiofStrategyLimits(max_lp_variables_per_problem=1)),
    )
    for item in requests:
        result = generate_rational_lift_strategy(item)
        assert result.status is not AiofStrategyStatus.SUCCESS
        assert result.strategy_result is None
        assert result.error is not None and result.error.message
