"""Focused independent-oracle tests for the M30 exact response module."""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.three_player_response as module
from repeated_poker.three_player_response import (
    ALGORITHM_VERSION,
    CAP_EXCEEDED,
    CONTRACT_VERSION,
    EXACT_CORRESPONDENCE_COMPLETE,
    INTERNAL_FAILURE,
    INVALID_INPUT,
    NUMERIC_FAILURE,
    STALE_INPUT,
    UNSUPPORTED_MODEL,
    ExactPayoffRow,
    ExactReducedGame,
    ExactResponseLimits,
    ResponseIdentityPins,
    exact_response_json,
    solve_three_player_response,
)


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    ROOT / "src" / "repeated_poker" / "three_player_cfr.py",
    ROOT / "src" / "repeated_poker" / "three_player_cfr_file_workflow.py",
    ROOT / "tests" / "test_three_player_cfr.py",
    ROOT / "tests" / "test_three_player_cfr_file_workflow.py",
    ROOT / "tests" / "test_three_player_cfr_public_workflow.py",
)


def identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rational(value: int | Fraction | str) -> str:
    if isinstance(value, str):
        return value
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def game(
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    payoffs: dict[tuple[str, str], tuple[object, object, object, object]],
    *,
    supplied_profile_identity: str | None = None,
    candidate_identity: str | None = None,
    search_mode_context_identity: str | None = None,
    run_context_identity: str | None = None,
    row_order: list[tuple[str, str]] | None = None,
    **overrides: object,
) -> ExactReducedGame:
    keys = row_order or [
        (o1_plan_id, o2_plan_id)
        for o1_plan_id in o1_plan_ids
        for o2_plan_id in o2_plan_ids
    ]
    rows = tuple(
        ExactPayoffRow(
            o1_plan_id,
            o2_plan_id,
            *(rational(value) for value in payoffs[(o1_plan_id, o2_plan_id)]),
        )
        for o1_plan_id, o2_plan_id in keys
    )
    values: dict[str, object] = {
        "o1_plan_ids": o1_plan_ids,
        "o2_plan_ids": o2_plan_ids,
        "payoff_rows": rows,
        "response_game_structure_identity": identity("structure"),
        "fixed_hero_identity": identity("hero"),
        "perfect_recall_evidence_identity": identity("recall"),
        "rake_convention_identity": identity("rake"),
        "supplied_profile_identity": supplied_profile_identity,
        "candidate_identity": candidate_identity,
        "search_mode_context_identity": search_mode_context_identity,
        "run_context_identity": run_context_identity,
    }
    values.update(overrides)
    return ExactReducedGame(**values)


def solve(candidate: ExactReducedGame, **kwargs: object):
    result = solve_three_player_response(candidate, **kwargs)
    assert result.partial_response is False
    return result


def success(candidate: ExactReducedGame, **kwargs: object) -> dict[str, object]:
    result = solve(candidate, **kwargs)
    assert result.status == EXACT_CORRESPONDENCE_COMPLETE
    assert result.error is None
    assert result.response is not None
    assert result.response["coverage"] == "complete"
    assert result.response["partial_response"] is False
    return result.response


def assert_no_partial(result, status: str) -> None:
    assert result.status == status
    assert result.response is None
    assert result.error is not None
    assert result.partial_response is False
    payload = result.to_dict()
    assert payload["response"] is None
    serialized = exact_response_json(result)
    for forbidden in (
        "support_cells",
        "hero_worst_witnesses",
        "pure_profile_unilateral_stability",
        "hero_min_joint_plan_stress",
    ):
        assert forbidden not in serialized


def _test_owned_pure_profile_oracle(
    candidate: ExactReducedGame,
) -> list[tuple[str, str, str]]:
    table = {
        (row.o1_plan_id, row.o2_plan_id): (
            Fraction(row.O1),
            Fraction(row.O2),
        )
        for row in candidate.payoff_rows
    }
    stable: list[tuple[str, str, str]] = []
    for i, o1_plan_id in enumerate(candidate.o1_plan_ids):
        for j, o2_plan_id in enumerate(candidate.o2_plan_ids):
            current_o1, current_o2 = table[(o1_plan_id, o2_plan_id)]
            best_o1 = max(
                table[(alternative, o2_plan_id)][0]
                for alternative in candidate.o1_plan_ids
            )
            best_o2 = max(
                table[(o1_plan_id, alternative)][1]
                for alternative in candidate.o2_plan_ids
            )
            residual_o1 = max(Fraction(0), best_o1 - current_o1)
            residual_o2 = max(Fraction(0), best_o2 - current_o2)
            if residual_o1 == 0 and residual_o2 == 0:
                stable.append(
                    (
                        f"O1:{i}|O2:{j}",
                        rational(residual_o1),
                        rational(residual_o2),
                    )
                )
    return stable


def one_by_one() -> ExactReducedGame:
    return game(("A",), ("L",), {("A", "L"): (-2, 1, 1, 0)})


def dominant() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (-2, 1, 1, 0),
            ("A", "R"): (-1, 1, 0, 0),
            ("B", "L"): (-1, 0, 1, 0),
            ("B", "R"): (0, 0, 0, 0),
        },
    )


def coordination() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (-2, 1, 1, 0),
            ("A", "R"): (-4, 0, 0, 4),
            ("B", "L"): (-4, 0, 0, 4),
            ("B", "R"): (-4, 2, 2, 0),
        },
    )


def matching_pennies() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (0, 1, -1, 0),
            ("A", "R"): (0, -1, 1, 0),
            ("B", "L"): (0, -1, 1, 0),
            ("B", "R"): (0, 1, -1, 0),
        },
    )


def coalition_counterexample() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (-2, 1, 1, 0),
            ("A", "R"): (-3, 3, 0, 0),
            ("B", "L"): (-3, 0, 3, 0),
            ("B", "R"): (-4, 2, 2, 0),
        },
    )


def fully_degenerate() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (-4, 0, 0, 4),
            ("A", "R"): (-2, 0, 0, 2),
            ("B", "L"): (1, 0, 0, -1),
            ("B", "R"): (3, 0, 0, -3),
        },
    )


def duplicate_cell_game() -> ExactReducedGame:
    return game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (0, 0, 0, 0),
            ("A", "R"): (0, 0, 0, 0),
            ("B", "L"): (0, 0, -1, 1),
            ("B", "R"): (0, 0, 0, 0),
        },
    )


def test_a1_one_by_one_exact_complete_and_all_residuals_zero():
    output = success(one_by_one())
    assert output["counts"]["pure_plans"] == {"O1": 1, "O2": 1}
    assert output["counts"]["support_pairs_total"] == 1
    assert output["counts"]["support_pairs_visited"] == 1
    assert output["counts"]["canonical_support_cells"] == 1
    assert output["hero_worst"] == output["hero_best"] == "-2"
    cell = output["support_cells"][0]
    assert cell["kind"] == "singleton"
    assert cell["dimension"] == 0
    assert cell["o1_mixture_polytope"]["vertices"] == [{"A": "1"}]
    assert cell["o2_mixture_polytope"]["vertices"] == [{"L": "1"}]
    pure = output["pure_profile_unilateral_stability"]
    assert pure["profile_count"] == 1
    assert pure["rows"][0]["unilateral_residual"] == {"O1": "0", "O2": "0"}
    assert output["independent_verification"]["status"] == "VERIFIED"
    assert output["independent_verification"]["solver_helpers_reused"] is False
    assert output["unilateral_residual_certificate"]["maximum"] == {
        "O1": "0",
        "O2": "0",
    }
    assert output["hero_worst_witnesses"][0]["unilateral_residual"] == {
        "O1": "0",
        "O2": "0",
    }


def test_independent_verifier_has_separate_utility_and_residual_evaluators():
    verifier_source = inspect.getsource(module._verify_correspondence)
    assert "_verifier_utility_at(" in verifier_source
    assert "_verifier_residuals(" in verifier_source
    assert "utility = _utility_at(" not in verifier_source
    assert "residuals = _unilateral_residuals(" not in verifier_source


def test_a2_strict_dominant_two_by_two_has_unique_pure_response():
    output = success(dominant())
    assert output["pure_profile_unilateral_stability"]["profile_count"] == 1
    row = output["pure_profile_unilateral_stability"]["rows"][0]
    assert row["plans"] == {"O1": "A", "O2": "L"}
    assert output["hero_worst"] == output["hero_best"] == "-2"
    points = {
        (
            tuple(witness["o1_mixture"].items()),
            tuple(witness["o2_mixture"].items()),
        )
        for witness in output["hero_worst_witnesses"]
    }
    assert points == {
        (
            (("A", "1"), ("B", "0")),
            (("L", "1"), ("R", "0")),
        )
    }


def test_a3_coordination_covers_all_responses_and_hero_interval():
    output = success(coordination())
    pure = output["pure_profile_unilateral_stability"]
    assert [row["profile_id"] for row in pure["rows"]] == [
        "O1:0|O2:0",
        "O1:1|O2:1",
    ]
    assert output["hero_worst"] == "-4"
    assert output["hero_best"] == "-2"
    assert output["utility_extrema"]["H"]["minimum"] == "-4"
    assert output["utility_extrema"]["H"]["maximum"] == "-2"
    assert output["counts"]["support_pairs_visited"] == 9
    assert output["counts"]["canonical_support_cells"] == 3
    assert {
        (
            tuple(cell["o1_mixture_polytope"]["vertices"][0].values()),
            tuple(cell["o2_mixture_polytope"]["vertices"][0].values()),
        )
        for cell in output["support_cells"]
    } == {
        (("1", "0"), ("1", "0")),
        (("0", "1"), ("0", "1")),
        (("2/3", "1/3"), ("2/3", "1/3")),
    }
    assert all(
        witness["extremum_value"] == "-4"
        for witness in output["hero_worst_witnesses"]
    )


def test_a4_matching_pennies_has_no_pure_subset_and_exact_half_mixture():
    output = success(matching_pennies())
    assert output["pure_profile_unilateral_stability"]["profile_count"] == 0
    assert output["counts"]["canonical_support_cells"] == 1
    cell = output["support_cells"][0]
    assert cell["kind"] == "singleton"
    assert cell["o1_mixture_polytope"]["vertices"] == [
        {"A": "1/2", "B": "1/2"}
    ]
    assert cell["o2_mixture_polytope"]["vertices"] == [
        {"L": "1/2", "R": "1/2"}
    ]
    assert output["hero_worst"] == output["hero_best"] == "0"


def test_a5_degenerate_continuum_is_support_cells_not_finite_equilibrium_rows():
    output = success(fully_degenerate())
    continuum = [
        cell for cell in output["support_cells"] if cell["kind"] == "continuum"
    ]
    assert continuum
    assert max(cell["dimension"] for cell in continuum) == 2
    full = next(cell for cell in continuum if cell["dimension"] == 2)
    assert full["o1_mixture_polytope"]["vertex_count"] == 2
    assert full["o2_mixture_polytope"]["vertex_count"] == 2
    assert output["counts"]["support_pairs_visited"] == 9
    assert output["counts"]["canonical_support_cells"] == 9
    assert output["counts"]["raw_nonempty_support_cells"] == 9
    assert output["hero_worst"] == "-4"
    assert output["hero_best"] == "3"
    serialized = exact_response_json(
        solve_three_player_response(fully_degenerate())
    )
    assert '"support_cells"' in serialized
    assert '"components"' not in serialized


def test_a6_ties_and_canonical_dedup_are_deterministic():
    first = success(duplicate_cell_game())
    second = success(duplicate_cell_game())
    assert first == second
    assert first["utility_extrema"]["O1"]["minimum"] == "0"
    assert first["utility_extrema"]["O1"]["maximum"] == "0"
    assert len(first["utility_extrema"]["O1"]["minimum_witnesses"]) >= 2
    cell_ids = [cell["support_cell_id"] for cell in first["support_cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    assert first["counts"]["exact_duplicate_support_cells"] == 2
    assert first["counts"]["raw_nonempty_support_cells"] == 7
    assert first["counts"]["canonical_support_cells"] == 5
    assert any(
        len(cell["source_support_pairs"]) > 1
        for cell in first["support_cells"]
    )
    assert all(
        item["outcome"]
        in {
            "new_support_cell",
            "exact_duplicate_support_cell",
            "empty_o1_mixture_polytope",
            "empty_o2_mixture_polytope",
        }
        for item in first["support_pair_audit"]
    )


def test_owned_one_by_one_and_two_by_two_pure_profile_oracle_is_exhaustive():
    for candidate in (
        one_by_one(),
        dominant(),
        coordination(),
        matching_pennies(),
        fully_degenerate(),
        duplicate_cell_game(),
    ):
        expected = _test_owned_pure_profile_oracle(candidate)
        actual = [
            (
                row["profile_id"],
                row["unilateral_residual"]["O1"],
                row["unilateral_residual"]["O2"],
            )
            for row in success(candidate)[
                "pure_profile_unilateral_stability"
            ]["rows"]
        ]
        assert actual == expected


def test_a7_a8_exact_residual_boundary_is_zero_and_never_negative():
    for candidate in (
        dominant(),
        coordination(),
        matching_pennies(),
        fully_degenerate(),
    ):
        output = success(candidate)
        for row in output["pure_profile_unilateral_stability"]["rows"]:
            assert Fraction(row["unilateral_residual"]["O1"]) >= 0
            assert Fraction(row["unilateral_residual"]["O2"]) >= 0
        for cell in output["support_cells"]:
            assert "epsilon" not in cell
    strict = dominant()
    output = success(strict)
    assert output["pure_profile_unilateral_stability"]["rows"][0][
        "unilateral_residual"
    ] == {"O1": "0", "O2": "0"}
    assert "VERIFIED_EPSILON_PROFILE_ONLY" not in exact_response_json(
        solve_three_player_response(strict)
    )


def test_a9_primary_unique_al_and_separate_coalition_stress_br():
    output = success(coalition_counterexample())
    pure = output["pure_profile_unilateral_stability"]["rows"]
    assert len(pure) == 1 and pure[0]["plans"] == {"O1": "A", "O2": "L"}
    assert output["hero_worst"] == output["hero_best"] == "-2"
    stress = output["hero_min_joint_plan_stress"]
    assert stress["hero_minimum"] == "-4"
    assert stress["witnesses"][0]["plans"] == {"O1": "B", "O2": "R"}
    assert stress["witnesses"][0]["utility"] == {
        "H": "-4",
        "O1": "2",
        "O2": "2",
        "R": "0",
    }
    assert stress["opponent_individual_rationality_not_required"] is True
    assert stress["transferable_utility_assumed"] is False
    assert stress["coalition_equilibrium_claim"] is False
    assert stress["primary_response_status_influence"] is False
    assert len(stress["diagnostic_identity"]) == 64


def test_a10_baseline_context_keeps_game_identity_and_changes_run_identity():
    first = success(
        replace(
            coordination(), supplied_profile_identity=identity("baseline-1")
        )
    )
    second = success(
        replace(
            coordination(), supplied_profile_identity=identity("baseline-2")
        )
    )
    assert first["response_game_identity"] == second["response_game_identity"]
    assert first["response_run_identity"] != second["response_run_identity"]
    assert first["support_cells"] == second["support_cells"]
    assert first["hero_worst"] == second["hero_worst"]


@pytest.mark.parametrize(
    ("pin_name", "actual_name"),
    [
        ("response_game_structure_identity", "response_game_structure_identity"),
        ("fixed_hero_identity", "fixed_hero_identity"),
        ("perfect_recall_evidence_identity", "perfect_recall_evidence_identity"),
        ("rake_convention_identity", "rake_convention_identity"),
        ("supplied_profile_identity", "supplied_profile_identity"),
        ("candidate_identity", "candidate_identity"),
        ("search_mode_context_identity", "search_mode_context_identity"),
        ("run_context_identity", "run_context_identity"),
    ],
)
def test_a12_all_source_context_stale_pins_fail_closed(pin_name: str, actual_name: str):
    candidate = replace(
        coordination(),
        supplied_profile_identity=identity("baseline"),
        candidate_identity=identity("candidate"),
        search_mode_context_identity=identity("search"),
        run_context_identity=identity("run"),
    )
    assert getattr(candidate, actual_name) != identity("stale")
    result = solve(
        candidate,
        pins=ResponseIdentityPins(**{pin_name: identity("stale")}),
    )
    assert_no_partial(result, STALE_INPUT)


def test_a12_payoff_config_game_and_run_identity_pins_reject_stale_values():
    baseline = success(coordination())
    for field in (
        "payoff_table_identity",
        "config_identity",
        "response_game_identity",
        "response_run_identity",
    ):
        result = solve(
            coordination(),
            pins=ResponseIdentityPins(**{field: identity(f"stale-{field}")}),
        )
        assert_no_partial(result, STALE_INPUT)
    matching_pins = ResponseIdentityPins(
        payoff_table_identity=baseline["content_identities"]["payoff_table"],
        config_identity=baseline["content_identities"]["config"],
        response_game_identity=baseline["response_game_identity"],
        response_run_identity=baseline["response_run_identity"],
    )
    assert success(coordination(), pins=matching_pins) == baseline


def test_a13_support_pair_preflight_cap_fires_before_support_materialization(
    monkeypatch,
):
    called = False

    def bomb(_size):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(module, "_support_subsets", bomb)
    result = solve(
        coordination(),
        limits=replace(ExactResponseLimits(), max_support_pairs=8),
    )
    assert_no_partial(result, CAP_EXCEEDED)
    assert called is False


def test_a13_checked_support_count_multiplication_for_three_by_three():
    ids1 = ("A", "B", "C")
    ids2 = ("L", "M", "R")
    payoffs = {
        (i, j): (0, 0, 0, 0)
        for i in ids1
        for j in ids2
    }
    result = solve(
        game(ids1, ids2, payoffs),
        limits=replace(
            ExactResponseLimits(),
            max_joint_pure_profiles=9,
            max_support_pairs=48,
        ),
    )
    assert_no_partial(result, CAP_EXCEEDED)


def test_a14_derived_vertex_denominator_bit_cap_discards_all_partial_cells():
    result = solve(
        matching_pennies(),
        limits=replace(
            ExactResponseLimits(),
            max_rational_denominator_bits=1,
        ),
    )
    assert_no_partial(result, CAP_EXCEEDED)


@pytest.mark.parametrize(
    "bad",
    [
        "-0",
        "+1",
        "01",
        "-01",
        "0/1",
        "1/0",
        "1/-2",
        "2/4",
        "1/02",
        " 1",
        "1 ",
        "",
        True,
        1,
        1.0,
    ],
)
def test_a15_noncanonical_rational_and_invalid_numeric_rejected(bad: object):
    candidate = one_by_one()
    row = replace(candidate.payoff_rows[0], H=bad)
    result = solve(replace(candidate, payoff_rows=(row,)))
    assert_no_partial(result, INVALID_INPUT)


def test_a14_input_numerator_and_denominator_bit_caps():
    numerator_game = game(
        ("A",), ("L",), {("A", "L"): ("8", "-8", "0", "0")}
    )
    result = solve(
        numerator_game,
        limits=replace(
            ExactResponseLimits(), max_rational_numerator_bits=3
        ),
    )
    assert_no_partial(result, CAP_EXCEEDED)
    denominator_game = game(
        ("A",),
        ("L",),
        {("A", "L"): ("-1/8", "1/8", "0", "0")},
    )
    result = solve(
        denominator_game,
        limits=replace(
            ExactResponseLimits(), max_rational_denominator_bits=3
        ),
    )
    assert_no_partial(result, CAP_EXCEEDED)


def test_a16_conservation_mismatch_fails_and_r_is_non_strategic():
    candidate = one_by_one()
    bad = replace(candidate.payoff_rows[0], R="1")
    assert_no_partial(
        solve(replace(candidate, payoff_rows=(bad,))), INVALID_INPUT
    )
    output = success(fully_degenerate())
    assert output["utility_extrema"]["R"]["minimum"] == "-3"
    assert output["utility_extrema"]["R"]["maximum"] == "4"
    assert "R is excluded from response conditions" in output[
        "unilateral_residual_semantics"
    ]
    assert all(
        set(row["unilateral_residual"]) == {"O1", "O2"}
        for row in output["pure_profile_unilateral_stability"]["rows"]
    )


def test_a17_perfect_recall_evidence_missing_and_stale_fail_closed():
    missing = replace(coordination(), perfect_recall_evidence_identity=None)
    assert_no_partial(solve(missing), UNSUPPORTED_MODEL)
    stale = solve(
        coordination(),
        pins=ResponseIdentityPins(
            perfect_recall_evidence_identity=identity("old-recall")
        ),
    )
    assert_no_partial(stale, STALE_INPUT)


def test_a20_row_permutation_is_canonical_but_plan_order_is_semantic():
    original = coordination()
    permuted = game(
        original.o1_plan_ids,
        original.o2_plan_ids,
        {
            (row.o1_plan_id, row.o2_plan_id): (row.H, row.O1, row.O2, row.R)
            for row in original.payoff_rows
        },
        row_order=[
            ("B", "R"),
            ("A", "R"),
            ("B", "L"),
            ("A", "L"),
        ],
    )
    first = success(original)
    second = success(permuted)
    assert first == second

    reordered_payoffs = {
        ("B", "L"): (-4, 0, 0, 4),
        ("B", "R"): (-4, 2, 2, 0),
        ("A", "L"): (-2, 1, 1, 0),
        ("A", "R"): (-4, 0, 0, 4),
    }
    reordered = success(
        game(("B", "A"), ("L", "R"), reordered_payoffs)
    )
    assert reordered["response_game_identity"] != first["response_game_identity"]


def test_a21_same_input_twice_is_byte_identical_and_strict_json():
    first_result = solve_three_player_response(coordination())
    second_result = solve_three_player_response(coordination())
    first = exact_response_json(first_result)
    second = exact_response_json(second_result)
    assert first == second
    assert "\n" not in first and "\r" not in first
    assert "NaN" not in first and "Infinity" not in first


@pytest.mark.parametrize(
    "limits",
    [
        replace(ExactResponseLimits(), max_output_records=1),
        replace(ExactResponseLimits(), max_output_bytes=1),
    ],
)
def test_a23_output_caps_are_no_truncation_no_partial(limits):
    result = solve(coordination(), limits=limits)
    assert_no_partial(result, CAP_EXCEEDED)


def test_duplicate_missing_unknown_rows_rectangular_and_empty_inputs():
    candidate = coordination()
    assert_no_partial(
        solve(replace(candidate, o1_plan_ids=())), INVALID_INPUT
    )
    assert_no_partial(
        solve(replace(candidate, o1_plan_ids=("A", "A"))), INVALID_INPUT
    )
    assert_no_partial(
        solve(replace(candidate, payoff_rows=candidate.payoff_rows[:-1])),
        INVALID_INPUT,
    )
    duplicate = candidate.payoff_rows[:-1] + (candidate.payoff_rows[0],)
    assert_no_partial(
        solve(replace(candidate, payoff_rows=duplicate)), INVALID_INPUT
    )
    unknown = replace(
        candidate.payoff_rows[0], o1_plan_id="UNKNOWN"
    )
    assert_no_partial(
        solve(
            replace(
                candidate,
                payoff_rows=(unknown,) + candidate.payoff_rows[1:],
            )
        ),
        INVALID_INPUT,
    )


def test_singular_systems_and_lower_dimensional_cells_are_complete():
    output = success(fully_degenerate())
    assert output["counts"]["exact_linear_systems_solved"] > 0
    assert any(cell["dimension"] == 0 for cell in output["support_cells"])
    assert any(cell["dimension"] == 1 for cell in output["support_cells"])
    assert any(cell["dimension"] == 2 for cell in output["support_cells"])
    assert (
        output["independent_verification"]["support_pairs_reconstructed"]
        == output["counts"]["support_pairs_total"]
    )


def test_valid_finite_game_zero_solver_cells_is_fail_closed(monkeypatch):
    def empty_solver(_table, o1_supports, o2_supports, _limits):
        audit = [
            {
                "support_pair": (o1_support, o2_support),
                "outcome": "empty_o1_mixture_polytope",
                "support_cell_id": None,
            }
            for o1_support in o1_supports
            for o2_support in o2_supports
        ]
        return [], audit, module._SolveCounters()

    monkeypatch.setattr(module, "_enumerate_support_cells", empty_solver)
    result = solve(coordination())
    assert_no_partial(result, NUMERIC_FAILURE)


def test_unexpected_failure_is_internal_and_never_leaks_exception(monkeypatch):
    monkeypatch.setattr(
        module, "_enumerate_support_cells", lambda *_a, **_k: 1 / 0
    )
    result = solve(coordination())
    assert_no_partial(result, INTERNAL_FAILURE)
    assert result.error.message == "unexpected exact response failure"
    assert "division" not in exact_response_json(result)


def test_coalition_fields_do_not_change_primary_response_status_or_cells():
    first = success(coalition_counterexample())
    changed = game(
        ("A", "B"),
        ("L", "R"),
        {
            ("A", "L"): (-20, 1, 1, 18),
            ("A", "R"): (-30, 3, 0, 27),
            ("B", "L"): (-30, 0, 3, 27),
            ("B", "R"): (-40, 2, 2, 36),
        },
    )
    second = success(changed)
    assert first["status"] == second["status"] == EXACT_CORRESPONDENCE_COMPLETE
    assert [
        (
            cell["o1_mixture_polytope"]["vertices"],
            cell["o2_mixture_polytope"]["vertices"],
        )
        for cell in first["support_cells"]
    ] == [
        (
            cell["o1_mixture_polytope"]["vertices"],
            cell["o2_mixture_polytope"]["vertices"],
        )
        for cell in second["support_cells"]
    ]
    assert first["hero_min_joint_plan_stress"]["hero_minimum"] == "-4"
    assert second["hero_min_joint_plan_stress"]["hero_minimum"] == "-40"


def test_contract_algorithm_and_limit_types_fail_with_stable_status():
    assert_no_partial(
        solve(replace(coordination(), contract_version="v2")),
        UNSUPPORTED_MODEL,
    )
    assert_no_partial(
        solve(replace(coordination(), algorithm_version="other")),
        UNSUPPORTED_MODEL,
    )
    assert_no_partial(
        solve(
            coordination(),
            limits=replace(ExactResponseLimits(), max_support_pairs=True),
        ),
        INVALID_INPUT,
    )


def test_a24_module_is_isolated_from_current_cfr_and_public_surface():
    source = inspect.getsource(module)
    assert "three_player_cfr" not in source
    assert "three_player_cfr_file_workflow" not in source
    assert "approximate" in (inspect.getdoc(module) or "").lower()
    assert all(path.exists() for path in PROTECTED)
    package_init = (ROOT / "src" / "repeated_poker" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "three_player_response" not in package_init


def test_success_output_has_required_contract_counts_extrema_and_identity_fields():
    output = success(coordination())
    assert output["contract_version"] == CONTRACT_VERSION
    assert output["algorithm_version"] == ALGORITHM_VERSION
    assert output["status"] == EXACT_CORRESPONDENCE_COMPLETE
    assert set(output["utility_extrema"]) == {"H", "O1", "O2", "R"}
    assert set(output["content_identities"]) == {
        "response_game_structure",
        "fixed_hero",
        "perfect_recall_evidence",
        "rake_convention",
        "payoff_table",
        "config",
        "supplied_profile",
        "candidate",
        "search_mode_context",
        "run_context",
    }
    assert output["counts"]["support_pairs_total"] == 9
    assert output["counts"]["support_pairs_visited"] == 9
    assert output["independent_verification"]["operations_used"] > 0
    assert output["limits"]["max_pure_plans_o1"] == 4


def test_identity_grammar_and_plan_id_grammar_are_strict():
    assert_no_partial(
        solve(replace(coordination(), fixed_hero_identity="hero")),
        INVALID_INPUT,
    )
    assert_no_partial(
        solve(replace(coordination(), o1_plan_ids=("bad id", "B"))),
        INVALID_INPUT,
    )


def test_output_witness_cap_failure_discards_primary_payload(monkeypatch):
    baseline = success(fully_degenerate())
    witness_records = sum(
        len(baseline["utility_extrema"][name]["minimum_witnesses"])
        + len(baseline["utility_extrema"][name]["maximum_witnesses"])
        for name in ("H", "O1", "O2", "R")
    )
    assert witness_records > 4
    base_records = 256 + baseline["counts"]["support_pairs_total"] + sum(
        8
        + len(cell["source_support_pairs"])
        + cell["o1_mixture_polytope"]["vertex_count"]
        + cell["o2_mixture_polytope"]["vertex_count"]
        for cell in baseline["support_cells"]
    )
    pure_called = False
    original = module._pure_stability_subset

    def tracked_pure(*args, **kwargs):
        nonlocal pure_called
        pure_called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_pure_stability_subset", tracked_pure)
    result = solve(
        fully_degenerate(),
        limits=replace(
            ExactResponseLimits(), max_output_records=base_records + 1
        ),
    )
    assert_no_partial(result, CAP_EXCEEDED)
    assert result.error.phase == "output"
    assert pure_called is False


def test_local_vertex_cap_is_enforced_before_cell_vertex_materialization():
    limits = replace(ExactResponseLimits(), max_vertices_per_cell=1)
    with pytest.raises(module._ResponseFailure) as raised:
        module._solver_vertices(
            [((Fraction(1), Fraction(1)), Fraction(1))],
            [
                ((Fraction(-1), Fraction(0)), Fraction(0)),
                ((Fraction(0), Fraction(-1)), Fraction(0)),
            ],
            2,
            limits,
            module._SolveCounters(),
        )
    assert raised.value.status == CAP_EXCEEDED
    assert "before vertex allocation" in str(raised.value)


def test_linear_system_and_verifier_operation_caps_fail_closed():
    assert_no_partial(
        solve(
            coordination(),
            limits=replace(
                ExactResponseLimits(), max_exact_linear_systems=1
            ),
        ),
        CAP_EXCEEDED,
    )
    assert_no_partial(
        solve(
            coordination(),
            limits=replace(
                ExactResponseLimits(), max_verifier_operations=1
            ),
        ),
        CAP_EXCEEDED,
    )
