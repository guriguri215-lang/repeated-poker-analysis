"""M35 known-board real-card three-player river/rake contract tests."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from repeated_poker import known_board_real_card_three_player_river as module
from repeated_poker.aiof_cards import (
    AiofContractError,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    canonicalize_exact_combo,
    card_id,
)
from repeated_poker.known_board_real_card_hu_river import (
    ComboBucketAssignment,
    ComboBucketMap,
)
from repeated_poker.three_player_candidate_repeated import (
    CANDIDATE_RESPONSE_FAILURE,
    ThreePlayerCandidateGenerationConfig,
    ThreePlayerCandidateRepeatedError,
    ThreePlayerCandidateRepeatedResult,
    ThreePlayerRepeatedConfig,
)


BOARD = ("2c", "3d", "4h", "5s", "9c")
ATTESTATION = module.GeneratedTreeAttestation(
    verifier="M35 focused-test verifier",
    verification_date="2026-07-25",
    evidence_version="m35-focused-test-v1",
)


def exact_range(*rows: tuple[str, float] | str) -> RangeSpec:
    normalized = tuple(
        (row, 1.0) if isinstance(row, str) else row for row in rows
    )
    return RangeSpec(
        tuple(
            RangeEntry(label, weight, WeightBasis.EXACT_COMBO_MASS)
            for label, weight in normalized
        )
    )


def class_range(label: str, weight: float = 1.0) -> RangeSpec:
    return RangeSpec(
        (RangeEntry(label, weight, WeightBasis.CLASS_TOTAL_MASS),)
    )


def distribution(**probabilities: str) -> tuple[module.ExactActionProbability, ...]:
    return tuple(
        module.ExactActionProbability(action, probability)
        for action, probability in probabilities.items()
    )


def profile(
    *,
    hero_buckets: tuple[str, ...],
    o1_buckets: tuple[str, ...],
    o2_buckets: tuple[str, ...],
    hero_bet: str = "0",
    o1_call: str = "1",
    o2_call_after_call: str = "1",
    o2_call_after_fold: str = "1",
) -> module.RealCardThreePlayerProfile:
    rows: list[module.RealCardProfileRow] = []
    for bucket in hero_buckets:
        rows.append(
            module.RealCardProfileRow(
                "H",
                bucket,
                "open",
                distribution(check=str(1 - Fraction(hero_bet)), bet=hero_bet),
            )
        )
    for bucket in o1_buckets:
        rows.extend(
            (
                module.RealCardProfileRow(
                    "O1",
                    bucket,
                    "after_hero_check",
                    distribution(check="1"),
                ),
                module.RealCardProfileRow(
                    "O1",
                    bucket,
                    "vs_hero_bet",
                    distribution(
                        call=o1_call,
                        fold=str(1 - Fraction(o1_call)),
                    ),
                ),
            )
        )
    for bucket in o2_buckets:
        rows.extend(
            (
                module.RealCardProfileRow(
                    "O2",
                    bucket,
                    "after_hero_o1_check",
                    distribution(check="1"),
                ),
                module.RealCardProfileRow(
                    "O2",
                    bucket,
                    "vs_hero_bet_o1_call",
                    distribution(
                        call=o2_call_after_call,
                        fold=str(1 - Fraction(o2_call_after_call)),
                    ),
                ),
                module.RealCardProfileRow(
                    "O2",
                    bucket,
                    "vs_hero_bet_o1_fold",
                    distribution(
                        call=o2_call_after_fold,
                        fold=str(1 - Fraction(o2_call_after_fold)),
                    ),
                ),
            )
        )
    return module.RealCardThreePlayerProfile(tuple(rows))


def request(
    *,
    hero: str = "AsAh",
    o1: str = "KsKh",
    o2: str = "QsQh",
    hero_range: RangeSpec | None = None,
    o1_range: RangeSpec | None = None,
    o2_range: RangeSpec | None = None,
    baseline_profile: module.RealCardThreePlayerProfile | None = None,
    hero_bet: str = "0",
    o1_call: str = "1",
    o2_call_after_call: str = "1",
    o2_call_after_fold: str = "1",
    shifts: tuple[str, ...] = ("1",),
    horizon: int = 3,
    rake_rate: str = "0",
    rake_cap: str | None = None,
    dead_cards: tuple[str, ...] = (),
    limits: module.KnownBoardRealCardThreePlayerLimits | None = None,
) -> module.KnownBoardRealCardThreePlayerRequest:
    hero = canonicalize_exact_combo(hero)
    o1 = canonicalize_exact_combo(o1)
    o2 = canonicalize_exact_combo(o2)
    return module.KnownBoardRealCardThreePlayerRequest(
        board=BOARD,
        dead_cards=dead_cards,
        hero_range=hero_range or exact_range(hero),
        o1_range=o1_range or exact_range(o1),
        o2_range=o2_range or exact_range(o2),
        baseline_profile=baseline_profile
        or profile(
            hero_buckets=(hero,),
            o1_buckets=(o1,),
            o2_buckets=(o2,),
            hero_bet=hero_bet,
            o1_call=o1_call,
            o2_call_after_call=o2_call_after_call,
            o2_call_after_fold=o2_call_after_fold,
        ),
        attestation=ATTESTATION,
        rake_rate=rake_rate,
        rake_cap=rake_cap,
        generation=ThreePlayerCandidateGenerationConfig(
            shift_amounts=shifts
        ),
        repeated=ThreePlayerRepeatedConfig(horizon=horizon),
        limits=limits or module.KnownBoardRealCardThreePlayerLimits(),
    )


def success(
    value: module.KnownBoardRealCardThreePlayerRequest,
) -> module.KnownBoardRealCardThreePlayerPayload:
    result = module.analyze_known_board_real_card_three_player_river(value)
    assert result.status is AiofStatus.SUCCESS, result.error_message
    assert result.payload is not None
    assert result.error_message is None
    return result.payload


def independent_triples(
    hero: tuple[tuple[str, Fraction], ...],
    o1: tuple[tuple[str, Fraction], ...],
    o2: tuple[tuple[str, Fraction], ...],
) -> dict[tuple[str, str, str], Fraction]:
    """Independent test-only exhaustive compatible-triple oracle."""

    raw: dict[tuple[str, str, str], Fraction] = {}
    for hero_combo, hero_mass in hero:
        for o1_combo, o1_mass in o1:
            for o2_combo, o2_mass in o2:
                canonical = tuple(
                    canonicalize_exact_combo(combo)
                    for combo in (hero_combo, o1_combo, o2_combo)
                )
                cards = tuple(
                    card_id(combo[index : index + 2])
                    for combo in canonical
                    for index in (0, 2)
                )
                if len(set(cards)) == 6:
                    raw[canonical] = (
                        hero_mass * o1_mass * o2_mass
                    )
    total = sum(raw.values(), Fraction())
    return {key: value / total for key, value in raw.items()}


def terminal_records(payload: module.KnownBoardRealCardThreePlayerPayload):
    response = payload.native_m32_result.analysis.baseline_response
    assert response.scenario_evaluation is not None
    return response.scenario_evaluation["terminal_records"]


def test_exact_joint_support_normalizes_and_matches_independent_oracle():
    hero_rows = (("AsAh", Fraction(2)), ("KsKh", Fraction(1)))
    o1_rows = (("QsQh", Fraction(3)), ("JsJh", Fraction(1)))
    o2_rows = (("TsTh", Fraction(1)), ("8s8h", Fraction(2)))
    prepared = module.prepare_known_board_three_player_support(
        board=BOARD,
        dead_cards=(),
        hero_range=exact_range(*[(combo, float(weight)) for combo, weight in hero_rows]),
        o1_range=exact_range(*[(combo, float(weight)) for combo, weight in o1_rows]),
        o2_range=exact_range(*[(combo, float(weight)) for combo, weight in o2_rows]),
    )
    actual = {
        (row.hero_combo, row.o1_combo, row.o2_combo): Fraction(
            row.probability_exact
        )
        for row in prepared.triples
    }
    assert actual == independent_triples(hero_rows, o1_rows, o2_rows)
    assert sum(actual.values(), Fraction()) == 1
    for player in ("H", "O1", "O2"):
        assert (
            sum(
                Fraction(row.probability_exact)
                for row in prepared.marginals
                if row.player_id == player
            )
            == 1
        )


def test_board_and_dead_blockers_are_separately_accounted():
    prepared = module.prepare_known_board_three_player_support(
        board=BOARD,
        dead_cards=("7c",),
        hero_range=exact_range("AsAh", "2c2h", "7c7d"),
        o1_range=exact_range("KsKh"),
        o2_range=exact_range("QsQh"),
    )
    hero = prepared.range_provenance[0]
    assert hero.initial_combo_count == 3
    assert hero.board_removed_combo_count == 1
    assert hero.dead_removed_combo_count == 1
    assert hero.surviving_combo_count == 1
    assert prepared.triples[0].hero_combo == "AsAh"


def test_hero_blocks_both_opponents_and_o1_o2_collision_changes_conditionals():
    hero_rows = (("AsAh", Fraction(1)), ("7s7h", Fraction(1)))
    o1_rows = (("AsKd", Fraction(1)), ("KcKh", Fraction(1)))
    o2_rows = (("AhQd", Fraction(1)), ("KdQh", Fraction(1)))
    prepared = module.prepare_known_board_three_player_support(
        board=BOARD,
        dead_cards=(),
        hero_range=exact_range("AsAh", "7s7h"),
        o1_range=exact_range("AsKd", "KcKh"),
        o2_range=exact_range("AhQd", "KdQh"),
    )
    actual = {
        (row.hero_combo, row.o1_combo, row.o2_combo): Fraction(
            row.probability_exact
        )
        for row in prepared.triples
    }
    oracle = independent_triples(hero_rows, o1_rows, o2_rows)
    assert actual == oracle
    assert prepared.private_collision_excluded_count == 4
    hero_as = next(
        Fraction(row.probability_exact)
        for row in prepared.marginals
        if row.player_id == "H" and row.combo == "AsAh"
    )
    o1_k = next(
        Fraction(row.probability_exact)
        for row in prepared.marginals
        if row.player_id == "O1"
        and row.combo == canonicalize_exact_combo("KcKh")
    )
    # Conditioning is joint: its admissible cell probability is not the
    # product of the two independently normalized seat marginals.
    joint = actual[
        tuple(
            canonicalize_exact_combo(combo)
            for combo in ("AsAh", "KcKh", "KdQh")
        )
    ]
    o2_b = next(
        Fraction(row.probability_exact)
        for row in prepared.marginals
        if row.player_id == "O2" and row.combo == "KdQh"
    )
    assert joint != hero_as * o1_k * o2_b


def test_class_169_input_reuses_existing_expansion_contract():
    prepared = module.prepare_known_board_three_player_support(
        board=BOARD,
        dead_cards=(),
        hero_range=class_range("AA"),
        o1_range=exact_range("KsKh"),
        o2_range=exact_range("QsQh"),
    )
    assert prepared.range_provenance[0].initial_combo_count == 6
    assert prepared.compatible_triple_count == 6
    assert {
        Fraction(row.probability_exact) for row in prepared.triples
    } == {Fraction(1, 6)}


def test_joint_blocker_weights_propagate_through_showdown_and_native_response():
    hero_rows = (("AsAh", Fraction(1)), ("7s7h", Fraction(1)))
    o1_rows = (("AsKd", Fraction(1)), ("KcKh", Fraction(1)))
    o2_rows = (("AhQd", Fraction(1)), ("KdQh", Fraction(1)))
    ranges = {
        "H": exact_range("AsAh", "7s7h"),
        "O1": exact_range("AsKd", "KcKh"),
        "O2": exact_range("AhQd", "KdQh"),
    }

    def grouped(rows, bucket):
        combos = tuple(
            sorted(canonicalize_exact_combo(combo) for combo, _ in rows)
        )
        return ComboBucketMap(
            bucket_ids=(bucket,),
            assignments=tuple(
                ComboBucketAssignment(combo, bucket) for combo in combos
            ),
        )

    candidate = module.KnownBoardRealCardThreePlayerRequest(
        board=BOARD,
        dead_cards=(),
        hero_range=ranges["H"],
        o1_range=ranges["O1"],
        o2_range=ranges["O2"],
        hero_combo_to_bucket=grouped(hero_rows, "H-all"),
        o1_combo_to_bucket=grouped(o1_rows, "O1-all"),
        o2_combo_to_bucket=grouped(o2_rows, "O2-all"),
        baseline_profile=profile(
            hero_buckets=("H-all",),
            o1_buckets=("O1-all",),
            o2_buckets=("O2-all",),
        ),
        attestation=ATTESTATION,
        generation=ThreePlayerCandidateGenerationConfig(shift_amounts=()),
    )
    payload = success(candidate)
    oracle = independent_triples(hero_rows, o1_rows, o2_rows)
    assert {
        (row.hero_combo, row.o1_combo, row.o2_combo): Fraction(
            row.probability_exact
        )
        for row in payload.prepared_support.triples
    } == oracle
    rank_rows = {
        (row.hero_combo, row.o1_combo, row.o2_combo): row
        for row in payload.showdown_rows
    }
    independently_weighted_hero = sum(
        probability
        * (
            Fraction(30, len(rank_rows[key].three_way_winners))
            - 10
            if "H" in rank_rows[key].three_way_winners
            else Fraction(-10)
        )
        for key, probability in oracle.items()
    )
    analysis = payload.native_m32_result.analysis
    assert analysis.baseline_exact_values["H"] == str(
        independently_weighted_hero
    )
    assert (
        Fraction(analysis.baseline_response.response["hero_worst"])
        == independently_weighted_hero
    )
    assert analysis.baseline_response.scenario_evaluation["counts"][
        "chance_outcomes"
    ] == len(oracle)


def test_impossible_support_and_duplicate_card_input_fail_closed():
    with pytest.raises(AiofContractError) as error:
        module.prepare_known_board_three_player_support(
            board=BOARD,
            dead_cards=(),
            hero_range=exact_range("AsAh"),
            o1_range=exact_range("AsKd"),
            o2_range=exact_range("AhQd"),
        )
    assert error.value.status is AiofStatus.EMPTY_COMPATIBLE_SUPPORT

    result = module.analyze_known_board_real_card_three_player_river(
        replace(request(), board=("2c", "2c", "4h", "5s", "9c"))
    )
    assert result.status is AiofStatus.INVALID_CARD_INPUT
    assert result.payload is None
    encoded = module.exact_known_board_real_card_three_player_json(result)
    assert '"payload":null' in encoded
    assert '"showdown_rows"' not in encoded


def test_unique_winner_and_three_way_tie_are_exact_ranked():
    unique = success(request()).showdown_rows[0]
    assert unique.three_way_winners == ("H",)
    assert dict(unique.heads_up_winners)["H-O1"] == ("H",)

    tied = success(
        request(hero="AsKd", o1="AhKc", o2="AdQd", shifts=())
    ).showdown_rows[0]
    assert tied.three_way_winners == ("H", "O1", "O2")
    assert dict(tied.heads_up_winners)["H-O1"] == ("H", "O1")


def test_fold_heads_up_three_way_uncalled_and_zero_rake_conserve():
    payload = success(request(shifts=()))
    records = terminal_records(payload)
    by_id = {row["node_id"]: row for row in records}
    three_way = next(
        row for key, row in by_id.items() if "both-call" in key
    )
    heads_up = next(
        row for key, row in by_id.items() if "O1-call-O2-fold" in key
    )
    fold = next(row for key, row in by_id.items() if "both-fold" in key)
    assert three_way["rake_amount"] == "0"
    assert heads_up["rake_amount"] == "0"
    assert fold["kind"] == "fold"
    assert fold["uncalled_return"] == {"H": "10", "O1": "0", "O2": "0"}
    for row in (three_way, heads_up, fold):
        assert sum(Fraction(value) for value in row["utility"].values()) == 0


@pytest.mark.parametrize(
    ("rake_rate", "rake_cap", "expected"),
    (("1/10", None, "6"), ("1/2", "2", "2")),
)
def test_positive_rake_below_cap_and_binding_cap_conserve(
    rake_rate: str, rake_cap: str | None, expected: str
):
    payload = success(
        request(rake_rate=rake_rate, rake_cap=rake_cap, shifts=())
    )
    record = next(
        row
        for row in terminal_records(payload)
        if "both-call" in row["node_id"]
    )
    assert record["rake_amount"] == expected
    assert sum(Fraction(value) for value in record["utility"].values()) == 0


def test_unique_on_path_response_value_and_complete_worst_best_interval():
    unique = success(
        request(
            hero_bet="1",
            o1_call="0",
            o2_call_after_call="0",
            o2_call_after_fold="0",
            shifts=(),
        )
    ).native_m32_result.analysis.baseline_response.response
    assert unique["hero_worst"] == unique["hero_best"] == "20"
    stable = unique["pure_profile_unilateral_stability"]["rows"]
    assert {row["plans"]["O1"] for row in stable} == {"O1:1"}
    assert {row["utility"]["H"] for row in stable} == {"20"}
    # Complete M30 output retains the two behaviorally distinct off-path O2
    # plans even though the realized response and Hero value are unique.
    assert len(unique["hero_worst_witnesses"]) == 2

    multiple = success(
        request(
            hero="AsKd",
            o1="AhKc",
            o2="QsQh",
            hero_bet="1",
            o1_call="0",
            o2_call_after_call="0",
            o2_call_after_fold="0",
            rake_rate="3/5",
            shifts=(),
        )
    ).native_m32_result.analysis.baseline_response.response
    assert Fraction(multiple["hero_worst"]) < Fraction(multiple["hero_best"])
    assert len(multiple["hero_worst_witnesses"]) >= 1
    assert len(multiple["hero_best_witnesses"]) >= 1


def test_baseline_identity_and_each_candidate_use_fresh_exact_response(monkeypatch):
    first_request = request()
    first = success(first_request)
    changed = success(replace(first_request, rake_rate="1/10"))
    assert first.baseline_identity != changed.baseline_identity

    original = module._m32._m31.evaluate_three_player_river_rake
    calls: list[object] = []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(
        module._m32._m31, "evaluate_three_player_river_rake", wrapped
    )
    payload = success(first_request)
    assert len(payload.native_m32_result.analysis.candidates) == 1
    assert len(calls) == 2  # baseline plus one fresh candidate response


def test_complete_profile_and_expected_baseline_identity_are_strict():
    value = request(shifts=())
    payload = success(value)
    assert (
        success(
            replace(
                value,
                expected_baseline_identity=payload.baseline_identity,
            )
        ).baseline_identity
        == payload.baseline_identity
    )
    stale = module.analyze_known_board_real_card_three_player_river(
        replace(value, expected_baseline_identity="0" * 64)
    )
    assert stale.status is AiofStatus.INVALID_INPUT
    assert stale.payload is None

    incomplete = module.RealCardThreePlayerProfile(
        value.baseline_profile.rows[:-1]
    )
    invalid = module.analyze_known_board_real_card_three_player_river(
        replace(value, baseline_profile=incomplete)
    )
    assert invalid.status is AiofStatus.INVALID_STRATEGY
    assert invalid.payload is None


def test_repeated_uplift_and_no_beneficial_commitment():
    profitable = success(request(horizon=3))
    analysis = profitable.native_m32_result.analysis
    candidate = analysis.candidates[0]
    assert (
        Fraction(candidate.exact_values["initial_profile"]["H"])
        > Fraction(candidate.exact_values["response"]["hero_worst"])
    )
    assert analysis.selector_report.rows[-1].selected_candidate_id is not None

    no_benefit = success(request(horizon=1))
    assert (
        no_benefit.native_m32_result.analysis.selector_report.rows[
            0
        ].selected_candidate_id
        is None
    )


def test_cap_is_checked_before_rank_allocation_and_no_partial_result(monkeypatch):
    def forbidden_rank(*_args, **_kwargs):
        raise AssertionError("ranking must not start after a preflight cap failure")

    monkeypatch.setattr(module, "evaluate_seven_card_hand", forbidden_rank)
    capped = request(
        hero_range=exact_range("AsAh", "7s7h"),
        baseline_profile=profile(
            hero_buckets=("AsAh",),
            o1_buckets=("KsKh",),
            o2_buckets=("QsQh",),
        ),
        limits=replace(
            module.KnownBoardRealCardThreePlayerLimits(),
            max_compatible_triples=1,
        ),
    )
    result = module.analyze_known_board_real_card_three_player_river(capped)
    assert result.status is AiofStatus.CAP_EXCEEDED
    assert result.payload is None
    assert '"showdown_rows"' not in module.exact_known_board_real_card_three_player_json(
        result
    )


def test_nested_response_failure_is_fail_closed_without_partial_payload(monkeypatch):
    failure = ThreePlayerCandidateRepeatedResult(
        status=CANDIDATE_RESPONSE_FAILURE,
        analysis=None,
        error=ThreePlayerCandidateRepeatedError(
            phase="candidate_response",
            message="test-only injected failure",
        ),
        partial_result=False,
    )
    monkeypatch.setattr(
        module._m32,
        "evaluate_three_player_candidate_repeated",
        lambda *_args, **_kwargs: failure,
    )
    result = module.analyze_known_board_real_card_three_player_river(request())
    assert result.status is AiofStatus.ORACLE_MISMATCH
    assert result.payload is None
    assert result.error_message is not None
    encoded = module.exact_known_board_real_card_three_player_json(result)
    assert '"payload":null' in encoded
    assert '"showdown_rows"' not in encoded


def test_deterministic_identity_and_serialization():
    first = module.analyze_known_board_real_card_three_player_river(request())
    second = module.analyze_known_board_real_card_three_player_river(request())
    assert first.status is AiofStatus.SUCCESS
    assert first.payload.identities == second.payload.identities
    encoded = module.exact_known_board_real_card_three_player_json(first)
    assert encoded == module.exact_known_board_real_card_three_player_json(second)
    parsed = json.loads(encoded)
    assert parsed["payload"]["identities"]["analysis"] == (
        first.payload.identities["analysis"]
    )
