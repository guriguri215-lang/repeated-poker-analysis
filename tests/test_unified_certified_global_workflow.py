"""Cross-module contract/oracle tests for the M40 unified workflow."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from repeated_poker import aiof_preflop_certified_global as m37
from repeated_poker import certified_global_optimizer as m36
from repeated_poker import known_board_real_card_hu_certified_global as m38
from repeated_poker import known_board_real_card_three_player_river as m35
from repeated_poker import three_player_certified_global as m39
from repeated_poker import three_player_river_rake as m31
from repeated_poker import unified_certified_global_workflow as module
from repeated_poker.aiof_cards import (
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
    canonicalize_exact_combo,
)
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    SuppliedProfile,
)
from repeated_poker.known_board_real_card_hu_river import (
    ActionProbability,
    ComboBucketAssignment,
    ComboBucketMap,
    RiverActionProfile,
    RiverProfileRow,
)
from repeated_poker.three_player_candidate_repeated import (
    ThreePlayerRepeatedConfig,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "repeated_poker" / (
    "unified_certified_global_workflow.py"
)
GUIDE = ROOT / "docs" / "unified_certified_global_workflow.md"
EXAMPLE = ROOT / "examples" / "unified_certified_global_workflow.py"
README = ROOT / "README.md"
ASSUMPTIONS = ROOT / "docs" / "assumptions_and_limitations.md"
SUCCESS = (m36.CERTIFIED_GLOBAL, m36.CERTIFIED_EPSILON_GLOBAL)


def exact_range(*labels: str) -> RangeSpec:
    return RangeSpec(
        tuple(
            RangeEntry(
                canonicalize_exact_combo(label),
                1.0,
                WeightBasis.EXACT_COMBO_MASS,
            )
            for label in labels
        )
    )


def dead_except(live_cards: tuple[str, ...]) -> tuple[str, ...]:
    live = {card_id(card) for card in live_cards}
    return tuple(
        card_from_id(value) for value in range(52) if value not in live
    )


def m37_request(
    *,
    reverse: bool = False,
    horizon: int = 2,
    integration_limits: m37.AiofPreflopCertifiedGlobalLimits | None = None,
    optimizer_limits: m36.CertifiedGlobalOptimizerLimits | None = None,
) -> m37.AiofPreflopCertifiedGlobalRequest:
    sb = "AsAh"
    bb = "KsKh"
    board = ("2c", "3d", "4h", "5s", "7c")
    dead = dead_except(("As", "Ah", "Ks", "Kh") + board)
    if reverse:
        dead = tuple(reversed(dead))
    return m37.AiofPreflopCertifiedGlobalRequest(
        game=HeadsUpChipEvGame(
            starting_stack_sb=10.0,
            starting_stack_bb=10.0,
            small_blind=0.5,
            big_blind=1.0,
            ante=0.0,
        ),
        sb_range=exact_range(sb),
        bb_range=exact_range(bb),
        dead_cards=dead,
        baseline_profile=SuppliedProfile(
            sb_shove=(ComboActionProbability(sb, 0.0),),
            bb_call=(ComboActionProbability(bb, 0.0),),
        ),
        hero_seat="sb",
        horizon=horizon,
        adaptation_opportunity=2,
        discount="1",
        absolute_gap_tolerance="0",
        relative_gap_tolerance="0",
        integration_limits=(
            integration_limits
            or m37.AiofPreflopCertifiedGlobalLimits()
        ),
        optimizer_limits=(
            optimizer_limits or m36.CertifiedGlobalOptimizerLimits()
        ),
    )


def distribution_float(
    **values: float,
) -> tuple[ActionProbability, ...]:
    return tuple(
        ActionProbability(action, probability)
        for action, probability in values.items()
    )


def m38_request() -> m38.KnownBoardRealCardHuCertifiedGlobalRequest:
    hero_profile = RiverActionProfile(
        (
            RiverProfileRow(
                "H",
                "after_oop_check",
                distribution_float(check=1.0, bet=0.0),
            ),
            RiverProfileRow(
                "H",
                "vs_oop_bet",
                distribution_float(
                    **{"call": 0.0, "fold": 1.0, "raise": 0.0}
                ),
            ),
        )
    )
    villain_profile = RiverActionProfile(
        (
            RiverProfileRow(
                "V",
                "oop_first",
                distribution_float(check=0.5, bet=0.5),
            ),
            RiverProfileRow(
                "V",
                "vs_ip_bet",
                distribution_float(call=0.5, fold=0.5),
            ),
            RiverProfileRow(
                "V",
                "vs_ip_raise",
                distribution_float(call=0.5, fold=0.5),
            ),
        )
    )
    return m38.KnownBoardRealCardHuCertifiedGlobalRequest(
        board=("Qc", "Jd", "9h", "7s", "4c"),
        hero_range=exact_range("AsAh", "2s2h"),
        villain_range=exact_range("KsKh"),
        baseline_hero_profile=hero_profile,
        hero_combo_to_bucket=ComboBucketMap(
            ("H",),
            (
                ComboBucketAssignment("AsAh", "H"),
                ComboBucketAssignment("2s2h", "H"),
            ),
        ),
        villain_combo_to_bucket=ComboBucketMap(
            ("V",), (ComboBucketAssignment("KsKh", "V"),)
        ),
        baseline_villain_profile=villain_profile,
        initial_commitment_hero=1.0,
        initial_commitment_villain=1.0,
        rake_rate=0.05,
        rake_cap=0.2,
        oop_bet_size=2.0,
        ip_bet_after_check_size=2.0,
        ip_raise_to_size=5.0,
        horizon=1,
        adaptation_opportunity=1,
        discount="1",
        absolute_gap_tolerance="1/64",
    )


def observation(suffix: str) -> m31.RiverObservation:
    return m31.RiverObservation(
        f"public-{suffix}",
        {
            "H": f"hero-{suffix}",
            "O1": f"o1-{suffix}",
            "O2": f"o2-{suffix}",
        },
    )


def abstract_scenario() -> m31.ThreePlayerRiverRakeScenario:
    showdown = m31.RiverTerminalNode(
        "hero-check-terminal",
        "showdown",
        (m31.AwardShare("H", "1"),),
    )
    o2_check = m31.RiverDecisionNode(
        "o2-check-after-hero",
        "O2",
        "O2_after_hero_check",
        (m31.RiverAction("check", "check", None, showdown),),
    )
    o1_check = m31.RiverDecisionNode(
        "o1-check-after-hero",
        "O1",
        "O1_after_hero_check",
        (m31.RiverAction("check", "check", None, o2_check),),
    )
    fold = m31.RiverTerminalNode("hero-bet-both-fold", "fold")
    o2_fold = m31.RiverDecisionNode(
        "o2-fold-after-hero",
        "O2",
        "O2_after_hero_bet",
        (m31.RiverAction("fold", "fold", None, fold),),
    )
    o1_fold = m31.RiverDecisionNode(
        "o1-fold-after-hero",
        "O1",
        "O1_after_hero_bet",
        (m31.RiverAction("fold", "fold", None, o2_fold),),
    )
    root = m31.RiverDecisionNode(
        "hero-two-actions",
        "H",
        "H_two_actions",
        (
            m31.RiverAction("check", "check", None, o1_check),
            m31.RiverAction("bet", "bet", "20", o1_fold),
        ),
    )
    return m31.ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=observation("m40"),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def m39_abstract_request(
    *,
    optimizer_limits: m36.CertifiedGlobalOptimizerLimits | None = None,
    absolute_gap_tolerance: object = "100",
) -> m39.AbstractThreePlayerCertifiedGlobalRequest:
    scenario = abstract_scenario()
    return m39.AbstractThreePlayerCertifiedGlobalRequest(
        scenario=scenario,
        baseline_fixed_hero_policy=m31.ExactBehaviorPolicy(
            {"H_two_actions": {"check": "1", "bet": "0"}}
        ),
        initial_profile=m31.OpponentInitialProfile(
            {
                "O1_after_hero_bet": {"fold": "1"},
                "O1_after_hero_check": {"check": "1"},
            },
            {
                "O2_after_hero_bet": {"fold": "1"},
                "O2_after_hero_check": {"check": "1"},
            },
        ),
        attestation=m31.create_perfect_recall_attestation(
            scenario,
            verifier="M40 test-owned cross-module oracle",
            verification_date="2026-07-26",
            evidence_version="m40-cross-module-v1",
            o1_confirmed=True,
            o2_confirmed=True,
        ),
        repeated=m39.ThreePlayerCertifiedRepeatedConfig(
            horizon=3,
            adaptation_opportunity=2,
            discount=0.5,
        ),
        absolute_gap_tolerance=absolute_gap_tolerance,
        optimizer_limits=(
            optimizer_limits or m36.CertifiedGlobalOptimizerLimits()
        ),
    )


def distribution_exact(
    **values: str,
) -> tuple[m35.ExactActionProbability, ...]:
    return tuple(
        m35.ExactActionProbability(action, probability)
        for action, probability in values.items()
    )


def m39_real_request(
) -> m39.KnownBoardRealCardThreePlayerCertifiedGlobalRequest:
    hero = "AsAh"
    o1 = "KsKh"
    o2 = "QsQh"
    profile = m35.RealCardThreePlayerProfile(
        (
            m35.RealCardProfileRow(
                "H", hero, "open", distribution_exact(check="1", bet="0")
            ),
            m35.RealCardProfileRow(
                "O1",
                o1,
                "after_hero_check",
                distribution_exact(check="1"),
            ),
            m35.RealCardProfileRow(
                "O1",
                o1,
                "vs_hero_bet",
                distribution_exact(call="1", fold="0"),
            ),
            m35.RealCardProfileRow(
                "O2",
                o2,
                "after_hero_o1_check",
                distribution_exact(check="1"),
            ),
            m35.RealCardProfileRow(
                "O2",
                o2,
                "vs_hero_bet_o1_call",
                distribution_exact(call="1", fold="0"),
            ),
            m35.RealCardProfileRow(
                "O2",
                o2,
                "vs_hero_bet_o1_fold",
                distribution_exact(call="1", fold="0"),
            ),
        )
    )
    source = m35.KnownBoardRealCardThreePlayerRequest(
        board=("2c", "3d", "4h", "5s", "9c"),
        dead_cards=(),
        hero_range=exact_range(hero),
        o1_range=exact_range(o1),
        o2_range=exact_range(o2),
        baseline_profile=profile,
        attestation=m35.GeneratedTreeAttestation(
            verifier="M40 test-owned real-card oracle",
            verification_date="2026-07-26",
            evidence_version="m40-real-card-v1",
        ),
        rake_rate="1/20",
        rake_cap="1/5",
        repeated=ThreePlayerRepeatedConfig(horizon=1),
    )
    return m39.KnownBoardRealCardThreePlayerCertifiedGlobalRequest(
        source=source,
        adaptation_opportunity=1,
        absolute_gap_tolerance="100",
    )


CASES = (
    (
        module.REAL_CARD_AIOF_PREFLOP,
        m37_request,
        m37.analyze_aiof_preflop_certified_global,
        m37.exact_aiof_preflop_certified_global_json,
    ),
    (
        module.KNOWN_BOARD_REAL_CARD_HU_RIVER,
        m38_request,
        m38.analyze_known_board_real_card_hu_certified_global,
        m38.exact_known_board_real_card_hu_certified_global_json,
    ),
    (
        module.ABSTRACT_THREE_PLAYER_RIVER,
        m39_abstract_request,
        m39.analyze_abstract_three_player_certified_global,
        m39.exact_three_player_certified_global_json,
    ),
    (
        module.KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER,
        m39_real_request,
        m39.analyze_known_board_real_card_three_player_certified_global,
        m39.exact_three_player_certified_global_json,
    ),
)


@pytest.fixture(scope="module")
def parity_results():
    output = {}
    for variant, factory, analyzer, serializer in CASES:
        nested_request = factory()
        direct = analyzer(nested_request)
        unified = module.analyze_unified_certified_global(
            module.UnifiedCertifiedGlobalRequest(variant, nested_request)
        )
        output[variant] = (direct, unified, serializer)
    return output


def test_01_four_variants_are_lossless_against_direct_public_analyzers(
    parity_results,
):
    """Test-owned projection compares native result, certificate, and bytes."""

    for variant, (direct, unified, serializer) in parity_results.items():
        assert direct.status in SUCCESS
        assert unified.status == direct.status
        assert unified.payload is not None
        assert unified.error is None
        assert unified.failure_evidence is None
        nested = unified.payload.nested_result
        assert type(nested) is type(direct)
        assert nested.to_dict() == direct.to_dict()
        assert serializer(nested).encode("utf-8") == serializer(direct).encode(
            "utf-8"
        )
        assert unified.optimizer_work_counters == (
            direct.optimizer_work_counters.to_dict()
        )
        assert unified.oracle_work_counters == (
            direct.oracle_work_counters.to_dict()
        )
        native_payload = direct.payload.native_optimizer_result.payload
        assert native_payload is not None
        assert native_payload.certificate.to_dict() == (
            nested.payload.native_optimizer_result.payload.certificate.to_dict()
        )
        assert set(unified.payload.identities) == {
            "contract_identity",
            "schema_identity",
            "variant_identity",
            "nested_request_identity",
            "workflow_identity",
            "nested_analysis_identity",
            "nested_result_identity",
            "analysis_identity",
        }


@pytest.mark.parametrize(("variant", "factory"), [(c[0], c[1]) for c in CASES])
def test_02_exactly_one_selected_public_analyzer_is_called(
    monkeypatch, variant, factory
):
    calls = Counter()
    targets = (
        (m37, "analyze_aiof_preflop_certified_global"),
        (m38, "analyze_known_board_real_card_hu_certified_global"),
        (m39, "analyze_abstract_three_player_certified_global"),
        (
            m39,
            "analyze_known_board_real_card_three_player_certified_global",
        ),
    )
    for owner, name in targets:
        original = getattr(owner, name)

        def spy(value, *, _original=original, _name=name):
            calls[_name] += 1
            return _original(value)

        monkeypatch.setattr(owner, name, spy)
    result = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(variant, factory())
    )
    assert result.status in SUCCESS
    assert sum(calls.values()) == 1


def test_03_r1_to_r8_cross_module_independent_acceptance_projection(
    parity_results,
):
    """Cross-module gate reuses native independently tested oracle fixtures."""

    m37_native = parity_results[module.REAL_CARD_AIOF_PREFLOP][0]
    m38_native = parity_results[module.KNOWN_BOARD_REAL_CARD_HU_RIVER][0]
    abstract = parity_results[module.ABSTRACT_THREE_PLAYER_RIVER][0]
    real_three = parity_results[
        module.KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER
    ][0]

    # R1/R2: exact card support and full AIoF behavior domain survive M37.
    prep37 = m37_native.payload.preparation
    assert prep37.prepared_ranges.compatible_pair_count == 1
    assert prep37.scenario.information_sets
    assert not any(
        "candidate" in field.name or "grid" in field.name
        for field in fields(m37.AiofPreflopCertifiedGlobalRequest)
    )

    # R3: M38 retains known board, blocker-conditioned joint rows, rake.
    prep38 = m38_native.payload.preparation
    assert set(prep38.board) == {"Qc", "Jd", "9h", "7s", "4c"}
    assert prep38.joint_rows
    assert Fraction(
        sum(
            (Fraction(value) for value in prep38.exact_joint_probabilities),
            Fraction(0),
        )
    ) == 1
    assert m38_native.payload.request.rake_rate == 0.05

    # R4/R5: abstract M31 keeps the supplied baseline and fresh complete M30
    # response; the test-owned two-branch arithmetic has exact zero uplift.
    prep39 = abstract.payload.preparation
    assert prep39.surface_kind == m39.ABSTRACT
    assert prep39.baseline_point.uplift == "0"
    assert abstract.oracle_work_counters.complete_response_records > 0
    assert abstract.oracle_work_counters.joint_profiles_evaluated > 0

    # R6: finite M32 remains available but is not accepted by M40/M36 paths.
    assert ThreePlayerRepeatedConfig(horizon=1).horizon == 1
    assert all(
        "candidate" not in field.name
        and "grid" not in field.name
        and "warm" not in field.name
        for field in fields(module.UnifiedCertifiedGlobalRequest)
    )

    # R7: M35 ordered real-card triples and rake/cap reach M39 end-to-end.
    prep_real = real_three.payload.preparation
    assert prep_real.surface_kind == m39.KNOWN_BOARD_REAL_CARD
    assert prep_real.real_card is not None
    assert prep_real.real_card.workload["compatible_triples"] == 1
    assert real_three.oracle_work_counters.terminal_records_prepared > 0

    # R8: every selected path carries a native global certificate over an
    # automatic non-candidate domain and a sound consumer bound contract.
    for native in (m37_native, m38_native, abstract, real_three):
        certificate = (
            native.payload.native_optimizer_result.payload.certificate
        )
        assert Fraction(certificate.incumbent_lower_bound) <= Fraction(
            certificate.valid_global_upper_bound
        )
        assert certificate.bound_contract_version
        assert native.optimizer_work_counters.oracle_bound_calls > 0
        assert native.partial_result is False


def test_04_wrong_pair_invalid_nested_limits_and_outer_pins_preflight(
    monkeypatch,
):
    calls = Counter()

    def forbidden(_request):
        calls["analyzer"] += 1
        raise AssertionError("nested analyzer must not run")

    monkeypatch.setattr(
        m37, "analyze_aiof_preflop_certified_global", forbidden
    )
    wrong_pair = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, m38_request()
        )
    )
    assert wrong_pair.status == m36.INVALID_INPUT

    bad_limits_request = replace(
        m37_request(), integration_limits="not-a-limit"
    )
    bad_limits = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, bad_limits_request
        )
    )
    assert bad_limits.status == m36.INVALID_INPUT

    malformed = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            m37_request(),
            pins=module.UnifiedCertifiedGlobalPins(
                contract_identity="not-an-identity"
            ),
        )
    )
    assert malformed.status == m36.INVALID_INPUT

    stale = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            m37_request(),
            pins=module.UnifiedCertifiedGlobalPins(
                variant_identity="0" * 64
            ),
        )
    )
    assert stale.status == m36.STALE_INPUT
    assert calls == Counter()


def test_05_nested_no_certificate_is_not_promoted_and_is_lossless():
    nested_request = m39_abstract_request(
        optimizer_limits=replace(
            m36.CertifiedGlobalOptimizerLimits(), max_oracle_calls=1
        ),
        absolute_gap_tolerance="0",
    )
    direct = m39.analyze_abstract_three_player_certified_global(
        nested_request
    )
    result = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.ABSTRACT_THREE_PLAYER_RIVER, nested_request
        )
    )
    assert direct.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert result.status == direct.status
    assert result.payload is None
    assert result.error is not None
    assert result.failure_evidence is not None
    assert result.failure_evidence.native_failure_result.to_dict() == (
        direct.to_dict()
    )
    assert result.failure_evidence.nested_error == {
        "phase": direct.error.phase,
        "message": direct.error.message,
        "cause_status": direct.error.cause_status,
    }
    assert result.optimizer_work_counters == (
        direct.optimizer_work_counters.to_dict()
    )
    assert result.oracle_work_counters == (
        direct.oracle_work_counters.to_dict()
    )
    assert result.partial_result is False


@pytest.mark.parametrize(
    "limits",
    (
        module.UnifiedCertifiedGlobalLimits(max_output_records=1),
        module.UnifiedCertifiedGlobalLimits(max_output_bytes=1),
    ),
)
def test_06_very_low_outer_cap_stops_before_analyzer_aggregate_and_serializer(
    monkeypatch, limits
):
    calls = Counter()

    def analyzer(_request):
        calls["analyzer"] += 1
        raise AssertionError

    def to_dict(_self):
        calls["to_dict"] += 1
        raise AssertionError

    def serializer(_result):
        calls["serializer"] += 1
        raise AssertionError

    monkeypatch.setattr(
        m37, "analyze_aiof_preflop_certified_global", analyzer
    )
    monkeypatch.setattr(
        m37.AiofPreflopCertifiedGlobalResult, "to_dict", to_dict
    )
    monkeypatch.setattr(
        m37, "exact_aiof_preflop_certified_global_json", serializer
    )
    result = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            m37_request(),
            limits=limits,
        )
    )
    assert result.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert result.payload is None
    assert result.error.phase == "output.preflight"
    assert result.partial_result is False
    assert calls == Counter()


def test_07_complete_wrapper_record_cap_n_and_n_minus_one():
    nested_request = m37_request()
    baseline = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, nested_request
        )
    )
    records, _ = module.measure_unified_certified_global_result(baseline)
    at_n = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            nested_request,
            limits=module.UnifiedCertifiedGlobalLimits(
                max_output_records=records
            ),
        )
    )
    below = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            nested_request,
            limits=module.UnifiedCertifiedGlobalLimits(
                max_output_records=records - 1
            ),
        )
    )
    assert at_n.status in SUCCESS
    assert module.measure_unified_certified_global_result(at_n)[0] == records
    assert below.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert below.payload is None
    assert below.failure_evidence is not None
    assert below.optimizer_work_counters == at_n.optimizer_work_counters
    assert below.oracle_work_counters == at_n.oracle_work_counters


def test_08_complete_wrapper_byte_cap_n_and_n_minus_one():
    nested_request = m37_request()
    cap = 1_000_000
    at_n = None
    for _ in range(8):
        result = module.analyze_unified_certified_global(
            module.UnifiedCertifiedGlobalRequest(
                module.REAL_CARD_AIOF_PREFLOP,
                nested_request,
                limits=module.UnifiedCertifiedGlobalLimits(
                    max_output_bytes=cap
                ),
            )
        )
        assert result.status in SUCCESS
        measured = module.measure_unified_certified_global_result(result)[1]
        at_n = result
        if measured == cap:
            break
        cap = measured
    assert at_n is not None
    assert module.measure_unified_certified_global_result(at_n)[1] == cap
    below = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            nested_request,
            limits=module.UnifiedCertifiedGlobalLimits(
                max_output_bytes=cap - 1
            ),
        )
    )
    assert below.status == m36.LIMIT_REACHED_NO_CERTIFICATE
    assert below.payload is None
    assert below.failure_evidence is not None
    assert below.optimizer_work_counters == at_n.optimizer_work_counters
    assert below.oracle_work_counters == at_n.oracle_work_counters


def test_09_all_outer_pins_raw_prefix_stale_and_post_dispatch_evidence():
    nested_request = m37_request()
    baseline = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, nested_request
        )
    )
    identities = baseline.payload.identities
    pins = module.UnifiedCertifiedGlobalPins(
        **{
            field.name: (
                identities[field.name]
                if index % 2 == 0
                else f"sha256:{identities[field.name]}"
            )
            for index, field in enumerate(
                fields(module.UnifiedCertifiedGlobalPins)
            )
        }
    )
    accepted = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            nested_request,
            pins=pins,
        )
    )
    assert accepted.status == baseline.status
    assert module.exact_unified_certified_global_json(accepted) == (
        module.exact_unified_certified_global_json(baseline)
    )

    stale = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP,
            nested_request,
            pins=replace(pins, nested_analysis_identity="0" * 64),
        )
    )
    assert stale.status == m36.STALE_INPUT
    assert stale.payload is None
    assert stale.failure_evidence is not None
    assert stale.failure_evidence.nested_status in SUCCESS
    assert stale.optimizer_work_counters == baseline.optimizer_work_counters


def test_10_semantic_permutation_is_byte_identical_and_changes_are_not():
    first = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, m37_request()
        )
    )
    permuted = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, m37_request(reverse=True)
        )
    )
    changed = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, m37_request(horizon=3)
        )
    )
    encoded = module.exact_unified_certified_global_json(first)
    assert encoded == module.exact_unified_certified_global_json(permuted)
    assert first.payload.identities == permuted.payload.identities
    assert encoded != module.exact_unified_certified_global_json(changed)
    assert first.payload.identities["analysis_identity"] != (
        changed.payload.identities["analysis_identity"]
    )

    other_variant = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.ABSTRACT_THREE_PLAYER_RIVER,
            m39_abstract_request(),
        )
    )
    assert encoded != module.exact_unified_certified_global_json(
        other_variant
    )
    assert first.payload.identities["variant_identity"] != (
        other_variant.payload.identities["variant_identity"]
    )


def test_11_strict_no_partial_wrapper_and_serializer():
    result = module.analyze_unified_certified_global(
        module.UnifiedCertifiedGlobalRequest(
            module.REAL_CARD_AIOF_PREFLOP, m37_request()
        )
    )
    encoded = module.exact_unified_certified_global_json(result)
    assert "\n" not in encoded and "\r" not in encoded
    assert json.loads(encoded) == result.to_dict()
    with pytest.raises(ValueError):
        module.UnifiedCertifiedGlobalResult(
            m36.INVALID_INPUT,
            None,
            module.UnifiedCertifiedGlobalError("x", "x"),
            None,
            {},
            {},
            partial_result=True,
        )


def test_12_public_example_is_two_process_byte_deterministic():
    command = [sys.executable, str(EXAMPLE)]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert first.stdout.endswith(b"\n")
    assert b"\r" not in first.stdout
    parsed = json.loads(first.stdout)
    assert parsed["payload"]["variant"] == module.REAL_CARD_AIOF_PREFLOP
    assert hashlib.sha256(first.stdout).hexdigest() == hashlib.sha256(
        second.stdout
    ).hexdigest()


def test_13_docs_crosswalk_claim_and_protected_boundaries_are_explicit():
    source = SOURCE.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assumptions = ASSUMPTIONS.read_text(encoding="utf-8")
    combined = "\n".join((source, guide, readme, assumptions))
    for token in (
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "exactly one",
        "no partial",
        "candidate",
        "grid",
        "sampling",
        "cross-variant",
        "equilibrium",
        "strategy advice",
        "human merge",
        "post-merge",
    ):
        assert token in combined
    assert "test_03_r1_to_r8_cross_module_independent_acceptance_projection" in guide
    assert "tests/test_aiof_preflop_certified_global.py" in guide
    assert "tests/test_known_board_real_card_hu_certified_global.py" in guide
    assert "tests/test_three_player_certified_global.py" in guide
