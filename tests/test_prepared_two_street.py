"""Focused contract, oracle, accounting, observation, and cap tests for M14."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

import repeated_poker
import repeated_poker.prepared_two_street as pts
from repeated_poker.exact_response import solve_exact_response
from repeated_poker.fixed_profile import evaluate_fixed_profile
from repeated_poker.game import (
    ChanceNode,
    HeroStrategy,
    TerminalNode,
    VillainStrategy,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    iter_terminals,
    validate_tree,
)
from repeated_poker.prepared_two_street import (
    PREPARED_CHANCE_NORMALIZATION_ID,
    PREPARED_INFORMATION_KEY_ID,
    PREPARED_TWO_STREET_BUILDER_ID,
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedActionEvent,
    PreparedActionKind,
    PreparedActionOption,
    PreparedBucket,
    PreparedChanceEdge,
    PreparedChanceEvent,
    PreparedContentIdentity,
    PreparedDataAttestation,
    PreparedDecisionMenu,
    PreparedHeadsUpChips,
    PreparedPlayer,
    PreparedRake,
    PreparedRoundCloseReason,
    PreparedShowdownValue,
    PreparedStreet,
    PreparedStreetCloseEvent,
    PreparedTransitionRow,
    PreparedTwoStreetLimits,
    PreparedTwoStreetSpec,
    PreparedTwoStreetStatus,
    build_prepared_two_street_game,
    prepared_public_history_id,
    prepared_semantic_sha256,
)


def _attestation() -> PreparedDataAttestation:
    return PreparedDataAttestation(
        source="focused synthetic fixture",
        bucket_semantics="abstract test buckets",
        conditional_probability_semantics="prepared conditional mass",
        observation_mapping="public actions/outcomes and own bucket only",
        perfect_recall_attested=True,
    )


def _passive(kind: PreparedActionKind) -> PreparedActionOption:
    return PreparedActionOption(kind=kind, size_id=None, raise_to=None, is_all_in=False)


def _aggressive(
    kind: PreparedActionKind, size_id: str, raise_to: float, *, all_in: bool = False
) -> PreparedActionOption:
    return PreparedActionOption(kind=kind, size_id=size_id, raise_to=raise_to, is_all_in=all_in)


def _action(
    street: str,
    player: PreparedPlayer,
    kind: PreparedActionKind,
    *,
    size_id: str | None = None,
    raise_to: float | None = None,
    all_in: bool = False,
    reopen: bool = True,
) -> PreparedActionEvent:
    return PreparedActionEvent(street, player, kind, size_id, raise_to, all_in, reopen)


def _identity(spec: PreparedTwoStreetSpec, raw: bytes = b"prepared-focused-input") -> PreparedContentIdentity:
    try:
        semantic = prepared_semantic_sha256(spec)
    except ValueError:
        # Invalid numeric fixtures are rejected before semantic identity is used.
        semantic = "0" * 64
    return PreparedContentIdentity(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=semantic,
    )


def _build(spec: PreparedTwoStreetSpec, *, raw: bytes = b"prepared-focused-input", limits=None):
    return build_prepared_two_street_game(
        spec,
        raw,
        _identity(spec, raw),
        PreparedTwoStreetLimits() if limits is None else limits,
    )


def _assert_failure(result, status: PreparedTwoStreetStatus):
    assert result.status is status
    assert result.build is None
    assert result.error is not None
    assert result.error.message
    assert result.error.phase


def _one_street_spec(*, root_actions=None) -> PreparedTwoStreetSpec:
    s = "river"
    root = ()
    v_check = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h_check = _action(s, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    v_bet = _action(
        s,
        PreparedPlayer.VILLAIN,
        PreparedActionKind.BET,
        size_id="open-2",
        raise_to=2.0,
    )
    h_call = _action(s, PreparedPlayer.HERO, PreparedActionKind.CALL)
    root_options = root_actions or (
        _passive(PreparedActionKind.CHECK),
        _aggressive(PreparedActionKind.BET, "open-2", 2.0),
    )
    menus = (
        PreparedDecisionMenu(prepared_public_history_id(root), s, PreparedPlayer.VILLAIN, root_options),
        PreparedDecisionMenu(
            prepared_public_history_id((v_check,)),
            s,
            PreparedPlayer.HERO,
            (_passive(PreparedActionKind.CHECK),),
        ),
        PreparedDecisionMenu(
            prepared_public_history_id((v_bet,)),
            s,
            PreparedPlayer.HERO,
            (_passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL)),
        ),
    )
    check_state = prepared_public_history_id(
        (v_check, h_check, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CHECK_CHECK))
    )
    call_state = prepared_public_history_id(
        (v_bet, h_call, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CALL))
    )
    return PreparedTwoStreetSpec(
        contract_version=PREPARED_TWO_STREET_CONTRACT_VERSION,
        attestation=_attestation(),
        starting_chips=PreparedHeadsUpChips(10.0, 10.0),
        initial_committed=PreparedHeadsUpChips(1.0, 1.0),
        rake=PreparedRake(0.05, 3.0),
        streets=(PreparedStreet(s, "River", PreparedPlayer.VILLAIN, 2.0),),
        hero_buckets=(PreparedBucket("H0", 1.0),),
        villain_buckets=(PreparedBucket("V0", 1.0),),
        decision_menus=menus,
        transition_id=None,
        transition_rows=(),
        showdown_values=(
            PreparedShowdownValue(check_state, "H0", "V0", 0.5),
            PreparedShowdownValue(call_state, "H0", "V0", 0.7),
        ),
    )


def _two_street_spec(*, probabilities=(0.6, 0.4)) -> PreparedTwoStreetSpec:
    s1, s2, transition = "flop", "turn", "deal-turn"
    v1 = _action(s1, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h1 = _action(s1, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    close1 = PreparedStreetCloseEvent(s1, PreparedRoundCloseReason.CHECK_CHECK)
    before_chance = (v1, h1, close1)
    menus = [
        PreparedDecisionMenu(prepared_public_history_id(()), s1, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK),)),
        PreparedDecisionMenu(prepared_public_history_id((v1,)), s1, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
    ]
    showdown_values = []
    for outcome, share in (("red", 0.8), ("black", 0.2)):
        chance = PreparedChanceEvent(transition, outcome)
        h2 = _action(s2, PreparedPlayer.HERO, PreparedActionKind.CHECK)
        v2 = _action(s2, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
        events = before_chance + (chance,)
        menus.append(
            PreparedDecisionMenu(prepared_public_history_id(events), s2, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),))
        )
        menus.append(
            PreparedDecisionMenu(prepared_public_history_id(events + (h2,)), s2, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK),))
        )
        showdown_values.append(
            PreparedShowdownValue(
                prepared_public_history_id(
                    events + (h2, v2, PreparedStreetCloseEvent(s2, PreparedRoundCloseReason.CHECK_CHECK))
                ),
                "H0",
                "V0",
                share,
            )
        )
    row = PreparedTransitionRow(
        transition,
        prepared_public_history_id(before_chance),
        "H0",
        "V0",
        (
            PreparedChanceEdge("red", "H0", "V0", probabilities[0]),
            PreparedChanceEdge("black", "H0", "V0", probabilities[1]),
        ),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        _attestation(),
        PreparedHeadsUpChips(10.0, 10.0),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.1, None),
        (
            PreparedStreet(s1, "First", PreparedPlayer.VILLAIN, 1.0),
            PreparedStreet(s2, "Second", PreparedPlayer.HERO, 1.0),
        ),
        (PreparedBucket("H0", 1.0),),
        (PreparedBucket("V0", 1.0),),
        tuple(menus),
        transition,
        (row,),
        tuple(showdown_values),
    )


def _initial_all_in_two_street_spec() -> PreparedTwoStreetSpec:
    s1, s2, transition = "flop", "turn", "deal-turn"
    close = PreparedStreetCloseEvent(s1, PreparedRoundCloseReason.ALL_IN_CALL)
    before_chance = (close,)
    row = PreparedTransitionRow(
        transition,
        prepared_public_history_id(before_chance),
        "H0",
        "V0",
        (
            PreparedChanceEdge("low", "H0", "V0", 0.25),
            PreparedChanceEdge("high", "H0", "V0", 0.75),
        ),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        _attestation(),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.0, None),
        (
            PreparedStreet(s1, "First", PreparedPlayer.VILLAIN, 1.0),
            PreparedStreet(s2, "Second", PreparedPlayer.HERO, 1.0),
        ),
        (PreparedBucket("H0", 1.0),),
        (PreparedBucket("V0", 1.0),),
        (),
        transition,
        (row,),
        (
            PreparedShowdownValue(
                prepared_public_history_id(before_chance + (PreparedChanceEvent(transition, "low"),)),
                "H0",
                "V0",
                0.0,
            ),
            PreparedShowdownValue(
                prepared_public_history_id(before_chance + (PreparedChanceEvent(transition, "high"),)),
                "H0",
                "V0",
                1.0,
            ),
        ),
    )


def _private_update_spec() -> PreparedTwoStreetSpec:
    base = _two_street_spec(probabilities=(0.5, 0.5))
    s1, s2, transition = "flop", "turn", "deal-turn"
    v1 = _action(s1, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h1 = _action(s1, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    before = (v1, h1, PreparedStreetCloseEvent(s1, PreparedRoundCloseReason.CHECK_CHECK))
    chance = PreparedChanceEvent(transition, "shared")
    h2 = _action(s2, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    v2 = _action(s2, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    public2 = before + (chance,)
    menus = (
        base.decision_menus[0],
        base.decision_menus[1],
        PreparedDecisionMenu(prepared_public_history_id(public2), s2, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
        PreparedDecisionMenu(prepared_public_history_id(public2 + (h2,)), s2, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK),)),
    )
    rows = tuple(
        PreparedTransitionRow(
            transition,
            prepared_public_history_id(before),
            hero,
            villain,
            (
                PreparedChanceEdge("shared", "H0", "V0", 0.5),
                PreparedChanceEdge("shared", "H1", "V1", 0.5),
            ),
        )
        for hero in ("H0", "H1")
        for villain in ("V0", "V1")
    )
    terminal_state = prepared_public_history_id(
        public2 + (h2, v2, PreparedStreetCloseEvent(s2, PreparedRoundCloseReason.CHECK_CHECK))
    )
    return replace(
        base,
        hero_buckets=(PreparedBucket("H0", 0.5), PreparedBucket("H1", 0.5)),
        villain_buckets=(PreparedBucket("V0", 0.5), PreparedBucket("V1", 0.5)),
        decision_menus=menus,
        transition_rows=rows,
        showdown_values=(
            PreparedShowdownValue(terminal_state, "H0", "V0", 0.25),
            PreparedShowdownValue(terminal_state, "H1", "V1", 0.75),
        ),
    )


def _strategy_for_tree(tree):
    hero = HeroStrategy(
        {info: {action: (1.0 if index == 0 else 0.0) for index, action in enumerate(actions)} for info, actions in collect_hero_info_sets(tree).items()}
    )
    villain = VillainStrategy(
        {info: {action: (1.0 if index == 0 else 0.0) for index, action in enumerate(actions)} for info, actions in collect_villain_info_sets(tree).items()}
    )
    return hero, villain


def test_exact_module_public_api_and_algorithm_ids_are_pinned():
    expected = [
        "PREPARED_TWO_STREET_CONTRACT_VERSION", "PREPARED_TWO_STREET_BUILDER_ID",
        "PREPARED_CHANCE_NORMALIZATION_ID", "PREPARED_INFORMATION_KEY_ID", "PreparedPlayer",
        "PreparedActionKind", "PreparedRoundCloseReason", "PreparedTwoStreetStatus",
        "PreparedTwoStreetLimits", "PreparedContentIdentity", "PreparedDataAttestation",
        "PreparedHeadsUpChips", "PreparedRake", "PreparedBucket", "PreparedActionOption",
        "PreparedActionEvent", "PreparedStreetCloseEvent", "PreparedChanceEvent", "PreparedStreet",
        "PreparedDecisionMenu", "PreparedChanceEdge", "PreparedTransitionRow", "PreparedShowdownValue",
        "PreparedTwoStreetSpec", "PreparedInformationSetKey", "PreparedInformationSetObservation",
        "PreparedChanceNormalizationRecord", "PreparedBuildCounts", "PreparedTwoStreetIdentity",
        "PreparedTwoStreetBuild", "PreparedBuildError", "PreparedTwoStreetBuildResult",
        "prepared_public_history_id", "prepared_semantic_sha256", "build_prepared_two_street_game",
    ]
    assert pts.__all__ == expected
    assert PREPARED_TWO_STREET_CONTRACT_VERSION == "betting-tree-v2-prepared-two-street-1"
    assert PREPARED_TWO_STREET_BUILDER_ID == "betting-tree-v2-prepared-two-street-builder-v1"
    assert PREPARED_CHANCE_NORMALIZATION_ID == "positive-fsum-normalize-v1"
    assert PREPARED_INFORMATION_KEY_ID == "canonical-private-public-recall-key-v1"
    assert "PreparedTwoStreetBuild" not in repeated_poker.__all__
    assert not hasattr(repeated_poker, "PREPARED_TWO_STREET_CONTRACT_VERSION")


def test_one_street_specialization_hand_counts_labels_and_keys():
    result = _build(_one_street_spec())
    assert result.status is PreparedTwoStreetStatus.SUCCESS
    assert result.error is None and result.build is not None
    build = result.build
    assert build.counts == pts.PreparedBuildCounts(1, 0, 1, 3, 3, 7, 3, 2, 1, 2, 2)
    assert isinstance(build.tree.root, ChanceNode)
    root_decision = build.tree.root.children[0][1]
    assert tuple(label for label, _ in root_decision.actions) == ("check", "bet::open-2")
    assert {item.key.player for item in build.information_sets} == {PreparedPlayer.HERO, PreparedPlayer.VILLAIN}
    validate_tree(build.tree)


def test_one_street_terminal_accounting_and_fold_rake_hand_oracle():
    build = _build(_one_street_spec()).build
    terminals = list(iter_terminals(build.tree.root))
    assert len(terminals) == 3
    for terminal in terminals:
        assert terminal.hero_ev + terminal.villain_ev + terminal.house_rake == pytest.approx(0.0)
    fold = next(terminal for terminal in terminals if terminal.house_rake == 0.0 and terminal.hero_ev < 0)
    assert fold.hero_ev == pytest.approx(-1.0)
    assert fold.villain_ev == pytest.approx(1.0)
    call = max(terminals, key=lambda terminal: terminal.house_rake)
    # Independent arithmetic: pot=6, rake=.3, Hero receives .7*5.7 and invested 3.
    assert call.hero_ev == pytest.approx(0.99)
    assert call.villain_ev == pytest.approx(-1.29)
    assert call.house_rake == pytest.approx(0.3)


def test_two_street_hand_oracle_fixed_profile_and_terminal_conservation():
    result = _build(_two_street_spec())
    assert result.status is PreparedTwoStreetStatus.SUCCESS
    build = result.build
    assert build.counts == pts.PreparedBuildCounts(1, 1, 3, 6, 2, 10, 6, 3, 3, 1, 1)
    hero, villain = _strategy_for_tree(build.tree)
    fixed = evaluate_fixed_profile(build.tree, hero, villain)
    # Independent helper-free hand oracle: pot=2, rake=.2, shares .8/.2.
    red = 1.8 * 0.8 - 1.0
    black = 1.8 * 0.2 - 1.0
    expected_hero = 0.6 * red + 0.4 * black
    assert fixed.hero_ev == pytest.approx(expected_hero)
    assert fixed.villain_ev == pytest.approx(-expected_hero - 0.2)
    assert fixed.house_rake == pytest.approx(0.2)
    for terminal in iter_terminals(build.tree.root):
        assert terminal.hero_ev + terminal.villain_ev + terminal.house_rake == pytest.approx(0.0)
        assert (10.0 + terminal.hero_ev) + (10.0 + terminal.villain_ev) + terminal.house_rake == pytest.approx(20.0)


def test_two_street_dp_and_enumerate_are_identical_on_tiny_tree():
    build = _build(_two_street_spec()).build
    hero, _ = _strategy_for_tree(build.tree)
    dp = solve_exact_response(build.tree, hero, method="dp")
    enum = solve_exact_response(build.tree, hero, method="enumerate")
    assert dp.villain_max_ev == pytest.approx(enum.villain_max_ev)
    assert dp.ev_h_worst == pytest.approx(enum.ev_h_worst)
    assert dp.ev_h_best == pytest.approx(enum.ev_h_best)
    assert dp.expected_house_rake_worst == pytest.approx(enum.expected_house_rake_worst)
    assert dp.best_response_strategies == enum.best_response_strategies


def test_chance_one_time_fsum_normalization_is_auditable():
    spec = _two_street_spec(probabilities=(0.6000000004, 0.4))
    record = _build(spec).build.chance_normalization[0]
    independent_raw = math.fsum(edge.probability for edge in spec.transition_rows[0].edges)
    independent_factor = 1.0 / independent_raw
    assert record.raw_sum == independent_raw
    assert record.normalization_factor == independent_factor
    assert record.effective_probabilities == tuple(
        edge.probability * independent_factor
        for edge in sorted(spec.transition_rows[0].edges, key=lambda e: (e.public_outcome_id, e.next_hero_bucket_id, e.next_villain_bucket_id))
    )
    assert math.fsum(record.effective_probabilities) == pytest.approx(1.0)


def test_initial_all_in_two_street_requires_transition_and_uses_outcome_values():
    spec = _initial_all_in_two_street_spec()
    result = _build(spec)
    assert result.status is PreparedTwoStreetStatus.SUCCESS
    assert result.error is None and result.build is not None
    build = result.build
    assert build.counts == pts.PreparedBuildCounts(1, 1, 3, 0, 2, 4, 2, 0, 0, 1, 1)
    assert len(build.chance_normalization) == 1
    record = build.chance_normalization[0]
    assert record.edge_identities == (("high", "H0", "V0"), ("low", "H0", "V0"))
    assert isinstance(build.tree.root, ChanceNode)
    transition_node = build.tree.root.children[0][1]
    assert isinstance(transition_node, ChanceNode)
    assert all(isinstance(child, TerminalNode) for _, child in transition_node.children)

    hero, villain = _strategy_for_tree(build.tree)
    fixed = evaluate_fixed_profile(build.tree, hero, villain)
    # Independent helper-free oracle: pot=2, no rake, low loses 1 and high wins 1.
    expected_hero = 0.25 * (2.0 * 0.0 - 1.0) + 0.75 * (2.0 * 1.0 - 1.0)
    assert fixed.hero_ev == pytest.approx(expected_hero)
    assert fixed.villain_ev == pytest.approx(-expected_hero)
    assert fixed.house_rake == 0.0


def test_initial_all_in_two_street_rejects_missing_and_extra_transition_rows():
    spec = _initial_all_in_two_street_spec()
    _assert_failure(
        _build(replace(spec, transition_rows=())),
        PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
    )
    extra = replace(spec.transition_rows[0], source_public_state_id="sha256:" + "0" * 64)
    _assert_failure(
        _build(replace(spec, transition_rows=spec.transition_rows + (extra,))),
        PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
    )


def test_public_outcome_is_separate_from_private_successor_updates_and_no_leakage():
    build = _build(_private_update_spec()).build
    second = [item for item in build.information_sets if item.key.street_id == "turn"]
    assert second
    # Same public outcome produces multiple private branches; histories remain private.
    assert len({item.key.public_history_id for item in second}) == 2  # before/after Hero check
    assert any(len(item.key.own_bucket_history) == 2 for item in second)
    for item in second:
        encoded = json.dumps(pts._canonical_value(item.key), sort_keys=True)
        if item.key.player is PreparedPlayer.HERO:
            assert "V0" not in encoded and "V1" not in encoded
        else:
            assert "H0" not in encoded and "H1" not in encoded
    # Opponent-only root differences pool at the first decision.
    first_villain = [item for item in build.information_sets if item.key.street_id == "flop" and item.key.player is PreparedPlayer.VILLAIN]
    assert len(first_villain) == 2  # one per own V bucket, not four matchups


def test_same_input_repeat_has_exact_identity_and_order():
    spec = _two_street_spec()
    first = _build(spec).build
    second = _build(spec).build
    assert first.identity == second.identity
    assert first.counts == second.counts
    assert first.chance_normalization == second.chance_normalization
    assert first.information_sets == second.information_sets


def test_two_street_identity_is_pinned_for_python_310_and_313_ci():
    build = _build(_two_street_spec()).build
    assert build.identity.semantic_sha256 == "2e7fda40c7d97bbe5ba317154eeb2bd80a0503394938e434bf31e1d402873d38"
    assert build.identity.ordered_tree_sha256 == "a92e2a233b8a0c350e670971be20fea8d8ee119fca794f08e99b0929d5aa9d86"
    assert build.identity.run_identity == "d897db55801a4b25302f729096da3f4aaedfa8b7050ad20c800176bf87d0b5a0"


def test_raw_and_semantic_hash_mismatch_fail_independently():
    spec = _one_street_spec()
    raw = b"prepared-focused-input"
    identity = _identity(spec, raw)
    bad_raw = replace(identity, raw_sha256="0" * 64)
    _assert_failure(build_prepared_two_street_game(spec, raw, bad_raw), PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH)
    bad_semantic = replace(identity, semantic_sha256="f" * 64)
    _assert_failure(build_prepared_two_street_game(spec, raw, bad_semantic), PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH)


@pytest.mark.parametrize(
    "mutator,status",
    [
        (lambda s: replace(s, starting_chips=PreparedHeadsUpChips(True, 10.0)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, starting_chips=PreparedHeadsUpChips(math.nan, 10.0)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, starting_chips=PreparedHeadsUpChips(math.inf, 10.0)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, initial_committed=PreparedHeadsUpChips(11.0, 1.0)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, rake=PreparedRake(-0.1, None)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, rake=PreparedRake(1.1, None)), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, showdown_values=(replace(s.showdown_values[0], hero_pot_share=1.1),) + s.showdown_values[1:]), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda s: replace(s, attestation=replace(s.attestation, perfect_recall_attested=False)), PreparedTwoStreetStatus.PERFECT_RECALL_ATTESTATION_MISSING),
    ],
)
def test_numeric_input_and_attestation_boundaries_fail_closed(mutator, status):
    _assert_failure(_build(mutator(_one_street_spec())), status)


def test_derived_nonfinite_and_positive_root_underflow_are_numeric_failures():
    huge = replace(_one_street_spec(), starting_chips=PreparedHeadsUpChips(1e308, 1e308))
    _assert_failure(_build(huge), PreparedTwoStreetStatus.NUMERIC_FAILURE)
    base = _one_street_spec()
    tiny = replace(
        base,
        hero_buckets=(PreparedBucket("Htiny", 5e-324), PreparedBucket("H0", 1.0)),
        villain_buckets=(PreparedBucket("Vtiny", 5e-324), PreparedBucket("V0", 1.0)),
    )
    _assert_failure(_build(tiny), PreparedTwoStreetStatus.NUMERIC_FAILURE)


@pytest.mark.parametrize(
    "edge_mutator,status",
    [
        (lambda edges: (), PreparedTwoStreetStatus.EMPTY_CHANCE_SUPPORT),
        (lambda edges: edges + (edges[0],), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=0.0), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=-0.1), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=math.nan), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=math.inf), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=-math.inf), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], probability=True), edges[1]), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda edges: (replace(edges[0], probability="0.6"), edges[1]), PreparedTwoStreetStatus.INVALID_INPUT),
        (lambda edges: (replace(edges[0], probability=0.7), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
        (lambda edges: (replace(edges[0], next_hero_bucket_id="UNKNOWN"), edges[1]), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT),
    ],
)
def test_chance_support_negative_matrix(edge_mutator, status):
    spec = _two_street_spec()
    row = replace(spec.transition_rows[0], edges=tuple(edge_mutator(spec.transition_rows[0].edges)))
    _assert_failure(_build(replace(spec, transition_rows=(row,))), status)


def test_missing_and_extra_transition_or_showdown_rows_fail_closed():
    spec = _two_street_spec()
    _assert_failure(_build(replace(spec, transition_rows=())), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT)
    extra_row = replace(spec.transition_rows[0], source_public_state_id="sha256:" + "0" * 64)
    _assert_failure(_build(replace(spec, transition_rows=spec.transition_rows + (extra_row,))), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT)
    _assert_failure(_build(replace(spec, showdown_values=spec.showdown_values[:-1])), PreparedTwoStreetStatus.INVALID_INPUT)
    extra_show = replace(spec.showdown_values[0], public_state_id="sha256:" + "1" * 64)
    _assert_failure(_build(replace(spec, showdown_values=spec.showdown_values + (extra_show,))), PreparedTwoStreetStatus.INVALID_INPUT)


def test_below_min_open_all_in_mismatch_and_same_amount_normal_allin_are_rejected():
    base = _one_street_spec()
    bad_open = _aggressive(PreparedActionKind.BET, "bad", 1.5)
    root = replace(base.decision_menus[0], actions=(bad_open, _passive(PreparedActionKind.CHECK)))
    _assert_failure(_build(replace(base, decision_menus=(root,) + base.decision_menus[1:])), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)
    wrong_flag = _aggressive(PreparedActionKind.BET, "open-2", 2.0, all_in=True)
    root = replace(base.decision_menus[0], actions=(wrong_flag, _passive(PreparedActionKind.CHECK)))
    _assert_failure(_build(replace(base, decision_menus=(root,) + base.decision_menus[1:])), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)
    duplicate_amount = replace(
        base.decision_menus[0],
        actions=(
            _aggressive(PreparedActionKind.BET, "normal", 2.0),
            _aggressive(PreparedActionKind.BET, "allin", 2.0, all_in=True),
            _passive(PreparedActionKind.CHECK),
        ),
    )
    _assert_failure(_build(replace(base, decision_menus=(duplicate_amount,) + base.decision_menus[1:])), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)


def test_below_min_non_allin_raise_is_rejected_without_tolerance_rescue():
    base = _one_street_spec()
    facing = replace(
        base.decision_menus[2],
        actions=(
            _aggressive(PreparedActionKind.RAISE, "too-small", 3.9999999999),
            _passive(PreparedActionKind.FOLD),
            _passive(PreparedActionKind.CALL),
        ),
    )
    _assert_failure(_build(replace(base, decision_menus=base.decision_menus[:2] + (facing,))), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)


def _unequal_allin_spec() -> PreparedTwoStreetSpec:
    s = "river"
    bet = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.BET, size_id="jam", raise_to=9.0, all_in=True)
    check = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    hero_check = _action(s, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    call = _action(s, PreparedPlayer.HERO, PreparedActionKind.CALL)
    menus = (
        PreparedDecisionMenu(prepared_public_history_id(()), s, PreparedPlayer.VILLAIN, (_aggressive(PreparedActionKind.BET, "jam", 9.0, all_in=True), _passive(PreparedActionKind.CHECK))),
        PreparedDecisionMenu(prepared_public_history_id((bet,)), s, PreparedPlayer.HERO, (_passive(PreparedActionKind.CALL), _passive(PreparedActionKind.FOLD))),
        PreparedDecisionMenu(prepared_public_history_id((check,)), s, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
    )
    showdown = (
        PreparedShowdownValue(
            prepared_public_history_id((bet, call, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.ALL_IN_CALL))),
            "H0", "V0", 0.5,
        ),
        PreparedShowdownValue(
            prepared_public_history_id((check, hero_check, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CHECK_CHECK))),
            "H0", "V0", 0.5,
        ),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(),
        PreparedHeadsUpChips(5.0, 10.0), PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.0, None), (PreparedStreet(s, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), menus, None, (), showdown,
    )


def test_short_call_refunds_excess_fold_refunds_first_and_no_side_pot():
    build = _build(_unequal_allin_spec()).build
    terminals = list(iter_terminals(build.tree.root))
    jam_showdown = next(t for t in terminals if t.house_rake == 0.0 and t.hero_ev == pytest.approx(0.0))
    assert jam_showdown.villain_ev == pytest.approx(0.0)  # matched 5/5 chop after 5-chip refund
    fold = next(t for t in terminals if t.hero_ev == pytest.approx(-1.0))
    assert fold.villain_ev == pytest.approx(1.0)
    assert fold.house_rake == 0.0


def _raise_chain_spec(*, third_raise: bool = False) -> PreparedTwoStreetSpec:
    s = "river"
    vbet = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.BET, size_id="b2", raise_to=2.0)
    hraise = _action(s, PreparedPlayer.HERO, PreparedActionKind.RAISE, size_id="r4", raise_to=4.0)
    vraise = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.RAISE, size_id="r6", raise_to=6.0)
    vcheck = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    hcheck = _action(s, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    menus = [
        PreparedDecisionMenu(prepared_public_history_id(()), s, PreparedPlayer.VILLAIN, (_aggressive(PreparedActionKind.BET, "b2", 2.0), _passive(PreparedActionKind.CHECK))),
        PreparedDecisionMenu(prepared_public_history_id((vcheck,)), s, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
        PreparedDecisionMenu(prepared_public_history_id((vbet,)), s, PreparedPlayer.HERO, (_aggressive(PreparedActionKind.RAISE, "r4", 4.0), _passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL))),
        PreparedDecisionMenu(prepared_public_history_id((vbet, hraise)), s, PreparedPlayer.VILLAIN, (_aggressive(PreparedActionKind.RAISE, "r6", 6.0), _passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL))),
    ]
    final_actions = [_passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL)]
    if third_raise:
        final_actions.insert(0, _aggressive(PreparedActionKind.RAISE, "r8", 8.0))
    menus.append(PreparedDecisionMenu(prepared_public_history_id((vbet, hraise, vraise)), s, PreparedPlayer.HERO, tuple(final_actions)))
    showdowns = [
        PreparedShowdownValue(prepared_public_history_id((vcheck, hcheck, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CHECK_CHECK))), "H0", "V0", 0.5),
    ]
    for prefix, player in (
        ((vbet,), PreparedPlayer.HERO),
        ((vbet, hraise), PreparedPlayer.VILLAIN),
        ((vbet, hraise, vraise), PreparedPlayer.HERO),
    ):
        call = _action(s, player, PreparedActionKind.CALL)
        showdowns.append(PreparedShowdownValue(prepared_public_history_id(prefix + (call, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CALL))), "H0", "V0", 0.5))
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(), PreparedHeadsUpChips(20.0, 20.0),
        PreparedHeadsUpChips(0.0, 0.0), PreparedRake(0.0, None),
        (PreparedStreet(s, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), tuple(menus), None, (), tuple(showdowns),
    )


def test_two_full_raises_exact_bound_and_third_raise_no_reopen():
    assert _build(_raise_chain_spec()).status is PreparedTwoStreetStatus.SUCCESS
    _assert_failure(_build(_raise_chain_spec(third_raise=True)), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)


def test_caller_lowered_full_raise_cap_fails_as_cap_without_fallback():
    limits = replace(PreparedTwoStreetLimits(), max_full_raises_per_street=1)
    _assert_failure(_build(_raise_chain_spec(), limits=limits), PreparedTwoStreetStatus.CAP_EXCEEDED)


def _short_allin_raise_spec(*, reopen_illegally: bool = False) -> PreparedTwoStreetSpec:
    s = "river"
    vbet = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.BET, size_id="b6", raise_to=6.0)
    hraise = _action(
        s, PreparedPlayer.HERO, PreparedActionKind.RAISE,
        size_id="short8", raise_to=8.0, all_in=True, reopen=False,
    )
    vcheck = _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    hcheck = _action(s, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    response = [_passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL)]
    if reopen_illegally:
        response.insert(0, _aggressive(PreparedActionKind.RAISE, "illegal-reopen", 12.0))
    menus = (
        PreparedDecisionMenu(
            prepared_public_history_id(()), s, PreparedPlayer.VILLAIN,
            (_aggressive(PreparedActionKind.BET, "b6", 6.0), _passive(PreparedActionKind.CHECK)),
        ),
        PreparedDecisionMenu(
            prepared_public_history_id((vcheck,)), s, PreparedPlayer.HERO,
            (_passive(PreparedActionKind.CHECK),),
        ),
        PreparedDecisionMenu(
            prepared_public_history_id((vbet,)), s, PreparedPlayer.HERO,
            (
                _aggressive(PreparedActionKind.RAISE, "short8", 8.0, all_in=True),
                _passive(PreparedActionKind.FOLD),
                _passive(PreparedActionKind.CALL),
            ),
        ),
        PreparedDecisionMenu(
            prepared_public_history_id((vbet, hraise)), s, PreparedPlayer.VILLAIN,
            tuple(response),
        ),
    )
    showdowns = (
        PreparedShowdownValue(
            prepared_public_history_id((vcheck, hcheck, PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CHECK_CHECK))),
            "H0", "V0", 0.5,
        ),
        PreparedShowdownValue(
            prepared_public_history_id((vbet, _action(s, PreparedPlayer.HERO, PreparedActionKind.CALL), PreparedStreetCloseEvent(s, PreparedRoundCloseReason.CALL))),
            "H0", "V0", 0.5,
        ),
        PreparedShowdownValue(
            prepared_public_history_id((vbet, hraise, _action(s, PreparedPlayer.VILLAIN, PreparedActionKind.CALL, reopen=False), PreparedStreetCloseEvent(s, PreparedRoundCloseReason.ALL_IN_CALL))),
            "H0", "V0", 0.5,
        ),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(),
        PreparedHeadsUpChips(8.0, 10.0), PreparedHeadsUpChips(0.0, 0.0),
        PreparedRake(0.0, None), (PreparedStreet(s, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), menus, None, (), showdowns,
    )


def test_valid_short_allin_raise_does_not_increment_or_reopen():
    result = _build(_short_allin_raise_spec())
    assert result.status is PreparedTwoStreetStatus.SUCCESS
    labels = [
        tuple(label for label, _ in node.actions)
        for node in iter_nodes(result.build.tree.root)
        if hasattr(node, "actions")
    ]
    assert ("fold", "call") in labels
    _assert_failure(_build(_short_allin_raise_spec(reopen_illegally=True)), PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR)


def test_accounting_fault_injections_fail_closed(monkeypatch):
    original_apply = pts._apply_action

    def double_count(*args, **kwargs):
        state, close = original_apply(*args, **kwargs)
        if state.pending_uncalled:
            state = replace(state, pot=state.pot + state.pending_uncalled)
        return state, close

    monkeypatch.setattr(pts, "_apply_action", double_count)
    _assert_failure(_build(_one_street_spec()), PreparedTwoStreetStatus.ACCOUNTING_MISMATCH)


def test_refund_omission_and_rake_twice_injections_fail_closed(monkeypatch):
    monkeypatch.setattr(pts, "_refund_uncalled", lambda state, tolerance: state)
    _assert_failure(_build(_unequal_allin_spec()), PreparedTwoStreetStatus.ACCOUNTING_MISMATCH)
    monkeypatch.undo()
    original = pts.make_equity_showdown_terminal

    def rake_twice(*args, **kwargs):
        terminal = original(*args, **kwargs)
        return replace(terminal, house_rake=terminal.house_rake * 2.0)

    monkeypatch.setattr(pts, "make_equity_showdown_terminal", rake_twice)
    _assert_failure(_build(_one_street_spec()), PreparedTwoStreetStatus.ACCOUNTING_MISMATCH)


def test_information_digest_collision_and_forged_observation_fail_closed(monkeypatch):
    monkeypatch.setattr(pts, "_information_digest", lambda key: "0" * 64)
    _assert_failure(_build(_one_street_spec()), PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH)
    monkeypatch.undo()
    original = pts._make_information_artifact

    def forged(*args, **kwargs):
        artifact = original(*args, **kwargs)
        return replace(artifact, public_observation_identity="obs:forged:V0")

    monkeypatch.setattr(pts, "_make_information_artifact", forged)
    _assert_failure(_build(_one_street_spec()), PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH)


@pytest.mark.parametrize(
    "fixture,field,exact,too_small",
    [
        (_one_street_spec, "max_actions_per_decision", 2, 1),
        (_one_street_spec, "max_depth_edges", 3, 2),
        (_one_street_spec, "max_decision_nodes", 3, 2),
        (_one_street_spec, "max_terminal_nodes", 3, 2),
        (_one_street_spec, "max_total_nodes", 7, 6),
        (_one_street_spec, "max_information_sets_per_player", 2, 1),
        (_one_street_spec, "max_information_sets_total", 3, 2),
        (_one_street_spec, "max_validation_trace_records", 3, 2),
        (_two_street_spec, "max_chance_outcomes_per_row", 2, 1),
        (_two_street_spec, "max_total_chance_edges", 3, 2),
    ],
)
def test_cap_boundaries_accept_exact_and_reject_cap_plus_one(fixture, field, exact, too_small):
    spec = fixture()
    exact_limits = replace(PreparedTwoStreetLimits(), **{field: exact})
    assert _build(spec, limits=exact_limits).status is PreparedTwoStreetStatus.SUCCESS
    small_limits = replace(PreparedTwoStreetLimits(), **{field: too_small})
    _assert_failure(_build(spec, limits=small_limits), PreparedTwoStreetStatus.CAP_EXCEEDED)


def test_root_and_transition_row_cap_boundaries():
    spec = _private_update_spec()
    assert _build(spec, limits=replace(PreparedTwoStreetLimits(), max_root_matchups=4, max_transition_rows=4)).status is PreparedTwoStreetStatus.SUCCESS
    _assert_failure(_build(spec, limits=replace(PreparedTwoStreetLimits(), max_root_matchups=3)), PreparedTwoStreetStatus.CAP_EXCEEDED)
    _assert_failure(_build(spec, limits=replace(PreparedTwoStreetLimits(), max_transition_rows=3)), PreparedTwoStreetStatus.CAP_EXCEEDED)


def test_large_checked_matchup_product_stops_before_materialization(monkeypatch):
    base = _one_street_spec()
    heroes = tuple(PreparedBucket(f"H{i}", 1.0 / 101.0) for i in range(101))
    villains = tuple(PreparedBucket(f"V{i}", 1.0 / 100.0) for i in range(100))
    spec = replace(base, hero_buckets=heroes, villain_buckets=villains, decision_menus=(), showdown_values=())
    calls = {"materialize": 0, "showdown": 0, "fold": 0}

    def no_materialize(*args, **kwargs):
        calls["materialize"] += 1
        raise AssertionError

    monkeypatch.setattr(pts, "_materialize", no_materialize)
    monkeypatch.setattr(pts, "make_equity_showdown_terminal", lambda *a, **k: calls.__setitem__("showdown", calls["showdown"] + 1))
    monkeypatch.setattr(pts, "make_fold_terminal", lambda *a, **k: calls.__setitem__("fold", calls["fold"] + 1))
    _assert_failure(_build(spec), PreparedTwoStreetStatus.CAP_EXCEEDED)
    assert calls == {"materialize": 0, "showdown": 0, "fold": 0}


def test_preflight_node_cap_calls_no_materializer_payoff_or_solver(monkeypatch):
    calls = {"materialize": 0, "showdown": 0, "fold": 0, "solver": 0}
    monkeypatch.setattr(pts, "_materialize", lambda *a, **k: calls.__setitem__("materialize", 1))
    monkeypatch.setattr(pts, "make_equity_showdown_terminal", lambda *a, **k: calls.__setitem__("showdown", 1))
    monkeypatch.setattr(pts, "make_fold_terminal", lambda *a, **k: calls.__setitem__("fold", 1))
    monkeypatch.setattr("repeated_poker.exact_response.solve_exact_response", lambda *a, **k: calls.__setitem__("solver", 1))
    _assert_failure(
        _build(_one_street_spec(), limits=replace(PreparedTwoStreetLimits(), max_total_nodes=6)),
        PreparedTwoStreetStatus.CAP_EXCEEDED,
    )
    assert calls == {"materialize": 0, "showdown": 0, "fold": 0, "solver": 0}


def test_all_public_failures_obey_no_partial_invariant(monkeypatch):
    failures = []
    spec = _one_street_spec()
    failures.append(_build(replace(spec, starting_chips=PreparedHeadsUpChips(-1.0, 10.0))))
    failures.append(_build(replace(spec, streets=spec.streets * 3)))
    failures.append(build_prepared_two_street_game(spec, b"wrong", _identity(spec)))
    failures.append(_build(spec, limits=replace(PreparedTwoStreetLimits(), max_total_nodes=6)))
    monkeypatch.setattr(pts, "_actual_counts", lambda tree: (0, 0, 0, 0, 0))
    failures.append(_build(spec))
    assert {result.status for result in failures} >= {
        PreparedTwoStreetStatus.INVALID_INPUT,
        PreparedTwoStreetStatus.UNSUPPORTED_MODEL,
        PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH,
        PreparedTwoStreetStatus.CAP_EXCEEDED,
        PreparedTwoStreetStatus.NON_REPRODUCIBLE,
    }
    for result in failures:
        _assert_failure(result, result.status)


def test_private_oracle_mismatch_seam_is_not_public_and_has_no_payload():
    assert "_oracle_comparison_result" not in pts.__all__
    assert pts._oracle_comparison_result((1.0, 2.0), (1.0, 2.0)) is None
    result = pts._oracle_comparison_result((1.0, 2.0), (1.0, 3.0))
    _assert_failure(result, PreparedTwoStreetStatus.ORACLE_MISMATCH)


def test_semantic_tuple_order_and_canonical_chance_edge_order_are_distinct():
    spec = _two_street_spec()
    reversed_input = replace(
        spec,
        transition_rows=(replace(spec.transition_rows[0], edges=tuple(reversed(spec.transition_rows[0].edges))),),
    )
    assert prepared_semantic_sha256(spec) != prepared_semantic_sha256(reversed_input)
    first = _build(spec).build
    second = _build(reversed_input).build
    # Chance edges are canonical-sorted before materialization, so the tree identity agrees.
    assert first.identity.ordered_tree_sha256 == second.identity.ordered_tree_sha256
    assert first.chance_normalization[0].edge_identities == second.chance_normalization[0].edge_identities


def test_factorized_root_chance_is_not_silently_normalized():
    base = _one_street_spec()
    h1 = PreparedBucket("H1", 0.5000000002)
    h0 = PreparedBucket("H0", 0.5)
    showdowns = base.showdown_values + tuple(
        replace(value, hero_bucket_id="H1") for value in base.showdown_values
    )
    spec = replace(base, hero_buckets=(h1, h0), showdown_values=showdowns)
    build = _build(spec).build
    probabilities = tuple(probability for probability, _ in build.tree.root.children)
    assert probabilities == (0.5000000002, 0.5)
    assert math.fsum(probabilities) == pytest.approx(1.0000000002)
    outside = replace(spec, hero_buckets=(PreparedBucket("H1", 0.500000002), h0))
    _assert_failure(_build(outside), PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT)


def test_dp_supports_pure_space_above_enumerator_cap_without_builder_materialization():
    base = _one_street_spec()
    buckets = tuple(PreparedBucket(f"V{i}", 1.0 / 17.0) for i in range(17))
    showdowns = tuple(
        replace(value, villain_bucket_id=bucket.bucket_id)
        for bucket in buckets
        for value in base.showdown_values
    )
    spec = replace(base, villain_buckets=buckets, showdown_values=showdowns)
    build = _build(spec).build
    assert build.counts.villain_pure_strategies == 2**17
    assert build.counts.villain_pure_strategies > 100_000
    hero, _ = _strategy_for_tree(build.tree)
    dp = solve_exact_response(build.tree, hero, method="dp", max_pure_strategies=100_000)
    assert dp.num_villain_pure_strategies == 2**17
    with pytest.raises(ValueError, match="safety limit"):
        solve_exact_response(build.tree, hero, method="enumerate", max_pure_strategies=100_000)


def test_limits_are_strict_positive_bounded_integers():
    spec = _one_street_spec()
    for bad in (0, -1, True, 2.5):
        _assert_failure(
            _build(spec, limits=replace(PreparedTwoStreetLimits(), max_total_nodes=bad)),
            PreparedTwoStreetStatus.INVALID_INPUT,
        )
    _assert_failure(
        _build(spec, limits=replace(PreparedTwoStreetLimits(), max_total_nodes=250_001)),
        PreparedTwoStreetStatus.INVALID_INPUT,
    )


def test_status_result_invariant_for_success_and_failure():
    success = _build(_one_street_spec())
    assert (success.status is PreparedTwoStreetStatus.SUCCESS) is (success.build is not None and success.error is None)
    failure = _build(replace(_one_street_spec(), decision_menus=()))
    assert failure.status is not PreparedTwoStreetStatus.SUCCESS
    assert failure.build is None and failure.error is not None and failure.error.message


def test_v1_public_contract_remains_version_one_and_builder_type_unchanged():
    from repeated_poker.scenario_io import SUPPORTED_FORMAT_VERSIONS, RiverScenarioBettingTree

    assert SUPPORTED_FORMAT_VERSIONS == ("1",)
    assert RiverScenarioBettingTree.__name__ == "RiverScenarioBettingTree"
