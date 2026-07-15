import itertools
import math

import pytest

from repeated_poker.aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    canonicalize_exact_combo,
    canonicalize_hand_class,
    card_from_id,
    card_id,
    expand_range,
    prepare_compatible_ranges,
)


def class_entry(label, weight=1.0):
    return RangeEntry(label, weight, WeightBasis.CLASS_TOTAL_MASS)


def combo_entry(label, weight=1.0):
    return RangeEntry(label, weight, WeightBasis.EXACT_COMBO_MASS)


def all_classes():
    ranks = "23456789TJQKA"
    result = [rank * 2 for rank in ranks]
    for high in range(1, len(ranks)):
        for low in range(high):
            result.extend((ranks[high] + ranks[low] + "s", ranks[high] + ranks[low] + "o"))
    return tuple(result)


def test_canonical_deck_has_52_unique_round_tripping_cards():
    cards = tuple(card_from_id(value) for value in range(52))
    assert len(cards) == len(set(cards)) == 52
    assert card_id("2c") == 0
    assert card_id("As") == 51
    assert tuple(card_id(card) for card in cards) == tuple(range(52))


@pytest.mark.parametrize("invalid", ["as", "AS", "10s", " As", "Xc", "2C", ""])
def test_card_parser_is_strict(invalid):
    with pytest.raises(AiofContractError) as error:
        card_id(invalid)
    assert error.value.status is AiofStatus.INVALID_CARD_INPUT
    with pytest.raises(AiofContractError):
        card_from_id(True)


def test_exact_combo_canonicalization_and_duplicate_card_rejection():
    assert canonicalize_exact_combo("KhAs") == "AsKh"
    assert canonicalize_exact_combo("AsKh") == "AsKh"
    with pytest.raises(AiofContractError):
        canonicalize_exact_combo("AsAs")


def test_169_classes_and_multiplicities_expand_to_1326():
    classes = all_classes()
    assert len(classes) == len(set(classes)) == 169
    assert all(canonicalize_hand_class(label) == label for label in classes)
    spec = RangeSpec(tuple(class_entry(label) for label in classes))
    expanded = expand_range(spec, (), AiofLimits())
    assert len(expanded.combos) == 1326
    assert len(expand_range(RangeSpec((class_entry("AA"),)), (), AiofLimits()).combos) == 6
    assert len(expand_range(RangeSpec((class_entry("AKs"),)), (), AiofLimits()).combos) == 4
    assert len(expand_range(RangeSpec((class_entry("AKo"),)), (), AiofLimits()).combos) == 12


@pytest.mark.parametrize(("label", "remaining"), [("AA", 3), ("AKs", 3), ("AKo", 9)])
def test_dead_ace_removes_combos_without_class_renormalization(label, remaining):
    expanded = expand_range(RangeSpec((class_entry(label, 12.0),)), ("As",), AiofLimits())
    pre_dead_multiplicity = 6 if label == "AA" else (4 if label.endswith("s") else 12)
    assert len(expanded.combos) == remaining
    assert {combo.raw_mass for combo in expanded.combos} == {12.0 / pre_dead_multiplicity}
    assert expanded.raw_mass_before_dead == pytest.approx(12.0)


@pytest.mark.parametrize(
    "entries",
    [
        (class_entry("AKs"), combo_entry("AsKs")),
        (combo_entry("AsKh"), combo_entry("KhAs")),
        (class_entry("AA"), class_entry("AA")),
    ],
)
def test_generated_duplicate_combos_fail_closed(entries):
    with pytest.raises(AiofContractError) as error:
        expand_range(RangeSpec(entries), (), AiofLimits())
    assert error.value.status is AiofStatus.DUPLICATE_COMBO


@pytest.mark.parametrize("label", ["AK", "KAs", "AAs", "AA+", "random", "AsK"])
def test_ambiguous_or_non_contract_range_syntax_is_rejected(label):
    with pytest.raises(AiofContractError) as error:
        expand_range(RangeSpec((class_entry(label),)), (), AiofLimits())
    assert error.value.status is AiofStatus.INVALID_RANGE


@pytest.mark.parametrize("weight", [True, 0.0, -1.0, math.nan, math.inf])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(AiofContractError) as error:
        expand_range(RangeSpec((class_entry("AA", weight),)), (), AiofLimits())
    assert error.value.status is AiofStatus.INVALID_RANGE


def test_wrong_weight_basis_is_rejected():
    with pytest.raises(AiofContractError):
        expand_range(
            RangeSpec((RangeEntry("AA", 1.0, WeightBasis.EXACT_COMBO_MASS),)),
            (),
            AiofLimits(),
        )


def test_compatible_joint_mass_and_single_normalization_match_hand_calculation():
    sb = RangeSpec((combo_entry("AsKh", 2.0), combo_entry("AcKd", 1.0)))
    bb = RangeSpec((combo_entry("QsJh", 3.0), combo_entry("AsJd", 5.0)))
    prepared = prepare_compatible_ranges(sb, bb, (), AiofLimits())
    # All except AsKh vs AsJd are compatible: 2*3 + 1*3 + 1*5 = 14.
    assert prepared.compatible_pair_count == 3
    assert prepared.compatible_raw_joint_mass == pytest.approx(14.0)
    assert prepared.normalization_factor == pytest.approx(1.0 / 14.0)
    assert math.fsum(item.probability for item in prepared.sb_marginals) == pytest.approx(1.0)
    assert math.fsum(item.probability for item in prepared.bb_marginals) == pytest.approx(1.0)


def test_full_support_pair_count_is_1624350_without_public_pair_list():
    spec = RangeSpec(tuple(class_entry(label) for label in all_classes()))
    prepared = prepare_compatible_ranges(spec, spec, (), AiofLimits())
    assert prepared.compatible_pair_count == 1_624_350
    assert not hasattr(prepared, "compatible_pairs")


def test_empty_support_and_caps_fail_before_success_payload_exists():
    with pytest.raises(AiofContractError) as error:
        expand_range(RangeSpec((combo_entry("AsKh"),)), ("As",), AiofLimits())
    assert error.value.status is AiofStatus.EMPTY_COMPATIBLE_SUPPORT
    with pytest.raises(AiofContractError) as error:
        expand_range(RangeSpec((class_entry("AA"),)), (), AiofLimits(max_exact_combos_per_side=5))
    assert error.value.status is AiofStatus.CAP_EXCEEDED
    with pytest.raises(AiofContractError) as error:
        prepare_compatible_ranges(
            RangeSpec((class_entry("AA"),)),
            RangeSpec((class_entry("KK"),)),
            (),
            AiofLimits(max_compatible_combo_pairs=1),
        )
    assert error.value.status is AiofStatus.CAP_EXCEEDED
    with pytest.raises(AiofContractError):
        expand_range(RangeSpec((class_entry("AA"), class_entry("KK"))), (), AiofLimits(max_range_entries_per_side=1))
    with pytest.raises(AiofContractError):
        expand_range(RangeSpec((class_entry("AA"),)), tuple(card_from_id(i) for i in range(44)), AiofLimits())


def test_surviving_combo_with_zero_compatible_marginal_is_not_silently_dropped():
    sb = RangeSpec((combo_entry("AsAh"), combo_entry("KcKd")))
    bb = RangeSpec((combo_entry("AsAh"),))
    with pytest.raises(AiofContractError) as error:
        prepare_compatible_ranges(sb, bb, (), AiofLimits())
    assert error.value.status is AiofStatus.ZERO_COMPATIBLE_MARGINAL
