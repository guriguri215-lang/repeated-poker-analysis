"""Lightweight documentation checks for the project docs."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WALKTHROUGH = _ROOT / "docs" / "mvp_walkthrough.md"
_ASSUMPTIONS = _ROOT / "docs" / "assumptions_and_limitations.md"
_EXAMPLES_GUIDE = _ROOT / "docs" / "examples_guide.md"
_FORMAT_REFERENCE = _ROOT / "docs" / "scenario_format_reference.md"
_STT_FORMAT_REFERENCE = _ROOT / "docs" / "stt_pushfold_format_reference.md"
_PUBLIC_OBSERVABLES = _ROOT / "docs" / "public_observables_and_adaptation.md"
_GUI_DESIGN = _ROOT / "docs" / "gui_input_design.md"
_PUBLIC_READINESS = _ROOT / "docs" / "public_readiness_checklist.md"
_PUBLICATION_POLICY = _ROOT / "docs" / "publication_policy.md"
_LICENSE = _ROOT / "LICENSE"
_PYPROJECT = _ROOT / "pyproject.toml"
_README = _ROOT / "README.md"
# Assembled from fragments so this file does not contain the literal marker.
_STALE_LICENSE_MARKER = "Propri" + "etary"


def test_walkthrough_file_exists():
    assert _WALKTHROUGH.is_file()


def test_assumptions_file_exists():
    assert _ASSUMPTIONS.is_file()


def test_examples_guide_file_exists():
    assert _EXAMPLES_GUIDE.is_file()


def test_public_readiness_file_exists():
    assert _PUBLIC_READINESS.is_file()


def test_readme_links_to_public_readiness():
    assert "docs/public_readiness_checklist.md" in _README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "No tokens, passwords, API keys",
        "No private paths",
        "not a full poker solver",
        "does not guarantee profitable play",
        "python scripts/check_mvp.py",
        "the MIT License file exists",
    ],
)
def test_public_readiness_contains_key_phrase(phrase):
    assert phrase in _PUBLIC_READINESS.read_text(encoding="utf-8")


def test_public_readiness_no_longer_defers_license_decision():
    assert "Decide license separately" not in _PUBLIC_READINESS.read_text(
        encoding="utf-8"
    )


def test_license_file_exists():
    assert _LICENSE.is_file()


@pytest.mark.parametrize(
    "phrase",
    ["MIT License", "Copyright (c) 2026 guriguri215-lang"],
)
def test_license_contains_phrase(phrase):
    assert phrase in _LICENSE.read_text(encoding="utf-8")


def test_publication_policy_file_exists():
    assert _PUBLICATION_POLICY.is_file()


def test_readme_links_to_license():
    assert "(LICENSE)" in _README.read_text(encoding="utf-8")


def test_readme_links_to_publication_policy():
    assert "docs/publication_policy.md" in _README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "MIT License",
        "experimental research / learning project",
        "not a full poker solver",
        "without warranty",
        "assumptions_and_limitations.md",
    ],
)
def test_publication_policy_contains_key_phrase(phrase):
    assert phrase in _PUBLICATION_POLICY.read_text(encoding="utf-8")


def test_pyproject_license_is_not_proprietary():
    assert _STALE_LICENSE_MARKER not in _PYPROJECT.read_text(encoding="utf-8")


def test_pyproject_license_points_to_license_file():
    assert 'license = { file = "LICENSE" }' in _PYPROJECT.read_text(encoding="utf-8")


def test_examples_guide_has_no_mojibake():
    text = _EXAMPLES_GUIDE.read_text(encoding="utf-8")
    # ASCII-only, so no mojibake glyphs and no U+FFFD replacement character.
    assert text.isascii()


def test_readme_links_to_walkthrough():
    assert "docs/mvp_walkthrough.md" in _README.read_text(encoding="utf-8")


def test_readme_links_to_assumptions():
    assert "docs/assumptions_and_limitations.md" in _README.read_text(encoding="utf-8")


def test_readme_links_to_examples_guide():
    assert "docs/examples_guide.md" in _README.read_text(encoding="utf-8")


def test_walkthrough_links_to_assumptions():
    assert "assumptions_and_limitations.md" in _WALKTHROUGH.read_text(encoding="utf-8")


def test_walkthrough_links_to_examples_guide():
    assert "examples_guide.md" in _WALKTHROUGH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "python scripts/check_mvp.py",
        "What this project is",
        "What this project is not",
        "Fastest way to run the MVP",
    ],
)
def test_readme_intro_contains_phrase(phrase):
    assert phrase in _README.read_text(encoding="utf-8")


# The Japanese title of the local reference project, built from code points so
# this test file stays ASCII. The local solver-folder name is assembled from
# fragments so this file does not contain the literal forbidden string.
_LOCAL_PROJECT_TITLE = "".join(
    chr(c)
    for c in (
        0x7E70, 0x308A, 0x8FD4, 0x3057, 0x30B2,
        0x30FC, 0x30E0, 0x306E, 0x89E3, 0x6790,
    )
)
_LOCAL_PROJECT_FOLDER = "poker " + "sim $EV"


@pytest.mark.parametrize("phrase", [_LOCAL_PROJECT_FOLDER, _LOCAL_PROJECT_TITLE])
def test_readme_does_not_reference_local_project(phrase):
    assert phrase not in _README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "before a GitHub repository is created",
        "No Git repository",
        "first commit",
    ],
)
def test_readme_has_no_stale_bootstrap_text(phrase):
    assert phrase not in _README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "run_candidate_analysis_pipeline",
        "python scripts/check_mvp.py",
    ],
)
def test_readme_describes_current_state(phrase):
    assert phrase in _README.read_text(encoding="utf-8")


def test_readme_is_ascii_only():
    # ASCII-only guarantees no mojibake glyphs, no replacement character, and no
    # smart quotes / dashes that render badly in some terminals.
    assert _README.read_text(encoding="utf-8").isascii()


# Forbidden code points: the mojibake glyph U+7AB6, the U+FFFD replacement
# character, curly single/double quotes, and en/em dashes. They are given as
# code points so this test file stays ASCII.
@pytest.mark.parametrize(
    "code_point",
    [0x7AB6, 0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2013, 0x2014],
)
def test_readme_has_no_mojibake_or_smart_punctuation(code_point):
    assert chr(code_point) not in _README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "T_deadline",
        "T_detect",
        "run_candidate_analysis_pipeline",
        "detected_adaptation_is_at_least_baseline",
        "not a full poker solver",
    ],
)
def test_walkthrough_contains_key_phrase(phrase):
    assert phrase in _WALKTHROUGH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "not legal, financial, gambling, or bankroll advice",
        "not a full poker solver",
        "T_deadline",
        "T_detect",
        "Malmuth-Harville Independent Chip Model",
        "Future-ICM / FGS / tournament-simulation backend",
        "not a strategic player",
        "does not guarantee profitable poker play",
        "comparable spot occurrence probability per physical hand",
    ],
)
def test_assumptions_contains_key_phrase(phrase):
    assert phrase in _ASSUMPTIONS.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "STT push/fold assumptions",
        "bystander prize EV delta",
        "not push/fold charts",
        "not simulate blind increases",
        "Future-ICM, FGS, and tournament simulation",
    ],
)
def test_assumptions_contains_stt_scope_phrase(phrase):
    assert phrase in _ASSUMPTIONS.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "analysis_pipeline.py",
        "nuts_chop_river.py",
        "candidate_filters.py",
        "value_bluff_river.py",
        "not real hand recommendations",
        "not full solver outputs",
        "MVP entry point",
    ],
)
def test_examples_guide_contains_key_phrase(phrase):
    assert phrase in _EXAMPLES_GUIDE.read_text(encoding="utf-8")


def test_format_reference_file_exists():
    assert _FORMAT_REFERENCE.is_file()


def test_public_observables_file_exists():
    assert _PUBLIC_OBSERVABLES.is_file()


def test_stt_format_reference_file_exists():
    assert _STT_FORMAT_REFERENCE.is_file()


def test_readme_links_to_format_reference():
    assert "docs/scenario_format_reference.md" in _README.read_text(encoding="utf-8")


def test_readme_links_to_stt_format_reference():
    assert "docs/stt_pushfold_format_reference.md" in _README.read_text(
        encoding="utf-8"
    )


def test_examples_guide_links_to_format_reference():
    assert "scenario_format_reference.md" in _EXAMPLES_GUIDE.read_text(encoding="utf-8")


def test_format_reference_is_ascii_only():
    # Match the README / examples-guide style: ASCII-only, so no mojibake glyphs,
    # no replacement character, and no smart quotes / dashes.
    assert _FORMAT_REFERENCE.read_text(encoding="utf-8").isascii()


def test_stt_format_reference_is_ascii_only():
    assert _STT_FORMAT_REFERENCE.read_text(encoding="utf-8").isascii()


@pytest.mark.parametrize(
    "phrase",
    [
        "format_version",
        "single-hand mode",
        "Hero-range-only mode",
        "showdown_matrix",
        "equity_matrix",
        "betting_tree",
        "Hero pot share before rake",
        "validate_river_scenario.py",
        "not a full poker solver",
        # The explicit baseline Villain profile (M2-T1) is a first-class input
        # field, documented with its provenance token.
        "baseline_villain_strategy",
        "baseline_villain_source",
        "not a scenario JSON field",
        "comparable spot occurrence probability",
        "--detection-comparable-spot-occurrence-probability-per-physical-hand",
    ],
)
def test_format_reference_contains_key_phrase(phrase):
    assert phrase in _FORMAT_REFERENCE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "stt_pushfold-1",
        "outcome_matrix",
        "sb_win_probability_matrix",
        "baseline_villain_source",
        "bystander prize EV delta",
        "Malmuth-Harville ICM",
        "not a push/fold chart",
        "not real-money advice",
        "validate_tree(..., allow_negative_residual=True)",
        "not a tournament simulation",
        "physical-hand conversion",
        "--detection-comparable-spot-occurrence-probability-per-physical-hand",
    ],
)
def test_stt_format_reference_contains_key_phrase(phrase):
    assert phrase in _STT_FORMAT_REFERENCE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        "cross-spot population frequency",
        "not a single-tree reach probability",
        "not a scenario JSON field",
        "physical-hand conversion",
        "--detection-comparable-spot-occurrence-probability-per-physical-hand",
    ],
)
def test_public_observables_contains_physical_conversion_scope(phrase):
    assert phrase in _PUBLIC_OBSERVABLES.read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", ["README", "FORMAT_REFERENCE"])
def test_explicit_baseline_villain_is_not_claimed_to_be_equilibrium(doc):
    # M2-T1: an explicit baseline_villain_strategy is a chosen comparison profile,
    # not an equilibrium. Both docs must state that non-claim in so many words so a
    # future edit cannot quietly upgrade it to an equilibrium assertion.
    path = {"README": _README, "FORMAT_REFERENCE": _FORMAT_REFERENCE}[doc]
    text = path.read_text(encoding="utf-8")
    assert "baseline_villain_strategy" in text
    assert "not an equilibrium claim" in text


def test_docs_mention_template_generator_script():
    # The generator is implemented, so the docs should reference the script.
    for doc in (_README, _EXAMPLES_GUIDE, _FORMAT_REFERENCE):
        assert "create_scenario_template.py" in doc.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "doc_name, stale_phrase",
    [
        # The template generator now exists, so it must no longer be described as
        # future "template-generation ... tooling" work.
        ("README", "template-generation"),
        ("EXAMPLES_GUIDE", "template-generation"),
        ("FORMAT_REFERENCE", "a template generator and a form/GUI input layer"),
    ],
)
def test_docs_have_no_stale_template_generator_wording(doc_name, stale_phrase):
    doc = {
        "README": _README,
        "EXAMPLES_GUIDE": _EXAMPLES_GUIDE,
        "FORMAT_REFERENCE": _FORMAT_REFERENCE,
    }[doc_name]
    assert stale_phrase not in doc.read_text(encoding="utf-8")


def test_gui_design_file_exists():
    assert _GUI_DESIGN.is_file()


def test_readme_links_to_gui_design():
    assert "docs/gui_input_design.md" in _README.read_text(encoding="utf-8")


def test_format_reference_links_to_gui_design():
    assert "gui_input_design.md" in _FORMAT_REFERENCE.read_text(encoding="utf-8")


def test_gui_design_is_ascii_only():
    # Match the other docs' style: ASCII-only, so no mojibake glyphs, no
    # replacement character, and no smart quotes / dashes.
    assert _GUI_DESIGN.read_text(encoding="utf-8").isascii()


@pytest.mark.parametrize(
    "phrase",
    [
        "GUI/form input design",
        "JSON remains the source of truth",
        "Validation panel",
        "Results summary",
        "not real-money advice",
        "real-card parser",
        "external solver import",
        "implementation phases",
        # The form-model section now covers more than single-hand / hero-range
        # (showdown-matrix and equity-matrix too), so its heading is mode-neutral.
        "### Scenario form model (supported modes)",
    ],
)
def test_gui_design_contains_key_phrase(phrase):
    assert phrase in _GUI_DESIGN.read_text(encoding="utf-8")


def test_gui_design_form_model_heading_is_not_stale():
    # The old heading named only single-hand / hero-range; it must not linger now
    # that the section also covers showdown-matrix and equity-matrix.
    assert "Scenario form model (single-hand and hero-range)" not in _GUI_DESIGN.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Scope consistency: docs must match the implemented feature set
# ---------------------------------------------------------------------------

_PLAN = _ROOT / "02_research_and_implementation_plan.md"


def test_non_goals_do_not_list_implemented_features():
    # CLI runners, file exports, and local GUI prototypes exist, so the old
    # blanket "CLI / file output / web app" non-goal must not linger.
    text = _ASSUMPTIONS.read_text(encoding="utf-8")
    assert "## Current non-goals" in text
    assert "CLI / file output / web app" not in text
    assert "STT push-fold game builder / runner" not in text


@pytest.mark.parametrize(
    "phrase",
    [
        # The web surface that remains out of scope is public hosting.
        "Publicly hosted web service",
        # The GUI freeze is an explicit current non-goal.
        "New GUI features",
        "bug fixes only",
    ],
)
def test_non_goals_reflect_current_scope(phrase):
    assert phrase in _ASSUMPTIONS.read_text(encoding="utf-8")


def test_readme_gui_section_is_table_plus_design_doc_reference():
    text = _README.read_text(encoding="utf-8")
    # The five-mode port table stays as the single GUI overview...
    assert "| Scenario mode | Command | Port | Editor | Analyze |" in text
    # ...while the repetitive per-mode paragraphs live in the design doc now.
    for stale in (
        "A separate Hero-range editor prototype is available",
        "A showdown-matrix editor prototype is available",
        "An equity-matrix editor prototype is available",
        "A betting-tree editor prototype is available",
    ):
        assert stale not in text


def test_readme_and_gui_design_note_gui_freeze():
    assert "frozen (bug fixes only)" in _README.read_text(encoding="utf-8")
    assert "frozen at this slice" in _GUI_DESIGN.read_text(encoding="utf-8")


def test_plan_architecture_notes_flat_layout_is_deliberate():
    # The planned subpackage split is documented as long-term shape; the
    # implemented flat layout must be flagged as a deliberate current choice.
    text = _PLAN.read_text(encoding="utf-8")
    assert "## Planned architecture" in text
    assert "flat module layout" in text
    assert "deliberate" in text
