"""Lightweight documentation checks for the project docs."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WALKTHROUGH = _ROOT / "docs" / "mvp_walkthrough.md"
_ASSUMPTIONS = _ROOT / "docs" / "assumptions_and_limitations.md"
_EXAMPLES_GUIDE = _ROOT / "docs" / "examples_guide.md"
_PUBLIC_READINESS = _ROOT / "docs" / "public_readiness_checklist.md"
_README = _ROOT / "README.md"


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
        "Decide license separately",
    ],
)
def test_public_readiness_contains_key_phrase(phrase):
    assert phrase in _PUBLIC_READINESS.read_text(encoding="utf-8")


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
        "not a strategic player",
        "does not guarantee profitable poker play",
    ],
)
def test_assumptions_contains_key_phrase(phrase):
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
