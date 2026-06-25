"""Lightweight documentation checks for the project docs."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WALKTHROUGH = _ROOT / "docs" / "mvp_walkthrough.md"
_ASSUMPTIONS = _ROOT / "docs" / "assumptions_and_limitations.md"
_README = _ROOT / "README.md"


def test_walkthrough_file_exists():
    assert _WALKTHROUGH.is_file()


def test_assumptions_file_exists():
    assert _ASSUMPTIONS.is_file()


def test_readme_links_to_walkthrough():
    assert "docs/mvp_walkthrough.md" in _README.read_text(encoding="utf-8")


def test_readme_links_to_assumptions():
    assert "docs/assumptions_and_limitations.md" in _README.read_text(encoding="utf-8")


def test_walkthrough_links_to_assumptions():
    assert "assumptions_and_limitations.md" in _WALKTHROUGH.read_text(encoding="utf-8")


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
