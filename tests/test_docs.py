"""Lightweight documentation checks for the MVP walkthrough."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WALKTHROUGH = _ROOT / "docs" / "mvp_walkthrough.md"
_README = _ROOT / "README.md"


def test_walkthrough_file_exists():
    assert _WALKTHROUGH.is_file()


def test_readme_links_to_walkthrough():
    assert "docs/mvp_walkthrough.md" in _README.read_text(encoding="utf-8")


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
