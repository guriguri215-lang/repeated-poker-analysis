"""Lightweight checks for the developer MVP check script.

These tests inspect the script source only; they do not execute it, because
running the full test suite and examples from within a test would be slow.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_mvp.py"
_README = _ROOT / "README.md"
_WALKTHROUGH = _ROOT / "docs" / "mvp_walkthrough.md"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_script_exists():
    assert _SCRIPT.is_file()


def test_script_does_not_use_shell_true():
    assert "shell=True" not in _script_text()


@pytest.mark.parametrize("forbidden", ["git ", "curl", "requests"])
def test_script_has_no_network_or_vcs_calls(forbidden):
    assert forbidden not in _script_text()


def test_script_uses_sys_executable():
    assert "sys.executable" in _script_text()


def test_script_uses_subprocess():
    assert "subprocess" in _script_text()


def test_readme_links_to_script():
    assert "scripts/check_mvp.py" in _README.read_text(encoding="utf-8")


def test_walkthrough_mentions_script():
    assert "scripts/check_mvp.py" in _WALKTHROUGH.read_text(encoding="utf-8")
