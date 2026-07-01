"""Tests for the shared local-GUI helpers (scripts/gui_common).

These are small, mode-agnostic primitives shared by the five ``serve_*_gui.py``
scripts. The per-GUI suites already exercise them indirectly; this file pins their
behaviour directly now that they live in a single module.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
import gui_common  # noqa: E402


# ---------------------------------------------------------------------------
# messages_payload
# ---------------------------------------------------------------------------


def test_messages_payload_maps_fields():
    messages = [
        SimpleNamespace(field="rake_rate", message="bad", severity="error"),
        SimpleNamespace(field="horizons", message="warn", severity="warning"),
    ]
    assert gui_common.messages_payload(messages) == [
        {"field": "rake_rate", "message": "bad", "severity": "error"},
        {"field": "horizons", "message": "warn", "severity": "warning"},
    ]


def test_messages_payload_empty():
    assert gui_common.messages_payload([]) == []


# ---------------------------------------------------------------------------
# as_text
# ---------------------------------------------------------------------------


def test_as_text_none_is_empty():
    assert gui_common.as_text(None) == ""


def test_as_text_stringifies():
    assert gui_common.as_text(0.5) == "0.5"
    assert gui_common.as_text("abc") == "abc"
    assert gui_common.as_text(3) == "3"


# ---------------------------------------------------------------------------
# require_bool
# ---------------------------------------------------------------------------


def test_require_bool_accepts_bools():
    assert gui_common.require_bool(True, "force") is True
    assert gui_common.require_bool(False, "force") is False


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_require_bool_rejects_non_bools(value):
    with pytest.raises(ValueError):
        gui_common.require_bool(value, "force")


# ---------------------------------------------------------------------------
# optional_horizon
# ---------------------------------------------------------------------------


def test_optional_horizon_none_is_skip():
    assert gui_common.optional_horizon(None) is None


def test_optional_horizon_accepts_positive_int():
    assert gui_common.optional_horizon(1) == 1
    assert gui_common.optional_horizon(7) == 7


@pytest.mark.parametrize("value", [True, 1.5, "3", 0, -1])
def test_optional_horizon_rejects(value):
    with pytest.raises(ValueError):
        gui_common.optional_horizon(value)


# ---------------------------------------------------------------------------
# optional_discount
# ---------------------------------------------------------------------------


def test_optional_discount_none_is_skip():
    assert gui_common.optional_discount(None) is None


def test_optional_discount_accepts_finite_positive():
    assert gui_common.optional_discount(1) == 1.0
    assert gui_common.optional_discount(0.9) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "value", [True, 0, -1, "0.9", math.inf, -math.inf, math.nan]
)
def test_optional_discount_rejects(value):
    with pytest.raises(ValueError):
        gui_common.optional_discount(value)


# ---------------------------------------------------------------------------
# equity_cell_value
# ---------------------------------------------------------------------------


def test_equity_cell_value_keeps_numbers():
    assert gui_common.equity_cell_value(0.5) == 0.5
    assert gui_common.equity_cell_value(1) == 1


def test_equity_cell_value_parses_numeric_strings():
    assert gui_common.equity_cell_value("0.5") == 0.5
    assert gui_common.equity_cell_value("1.5") == 1.5
    assert gui_common.equity_cell_value("-0.1") == pytest.approx(-0.1)


def test_equity_cell_value_parses_non_finite_strings_for_the_validator():
    # "nan" / "inf" become floats so the form validator rejects them as
    # non-finite, rather than being silently kept as strings.
    assert math.isnan(gui_common.equity_cell_value("nan"))
    assert gui_common.equity_cell_value("inf") == math.inf


def test_equity_cell_value_keeps_bool_for_the_validator():
    # A bool is left as-is so the validator rejects a boolean cell; it must not be
    # treated as its int value.
    assert gui_common.equity_cell_value(True) is True


@pytest.mark.parametrize("value", ["", "abc", None])
def test_equity_cell_value_keeps_non_numeric_unchanged(value):
    # Kept unchanged (never rounded to a default like 0.5) so the validator flags
    # it rather than the conversion raising.
    assert gui_common.equity_cell_value(value) is value
