"""Tests for the single-hand scenario GUI prototype (serve_single_hand_gui)."""

import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from repeated_poker import (  # noqa: E402
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)
import serve_single_hand_gui as gui  # noqa: E402


def _loaded_form() -> dict:
    return gui.api_load({"path": str(_SINGLE_HAND)})["form"]


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# api_load
# ---------------------------------------------------------------------------


def test_load_single_hand():
    result = gui.api_load({"path": str(_SINGLE_HAND)})
    assert result["ok"] is True
    assert result["mode"] == "single-hand"
    assert result["form"]["scenario_id"] == "nuts_chop_steal_bet98"
    assert result["form"]["bet_size"] == 98.0


def test_load_rejects_non_single_hand():
    with pytest.raises(ValueError):
        gui.api_load({"path": str(_MATRIX)})


def test_load_missing_path():
    with pytest.raises(ValueError):
        gui.api_load({})


def test_load_missing_file():
    with pytest.raises(ValueError):
        gui.api_load({"path": str(_SCENARIOS / "does_not_exist.json")})


def test_load_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        gui.api_load({"path": str(bad)})


# ---------------------------------------------------------------------------
# api_validate
# ---------------------------------------------------------------------------


def test_validate_valid_form():
    result = gui.api_validate({"form": _loaded_form()})
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["messages"] == []


def test_validate_invalid_form_returns_messages():
    form = _loaded_form()
    form["bet_size"] = "0"  # bet_size must be positive
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "bet_size" for m in result["messages"])


def test_validate_bad_value_is_error():
    form = _loaded_form()
    form["bet_size"] = "abc"  # not a number -> parse error, not a message
    with pytest.raises(ValueError):
        gui.api_validate({"form": form})


def test_validate_rejects_non_string_format_version():
    # A numeric format_version must not be coerced to "1"; it stays invalid and is
    # reported on the format_version field.
    form = _loaded_form()
    form["format_version"] = 1
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "format_version" for m in result["messages"])


def test_validate_accepts_string_format_version_one():
    form = _loaded_form()
    form["format_version"] = "1"
    result = gui.api_validate({"form": form})
    assert result["valid"] is True


# ---------------------------------------------------------------------------
# api_save
# ---------------------------------------------------------------------------


def test_save_new_file(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["scenario_id"] = "edited_via_gui"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is True
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["scenario_id"] == "edited_via_gui"


def test_save_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    with pytest.raises(ValueError):
        gui.api_save({"path": str(out), "form": _loaded_form()})
    assert out.read_text(encoding="utf-8") == "ORIGINAL"


def test_save_force_overwrites(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    result = gui.api_save({"path": str(out), "form": _loaded_form(), "force": True})
    assert result["ok"] is True
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


def test_save_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deeper" / "out.json"
    result = gui.api_save({"path": str(out), "form": _loaded_form()})
    assert result["ok"] is True
    assert out.is_file()


def test_save_strict_json(tmp_path):
    out = tmp_path / "out.json"
    result = gui.api_save(
        {"path": str(out), "form": _loaded_form(), "strict_json": True}
    )
    assert result["ok"] is True
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


def test_save_invalid_form_does_not_write(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["baseline_call_probability"] = "0.3"  # call + fold no longer sums to 1
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert result["messages"]
    assert not out.exists()


def test_save_rejects_invalid_format_version(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["format_version"] = 1  # numeric, unsupported
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert any(m["field"] == "format_version" for m in result["messages"])
    assert not out.exists()


def test_save_force_string_is_rejected(tmp_path):
    # "false" is truthy under bool(); the API must reject non-boolean force and
    # leave the existing file untouched.
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    with pytest.raises(ValueError):
        gui.api_save({"path": str(out), "form": _loaded_form(), "force": "false"})
    assert out.read_text(encoding="utf-8") == "ORIGINAL"


def test_save_strict_json_string_is_rejected(tmp_path):
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        gui.api_save({"path": str(out), "form": _loaded_form(), "strict_json": "true"})


def test_save_missing_path():
    with pytest.raises(ValueError):
        gui.api_save({"form": _loaded_form()})


def test_save_output_directory_errors(tmp_path):
    with pytest.raises(ValueError):
        gui.api_save({"path": str(tmp_path), "form": _loaded_form()})


# ---------------------------------------------------------------------------
# HTML page + live HTTP server (ephemeral port)
# ---------------------------------------------------------------------------


def test_page_contains_expected_labels():
    assert "scenario_id" in gui._PAGE
    assert "baseline_call_probability" in gui._PAGE
    assert "Validate" in gui._PAGE
    assert "Save JSON" in gui._PAGE
    assert "single-hand" in gui._PAGE


def _serve_in_background():
    server = gui.build_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, thread, port


def _post(port, route, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - localhost only
        return json.loads(resp.read().decode("utf-8"))


def test_http_server_serves_page_and_api():
    server, thread, port = _serve_in_background()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:  # noqa: S310
            assert resp.status == 200
            page = resp.read().decode("utf-8")
        assert "Single-hand scenario form" in page

        loaded = _post(port, "/api/load", {"path": str(_SINGLE_HAND)})
        assert loaded["ok"] is True
        assert loaded["form"]["scenario_id"] == "nuts_chop_steal_bet98"

        validated = _post(port, "/api/validate", {"form": loaded["form"]})
        assert validated["valid"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_server_error_response_has_no_traceback():
    server, thread, port = _serve_in_background()
    try:
        # A bad load path is a clean ValueError -> {"ok": false, "error": ...}.
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/load",
            data=json.dumps({"path": str(_MATRIX)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)  # noqa: S310
            raise AssertionError("expected an HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
        assert body["ok"] is False
        assert "single-hand" in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
