"""Tests for the Hero-range-only scenario GUI prototype (serve_hero_range_gui)."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from repeated_poker import (  # noqa: E402
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)
import serve_hero_range_gui as gui  # noqa: E402


def _loaded_form() -> dict:
    return gui.api_load({"path": str(_HERO_RANGE)})["form"]


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# api_load
# ---------------------------------------------------------------------------


def test_load_hero_range():
    result = gui.api_load({"path": str(_HERO_RANGE)})
    assert result["ok"] is True
    assert result["mode"] == "hero-range"
    form = result["form"]
    assert form["scenario_id"] == "abstract_range_steal_bet98"
    assert form["bet_size"] == 98.0
    assert [h["hand_id"] for h in form["hands"]] == ["chop_fold_candidate", "hero_winner"]
    assert form["hands"][0]["showdown"] == "chop"


@pytest.mark.parametrize("path", [_SINGLE_HAND, _MATRIX])
def test_load_rejects_non_hero_range(path):
    with pytest.raises(ValueError):
        gui.api_load({"path": str(path)})


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


def test_validate_detects_non_positive_weight():
    form = _loaded_form()
    form["hands"][0]["weight"] = "0"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hands[0].weight" for m in result["messages"])


def test_validate_detects_duplicate_hand_id():
    form = _loaded_form()
    form["hands"][1]["hand_id"] = form["hands"][0]["hand_id"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hands[1].hand_id" for m in result["messages"])


def test_validate_detects_weights_not_summing_to_one():
    form = _loaded_form()
    form["hands"][0]["weight"] = "0.5"
    form["hands"][1]["weight"] = "0.2"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hands" for m in result["messages"])


def test_validate_detects_probability_sum_mismatch():
    form = _loaded_form()
    form["hands"][0]["baseline_call_probability"] = "0.4"
    form["hands"][0]["baseline_fold_probability"] = "0.4"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "hands[0].baseline_call_probability" for m in result["messages"]
    )


def test_validate_malformed_hand_entry_is_message_not_exception():
    form = _loaded_form()
    form["hands"] = [None]
    result = gui.api_validate({"form": form})
    assert result["ok"] is True
    assert result["valid"] is False
    assert any(m["field"] == "hands[0]" for m in result["messages"])


def test_validate_bad_value_is_error():
    form = _loaded_form()
    form["hands"][0]["weight"] = "abc"  # parse error, not a validation message
    with pytest.raises(ValueError):
        gui.api_validate({"form": form})


# ---------------------------------------------------------------------------
# api_save
# ---------------------------------------------------------------------------


def test_save_new_file(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["scenario_id"] = "edited_hero_range"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is True
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["scenario_id"] == "edited_hero_range"
    assert [h["hand_id"] for h in data["hero_range"]] == ["chop_fold_candidate", "hero_winner"]


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


def test_save_strict_json(tmp_path):
    out = tmp_path / "out.json"
    result = gui.api_save({"path": str(out), "form": _loaded_form(), "strict_json": True})
    assert result["ok"] is True
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


def test_save_invalid_form_does_not_write(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["hands"][0]["weight"] = "0.5"
    form["hands"][1]["weight"] = "0.2"  # weights no longer sum to 1
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert result["messages"]
    assert not out.exists()


def test_save_force_string_is_rejected(tmp_path):
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


def test_save_preserves_invalid_format_version_for_parser(tmp_path):
    # A numeric format_version is not coerced to "1"; the parser rejects it on save.
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["format_version"] = 1
    # Not a validation message (the validator flags format_version), so it returns
    # ok False with messages rather than writing.
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert any(m["field"] == "format_version" for m in result["messages"])
    assert not out.exists()


# ---------------------------------------------------------------------------
# api_analyze
# ---------------------------------------------------------------------------


def test_analyze_valid_form():
    result = gui.api_analyze({"form": _loaded_form()})
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["scenario_id"] == "abstract_range_steal_bet98"
    assert isinstance(result["generated_count"], int)
    assert isinstance(result["kept_count"], int)
    assert isinstance(result["excluded_count"], int)
    assert isinstance(result["markdown_summary"], str)
    assert result["markdown_summary"]


def test_analyze_render_markdown_false_omits_summary():
    result = gui.api_analyze({"form": _loaded_form(), "render_markdown": False})
    assert result["ok"] is True
    assert result["markdown_summary"] is None


def test_analyze_horizon_and_discount_overrides_apply():
    result = gui.api_analyze({"form": _loaded_form(), "horizon": 25, "discount": 0.9})
    assert result["ok"] is True
    assert result["horizon"] == 25
    assert result["discount"] == 0.9


def test_analyze_invalid_form_does_not_analyze():
    form = _loaded_form()
    form["hands"][0]["weight"] = "0.5"
    form["hands"][1]["weight"] = "0.2"  # weights no longer sum to 1
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert result["valid"] is False
    assert result["messages"]
    assert "markdown_summary" not in result


def test_analyze_bad_value_is_error():
    form = _loaded_form()
    form["hands"][0]["weight"] = "abc"
    with pytest.raises(ValueError):
        gui.api_analyze({"form": form})


def test_analyze_invalid_render_markdown_type():
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "render_markdown": "yes"})


@pytest.mark.parametrize("bad_horizon", [0, -5, 1.5, True])
def test_analyze_invalid_horizon(bad_horizon):
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "horizon": bad_horizon})


@pytest.mark.parametrize("bad_discount", [0, -1.0, float("inf"), float("nan"), "x", True])
def test_analyze_invalid_discount(bad_discount):
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "discount": bad_discount})


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------


def test_page_contains_expected_elements():
    page = gui._PAGE
    assert "hero-range" in page.lower()
    assert "hand_id" in page
    assert "weight" in page
    assert "showdown" in page
    assert "baseline_call_probability" in page
    assert "add_hand_btn" in page
    assert ">Add hand<" in page
    assert "remove_hand" in page
    assert ">Validate<" in page
    assert ">Save JSON<" in page
    assert "hands_body" in page


def test_page_contains_analyze_elements():
    page = gui._PAGE
    assert "analyze_btn" in page
    assert ">Analyze<" in page
    assert "analysis_result" in page
    assert "analysis_counts" in page
    assert "analysis_summary" in page
    assert "opt_horizon" in page
    assert "opt_discount" in page
    assert "render_markdown" in page


def test_page_wires_analyze_options_into_payload():
    page = gui._PAGE
    # The override fields and markdown toggle are wired into the analyze request.
    assert "payload.horizon" in page
    assert "payload.discount" in page
    assert "render_markdown: document.getElementById" in page


def test_analyze_clears_stale_messages_and_result():
    # The analyze handler clears validation messages and the previous analysis
    # result on each run (so a parse error or re-run leaves no stale output).
    page = gui._PAGE
    assert "clearMessagesAndAnalysis" in page
    assert "clearAnalysisResult" in page


def test_analyze_clears_before_client_side_parse_error():
    # The clear must happen before the horizon / discount parse-error returns, so
    # those early returns cannot leave stale messages or analysis result behind.
    page = gui._PAGE
    start = page.index('getElementById("analyze_btn").onclick')
    clear_idx = page.index("clearMessagesAndAnalysis()", start)
    parse_idx = page.index('parseOption("opt_horizon"', start)
    assert clear_idx < parse_idx


def test_load_clears_stale_analysis_result():
    # Loading a different scenario clears a prior analysis result (not just the
    # messages), so a stale scenario_id from an earlier Analyze cannot linger.
    page = gui._PAGE
    start = page.index('getElementById("load_btn").onclick')
    end = page.index("/api/load", start)
    assert "clearMessagesAndAnalysis()" in page[start:end]


# ---------------------------------------------------------------------------
# Live HTTP server (ephemeral port)
# ---------------------------------------------------------------------------


def _serve_in_background():
    server = gui.build_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


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
        assert "Hero-range scenario bucket editor" in page

        loaded = _post(port, "/api/load", {"path": str(_HERO_RANGE)})
        assert loaded["ok"] is True
        assert len(loaded["form"]["hands"]) == 2

        validated = _post(port, "/api/validate", {"form": loaded["form"]})
        assert validated["valid"] is True

        analyzed = _post(port, "/api/analyze", {"form": loaded["form"]})
        assert analyzed["ok"] is True
        assert isinstance(analyzed["markdown_summary"], str)
        assert "generated_count" in analyzed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_server_error_response_has_no_traceback():
    server, thread, port = _serve_in_background()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/load",
            data=json.dumps({"path": str(_SINGLE_HAND)}).encode("utf-8"),
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
        assert "Hero-range-only" in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
