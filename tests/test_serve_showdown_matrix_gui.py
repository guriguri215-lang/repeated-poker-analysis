"""Tests for the showdown-matrix scenario GUI prototype (serve_showdown_matrix_gui)."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SHOWDOWN_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_EQUITY_MATRIX = _SCENARIOS / "range_equity_steal_bet98.json"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from repeated_poker import (  # noqa: E402
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)
import serve_showdown_matrix_gui as gui  # noqa: E402


def _loaded_form() -> dict:
    return gui.api_load({"path": str(_SHOWDOWN_MATRIX)})["form"]


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# api_load
# ---------------------------------------------------------------------------


def test_load_showdown_matrix():
    result = gui.api_load({"path": str(_SHOWDOWN_MATRIX)})
    assert result["ok"] is True
    assert result["mode"] == "showdown-matrix"
    form = result["form"]
    assert form["scenario_id"] == "range_matrix_steal_bet98"
    assert form["bet_size"] == 98.0
    assert [b["hand_id"] for b in form["hero_buckets"]] == ["hero_chop", "hero_strong"]
    assert [b["hand_id"] for b in form["villain_buckets"]] == [
        "villain_chop",
        "villain_strong",
    ]
    assert form["showdown_matrix"]["hero_chop"]["villain_strong"] == "villain"
    assert form["showdown_matrix"]["hero_strong"]["villain_chop"] == "hero"


@pytest.mark.parametrize(
    "path", [_SINGLE_HAND, _HERO_RANGE, _EQUITY_MATRIX, _BETTING_TREE]
)
def test_load_rejects_non_showdown_matrix(path):
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


def test_validate_detects_non_positive_hero_weight():
    form = _loaded_form()
    form["hero_buckets"][0]["weight"] = "0"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hero_buckets[0].weight" for m in result["messages"])


def test_validate_detects_duplicate_hero_id():
    form = _loaded_form()
    form["hero_buckets"][1]["hand_id"] = form["hero_buckets"][0]["hand_id"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hero_buckets[1].hand_id" for m in result["messages"])


def test_validate_detects_duplicate_villain_id():
    form = _loaded_form()
    form["villain_buckets"][1]["hand_id"] = form["villain_buckets"][0]["hand_id"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "villain_buckets[1].hand_id" for m in result["messages"])


def test_validate_detects_probability_sum_mismatch():
    form = _loaded_form()
    form["hero_buckets"][0]["baseline_call_probability"] = "0.4"
    form["hero_buckets"][0]["baseline_fold_probability"] = "0.4"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "hero_buckets[0].baseline_call_probability"
        for m in result["messages"]
    )


def test_validate_detects_invalid_matrix_cell():
    form = _loaded_form()
    form["showdown_matrix"]["hero_chop"]["villain_chop"] = "split"  # not a result
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "showdown_matrix[hero_chop][villain_chop]"
        for m in result["messages"]
    )


def test_validate_detects_missing_matrix_cell():
    form = _loaded_form()
    del form["showdown_matrix"]["hero_chop"]["villain_strong"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"].startswith("showdown_matrix") for m in result["messages"])


def test_validate_malformed_hero_bucket_entry_is_message_not_exception():
    form = _loaded_form()
    form["hero_buckets"] = [None]
    result = gui.api_validate({"form": form})
    assert result["ok"] is True
    assert result["valid"] is False
    assert any(m["field"] == "hero_buckets[0]" for m in result["messages"])


def test_validate_bad_value_is_error():
    form = _loaded_form()
    form["hero_buckets"][0]["weight"] = "abc"  # parse error, not a validation message
    with pytest.raises(ValueError):
        gui.api_validate({"form": form})


# ---------------------------------------------------------------------------
# api_save
# ---------------------------------------------------------------------------


def test_save_new_file(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["scenario_id"] = "edited_showdown_matrix"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is True
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["scenario_id"] == "edited_showdown_matrix"
    assert [h["hand_id"] for h in data["hero_range"]] == ["hero_chop", "hero_strong"]
    assert data["showdown_matrix"]["hero_strong"]["villain_chop"] == "hero"


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
    form["hero_buckets"][0]["weight"] = "0.5"
    form["hero_buckets"][1]["weight"] = "0.2"  # weights no longer sum to 1
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
    # A numeric format_version is not coerced to "1"; the validator flags it, so the
    # save returns ok False with messages rather than writing.
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["format_version"] = 1
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
    assert result["scenario_id"] == "range_matrix_steal_bet98"
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
    form["hero_buckets"][0]["weight"] = "0.5"
    form["hero_buckets"][1]["weight"] = "0.2"  # weights no longer sum to 1
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert result["valid"] is False
    assert result["messages"]
    assert "markdown_summary" not in result


def test_analyze_bad_value_is_error():
    form = _loaded_form()
    form["hero_buckets"][0]["weight"] = "abc"
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
    assert "showdown-matrix" in page.lower()
    assert "hero_buckets" in page
    assert "villain_buckets" in page
    assert "hero_body" in page
    assert "villain_body" in page
    assert "matrix_body" in page
    assert "matrix_head" in page
    assert "add_hero_btn" in page
    assert "add_villain_btn" in page
    assert ">Add hero bucket<" in page
    assert ">Add villain bucket<" in page
    assert "remove_bucket" in page
    assert ">Rebuild matrix<" in page
    assert ">Validate<" in page
    assert ">Save JSON<" in page
    assert "showdown_matrix" in page


def test_page_has_no_innerhtml_injection():
    # All DOM updates use createElement / textContent / appendChild / removeChild;
    # the page must never assign innerHTML (no HTML-string injection path).
    assert "innerHTML" not in gui._PAGE


def test_page_has_matrix_rebuild_helper():
    page = gui._PAGE
    assert "function rebuildMatrix" in page
    assert "function collectMatrix" in page


def test_page_collects_matrix_into_payload():
    page = gui._PAGE
    # The form sent to the API carries the matrix collected from the grid.
    assert "form.showdown_matrix = collectMatrix()" in page
    assert "form.hero_buckets = collectBuckets" in page
    assert "form.villain_buckets = collectBuckets" in page


def test_page_add_remove_handlers_rebuild_matrix():
    page = gui._PAGE
    # Adding a hero / villain bucket and removing a row rebuild the matrix grid.
    assert "addHero(); rebuildMatrix();" in page
    assert "addVillain(); rebuildMatrix();" in page
    assert "rebuildMatrix();" in page  # also called from a row's Remove handler


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
        assert "Showdown-matrix scenario editor" in page

        loaded = _post(port, "/api/load", {"path": str(_SHOWDOWN_MATRIX)})
        assert loaded["ok"] is True
        assert len(loaded["form"]["hero_buckets"]) == 2
        assert len(loaded["form"]["villain_buckets"]) == 2

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
            data=json.dumps({"path": str(_HERO_RANGE)}).encode("utf-8"),
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
        assert "showdown_matrix" in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
