"""Tests for the betting-tree scenario GUI prototype (serve_betting_tree_gui)."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_BETTING_TREE = _SCENARIOS / "range_equity_betting_tree_bet98.json"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_HERO_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"
_SHOWDOWN_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"
_EQUITY_MATRIX = _SCENARIOS / "range_equity_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from repeated_poker import (  # noqa: E402
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)
import serve_betting_tree_gui as gui  # noqa: E402


def _loaded_form() -> dict:
    return gui.api_load({"path": str(_BETTING_TREE)})["form"]


def _showdown_form() -> dict:
    """The equity sample re-cast as a showdown-matrix betting-tree form."""

    form = _loaded_form()
    form["matrix_type"] = "showdown"
    for row in form["matrix"].values():
        for villain_id in row:
            row[villain_id] = "hero"
    return form


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# api_load
# ---------------------------------------------------------------------------


def test_load_betting_tree():
    result = gui.api_load({"path": str(_BETTING_TREE)})
    assert result["ok"] is True
    assert result["mode"] == "betting-tree"
    form = result["form"]
    assert form["scenario_id"] == "range_equity_betting_tree_bet98"
    assert form["matrix_type"] == "equity"
    assert form["bet_size"] == 98.0
    assert form["betting_tree"] == {
        "oop_bet_size": 98.0,
        "ip_bet_after_check_size": 98.0,
        "ip_raise_size": 196.0,
    }
    assert [b["hand_id"] for b in form["hero_buckets"]] == ["hero_medium", "hero_strong"]
    assert [b["hand_id"] for b in form["villain_buckets"]] == [
        "villain_weak",
        "villain_strong",
    ]
    assert form["matrix"]["hero_strong"]["villain_weak"] == 0.9
    # Hero buckets expose the two decision-point distributions.
    assert set(form["hero_buckets"][0]) == {
        "hand_id",
        "weight",
        "after_oop_check_check_probability",
        "after_oop_check_bet_probability",
        "vs_oop_bet_call_probability",
        "vs_oop_bet_fold_probability",
        "vs_oop_bet_raise_probability",
    }


@pytest.mark.parametrize(
    "path", [_SINGLE_HAND, _HERO_RANGE, _SHOWDOWN_MATRIX, _EQUITY_MATRIX]
)
def test_load_rejects_non_betting_tree(path):
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


def test_validate_valid_equity_form():
    result = gui.api_validate({"form": _loaded_form()})
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["messages"] == []


def test_validate_valid_showdown_form():
    result = gui.api_validate({"form": _showdown_form()})
    assert result["valid"] is True
    assert result["messages"] == []


def test_validate_non_positive_size():
    form = _loaded_form()
    form["betting_tree"]["ip_bet_after_check_size"] = "0"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "betting_tree.ip_bet_after_check_size" for m in result["messages"]
    )


def test_validate_ip_raise_not_greater_than_oop_bet():
    form = _loaded_form()
    form["betting_tree"]["ip_raise_size"] = "50"  # <= oop_bet_size 98
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "betting_tree.ip_raise_size" for m in result["messages"])


def test_validate_bet_size_must_equal_oop_bet_size():
    form = _loaded_form()
    form["bet_size"] = "50"  # != oop_bet_size 98
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "bet_size" for m in result["messages"])


def test_validate_after_oop_check_sum_mismatch():
    form = _loaded_form()
    form["hero_buckets"][0]["after_oop_check_check_probability"] = "0.4"
    form["hero_buckets"][0]["after_oop_check_bet_probability"] = "0.4"  # sum 0.8
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "hero_buckets[0].after_oop_check_check_probability"
        for m in result["messages"]
    )


def test_validate_vs_oop_bet_sum_mismatch():
    form = _loaded_form()
    form["hero_buckets"][0]["vs_oop_bet_call_probability"] = "0.4"
    form["hero_buckets"][0]["vs_oop_bet_fold_probability"] = "0.4"
    form["hero_buckets"][0]["vs_oop_bet_raise_probability"] = "0.0"  # sum 0.8
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "hero_buckets[0].vs_oop_bet_call_probability"
        for m in result["messages"]
    )


def test_validate_negative_probability():
    form = _loaded_form()
    form["hero_buckets"][0]["vs_oop_bet_raise_probability"] = "-0.5"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "hero_buckets[0].vs_oop_bet_raise_probability"
        for m in result["messages"]
    )


def test_validate_duplicate_hero_id():
    form = _loaded_form()
    form["hero_buckets"][1]["hand_id"] = form["hero_buckets"][0]["hand_id"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "hero_buckets[1].hand_id" for m in result["messages"])


def test_validate_duplicate_villain_id():
    form = _loaded_form()
    form["villain_buckets"][1]["hand_id"] = form["villain_buckets"][0]["hand_id"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "villain_buckets[1].hand_id" for m in result["messages"])


def test_validate_invalid_matrix_type():
    form = _loaded_form()
    form["matrix_type"] = "split"
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"] == "matrix_type" for m in result["messages"])


def test_validate_showdown_invalid_cell():
    form = _showdown_form()
    form["matrix"]["hero_medium"]["villain_weak"] = "split"  # not chop/hero/villain
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "showdown_matrix[hero_medium][villain_weak]"
        for m in result["messages"]
    )


@pytest.mark.parametrize("bad_cell", ["1.5", "abc"])
def test_validate_equity_invalid_cell(bad_cell):
    form = _loaded_form()
    form["matrix"]["hero_medium"]["villain_weak"] = bad_cell
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(
        m["field"] == "equity_matrix[hero_medium][villain_weak]"
        for m in result["messages"]
    )


def test_validate_missing_matrix_cell():
    form = _loaded_form()
    del form["matrix"]["hero_medium"]["villain_strong"]
    result = gui.api_validate({"form": form})
    assert result["valid"] is False
    assert any(m["field"].startswith("equity_matrix") for m in result["messages"])


def test_validate_malformed_hero_bucket_entry_is_message_not_exception():
    form = _loaded_form()
    form["hero_buckets"] = [None]
    result = gui.api_validate({"form": form})
    assert result["ok"] is True
    assert result["valid"] is False
    assert any(m["field"] == "hero_buckets[0]" for m in result["messages"])


def test_validate_bad_value_is_error():
    form = _loaded_form()
    form["hero_buckets"][0]["weight"] = "abc"  # parse error, not a message
    with pytest.raises(ValueError):
        gui.api_validate({"form": form})


# ---------------------------------------------------------------------------
# api_save
# ---------------------------------------------------------------------------


def test_save_new_file(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["scenario_id"] = "edited_betting_tree"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is True
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["scenario_id"] == "edited_betting_tree"
    assert data["betting_tree"]["ip_raise_size"] == 196.0
    assert "equity_matrix" in data


def test_save_showdown_variant_new_file(tmp_path):
    out = tmp_path / "sd.json"
    result = gui.api_save({"path": str(out), "form": _showdown_form()})
    assert result["ok"] is True
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert "showdown_matrix" in data
    assert "equity_matrix" not in data


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
    form["betting_tree"]["ip_raise_size"] = "50"  # <= oop_bet_size
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
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["format_version"] = 1  # numeric, not coerced to "1"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert any(m["field"] == "format_version" for m in result["messages"])
    assert not out.exists()


def test_save_does_not_round_invalid_equity_cell(tmp_path):
    out = tmp_path / "out.json"
    form = _loaded_form()
    form["matrix"]["hero_medium"]["villain_weak"] = "1.5"
    result = gui.api_save({"path": str(out), "form": form})
    assert result["ok"] is False
    assert any(
        m["field"] == "equity_matrix[hero_medium][villain_weak]"
        for m in result["messages"]
    )
    assert not out.exists()


# ---------------------------------------------------------------------------
# api_analyze
# ---------------------------------------------------------------------------


def test_analyze_valid_equity_form():
    result = gui.api_analyze({"form": _loaded_form()})
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["scenario_id"] == "range_equity_betting_tree_bet98"
    assert isinstance(result["generated_count"], int)
    assert isinstance(result["kept_count"], int)
    assert isinstance(result["excluded_count"], int)
    assert isinstance(result["markdown_summary"], str)
    assert result["markdown_summary"]


def test_analyze_valid_showdown_form():
    result = gui.api_analyze({"form": _showdown_form()})
    assert result["ok"] is True
    assert result["valid"] is True
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


def test_analyze_invalid_size_does_not_analyze():
    form = _loaded_form()
    form["betting_tree"]["ip_raise_size"] = "50"  # <= oop_bet_size
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert result["valid"] is False
    assert any(m["field"] == "betting_tree.ip_raise_size" for m in result["messages"])
    assert "markdown_summary" not in result


def test_analyze_invalid_distribution_does_not_analyze():
    form = _loaded_form()
    form["hero_buckets"][0]["vs_oop_bet_call_probability"] = "0.4"
    form["hero_buckets"][0]["vs_oop_bet_fold_probability"] = "0.4"
    form["hero_buckets"][0]["vs_oop_bet_raise_probability"] = "0.0"  # sum 0.8
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert any(
        m["field"] == "hero_buckets[0].vs_oop_bet_call_probability"
        for m in result["messages"]
    )


@pytest.mark.parametrize("bad_cell", ["1.5", "abc"])
def test_analyze_invalid_equity_cell_does_not_analyze(bad_cell):
    form = _loaded_form()
    form["matrix"]["hero_medium"]["villain_weak"] = bad_cell
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert result["valid"] is False
    assert any(
        m["field"] == "equity_matrix[hero_medium][villain_weak]"
        for m in result["messages"]
    )
    assert "markdown_summary" not in result


def test_analyze_invalid_showdown_cell_does_not_analyze():
    form = _showdown_form()
    form["matrix"]["hero_medium"]["villain_weak"] = "split"  # not chop/hero/villain
    result = gui.api_analyze({"form": form})
    assert result["ok"] is False
    assert any(
        m["field"] == "showdown_matrix[hero_medium][villain_weak]"
        for m in result["messages"]
    )


def test_analyze_bad_value_is_error():
    form = _loaded_form()
    form["hero_buckets"][0]["weight"] = "abc"
    with pytest.raises(ValueError):
        gui.api_analyze({"form": form})


def test_analyze_invalid_render_markdown_type():
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "render_markdown": "yes"})


@pytest.mark.parametrize("bad_horizon", [0, -5, 1.5, True, "x"])
def test_analyze_invalid_horizon(bad_horizon):
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "horizon": bad_horizon})


@pytest.mark.parametrize(
    "bad_discount", [0, -1.0, float("inf"), float("nan"), True, "x"]
)
def test_analyze_invalid_discount(bad_discount):
    with pytest.raises(ValueError):
        gui.api_analyze({"form": _loaded_form(), "discount": bad_discount})


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------


def test_page_contains_expected_elements():
    page = gui._PAGE
    assert "betting-tree" in page.lower()
    assert "matrix_type" in page
    assert "oop_bet_size" in page
    assert "ip_bet_after_check_size" in page
    assert "ip_raise_size" in page
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
    assert "after_oop_check_check_probability" in page
    assert "vs_oop_bet_raise_probability" in page


def test_page_has_no_innerhtml_injection():
    assert "innerHTML" not in gui._PAGE


def test_page_has_matrix_type_selector():
    page = gui._PAGE
    assert '<select id="matrix_type">' in page
    assert '<option value="showdown">' in page
    assert '<option value="equity">' in page


def test_page_has_matrix_rebuild_helper():
    page = gui._PAGE
    assert "function rebuildMatrix" in page
    assert "function collectMatrix" in page
    # matrix_type change rebuilds the grid.
    assert 'getElementById("matrix_type").onchange' in page


def test_page_collects_matrix_and_sizes_into_payload():
    page = gui._PAGE
    assert "form.matrix = collectMatrix()" in page
    assert "form.matrix_type = currentMatrixType()" in page
    assert "form.betting_tree = betting" in page
    assert "form.hero_buckets = collectBuckets" in page
    assert "form.villain_buckets = collectBuckets" in page


def test_page_add_remove_handlers_rebuild_matrix():
    page = gui._PAGE
    assert "addHero(); rebuildMatrix();" in page
    assert "addVillain(); rebuildMatrix();" in page


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
    assert "payload.horizon" in page
    assert "payload.discount" in page
    assert "render_markdown: document.getElementById" in page


def test_analyze_clears_stale_messages_and_result():
    page = gui._PAGE
    assert "clearMessagesAndAnalysis" in page
    assert "clearAnalysisResult" in page


def test_analyze_clears_before_client_side_parse_error():
    # The clear must happen before the horizon / discount parse-error returns.
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
        assert "Betting-tree scenario editor" in page

        loaded = _post(port, "/api/load", {"path": str(_BETTING_TREE)})
        assert loaded["ok"] is True
        assert len(loaded["form"]["hero_buckets"]) == 2
        assert loaded["form"]["matrix_type"] == "equity"

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
            data=json.dumps({"path": str(_EQUITY_MATRIX)}).encode("utf-8"),
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
        assert "betting_tree" in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
