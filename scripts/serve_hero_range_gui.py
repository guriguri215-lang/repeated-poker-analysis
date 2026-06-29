#!/usr/bin/env python3
"""Serve a minimal local GUI for editing a Hero-range-only scenario (prototype v1).

Usage:

    python scripts/serve_hero_range_gui.py --port 8001
    # then open http://127.0.0.1:8001/ in a browser

This is a pre-GUI-product prototype: a local-only web page that loads a
Hero-range-only scenario JSON into a form (top-level fields plus a table of
weighted Hero buckets), lets you add / remove / edit buckets, validates the form,
and saves it back to JSON. It is the Hero-range counterpart of
``scripts/serve_single_hand_gui.py`` and uses the standard library only
(``http.server`` plus inline HTML / CSS / vanilla JavaScript); it adds no
framework or dependency, and no new solver, model, or analysis logic.

Scope (v1): Hero-range-only mode and bucket editing only. Single-hand, matrix
(showdown / equity), and betting-tree scenarios are rejected. Running the analysis
pipeline, charts, real-card parsing, and external solver imports are all out of
scope. It reuses the existing form model, parser, and game builder as the source
of truth.

Endpoints:

* ``GET  /``            -- the bucket-editor page;
* ``POST /api/load``    -- ``{"path": "..."}`` -> the form fields of that
  Hero-range-only scenario (a non-Hero-range scenario is rejected);
* ``POST /api/validate``-- ``{"form": {...}}`` -> field-level validation messages;
* ``POST /api/save``    -- ``{"path", "form", "force", "strict_json"}`` -> writes
  the JSON only when the form validates and its ``to_dict`` re-parses and rebuilds.

Security / safety: it binds to ``127.0.0.1`` by default and makes no external
calls. It reads and writes only the local paths you type, refuses to overwrite an
existing file unless the overwrite box is checked, requires real booleans for the
save options, keeps the raw ``format_version`` (no coercion), and returns short
``error`` messages (never a traceback).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling CLIs

from repeated_poker import (  # noqa: E402  (path is set up above)
    HeroRangeHandForm,
    HeroRangeScenarioForm,
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    hero_range_form_from_dict,
    hero_range_form_to_dict,
    river_scenario_from_dict,
    validate_hero_range_form,
)
from repeated_poker.report_export import _dump_json  # noqa: E402

# Reuse the loader, the safe file writer, and the value parsing from the sibling
# CLIs rather than duplicating them.
from inspect_scenario_form import _load_scenario_dict  # noqa: E402
from roundtrip_scenario_form import _write_output  # noqa: E402
from edit_scenario_form import _NO_CAP_VALUES, _to_float, _to_number_list  # noqa: E402

# Top-level (non-hand) flat fields shown in the form.
_TOP_FIELDS = [
    "scenario_id",
    "description",
    "rake_rate",
    "rake_cap",
    "initial_commitment_hero",
    "initial_commitment_villain",
    "bet_size",
    "shift_amounts",
    "horizons",
    "discount",
]
# Per-bucket fields shown in each table row.
_HAND_FIELDS = [
    "hand_id",
    "weight",
    "showdown",
    "baseline_call_probability",
    "baseline_fold_probability",
]


def _as_text(value) -> str:
    return "" if value is None else str(value)


def _require_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _hand_from_payload(raw, index: int) -> HeroRangeHandForm:
    """Convert one bucket dict from the browser into a HeroRangeHandForm."""

    hand = HeroRangeHandForm()
    if "hand_id" in raw:
        hand.hand_id = _as_text(raw["hand_id"])
    if "showdown" in raw:
        hand.showdown = _as_text(raw["showdown"])
    if "weight" in raw:
        hand.weight = _to_float(f"hands[{index}].weight", _as_text(raw["weight"]))
    if "baseline_call_probability" in raw:
        hand.baseline_call_probability = _to_float(
            f"hands[{index}].baseline_call_probability",
            _as_text(raw["baseline_call_probability"]),
        )
    if "baseline_fold_probability" in raw:
        hand.baseline_fold_probability = _to_float(
            f"hands[{index}].baseline_fold_probability",
            _as_text(raw["baseline_fold_probability"]),
        )
    return hand


def _form_from_payload(payload) -> HeroRangeScenarioForm:
    """Build a HeroRangeScenarioForm from the browser's flat dict.

    Top-level numeric fields and per-bucket weights / probabilities are converted
    with the same rules as the edit CLI (floats, optional ``rake_cap``,
    comma-separated ``shift_amounts`` / ``horizons``). ``format_version`` is kept
    raw (an unsupported value is reported by the validator / parser, not coerced).
    A non-object bucket entry is kept as-is so ``validate_hero_range_form`` reports
    it as a ``hands[i]`` message instead of the conversion raising. A bad value
    inside a bucket dict (for example a non-numeric weight) raises
    :class:`ValueError`.
    """

    if not isinstance(payload, dict):
        raise ValueError("form must be a JSON object")
    form = HeroRangeScenarioForm()
    if "format_version" in payload:
        form.format_version = payload["format_version"]
    if "scenario_id" in payload:
        form.scenario_id = _as_text(payload["scenario_id"])
    if "description" in payload:
        form.description = _as_text(payload["description"])
    if "rake_rate" in payload:
        form.rake_rate = _to_float("rake_rate", _as_text(payload["rake_rate"]))
    if "rake_cap" in payload:
        raw_cap = _as_text(payload["rake_cap"])
        form.rake_cap = (
            None if raw_cap.strip().lower() in _NO_CAP_VALUES else _to_float("rake_cap", raw_cap)
        )
    if "initial_commitment_hero" in payload:
        form.initial_commitment_hero = _to_float(
            "initial_commitment_hero", _as_text(payload["initial_commitment_hero"])
        )
    if "initial_commitment_villain" in payload:
        form.initial_commitment_villain = _to_float(
            "initial_commitment_villain", _as_text(payload["initial_commitment_villain"])
        )
    if "bet_size" in payload:
        form.bet_size = _to_float("bet_size", _as_text(payload["bet_size"]))
    if "shift_amounts" in payload:
        form.shift_amounts = _to_number_list(
            "shift_amounts", _as_text(payload["shift_amounts"]), float
        )
    if "horizons" in payload:
        form.horizons = _to_number_list("horizons", _as_text(payload["horizons"]), int)
    if "discount" in payload:
        form.discount = _to_float("discount", _as_text(payload["discount"]))
    if "hands" in payload:
        hands_raw = payload["hands"]
        if not isinstance(hands_raw, list):
            raise ValueError("hands must be a list")
        hands = []
        for index, entry in enumerate(hands_raw):
            if not isinstance(entry, dict):
                # Keep a malformed entry as-is; the validator reports it as a
                # hands[i] message rather than the conversion raising.
                hands.append(entry)
            else:
                hands.append(_hand_from_payload(entry, index))
        form.hands = hands
    return form


def _form_to_payload(form: HeroRangeScenarioForm) -> dict:
    """Flatten a form into the string-friendly dict the browser displays."""

    return {
        "format_version": form.format_version,
        "scenario_id": form.scenario_id,
        "description": form.description,
        "rake_rate": form.rake_rate,
        "rake_cap": "" if form.rake_cap is None else form.rake_cap,
        "initial_commitment_hero": form.initial_commitment_hero,
        "initial_commitment_villain": form.initial_commitment_villain,
        "bet_size": form.bet_size,
        "shift_amounts": ",".join(str(value) for value in form.shift_amounts),
        "horizons": ",".join(str(value) for value in form.horizons),
        "discount": form.discount,
        "hands": [
            {
                "hand_id": hand.hand_id,
                "weight": hand.weight,
                "showdown": hand.showdown,
                "baseline_call_probability": hand.baseline_call_probability,
                "baseline_fold_probability": hand.baseline_fold_probability,
            }
            for hand in form.hands
        ],
    }


def _messages_payload(messages) -> list:
    return [
        {"field": m.field, "message": m.message, "severity": m.severity}
        for m in messages
    ]


def api_load(payload) -> dict:
    """Load a Hero-range-only scenario file into form fields for the browser."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    data = _load_scenario_dict(path.strip())
    mode = detect_scenario_form_mode(data)
    if mode != "hero-range":
        raise ValueError(
            f"hero-range GUI supports Hero-range-only mode; this scenario is {mode} mode"
        )
    form = hero_range_form_from_dict(data)
    return {"ok": True, "mode": mode, "form": _form_to_payload(form)}


def api_validate(payload) -> dict:
    """Validate the form the browser sent, returning field-level messages."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    form = _form_from_payload(payload.get("form"))
    messages = validate_hero_range_form(form)
    return {"ok": True, "valid": not messages, "messages": _messages_payload(messages)}


def api_save(payload) -> dict:
    """Validate, round-trip, and write the form the browser sent.

    Returns ``{"ok": False, ...}`` with messages when the form is invalid (nothing
    is written). Raises :class:`ValueError` for a missing path, a non-boolean
    option, a bad value, a failed round-trip, or a write problem.
    """

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    force = _require_bool(payload.get("force", False), "force")
    strict = _require_bool(payload.get("strict_json", False), "strict_json")

    form = _form_from_payload(payload.get("form"))
    messages = validate_hero_range_form(form)
    if messages:
        return {
            "ok": False,
            "valid": False,
            "messages": _messages_payload(messages),
            "error": "form has validation messages; not saved",
        }

    try:
        out_dict = hero_range_form_to_dict(form)
        scenario = river_scenario_from_dict(out_dict)
        build_river_steal_game_from_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - surface as a clean round-trip error
        raise ValueError(f"round-trip failed, not saved: {exc}")

    text = _dump_json(out_dict, strict)
    _write_output(text, path.strip(), force, print_func=lambda *_args, **_kwargs: None)
    return {"ok": True, "path": path.strip()}


_API = {
    "/api/load": api_load,
    "/api/validate": api_validate,
    "/api/save": api_save,
}

# Inline single-file page: HTML + CSS + vanilla JS, no external resources.
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hero-range scenario bucket editor (prototype)</title>
<style>
  body { font-family: sans-serif; max-width: 880px; margin: 1.5rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; }
  fieldset { margin: 1rem 0; }
  label { display: block; margin: 0.4rem 0; }
  label span { display: inline-block; width: 16rem; }
  input[type=text] { width: 22rem; }
  table { border-collapse: collapse; margin: 0.5rem 0; }
  th, td { border: 1px solid #ccc; padding: 0.25rem 0.4rem; }
  td input[type=text] { width: 10rem; }
  .row { margin: 0.5rem 0; }
  button { margin-right: 0.5rem; }
  #status { margin: 0.75rem 0; font-weight: bold; }
  #messages { color: #a00; white-space: pre-wrap; }
  .hint { color: #555; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>Hero-range scenario bucket editor (local prototype)</h1>
<p class="hint">Local-only Hero-range-only editor. Load a scenario JSON, edit the
weighted Hero buckets, validate, and save. Single-hand / matrix / betting-tree
scenarios are not supported here, and the analysis pipeline is not run.</p>

<fieldset>
  <legend>Load</legend>
  <div class="row">
    <label><span>scenario path</span>
      <input type="text" id="load_path" placeholder="examples/scenarios/abstract_range_steal_bet98.json"></label>
    <button id="load_btn">Load</button>
  </div>
</fieldset>

<fieldset>
  <legend>Top-level fields</legend>
  <label><span>scenario_id</span><input type="text" id="scenario_id"></label>
  <label><span>description</span><input type="text" id="description"></label>
  <label><span>rake_rate</span><input type="text" id="rake_rate"></label>
  <label><span>rake_cap (blank = no cap)</span><input type="text" id="rake_cap"></label>
  <label><span>initial_commitment_hero</span><input type="text" id="initial_commitment_hero"></label>
  <label><span>initial_commitment_villain</span><input type="text" id="initial_commitment_villain"></label>
  <label><span>bet_size</span><input type="text" id="bet_size"></label>
  <label><span>shift_amounts (comma list)</span><input type="text" id="shift_amounts"></label>
  <label><span>horizons (comma list)</span><input type="text" id="horizons"></label>
  <label><span>discount</span><input type="text" id="discount"></label>
</fieldset>

<fieldset>
  <legend>Hero buckets</legend>
  <p class="hint">Each bucket is a weighted abstract Hero hand. Weights must sum to
  1; hand_id values must be unique; showdown is hero / villain / chop; the call and
  fold probabilities must sum to 1.</p>
  <table>
    <thead>
      <tr>
        <th>hand_id</th><th>weight</th><th>showdown</th>
        <th>baseline_call_probability</th><th>baseline_fold_probability</th><th></th>
      </tr>
    </thead>
    <tbody id="hands_body"></tbody>
  </table>
  <button id="add_hand_btn">Add hand</button>
</fieldset>

<fieldset>
  <legend>Save</legend>
  <div class="row">
    <label><span>output path</span><input type="text" id="save_path"></label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="force"> overwrite existing file</label>
    <label><input type="checkbox" id="strict_json"> strict JSON</label>
  </div>
  <button id="validate_btn">Validate</button>
  <button id="save_btn">Save JSON</button>
</fieldset>

<div id="status"></div>
<div id="messages"></div>

<script>
var TOP_FIELDS = ["scenario_id","description","rake_rate","rake_cap",
  "initial_commitment_hero","initial_commitment_villain","bet_size",
  "shift_amounts","horizons","discount"];
var HAND_FIELDS = ["hand_id","weight","showdown",
  "baseline_call_probability","baseline_fold_probability"];
var formatVersion = "1";

function makeHandRow(hand) {
  hand = hand || {};
  var tr = document.createElement("tr");
  HAND_FIELDS.forEach(function (f) {
    var td = document.createElement("td");
    var input = document.createElement("input");
    input.type = "text";
    input.setAttribute("data-field", f);
    var v = hand[f];
    input.value = (v === null || v === undefined) ? "" : v;
    td.appendChild(input);
    tr.appendChild(td);
  });
  var td = document.createElement("td");
  var remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove_hand";
  remove.textContent = "Remove";
  remove.onclick = function () { tr.parentNode.removeChild(tr); };
  td.appendChild(remove);
  tr.appendChild(td);
  return tr;
}
function addHand(hand) {
  document.getElementById("hands_body").appendChild(makeHandRow(hand));
}
function clearHands() {
  var body = document.getElementById("hands_body");
  while (body.firstChild) { body.removeChild(body.firstChild); }
}
function collectForm() {
  var form = {format_version: formatVersion};
  TOP_FIELDS.forEach(function (f) { form[f] = document.getElementById(f).value; });
  form.hands = [];
  var rows = document.querySelectorAll("#hands_body tr");
  rows.forEach(function (row) {
    var hand = {};
    row.querySelectorAll("input[data-field]").forEach(function (input) {
      hand[input.getAttribute("data-field")] = input.value;
    });
    form.hands.push(hand);
  });
  return form;
}
function fillForm(form) {
  formatVersion = form.format_version || "1";
  TOP_FIELDS.forEach(function (f) {
    var v = form[f];
    document.getElementById(f).value = (v === null || v === undefined) ? "" : v;
  });
  clearHands();
  (form.hands || []).forEach(function (h) { addHand(h); });
}
function setStatus(text, isError) {
  var s = document.getElementById("status");
  s.textContent = text;
  s.style.color = isError ? "#a00" : "#070";
}
function showMessages(messages) {
  var box = document.getElementById("messages");
  if (!messages || !messages.length) { box.textContent = ""; return; }
  box.textContent = messages.map(function (m) {
    return "[" + m.severity + "] " + m.field + ": " + m.message;
  }).join("\\n");
}
function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  }).then(function (r) { return r.json(); });
}

document.getElementById("add_hand_btn").onclick = function () { addHand(); };

document.getElementById("load_btn").onclick = function () {
  showMessages([]);
  post("/api/load", {path: document.getElementById("load_path").value})
    .then(function (res) {
      if (!res.ok) { setStatus("error: " + res.error, true); return; }
      fillForm(res.form);
      if (!document.getElementById("save_path").value) {
        document.getElementById("save_path").value = document.getElementById("load_path").value;
      }
      setStatus("loaded (" + res.mode + ")", false);
    })
    .catch(function () { setStatus("request failed", true); });
};

document.getElementById("validate_btn").onclick = function () {
  post("/api/validate", {form: collectForm()})
    .then(function (res) {
      if (!res.ok) { setStatus("error: " + res.error, true); showMessages([]); return; }
      showMessages(res.messages);
      setStatus(res.valid ? "valid" : (res.messages.length + " validation message(s)"), !res.valid);
    })
    .catch(function () { setStatus("request failed", true); });
};

document.getElementById("save_btn").onclick = function () {
  post("/api/save", {
    path: document.getElementById("save_path").value,
    form: collectForm(),
    force: document.getElementById("force").checked,
    strict_json: document.getElementById("strict_json").checked
  })
    .then(function (res) {
      if (!res.ok) {
        setStatus("error: " + (res.error || "not saved"), true);
        showMessages(res.messages || []);
        return;
      }
      showMessages([]);
      setStatus("saved " + res.path, false);
    })
    .catch(function () { setStatus("request failed", true); });
};
</script>
</body>
</html>
"""


def make_handler():
    """Return a request handler class bound to the API functions and page."""

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                raise ValueError("request body must be valid JSON")

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in ("/", "/index.html"):
                body = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            handler = _API.get(self.path)
            if handler is None:
                self._send_json({"ok": False, "error": "not found"}, 404)
                return
            try:
                payload = self._read_json()
                self._send_json(handler(payload))
            except ValueError as exc:
                # Expected, user-facing error: short message, no traceback.
                self._send_json({"ok": False, "error": str(exc)}, 400)
            except Exception:  # noqa: BLE001 - never leak a traceback to the client
                self._send_json({"ok": False, "error": "internal error"}, 500)

        def log_message(self, *_args):  # silence the default per-request logging
            pass

    return _Handler


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create (but do not start) the local GUI server bound to ``host:port``."""

    return ThreadingHTTPServer((host, port), make_handler())


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Serve a minimal local GUI for editing a Hero-range-only scenario."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host to bind (default 127.0.0.1, local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="port to bind (default 8001)",
    )
    return parser.parse_args(argv)


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        server = build_server(args.host, args.port)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    host, port = server.server_address[0], server.server_address[1]
    print(f"serving hero-range scenario GUI at http://{host}:{port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
