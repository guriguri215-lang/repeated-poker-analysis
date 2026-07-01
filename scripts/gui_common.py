#!/usr/bin/env python3
"""Shared helpers for the local scenario GUI prototypes.

The single-hand, Hero-range, showdown-matrix, equity-matrix, and betting-tree GUI
scripts (``scripts/serve_*_gui.py``) are all the same small local-only web app: a
standard-library ``http.server`` that serves one inline HTML page at ``GET /`` and
dispatches a handful of JSON POST routes. This module factors out the parts that
are shared across those scripts so there is a single place to read and fix them,
and so a new GUI can reuse them instead of copying the scaffolding again.

What lives here is deliberately small and mode-agnostic: the request-handler
factory and server builder, plus a few tiny payload / option primitives -- the
JSON message shape, string coercion, the boolean-flag guard, the analyze-option
horizon / discount validators, and the equity-cell soft-parse shared by the two
equity matrices. Each GUI keeps its own ``_PAGE`` (the inline HTML/CSS/JS) and its
own ``_API`` mapping and ``api_*`` functions -- those carry the per-mode form
fields and validation and are not shared here. No new dependency: standard library
only.
"""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def messages_payload(messages) -> list:
    """Convert form-validation messages into the browser's JSON shape."""

    return [
        {"field": m.field, "message": m.message, "severity": m.severity}
        for m in messages
    ]


def as_text(value) -> str:
    """Render a payload value as a string, treating ``None`` as empty."""

    return "" if value is None else str(value)


def require_bool(value, name: str) -> bool:
    """Return ``value`` if it is a real ``bool``, else raise :class:`ValueError`.

    A string such as ``"false"`` must not silently enable an overwrite or change a
    serialiser option, so the GUIs require an actual boolean for their flags.
    """

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def optional_horizon(value):
    """Validate an optional horizon override (a positive int, ``None`` to skip).

    The GUIs' analyze options share this so the accept/reject behaviour (a real
    ``int`` at least 1; ``bool`` and ``float`` rejected) lives in one place.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("horizon must be an integer")
    if value < 1:
        raise ValueError("horizon must be at least 1")
    return value


def optional_discount(value):
    """Validate an optional discount override (a finite positive number).

    The GUIs' analyze options share this so the accept/reject behaviour (a real
    ``int`` / ``float`` that is finite and positive; ``bool`` rejected) lives in
    one place.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("discount must be a number")
    discount = float(value)
    if not math.isfinite(discount) or discount <= 0:
        raise ValueError("discount must be a finite positive number")
    return discount


def equity_cell_value(raw):
    """Convert one equity matrix cell to a float when possible, else keep it.

    Equity cells are numbers (the Hero pot share before rake), so a numeric string
    from the browser is parsed to ``float`` -- including ``"nan"`` / ``"inf"`` and
    out-of-range values, which stay as floats so the form validator flags them. A
    value that does not parse as a number (an empty string, ``"abc"``, a bool,
    ``None``) is kept unchanged so the validator reports it as a bad cell rather
    than the conversion raising or the value being silently coerced (in particular
    it is never rounded to a default like ``0.5``).

    Shared by the equity-matrix and betting-tree GUIs, whose matrices both hold
    equity cells; keeping it here is the single source for that soft-parse.
    """

    if isinstance(raw, bool):
        return raw  # keep so the validator rejects a boolean cell
    if isinstance(raw, (int, float)):
        return raw
    text = as_text(raw).strip()
    try:
        return float(text)
    except ValueError:
        return raw


def make_handler(api, page):
    """Return a request handler class bound to an ``api`` mapping and ``page``.

    ``api`` maps a POST path (for example ``"/api/load"``) to a function taking the
    decoded JSON payload and returning a JSON-serialisable result; ``page`` is the
    HTML served at ``GET /``. The handler keeps the GUIs' shared safety behaviour:
    a :class:`ValueError` becomes a short ``400`` error message, any other
    exception becomes a generic ``500`` "internal error" with no traceback, and the
    default per-request logging is silenced.
    """

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
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            handler = api.get(self.path)
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


def build_server(host: str, port: int, api, page) -> ThreadingHTTPServer:
    """Create (but do not start) a local GUI server bound to ``host:port``."""

    return ThreadingHTTPServer((host, port), make_handler(api, page))
