"""Content scan to guard against accidental disclosure before going public.

The forbidden patterns are assembled from fragments (or code points) at runtime
so that this test file does not itself contain the literal strings it scans for.
Only tracked text sources are scanned (``.py`` and ``.md``); compiled caches are
ignored because they are not committed.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _scanned_files():
    files = [_ROOT / "README.md"]
    files += sorted((_ROOT / "docs").glob("*.md"))
    files += sorted((_ROOT / "examples").glob("*.py"))
    files += sorted((_ROOT / "src").rglob("*.py"))
    files += sorted((_ROOT / "tests").glob("*.py"))
    files += sorted((_ROOT / "scripts").glob("*.py"))
    return files


# The Japanese title of the local reference project, built from code points so
# this test file stays ASCII and does not contain the literal string.
_LOCAL_PROJECT_TITLE = "".join(
    chr(c)
    for c in (
        0x7E70, 0x308A, 0x8FD4, 0x3057, 0x30B2,
        0x30FC, 0x30E0, 0x306E, 0x89E3, 0x6790,
    )
)

# Forbidden literals, assembled from fragments so they do not appear verbatim
# in this file.
_FORBIDDEN_STRINGS = [
    "C:" + "\\Users\\" + "gurig",
    "poker " + "sim $EV",
    _LOCAL_PROJECT_TITLE,
    "GITHUB" + "_TOKEN",
    "API" + "_KEY",
    "PASS" + "WORD",
    "SEC" + "RET",
    ".claude/" + "settings.local.json",
]

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.name))
def test_no_forbidden_strings(path):
    text = path.read_text(encoding="utf-8")
    found = [needle for needle in _FORBIDDEN_STRINGS if needle in text]
    assert not found, f"{path} contains forbidden strings: {found}"


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.name))
def test_no_email_addresses(path):
    text = path.read_text(encoding="utf-8")
    matches = _EMAIL_PATTERN.findall(text)
    assert not matches, f"{path} contains email-like strings: {matches}"
