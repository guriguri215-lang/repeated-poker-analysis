"""Inspect or run one strict three-player CFR file-v1 JSON document."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.three_player_cfr_file_workflow import (
    ThreePlayerCfrFileError,
    ThreePlayerCfrFileResult,
    ThreePlayerCfrFileStatus,
    process_three_player_cfr_file,
    three_player_cfr_file_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a bounded three-player CFR-style diagnostic document."
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
    except OSError:
        result = ThreePlayerCfrFileResult(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            None,
            ThreePlayerCfrFileError("input", "cannot read input file", None),
        )
    else:
        result = process_three_player_cfr_file(raw)
    print(three_player_cfr_file_json(result))
    return 0 if result.status is ThreePlayerCfrFileStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
