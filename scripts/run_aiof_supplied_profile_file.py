"""Inspect or run one strict AIoF supplied-profile v1 JSON document."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.aiof_supplied_profile_file_workflow import (
    AiofSuppliedProfileFileError,
    AiofSuppliedProfileFileResult,
    AiofSuppliedProfileFileStatus,
    aiof_supplied_profile_file_json,
    process_aiof_supplied_profile_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a bounded exact AIoF supplied-profile document."
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
    except OSError:
        result = AiofSuppliedProfileFileResult(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            None,
            AiofSuppliedProfileFileError("input", "cannot read input file", None),
        )
    else:
        result = process_aiof_supplied_profile_file(raw)
    print(aiof_supplied_profile_file_json(result))
    return 0 if result.status is AiofSuppliedProfileFileStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
