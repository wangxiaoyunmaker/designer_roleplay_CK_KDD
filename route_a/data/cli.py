"""Command-line interface for Route A data preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .build import build_dataset, inspect_dataset, validate_dataset
from .errors import DataContractError
from .io import canonical_json_bytes, read_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m route_a.data",
        description="Build and validate Route A training datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a draft or formal dataset")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--mode", choices=("draft", "formal"), required=True)
    build.add_argument("--cleaning-file", type=Path)
    build.add_argument("--config-path", type=Path)
    build.add_argument("--mapping-path", type=Path)

    validate = subparsers.add_parser("validate", help="Validate a built dataset")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--allow-draft-uncleaned", action="store_true")

    inspect = subparsers.add_parser("inspect", help="Print manifest and validation summary")
    inspect.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def _print_json(value) -> None:
    print(canonical_json_bytes(value).decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            dataset_dir = build_dataset(
                args.root,
                mode=args.mode,
                cleaning_file=args.cleaning_file,
                config_path=args.config_path,
                mapping_path=args.mapping_path,
            )
            manifest = read_json(dataset_dir / "dataset_manifest.json")
            _print_json(
                {
                    "status": "PASS",
                    "tier": manifest["tier"],
                    "dataset_dir": str(dataset_dir),
                    "output_counts": manifest["output_counts"],
                }
            )
        elif args.command == "validate":
            _print_json(
                validate_dataset(
                    args.dataset_dir,
                    allow_draft=args.allow_draft_uncleaned,
                )
            )
        else:
            _print_json(inspect_dataset(args.dataset_dir))
    except DataContractError as error:
        print(f"data contract error: {error}", file=sys.stderr)
        return 2
    return 0
