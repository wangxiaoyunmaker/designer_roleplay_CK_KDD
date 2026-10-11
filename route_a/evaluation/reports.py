"""Deterministic JSON/CSV report writers and result checksums."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .io import sha256_file


def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    Path(path).write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_checksums(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")
