"""Canonical access to the frozen C-K annotation ontology."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from route_a.data.io import read_json


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "state_ontology_v2.json"


def load_kc_state_contract(path: Path | None = None) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    """Load the six planner-eligible states and their shared prompt contract."""
    raw = read_json(path or default_ontology_path())
    order = tuple(raw.get("kc_state_order", ()))
    rows = raw.get("states")
    if len(order) != 6 or len(set(order)) != 6 or "V" in order or not isinstance(rows, list):
        raise ValueError("ontology must define six unique non-V KC states")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    required = {"id", "display_symbol", "display_name", "definition", "core_criterion", "prompt_action"}
    result: dict[str, dict[str, str]] = {}
    for state in order:
        row = by_id.get(state)
        if not isinstance(row, dict) or row.get("planner_eligible") is not True or required - set(row):
            raise ValueError(f"ontology lacks a complete planner state contract for {state}")
        result[state] = {key: str(row[key]) for key in required}
    return order, result
