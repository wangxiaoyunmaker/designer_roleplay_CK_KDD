"""Deterministic, item-level V30 design-orientation rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from route_a.data.io import read_json, sha256_file


class ValueProfileError(ValueError):
    """Raised when a frozen V30 value-profile resource is invalid."""


@dataclass(frozen=True)
class ValueProfileLexicon:
    path: Path
    sha256: str
    items: tuple[dict[str, Any], ...]
    templates: dict[str, str]
    buckets: dict[int, tuple[str, str]]


_ROOT_FIELDS = {
    "schema_version", "lexicon_id", "status", "approved_on", "source_instrument",
    "interpretation", "score_buckets", "render_templates", "items",
}
_ITEM_FIELDS = {"item_id", "order", "dimension", "agree_pole", "disagree_pole"}
_COMPACT_ROOT_FIELDS = {"schema_version", "lexicon_id", "status", "source_long_lexicon", "items"}
_COMPACT_ITEM_FIELDS = {"item_id", "order", "agree_short", "disagree_short"}


def load_value_profile_lexicon(path: Path) -> ValueProfileLexicon:
    """Load a reviewed V30 lexicon without reading any participant responses."""
    resolved = Path(path).resolve()
    raw = read_json(resolved)
    if not isinstance(raw, dict) or set(raw) != _ROOT_FIELDS:
        raise ValueProfileError("value-profile lexicon fields do not match schema")
    if raw["schema_version"] != "1.0.0" or raw["status"] != "approved":
        raise ValueProfileError("value-profile lexicon must be approved schema 1.0.0")
    source = raw["source_instrument"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueProfileError("value-profile source instrument is invalid")
    source_sha = source["sha256"]
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueProfileError("value-profile source SHA is invalid")
    templates = raw["render_templates"]
    if not isinstance(templates, dict) or set(templates) != {"strong", "moderate"}:
        raise ValueProfileError("value-profile templates are invalid")
    if any(not isinstance(value, str) or "{pole}" not in value for value in templates.values()):
        raise ValueProfileError("value-profile templates must contain {pole}")
    buckets: dict[int, tuple[str, str]] = {}
    if not isinstance(raw["score_buckets"], list) or len(raw["score_buckets"]) != 4:
        raise ValueProfileError("value-profile score buckets are invalid")
    for bucket in raw["score_buckets"]:
        if not isinstance(bucket, dict) or set(bucket) != {"scores", "pole", "strength"}:
            raise ValueProfileError("value-profile score bucket fields are invalid")
        pole, strength = bucket["pole"], bucket["strength"]
        if pole not in {"agree_pole", "disagree_pole"} or strength not in templates:
            raise ValueProfileError("value-profile score bucket semantics are invalid")
        scores = bucket["scores"]
        if not isinstance(scores, list) or not scores:
            raise ValueProfileError("value-profile score bucket is empty")
        for score in scores:
            if type(score) is not int or score not in range(1, 7) or score in buckets:
                raise ValueProfileError("value-profile scores must cover 1..6 exactly once")
            buckets[score] = (pole, strength)
    if set(buckets) != set(range(1, 7)):
        raise ValueProfileError("value-profile score buckets must cover 1..6")
    items = raw["items"]
    if not isinstance(items, list) or len(items) != 30:
        raise ValueProfileError("value-profile lexicon must contain 30 items")
    expected_ids = [f"V{index:02d}" for index in range(1, 31)]
    if [row.get("item_id") for row in items] != expected_ids:
        raise ValueProfileError("value-profile item IDs must be V01 through V30")
    if [row.get("order") for row in items] != list(range(1, 31)):
        raise ValueProfileError("value-profile item order must be 1 through 30")
    if any(not isinstance(row, dict) or set(row) != _ITEM_FIELDS for row in items):
        raise ValueProfileError("value-profile item fields are invalid")
    if any(
        not isinstance(row[field], str) or not row[field].strip()
        for row in items for field in ("dimension", "agree_pole", "disagree_pole")
    ):
        raise ValueProfileError("value-profile item text must be nonempty")
    if len({row["dimension"] for row in items}) != 30:
        raise ValueProfileError("value-profile dimensions must be unique")
    return ValueProfileLexicon(
        path=resolved,
        sha256=sha256_file(resolved),
        items=tuple(items), templates=dict(templates), buckets=buckets,
    )


def render_value_profile(scores: Iterable[int], lexicon: ValueProfileLexicon) -> str:
    """Render all 30 V30 items in frozen order, using no model or private text."""
    values = tuple(scores)
    if len(values) != 30 or any(type(score) is not int or score not in range(1, 7) for score in values):
        raise ValueProfileError("V30 profile rendering requires 30 integer scores in [1,6]")
    lines = []
    for score, item in zip(values, lexicon.items):
        pole_name, strength = lexicon.buckets[score]
        pole = item[pole_name]
        lines.append(f"- {item['dimension']}：{lexicon.templates[strength].format(pole=pole)}")
    return "\n".join(lines)


def load_compact_value_profile_lexicon(
    path: Path, *, allow_proposed: bool = False,
) -> dict[str, Any]:
    """Load the no-strength short-form V30 vocabulary.

    It is intentionally a separate resource: the long, approved lexicon remains
    the source of truth for score direction and for active-cue strength wording.
    """
    raw = read_json(Path(path).resolve())
    if not isinstance(raw, dict) or set(raw) != _COMPACT_ROOT_FIELDS:
        raise ValueProfileError("compact value-profile lexicon fields do not match schema")
    if raw["schema_version"] != "1.0.0" or raw["status"] not in {"approved", "proposed"}:
        raise ValueProfileError("compact value-profile lexicon has invalid status")
    if raw["status"] != "approved" and not allow_proposed:
        raise ValueProfileError("compact value-profile lexicon is not approved")
    source = raw["source_long_lexicon"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueProfileError("compact value-profile source linkage is invalid")
    items = raw["items"]
    expected_ids = [f"V{index:02d}" for index in range(1, 31)]
    if (
        not isinstance(items, list)
        or len(items) != 30
        or [item.get("item_id") for item in items] != expected_ids
        or [item.get("order") for item in items] != list(range(1, 31))
        or any(not isinstance(item, dict) or set(item) != _COMPACT_ITEM_FIELDS for item in items)
        or any(not isinstance(item[field], str) or not item[field].strip() for item in items for field in ("agree_short", "disagree_short"))
    ):
        raise ValueProfileError("compact value-profile items are invalid")
    return raw


def render_compact_value_profile(
    scores: Iterable[int], *, long_lexicon: ValueProfileLexicon, compact_lexicon: dict[str, Any],
) -> str:
    """Render all 30 values as short direction phrases without strength labels."""
    values = tuple(scores)
    if len(values) != 30 or any(type(score) is not int or score not in range(1, 7) for score in values):
        raise ValueProfileError("compact V30 profile rendering requires 30 integer scores in [1,6]")
    source = compact_lexicon["source_long_lexicon"]
    if source["sha256"] != long_lexicon.sha256:
        raise ValueProfileError("compact and long V30 lexicons do not share the frozen source")
    lines = []
    for score, item in zip(values, compact_lexicon["items"]):
        pole_name, _ = long_lexicon.buckets[score]
        short_key = "agree_short" if pole_name == "agree_pole" else "disagree_short"
        lines.append(f"- {item[short_key]}")
    return "\n".join(lines)
