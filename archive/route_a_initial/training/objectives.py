"""Framework-neutral Route A objective helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping


def auxiliary_class_weights(
    counts: Mapping[str, int],
    *,
    minimum: float = 0.5,
    maximum: float = 2.0,
) -> dict[str, float]:
    if not counts or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid class-weight inputs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("class counts must be positive integers")
    total = sum(counts.values())
    classes = len(counts)
    return {
        label: min(maximum, max(minimum, math.sqrt(total / (classes * count))))
        for label, count in sorted(counts.items())
    }


def stable_candidate_seed(run_seed: int, candidate_kind: str, candidate_index: int) -> int:
    """Derive a candidate seed without participant identity or iteration order."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError("run_seed must be an integer")
    if not isinstance(candidate_kind, str) or not candidate_kind:
        raise TypeError("candidate_kind must be a nonempty string")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise TypeError("candidate_index must be a nonnegative integer")
    payload = f"{run_seed}|{candidate_kind}|{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
