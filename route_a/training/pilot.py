"""Frozen STOP-strength pilot selection for the single development fold."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def choose_stop_fraction(
    results: Mapping[float, Mapping[str, Any]],
    *,
    natural_fraction: float = 0.1275,
    macro_f1_tolerance: float = 0.01,
) -> tuple[float, dict[str, Any]]:
    expected = {natural_fraction, 0.2, 0.25}
    if set(results) != expected:
        raise ValueError(f"STOP pilot must cover exactly {sorted(expected)}")
    baseline = results[natural_fraction]
    baseline_f1 = baseline.get("macro_f1_six_states")
    if not isinstance(baseline_f1, (int, float)) or isinstance(baseline_f1, bool):
        raise ValueError("natural STOP pilot is missing macro_f1_six_states")
    checks = {}
    eligible = []
    for fraction, metrics in sorted(results.items()):
        required = (
            metrics.get("rollout_gate_passed") is True
            and isinstance(metrics.get("macro_f1_six_states"), (int, float))
            and not isinstance(metrics.get("macro_f1_six_states"), bool)
            and float(metrics["macro_f1_six_states"]) >= float(baseline_f1) - macro_f1_tolerance
        )
        checks[str(fraction)] = {
            "rollout_gate_passed": metrics.get("rollout_gate_passed") is True,
            "state_macro_f1_non_degraded": (
                isinstance(metrics.get("macro_f1_six_states"), (int, float))
                and float(metrics["macro_f1_six_states"]) >= float(baseline_f1) - macro_f1_tolerance
            ),
        }
        if required:
            eligible.append(fraction)
    if not eligible:
        raise ValueError("no STOP pilot setting passes rollout and state-F1 gates")
    selected = min(eligible)
    return selected, {
        "selected_stop_batch_fraction": selected,
        "selection_rule": "smallest_fraction_passing_rollout_and_macro_f1_non_degradation",
        "natural_fraction": natural_fraction,
        "macro_f1_tolerance": macro_f1_tolerance,
        "checks": checks,
        "q7_gold_used": False,
    }
