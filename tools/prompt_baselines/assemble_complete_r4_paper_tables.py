#!/usr/bin/env python3
"""Assemble the two complete baseline/full and GPT-4o ablation paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


METRIC_COLUMNS = (
    "bge_own_user_margin", "exact_match_rate", "mean_length_error", "mean_kc_nes",
    "mean_kc_own_user_auc", "mean_kc_own_user_margin",
    "pairwise_kc_diversity_retention", "trajectory_geometry_correlation",
    "unique_sequences", "top_share", "human_expression_f1",
    "human_expression_precision", "human_expression_recall",
    "profile_polarity_precision", "expression_coverage", "own_profile_advantage",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def value_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("value summary has no rows")
    output = {}
    for row in rows:
        if row.get("segment_id") != "seg2":
            continue
        system = str(row["system_id"])
        if system in output:
            raise ValueError(f"duplicate value row: {system}")
        output[system] = dict(row)
    return output


def combine(
    *, label: str, family: str, model: str, arm: str, sequence: Mapping[str, Any] | None,
    bge: Mapping[str, Any] | None, value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = {"system": label, "family": family, "model": model, "arm": arm}
    if sequence is None:
        row.update({"n_sequence": None, **{column: None for column in METRIC_COLUMNS}})
        row["status"] = "PENDING_GENERATION"
        return row
    row.update({
        "n_sequence": sequence.get("n"),
        "bge_own_user_margin": bge.get("mean_bge_own_user_margin") if bge else None,
        "exact_match_rate": sequence.get("exact_match_rate"),
        "mean_length_error": sequence.get("mean_length_error"),
        "mean_kc_nes": sequence.get("mean_kc_nes"),
        "mean_kc_own_user_auc": sequence.get("mean_kc_own_user_auc"),
        "mean_kc_own_user_margin": sequence.get("mean_kc_own_user_margin"),
        "pairwise_kc_diversity_retention": sequence.get("pairwise_kc_diversity_retention"),
        "trajectory_geometry_correlation": sequence.get("trajectory_geometry_correlation"),
        "unique_sequences": sequence.get("unique_sequences"),
        "top_share": sequence.get("top_share"),
    })
    for column in METRIC_COLUMNS[10:]:
        row[column] = value.get(column) if value else None
    row["n_value"] = value.get("instances") if value else None
    row["status"] = "COMPLETE" if bge is not None and value is not None else "INCOMPLETE_METRICS"
    return row


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def panel_a(rows: list[Mapping[str, Any]]) -> list[str]:
    output = [
        "| Model | Arm | BGE | EM | LenErr | KC-NES | KC AUC | KC margin | KCDR | TGC | Unique | Top share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            f"| {row['model']} | {row['arm']} | {fmt(row['bge_own_user_margin'])} | "
            f"{fmt(row['exact_match_rate'])} | {fmt(row['mean_length_error'])} | "
            f"{fmt(row['mean_kc_nes'])} | {fmt(row['mean_kc_own_user_auc'])} | "
            f"{fmt(row['mean_kc_own_user_margin'])} | "
            f"{fmt(row['pairwise_kc_diversity_retention'])} | "
            f"{fmt(row['trajectory_geometry_correlation'])} | "
            f"{fmt(row['unique_sequences'], 0)} | {fmt(row['top_share'])} |"
        )
    return output


def panel_b(rows: list[Mapping[str, Any]]) -> list[str]:
    output = [
        "| Model | Arm | Human F1 | Precision | Recall | Profile polarity precision | Expression coverage | Own-profile advantage | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            f"| {row['model']} | {row['arm']} | {fmt(row['human_expression_f1'])} | "
            f"{fmt(row['human_expression_precision'])} | {fmt(row['human_expression_recall'])} | "
            f"{fmt(row['profile_polarity_precision'])} | {fmt(row['expression_coverage'])} | "
            f"{fmt(row['own_profile_advantage'])} | {fmt(row.get('n_value'), 0)} |"
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sequence", type=Path, required=True)
    parser.add_argument("--baseline-bge", type=Path, required=True)
    parser.add_argument("--r4", type=Path, required=True)
    parser.add_argument("--baseline-value", type=Path, required=True)
    parser.add_argument("--r4-value", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_sequence = read_json(args.baseline_sequence)["systems"]
    baseline_bge = read_json(args.baseline_bge)["systems"]
    r4 = read_json(args.r4)["conditions"]
    baseline_value = value_map(read_json(args.baseline_value))
    r4_value = value_map(read_json(args.r4_value))

    comparison_rows = []
    baseline_models = (
        ("Qwen3-8B", "qwen3_8b"),
        ("Llama-3.1-8B", "llama_3_1_8b_instruct"),
        ("Gemini 3 Flash", "gemini_3_flash"),
        ("GPT-4o", "gpt4o"),
    )
    for model_label, model_id in baseline_models:
        for arm in ("sp", "pp", "icl"):
            system = f"{model_id}__{arm}"
            comparison_rows.append(combine(
                label=system, family="baseline", model=model_label, arm=arm.upper(),
                sequence=baseline_sequence[system], bge=baseline_bge[system],
                value=baseline_value[system],
            ))

    full_specs = (
        ("Qwen3-8B", "qwen3_8b__aligned_full", "r4_qwen3_8b_full__sp"),
        ("Llama-3.1-8B", "llama_3_1_8b_instruct__aligned_full", "r4_llama_3_1_8b_full__sp"),
        ("Gemini 3 Flash", "gemini_3_flash__aligned_full", "r4_gemini_3_flash_full__sp"),
        ("GPT-4o", "gpt4o__aligned_full", "r4_gpt4o_full__sp"),
        ("Claude Sonnet 4.6", "claude_sonnet_4_6__aligned_full", "r4_claude_sonnet_4_6_full__sp"),
        ("Qwen3-235B-A22B", "qwen3_235b_a22b_instruct_2507__aligned_full", "r4_qwen3_235b_full__sp"),
    )
    for model_label, condition, value_system in full_specs:
        comparison_rows.append(combine(
            label=condition, family="ours", model=model_label, arm="Ours",
            sequence=r4.get(condition), bge=r4.get(condition), value=r4_value.get(value_system),
        ))

    ablation_specs = (
        ("Full", "gpt4o__aligned_full", "r4_gpt4o_full__sp"),
        ("w/o Q2", "gpt4o__aligned_no_q2", "r4_gpt4o_no_q2__sp"),
        ("w/o Full30", "gpt4o__aligned_no_full30", "r4_gpt4o_no_full30__sp"),
        ("w/o active4", "gpt4o__aligned_no_active4", "r4_gpt4o_no_active4__sp"),
        ("Shuffled V30", "gpt4o__aligned_shuffled", "r4_gpt4o_shuffled__sp"),
    )
    ablation_rows = [combine(
        label=condition, family="ablation", model="GPT-4o", arm=label,
        sequence=r4[condition], bge=r4[condition], value=r4_value.get(value_system),
    ) for label, condition, value_system in ablation_specs]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ("system", "family", "model", "arm", "status", "n_sequence", "n_value", *METRIC_COLUMNS)
    for name, rows in (("table1_baselines_vs_ours", comparison_rows), ("table2_gpt4o_ablation", ablation_rows)):
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    markdown = [
        "# Complete R4 Paper Tables", "",
        "All automatic metrics use Seg2 and the fixed D011--D162 cohort (n=152).",
        "Human F1/Precision/Recall denote automatic human-expression alignment from the frozen local two-pole probe, not manual ratings.",
        "", "## Table 1. Ordinary prompt baselines versus the full method", "",
        "### Panel A. Text and trajectory metrics", "", *panel_a(comparison_rows), "",
        "### Panel B. Value-expression metrics", "", *panel_b(comparison_rows), "",
        "## Table 2. GPT-4o ablation", "", "### Panel A. Text and trajectory metrics", "",
        *panel_a(ablation_rows), "", "### Panel B. Value-expression metrics", "",
        *panel_b(ablation_rows), "",
    ]
    (args.output_dir / "paper_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "r4_complete_paper_tables_v1",
        "sources": {key: str(value) for key, value in vars(args).items() if key != "output_dir"},
        "rows": {"table1": len(comparison_rows), "table2": len(ablation_rows)},
        "pending": [row["system"] for row in comparison_rows + ablation_rows if row["status"] != "COMPLETE"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
