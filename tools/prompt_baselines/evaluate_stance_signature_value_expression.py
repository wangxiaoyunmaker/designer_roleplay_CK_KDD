"""Audit value expression in a stance/signature pilot with a frozen NLI probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

from tools.prompt_baselines.run_stance_signature_pilot import paired_bootstrap, read_jsonl


ITEM_IDS = tuple(f"V{index:02d}" for index in range(1, 31))


def binary_profile(values: object) -> dict[str, str]:
    if not isinstance(values, list) or len(values) != 30:
        raise ValueError("V30 profile must contain 30 values")
    if any(type(value) is not int or value not in range(1, 7) for value in values):
        raise ValueError("V30 values must be integers in [1, 6]")
    return {
        item_id: "disagree" if value <= 3 else "agree"
        for item_id, value in zip(ITEM_IDS, values)
    }


def expression_metrics(
    expression: Mapping[str, str],
    *,
    participant_id: str,
    profiles: Mapping[str, Mapping[str, str]],
) -> dict[str, float | None]:
    if participant_id not in profiles:
        raise ValueError("expression participant is absent from profiles")
    if set(expression) != set(ITEM_IDS):
        raise ValueError("expression must cover all V30 items")
    claims = {
        item_id: stance for item_id, stance in expression.items()
        if stance != "not_expressed"
    }
    coverage = len(claims) / len(ITEM_IDS)
    if not claims:
        return {"coverage": coverage, "profile_precision": None, "own_profile_advantage": None}

    def alignment(profile: Mapping[str, str]) -> float:
        return sum(profile[item_id] == stance for item_id, stance in claims.items()) / len(claims)

    own = alignment(profiles[participant_id])
    alternatives = [alignment(profile) for other_id, profile in profiles.items() if other_id != participant_id]
    if not alternatives:
        raise ValueError("own-profile advantage requires at least two profiles")
    return {
        "coverage": coverage,
        "profile_precision": own,
        "own_profile_advantage": own - fmean(alternatives),
    }


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]], *, n_resamples: int, seed: int,
    reference_condition: str = "base",
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        condition = str(row["condition"])
        participant = str(row["participant_id"])
        if participant in by_condition.setdefault(condition, {}):
            raise ValueError(f"duplicate metric row: {condition}/{participant}")
        by_condition[condition][participant] = row

    summaries: dict[str, Any] = {}
    for condition, participant_rows in sorted(by_condition.items()):
        values = list(participant_rows.values())
        summaries[condition] = {"n": len(values)}
        for metric in ("coverage", "profile_precision", "own_profile_advantage"):
            observed = [float(row[metric]) for row in values if row[metric] is not None]
            summaries[condition][f"mean_{metric}"] = fmean(observed) if observed else None
            summaries[condition][f"{metric}_n"] = len(observed)

    comparisons: dict[str, Any] = {}
    if reference_condition in by_condition:
        for condition in sorted(by_condition):
            if condition == reference_condition:
                continue
            for metric in ("coverage", "profile_precision", "own_profile_advantage"):
                common = sorted(
                    participant for participant in set(by_condition[condition]) & set(by_condition[reference_condition])
                    if by_condition[condition][participant][metric] is not None
                    and by_condition[reference_condition][participant][metric] is not None
                )
                if not common:
                    comparisons[f"{condition}-{reference_condition}/{metric}"] = None
                    continue
                left = {participant: float(by_condition[condition][participant][metric]) for participant in common}
                right = {participant: float(by_condition[reference_condition][participant][metric]) for participant in common}
                comparisons[f"{condition}-{reference_condition}/{metric}"] = paired_bootstrap(
                    left, right, n_resamples=n_resamples, seed=seed
                )
    return {"conditions": summaries, "paired_comparisons": comparisons}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--realizer-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", default="base,active4,full")
    parser.add_argument("--reference-condition", default="base")
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("output directory must not already exist")
    if not 0.5 <= args.theta <= 0.999:
        raise ValueError("theta must lie in [0.5, 0.999]")

    participant_rows = read_jsonl(args.participants)
    profiles = {row["participant_id"]: binary_profile(row["v30"]) for row in participant_rows}
    lexicon_payload = json.loads(args.lexicon.read_text(encoding="utf-8"))
    lexicon = lexicon_payload.get("items")
    if not isinstance(lexicon, list) or [item.get("item_id") for item in lexicon] != list(ITEM_IDS):
        raise ValueError("lexicon must contain V01 through V30")

    allowed_conditions = {item.strip() for item in args.conditions.split(",") if item.strip()}
    pilot_rows = [row for row in read_jsonl(args.pilot) if row.get("condition") in allowed_conditions]
    if not pilot_rows:
        raise ValueError("pilot contains no requested conditions")

    sys.path.insert(0, str(args.realizer_root.resolve()))
    module = importlib.import_module("route_a.evaluation.constrained_choice_probe")
    probe = module.ConstrainedChoiceProbe(
        model_path=str(args.model), device=args.device, use_prefix_cache=True
    )
    probe.theta = args.theta

    expression_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    total = len(pilot_rows)
    for index, row in enumerate(pilot_rows, 1):
        participant = str(row["pid"])
        condition = str(row["condition"])
        text = str(row["nl_answer"])
        expression = probe.score_expression_twopole(text, lexicon)
        metrics = expression_metrics(expression, participant_id=participant, profiles=profiles)
        expression_rows.append({
            "schema_version": "stance_signature_value_expression_v1",
            "evaluation_tier": "POST_HOC_DIAGNOSTIC",
            "condition": condition,
            "participant_id": participant,
            "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "items": [{"item_id": item_id, "stance": expression[item_id]} for item_id in ITEM_IDS],
            "source_text_retained": False,
        })
        metric_rows.append({"condition": condition, "participant_id": participant, **metrics})
        if index % 10 == 0 or index == total:
            print(f"[value-expression] {index}/{total}", flush=True)

    summary = summarize_metric_rows(
        metric_rows,
        n_resamples=args.bootstrap,
        seed=args.seed,
        reference_condition=args.reference_condition,
    )
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "expressions.jsonl", expression_rows)
    write_jsonl(args.output_dir / "metrics.jsonl", metric_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", {
        "schema_version": "stance_signature_value_expression_manifest_v1",
        "evaluation_tier": "POST_HOC_DIAGNOSTIC",
        "q7_human_text_used": False,
        "training_or_selection_allowed": False,
        "pilot_sha256": sha256_file(args.pilot),
        "participants_sha256": sha256_file(args.participants),
        "lexicon_sha256": sha256_file(args.lexicon),
        "probe_model": str(args.model),
        "probe_protocol": module.PROBE_PROTOCOL_TWOPOLE,
        "theta": args.theta,
        "conditions": sorted(allowed_conditions),
        "reference_condition": args.reference_condition,
        "records": len(expression_rows),
        "source_text_retained": False,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
