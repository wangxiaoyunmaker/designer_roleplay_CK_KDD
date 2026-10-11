#!/usr/bin/env python3
"""Generate deterministic, public-safe reproducibility audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_ROOT = REPOSITORY_ROOT / "release" / "public_repo"
DEFAULT_PRIVATE_ROOT = Path(
    "/Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案"
)

# Transcribed from docs/notes/PKVC_30节点本体.md, the locked 30-axis ontology.
TRAIT_POLARITY = {
    "数据 vs 直觉": {"positive": "重直觉立场", "negative": "重数据立场"},
    "天赋信仰": {"positive": "天赋决定论", "negative": "后天努力论"},
    "AI 替代论": {"positive": "AI 替代主义", "negative": "岗位保护主义"},
    "技术支持": {"positive": "技术激进主义", "negative": "技术审慎主义"},
    "复杂工具信仰": {"positive": "工具门槛论", "negative": "工具无关论"},
    "混沌与灵感": {"positive": "包容失控派", "negative": "验证至上派"},
    "黑盒透明度": {"positive": "结果至上派", "negative": "过程透明派"},
    "系统微模块": {"positive": "解耦优先", "negative": "整体感优先"},
    "完美主义/MVP": {"positive": "完美主义", "negative": "速度优先"},
    "创新代价": {"positive": "创新激进派", "negative": "惯例守护派"},
    "审美反民粹": {"positive": "审美精英主义", "negative": "数据民粹妥协"},
    "情感附加值": {"positive": "仪式感优先", "negative": "效率优先"},
    "装饰合法性": {"positive": "装饰合法派", "negative": "形式追随功能"},
    "时代 vs 永恒": {"positive": "时代潮流派", "negative": "永恒经典派"},
    "本土文化妥协": {"positive": "本土深耕派", "negative": "全球通用派"},
    "仿生与有机": {"positive": "有机自然派", "negative": "几何工业派"},
    "独裁 vs 委员会": {"positive": "独裁愿景派", "negative": "民主共创派"},
    "秩序与防错": {"positive": "防错约束派", "negative": "探索开放派"},
    "创作者署名": {"positive": "作者性优先", "negative": "商业去人格化"},
    "资产开源": {"positive": "开源道德派", "negative": "竞争壁垒派"},
    "封闭生态竞争": {"positive": "封闭生态派", "negative": "开放兼容派"},
    "风险规避": {"positive": "风险规避", "negative": "风险偏好"},
    "团队依恋": {"positive": "独立打磨派", "negative": "团队共创派"},
    "商业操纵伦理": {"positive": "商业操纵容忍", "negative": "用户伦理底线"},
    "数据隐私交换": {"positive": "体验换隐私", "negative": "隐私保护派"},
    "弱势群体优先": {"positive": "边缘群体优先", "negative": "多数效率优先"},
    "股东服务优先": {"positive": "股东忠诚派", "negative": "用户忠诚派"},
    "政治中立性": {"positive": "政治中立派", "negative": "政治表达派"},
    "系统责任归属": {"positive": "系统归责派", "negative": "用户归责派"},
    "救世主情结": {"positive": "救世主情结", "negative": "工具中立派"},
}

PREDICATE_PRIORITY = {
    "strongly_forces_transition_to": 0,
    "guides_transition_to": 1,
    "strictly_inhibits": 2,
    "weakly_shifts_to": 3,
}


PROMPT_MANIFEST_ROWS = (
    ("B0", "baseline", "docs/briefs/paper_appendix_prompts.md", "public", "Zero-shot module ladder baseline."),
    ("B1", "baseline", "docs/briefs/paper_appendix_prompts.md", "public", "Module ladder: add pi."),
    ("B2", "baseline", "docs/briefs/paper_appendix_prompts.md", "public", "Module ladder: add stance."),
    ("B3", "baseline", "docs/briefs/paper_appendix_prompts.md", "public", "Module ladder: add self behavior."),
    ("B4", "baseline", "docs/briefs/paper_appendix_prompts.md", "public", "v2_anchor full configuration with CAPS routing."),
    ("v2_no_caps", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Remove the CAPS routing block."),
    ("v2_no_seg3_rule", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Remove the explicit seg3 judgment rule."),
    ("v2_no_beta_summary", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Remove the CAPS beta-strength summary."),
    ("v2_no_pkvc_def", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Remove the PKVC framework primer."),
    ("v2_plus_peer", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Add same-segment peer retrieval."),
    ("v2_hybrid", "ablation", "docs/briefs/paper_appendix_prompts.md", "public", "Use v2_clean for seg3 and v2_anchor elsewhere."),
    ("extractor_v7g_ablation", "ablation", "sft_pipeline/config/prompts/extractor_v7g_ablation.md", "public", "Extractor ablation prompt."),
    ("extractor_v8_nomarker", "ablation", "sft_pipeline/config/prompts/extractor_v8_nomarker.md", "public", "Extractor variant without marker tables."),
    ("extractor_v8_nosc", "ablation", "sft_pipeline/config/prompts/extractor_v8_nosc.md", "public", "Extractor variant without self-checks."),
    ("extractor_v8.7", "extractor", "sft_pipeline/config/prompts/extractor_v8.7.md", "public", "Released v8.7 extraction prompt."),
    ("sft_pipeline_overview", "sft", "sft_pipeline/README.md", "public", "SFT pipeline overview and run interface."),
    ("sft_data_format", "sft", "docs/briefs/sft_data_format_v1.md", "public", "SFT training data and prompt contract."),
    ("sft_expand_q1q6_v3", "sft", "sft_pipeline/config/prompts/expand_q1q6_v3.md", "public", "Q1-Q6 explanation expansion prompt."),
)


def _data_root(root: Path) -> Path:
    return root / "data" if (root / "data").is_dir() else root


def build_trait_items_table() -> pd.DataFrame:
    """Return the locked 30-axis ontology with its scoring polarity."""
    return pd.DataFrame(
        [
            {
                "trait_order": index,
                "trait_id": f"T{index:02d}",
                "trait_name": trait_name,
                "positive_label": labels["positive"],
                "negative_label": labels["negative"],
                "scoring_note": (
                    "Ontology tiering: Likert 1-2 = negative pole, 3-4 = neutral/weak, "
                    "5-6 = positive pole. Runtime CAPS activation uses score >=4 for "
                    "positive-direction procedures and score <=3 for negative-direction procedures."
                ),
            }
            for index, (trait_name, labels) in enumerate(TRAIT_POLARITY.items(), start=1)
        ]
    )


def build_prompt_manifest_table() -> pd.DataFrame:
    """Return public release paths for prompts and their supporting documentation."""
    return pd.DataFrame(
        PROMPT_MANIFEST_ROWS,
        columns=("prompt_id", "category", "path", "status", "notes"),
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_scale_matrix(root: Path) -> pd.DataFrame:
    """Return 163 designer profiles, indexed by public user ID."""
    data_root = _data_root(root)
    responses = pd.read_csv(data_root / "scale_responses.csv", encoding="utf-8")
    trait_order = list(dict.fromkeys(responses["question_node"]))
    matrix = responses.pivot(index="user_id", columns="question_node", values="score")
    matrix = matrix.reindex(columns=trait_order).sort_index()

    q7_rows = _read_jsonl(data_root / "processed" / "q7_extracted_v8.7_full.jsonl")
    q7_users = {row["user_id"] for row in q7_rows}
    matrix["has_q7_gold"] = matrix.index.isin(q7_users)
    matrix.index.name = "user_id"
    return matrix


def build_caps_procedure_table(root: Path) -> pd.DataFrame:
    """Load the locked 36 CAPS quadruples in source order."""
    data_root = _data_root(root)
    with (data_root / "processed" / "legacy" / "caps_quadruples.json").open(
        encoding="utf-8"
    ) as source:
        procedures = json.load(source)
    return pd.DataFrame(procedures)


def _is_active(score: float, direction: str) -> bool:
    return (direction == "+" and score >= 4) or (direction == "-" and score <= 3)


def select_activated_rules(
    score_row: Mapping[str, float], procedures: Sequence[dict]
) -> list[dict]:
    """Choose active rules using the runtime PKVC selection procedure."""
    by_class: dict[str, list[dict]] = {anchor: [] for anchor in ("P", "K", "V", "C")}
    for procedure in procedures:
        score = float(score_row[procedure["S_item"]])
        if procedure["O"] in by_class and _is_active(score, procedure["S_direction"]):
            by_class[procedure["O"]].append(procedure)

    selected: list[dict] = []
    for anchor in ("P", "K", "V", "C"):
        candidates = by_class[anchor]
        if candidates:
            top = max(
                candidates,
                key=lambda item: (
                    abs(float(item["S_coef"])),
                    -PREDICATE_PRIORITY.get(item.get("P", ""), 9),
                ),
            )
            selected.append({**top, "_class_anchor": anchor})

    active_procedures = [procedure for candidates in by_class.values() for procedure in candidates]
    if active_procedures:
        global_top = max(active_procedures, key=lambda item: abs(float(item["S_coef"])))
        if not any(procedure["id"] == global_top["id"] for procedure in selected):
            selected.append({**global_top, "_class_anchor": "overall_top"})

    rows = []
    for rank, procedure in enumerate(selected, start=1):
        score = float(score_row[procedure["S_item"]])
        rows.append(
            {
                **procedure,
                "rank": rank,
                "class_anchor": procedure["_class_anchor"],
                "trait_score": score,
                "active": True,
            }
        )
    return rows


def build_gold_audit(public_root: Path, private_root: Path) -> dict:
    """Create a summary of private expert gold without retaining source IDs or text."""
    users = pd.read_csv(_data_root(public_root) / "users.csv", encoding="utf-8")
    parsed_path = private_root / "data" / "processed" / "expert_gold" / "parsed.json"
    evaluation_path = private_root / "reports" / "datav3" / "eval_vs_gold_v8.6.json"
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    id_map = {
        row["user_id"]: f"G{index:03d}"
        for index, row in enumerate(parsed["users"], start=1)
    }
    gold_users = []
    for user in parsed["users"]:
        segments = {
            segment: {
                "dominant": user[segment]["dominant"],
                "sequence": user[segment]["sequence"],
            }
            for segment in ("seg1", "seg2", "seg3")
            if segment in user
        }
        gold_users.append({"user_id": id_map[user["user_id"]], "segments": segments})

    result_fields = (
        "seg",
        "gold_dominant",
        "gold_sequence",
        "pred_dominant",
        "pred_sequence",
        "dom_match",
        "seq_edit",
        "seq_jaccard",
    )
    results = [
        {"user_id": id_map[row["user_id"]], **{key: row[key] for key in result_fields}}
        for row in evaluation["results"]
    ]
    return {
        "schema_version": 1,
        "public_designer_count": int(users["user_id"].nunique()),
        "expert_gold": gold_users,
        "evaluation": {
            "version": evaluation["version"],
            "n": evaluation["n"],
            "overall_dom_acc": evaluation["overall_dom_acc"],
            "overall_avg_edit": evaluation["overall_avg_edit"],
            "overall_avg_jacc": evaluation["overall_avg_jacc"],
            "per_seg": evaluation["per_seg"],
            "confusion": evaluation["confusion"],
            "results": results,
        },
    }


def build_all(
    output_dir: Path,
    public_root: Path = DEFAULT_PUBLIC_ROOT,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
) -> dict[str, int]:
    """Write the reproducibility audit artifacts and return their deterministic counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_scale_matrix(public_root)
    procedures = build_caps_procedure_table(public_root)
    procedure_rows = procedures.to_dict(orient="records")

    matrix.reset_index().to_csv(
        output_dir / "trait_response_matrix_30d.csv", index=False, encoding="utf-8"
    )
    build_trait_items_table().to_csv(
        output_dir / "trait_items_30.csv", index=False, encoding="utf-8"
    )
    build_prompt_manifest_table().to_csv(
        output_dir / "prompt_manifest.csv", index=False, encoding="utf-8"
    )
    procedures.to_csv(output_dir / "procedures_36.csv", index=False, encoding="utf-8")

    activated_rows = []
    for user_id, scores in matrix.drop(columns="has_q7_gold").iterrows():
        for rule in select_activated_rules(scores.to_dict(), procedure_rows):
            activated_rows.append(
                {
                    "user_id": user_id,
                    "rank": rule["rank"],
                    "class_anchor": rule["class_anchor"],
                    "procedure_id": rule["id"],
                    "trait": rule["S_item"],
                    "trait_score": rule["trait_score"],
                    "polarity": rule["S_direction"],
                    "active": rule["active"],
                    "S_coef": rule["S_coef"],
                    "transition": rule["P"],
                    "O_subtype": rule["O_subtype"],
                }
            )
    pd.DataFrame(activated_rows).to_csv(
        output_dir / "activated_rules_by_designer.csv", index=False, encoding="utf-8"
    )

    audit = build_gold_audit(public_root, private_root)
    (output_dir / "gold_label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    q7_segments = sum(matrix["has_q7_gold"])
    return {
        "trait_matrix_rows": len(matrix),
        "trait_matrix_cols": len(matrix.columns),
        "q7_segments": len(
            _read_jsonl(_data_root(public_root) / "processed" / "q7_extracted_v8.7_full.jsonl")
        ),
        "caps_procedures": len(procedures),
        "activated_rules": len(activated_rows),
        "q7_designers": int(q7_segments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    args = parser.parse_args()
    print(json.dumps(build_all(output_dir=args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
