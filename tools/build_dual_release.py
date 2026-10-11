#!/usr/bin/env python3
"""Build public GitHub artifact plus restricted data pack for RolePlay Designer."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

DEFAULT_SOURCE = Path("/Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案")
DEFAULT_OUT = Path("release")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".yaml", ".yml", ".toml"}
ID_COLUMNS = {"user_id", "wechat_id", "designer_id", "﻿user_id", "﻿wechat_id"}
DROP_PUBLIC_COLUMNS = {
    "duration_seconds",
    "device_type",
    "submitted_at",
    "created_at",
    "scale_duration_seconds",
    "scale_completed_at",
    "task_duration_seconds",
    "task_completed_at",
    "design_field_other",
    "education_background_other",
    "work_context_other",
    "service_target_other",
}
PUBLIC_PROCESSED_FILES = [
    "user_factor_scores.csv",
    "q1q6_option_labels.jsonl",
    "q2_training_labels.jsonl",
    "q1q6_extracted_v8.7_full.jsonl",
    "q7_extracted_v8.7_full.jsonl",
    "quad_extracted_163_fixed.jsonl",
    "q1q6_expansion_outputs.jsonl",
    "sft_train.jsonl",
    "sft_val.jsonl",
    "typology_soft_assignment_per_qid_v3.json",
]
LEGACY_PROCESSED_FILES = [
    "caps_quadruples.json",
    "quad_templates_v1_robust.jsonl",
    "q1q6_option_inventory.json",
    "q1q6_option_inventory.md",
]
PUBLIC_REPRODUCIBILITY_FILES = [
    Path("docs/reproducibility/PIPELINE.md"),
    Path("docs/reproducibility/ARTIFACT_INVENTORY.md"),
    Path("docs/reproducibility/METRICS_AUDIT.md"),
    Path("artifacts/reproducibility/trait_items_30.csv"),
    Path("artifacts/reproducibility/procedures_36.csv"),
    Path("artifacts/reproducibility/activated_rules_by_designer.csv"),
    Path("artifacts/reproducibility/prompt_manifest.csv"),
]
PUBLIC_SMOKE_OVERLAY = REPOSITORY_ROOT / "tools" / "public_release_overlay" / "smoke"
PRIVATE_EXPERT_GOLD_FILES = [
    "expert_review_quad_candidates_v1.csv",
    "hand_label_candidates.jsonl",
    "hand_label_candidates.md",
    "hand_labels.json",
]
EXCLUDED_NAMES = {
    ".DS_Store",
    ".env",
    ".env.local",
    ".pytest_cache",
    ".claude",
    ".venv",
    "__pycache__",
    "cache",
    "reports",
    "remote_backup_20260522",
    "anonymize_pii.py",
    "extract_caps_quadruples.py",
}
EXCLUDED_SUBSTRINGS = {
    "embedding_cache",
    ".bak",
}
SECRET_PATTERNS = {
    "openai-key": re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    "anthropic-key": re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    "google-key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "phone-like": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "wxid": re.compile(r"wxid_[A-Za-z0-9_]+"),
}


def normalize_key(key: str) -> str:
    return key.lstrip("\ufeff")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{normalize_key(k): v for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def anonymize_user_ids(
    rows: list[dict[str, str]],
    id_columns: Iterable[str],
    mapping: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    mapping = dict(mapping or {})
    normalized_columns = {normalize_key(col) for col in id_columns}
    for row in rows:
        for key, value in row.items():
            if normalize_key(key) in normalized_columns and value and value not in mapping:
                mapping[value] = f"D{len(mapping) + 1:03d}"

    out: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        for key, value in row.items():
            if normalize_key(key) in normalized_columns and value in mapping:
                new_row[key] = mapping[value]
        out.append(new_row)
    return out, mapping


def replace_ids_in_text(text: str, mapping: dict[str, str]) -> str:
    for raw_id, anon_id in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if raw_id:
            text = text.replace(raw_id, anon_id)
    return text


def replace_ids_in_obj(obj, mapping: dict[str, str]):
    if isinstance(obj, str):
        return mapping.get(obj, replace_ids_in_text(obj, mapping))
    if isinstance(obj, list):
        return [replace_ids_in_obj(item, mapping) for item in obj]
    if isinstance(obj, dict):
        return {
            replace_ids_in_text(str(key), mapping): replace_ids_in_obj(value, mapping)
            for key, value in obj.items()
        }
    return obj


def filter_public_columns(row: dict[str, str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for raw_key, value in row.items():
        key = normalize_key(raw_key)
        if key in DROP_PUBLIC_COLUMNS:
            continue
        if key == "wechat_id":
            filtered["user_id"] = value
            continue
        if key.endswith("_other"):
            continue
        filtered[key] = value
    return filtered


def should_skip_path(path: Path) -> bool:
    if path.name.startswith("._"):
        return True
    parts = set(path.parts)
    if parts & EXCLUDED_NAMES:
        return True
    return any(token in path.name for token in EXCLUDED_SUBSTRINGS)


def copy_tree_filtered(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            candidate = Path(directory) / name
            if should_skip_path(candidate):
                ignored.add(name)
        return ignored
    shutil.copytree(src, dst, ignore=ignore)


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_mapping_from_users(source: Path) -> dict[str, str]:
    users = read_csv_rows(source / "data_v3" / "users.csv")
    mapping: dict[str, str] = {}
    for row in users:
        raw_id = row.get("wechat_id") or row.get("user_id")
        if raw_id and raw_id not in mapping:
            mapping[raw_id] = f"D{len(mapping) + 1:03d}"
    return mapping


def write_public_csv(source_path: Path, dest_path: Path, mapping: dict[str, str]) -> None:
    rows = read_csv_rows(source_path)
    rows, _ = anonymize_user_ids(rows, ID_COLUMNS, mapping)
    public_rows = [filter_public_columns(row) for row in rows]
    public_rows = [replace_ids_in_obj(row, mapping) for row in public_rows]
    write_csv_rows(dest_path, public_rows)


def write_public_jsonl(source_path: Path, dest_path: Path, mapping: dict[str, str]) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as fin, dest_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = replace_ids_in_obj(json.loads(line), mapping)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_public_json(source_path: Path, dest_path: Path, mapping: dict[str, str]) -> None:
    obj = json.loads(source_path.read_text(encoding="utf-8"))
    obj = replace_ids_in_obj(obj, mapping)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_public_data(source: Path, public_dir: Path, mapping: dict[str, str]) -> None:
    data_src = source / "data_v3"
    data_dst = public_dir / "data"
    for name in ["users.csv", "scale_responses.csv", "task_responses.csv", "task_responses_expanded.csv", "user_cluster_efa.csv"]:
        src = data_src / name
        if src.exists():
            write_public_csv(src, data_dst / name, mapping)

    processed_src = data_src / "processed"
    processed_dst = data_dst / "processed"
    for name in PUBLIC_PROCESSED_FILES:
        src = processed_src / name
        if not src.exists():
            continue
        if src.suffix == ".csv":
            write_public_csv(src, processed_dst / name, mapping)
        elif src.suffix == ".jsonl":
            write_public_jsonl(src, processed_dst / name, mapping)
        elif src.suffix == ".json":
            write_public_json(src, processed_dst / name, mapping)
        else:
            shutil.copy2(src, processed_dst / name)

    legacy_src = source / "data" / "processed"
    for name in LEGACY_PROCESSED_FILES:
        src = legacy_src / name
        if not src.exists():
            continue
        if src.suffix == ".csv":
            write_public_csv(src, processed_dst / "legacy" / name, mapping)
        elif src.suffix == ".jsonl":
            write_public_jsonl(src, processed_dst / "legacy" / name, mapping)
        elif src.suffix == ".json":
            write_public_json(src, processed_dst / "legacy" / name, mapping)
        else:
            shutil.copy2(src, processed_dst / "legacy" / name)


def write_mapping_csv(path: Path, mapping: dict[str, str]) -> None:
    rows = [{"public_user_id": anon, "source_user_id": raw} for raw, anon in mapping.items()]
    write_csv_rows(path, rows)


def public_readme() -> str:
    return """
    # Designer PKVC Role-Play

    This is the public, GitHub-ready artifact for the Designer PKVC role-play project.

    It contains:

    - `sft_pipeline/`: data preparation, PKVC extraction, CAPS routing, evaluation, and SFT utilities.
    - `data/`: anonymized 163-designer dataset files and lightweight processed artifacts.
    - `docs/`: method notes, prompt references, and data format descriptions.
    - `scripts/`: reproducibility and analysis entry points from the research codebase.
    - `smoke/`: isolated Qwen3-8B BF16 LoRA delivery smoke test.

    It intentionally excludes:

    - raw participant identifiers,
    - timestamp/device fields,
    - `.env` or API keys,
    - embedding caches,
    - virtual environments,
    - generated reports and model checkpoints.

    Start with `data/README.md` and `sft_pipeline/README.md`.
    """


def data_readme(mapping: dict[str, str]) -> str:
    return f"""
    # Dataset Card

    Public dataset size: {len(mapping)} anonymized designers.

    User IDs are deterministic release IDs (`D001`, `D002`, ...). The private mapping from
    release IDs to original participant IDs is not included in this public artifact.

    ## Included Files

    - `users.csv`: coarse participant metadata after removing raw IDs, timestamps, durations,
      devices, and free-text demographic "other" fields.
    - `scale_responses.csv`: 30-item trait scale responses with anonymized `user_id`.
    - `task_responses.csv`: raw task answers with anonymized `user_id` and public-safe columns.
    - `task_responses_expanded.csv`: expanded task answers where available.
    - `user_cluster_efa.csv`: cluster assignments.
    - `processed/`: lightweight extracted labels, SFT splits, factor scores, and CAPS artifacts.

    ## Privacy Notes

    This public release removes direct IDs and common re-identification fields. Before publishing,
    manually review free-form Chinese responses for names, institutions, companies, or other
    identifying details that automated pattern checks may not catch.
    """


def requirements_text() -> str:
    return """
    pandas>=2.0
    numpy>=1.24
    scikit-learn>=1.3
    scipy>=1.10
    pyyaml>=6.0
    openpyxl>=3.1
    jsonschema>=4.0
    openai>=1.0
    anthropic>=0.25
    torch>=2.1
    transformers>=4.51.0
    peft>=0.10
    accelerate>=1.0
    huggingface-hub>=0.30
    factor-analyzer>=0.5
    matplotlib>=3.7
    networkx>=3.0
    """


def gitignore_text() -> str:
    return """
    .env
    .env.local
    .DS_Store
    __pycache__/
    *.py[cod]
    .venv/
    venv/
    reports/
    preds/
    ckpt/
    *.pt
    *.bin
    *.safetensors
    *embedding_cache*
    uid_mapping.csv
    private_data_pack/
    """


def license_text() -> str:
    return """
    MIT License

    Copyright (c) 2026 RolePlay Designer authors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """


def copy_public_sources(source: Path, public_dir: Path) -> None:
    for name in ["sft_pipeline", "scripts"]:
        src = source / name
        if src.exists():
            copy_tree_filtered(src, public_dir / name)
    docs_dst = public_dir / "docs"
    for sub in ["briefs", "training_data_formats", "notes"]:
        src = source / "docs" / sub
        if src.exists():
            copy_tree_filtered(src, docs_dst / sub)


def copy_public_reproducibility(public_dir: Path) -> None:
    validate_public_reproducibility_files()
    for relative_path in PUBLIC_REPRODUCIBILITY_FILES:
        src = REPOSITORY_ROOT / relative_path
        dst = public_dir / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_public_smoke_overlay(public_dir: Path) -> None:
    if not PUBLIC_SMOKE_OVERLAY.is_dir():
        raise FileNotFoundError(f"public smoke overlay missing: {PUBLIC_SMOKE_OVERLAY}")
    copy_tree_filtered(PUBLIC_SMOKE_OVERLAY, public_dir / "smoke")


def validate_public_reproducibility_files() -> None:
    missing = [
        str(REPOSITORY_ROOT / relative_path)
        for relative_path in PUBLIC_REPRODUCIBILITY_FILES
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "required public reproducibility artifacts missing:\n" + "\n".join(missing)
        )


def sanitize_text_tree(path: Path, mapping: dict[str, str]) -> None:
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = replace_ids_in_text(text, mapping)
        for label, pattern in SECRET_PATTERNS.items():
            if label.endswith("-key"):
                sanitized = pattern.sub(f"{label.upper()}_REDACTED", sanitized)
        sanitized = sanitized.replace("sk-...", "OPENAI_KEY_PLACEHOLDER")
        sanitized = sanitized.replace("sk-xxx", "OPENAI_KEY_PLACEHOLDER")
        if sanitized != text:
            file.write_text(sanitized, encoding="utf-8")


def build_public_release(source: Path, out: Path) -> None:
    public_dir = out / "public_repo"
    validate_public_reproducibility_files()
    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir(parents=True)

    mapping = build_mapping_from_users(source)
    copy_public_sources(source, public_dir)
    copy_public_reproducibility(public_dir)
    copy_public_smoke_overlay(public_dir)
    write_public_data(source, public_dir, mapping)
    sanitize_text_tree(public_dir, mapping)
    write_text_file(public_dir / "README.md", public_readme())
    write_text_file(public_dir / "data" / "README.md", data_readme(mapping))
    write_text_file(public_dir / "requirements.txt", requirements_text())
    write_text_file(public_dir / ".gitignore", gitignore_text())
    write_text_file(public_dir / "LICENSE", license_text())

    leaks = scan_for_public_leaks(public_dir, forbidden_values=mapping.keys())
    if leaks:
        raise RuntimeError("Public release leak check failed:\n" + "\n".join(leaks[:50]))


def copy_private_file(src: Path, dst: Path) -> None:
    if not src.exists() or should_skip_path(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_private_pack(source: Path, out: Path) -> None:
    private_dir = out / "private_data_pack"
    if private_dir.exists():
        shutil.rmtree(private_dir)
    private_dir.mkdir(parents=True)

    mapping = build_mapping_from_users(source)
    write_mapping_csv(private_dir / "uid_mapping.csv", mapping)
    write_text_file(
        private_dir / "README_RESTRICTED.md",
        """
        # Restricted Data Pack

        This directory is not GitHub-ready. It contains raw participant data and the
        release anonymization mapping. Share only with authorized collaborators.

        Excluded even from this private pack: `.env`, API keys, virtual environments,
        embedding caches, generated reports, and model checkpoints.
        """,
    )

    for name in ["users.csv", "scale_responses.csv", "task_responses.csv", "task_responses_expanded.csv", "user_cluster_efa.csv"]:
        copy_private_file(source / "data_v3" / name, private_dir / "raw_data_v3" / name)
    for file in (source / "data_v3" / "processed").glob("*"):
        if file.is_file() and not should_skip_path(file):
            copy_private_file(file, private_dir / "processed_data_v3" / file.name)
    for name in ["users.csv", "scale_responses.csv", "task_responses.csv", "user_factor_scores.csv", "user_cluster_efa.csv"]:
        copy_private_file(source / "data" / name, private_dir / "legacy_data_79" / name)
    for name in PRIVATE_EXPERT_GOLD_FILES:
        copy_private_file(
            source / "data" / "processed" / name,
            private_dir / "expert_gold_raw" / name,
        )

    manifest = []
    for file in sorted(private_dir.rglob("*")):
        if file.is_file():
            manifest.append({"path": str(file.relative_to(private_dir)), "bytes": file.stat().st_size})
    (private_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scan_for_public_leaks(path: Path, forbidden_values: Iterable[str] | None = None) -> list[str]:
    leaks: list[str] = []
    forbidden = [value for value in (forbidden_values or []) if value]
    for file in sorted(path.rglob("*")):
        rel = file.relative_to(path)
        if should_skip_path(file):
            leaks.append(f"forbidden path: {rel}")
            continue
        if not file.is_file() or file.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                leaks.append(f"{label} pattern in {rel}")
        for value in forbidden:
            if value in text:
                leaks.append(f"source id value in {rel}: {value[:12]}")
                break
    return leaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", "--output", dest="out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="rebuild only the GitHub-safe public artifact; leave the private pack untouched",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    if args.check_only:
        mapping = build_mapping_from_users(source)
        public_dir = out / "public_repo"
        leaks = scan_for_public_leaks(public_dir, forbidden_values=mapping.keys())
        if leaks:
            print("Public release leak check failed:")
            for leak in leaks[:100]:
                print(f"- {leak}")
            return 1
        print("Public release leak check passed")
        return 0
    build_public_release(source, out)
    if not args.public_only:
        build_private_pack(source, out)
    print(f"Built public repo: {out / 'public_repo'}")
    if not args.public_only:
        print(f"Built private data pack: {out / 'private_data_pack'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
