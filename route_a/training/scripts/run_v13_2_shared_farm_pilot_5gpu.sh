#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT [GPU_CSV]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
GPU_CSV="${4:-0,2,3,6,7}"
PYTHON_BIN="${ROUTE_A_PYTHON:-python}"
BASE_CONFIG="route_a/configs/training_v13_2_shared_farm.json"
IFS=',' read -r -a GPUS <<<"${GPU_CSV}"
NAMES=(structural semantic shared_r2 shared_r4 shared_r8)
VARIANTS=(structural semantic shared shared shared)
RANKS=(4 4 2 4 8)

if [[ "${#GPUS[@]}" -ne 5 ]]; then
  echo "GPU_CSV must contain exactly five physical GPU indices" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -ne 5 ]]; then
  echo "GPU_CSV contains duplicate indices" >&2
  exit 2
fi
if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "Shared FaRM pilot requires a /formal dataset" >&2
  exit 2
fi
if [[ -e "${ROOT}" ]]; then
  echo "output root already exists: ${ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "training interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
AVAILABLE_KIB="$(df -Pk /data/wy | awk 'NR==2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ || "${AVAILABLE_KIB}" -lt 104857600 ]]; then
  echo "Shared FaRM pilot requires at least 100 GiB free on /data" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU index: ${gpu}" >&2
    exit 2
  fi
  free="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "${free}" =~ ^[0-9]+$ || "${free}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free:-unknown} MiB free; require >=60000 MiB" >&2
    exit 2
  fi
done

mkdir -p "${ROOT}/frozen"
cp route_a/configs/state_ontology_v2.json "${ROOT}/frozen/"
cp route_a/configs/choice_state_mapping_v1.json "${ROOT}/frozen/"
cp route_a/configs/v30_value_profile_lexicon_v1.json "${ROOT}/frozen/"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${ROOT}/frozen" <<'PY'
import copy
import json
import pathlib
import sys

source = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out = pathlib.Path(sys.argv[2])
settings = (
    ("structural", "structural", 4),
    ("semantic", "semantic", 4),
    ("shared_r2", "shared", 2),
    ("shared_r4", "shared", 4),
    ("shared_r8", "shared", 8),
)
for name, variant, rank in settings:
    config = copy.deepcopy(source)
    config["experiment_id"] = f"route_a_v13_2_shared_farm_pilot_{name}"
    config["shared_farm"]["variant"] = variant
    config["shared_farm"]["interaction_rank"] = rank
    config["shared_farm"]["base_max_updates"] = 100
    config["shared_farm"]["farm_max_updates"] = 100
    config["shared_farm"]["joint_max_updates"] = 200
    config["planner"]["enforce_rollout_gate"] = False
    config["planner"]["rollout_evaluation_interval"] = 25
    (out / f"{name}.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
PY

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "physical_gpus=${GPU_CSV}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "outer_fold=0"
  echo "inner_fold=0"
  echo "base_updates=100"
  echo "farm_updates=100"
  echo "joint_updates=200"
  echo "q7_gold_used=false"
} >"${ROOT}/provenance.txt"

run_variant() {
  local index="$1"
  local name="${NAMES[$index]}"
  local gpu="${GPUS[$index]}"
  local runtime="${ROOT}/${name}-runtime"
  mkdir -p "${runtime}/tmp" "${runtime}/torch-cache"
  TMPDIR="${runtime}/tmp" TEMP="${runtime}/tmp" TMP="${runtime}/tmp" \
  TORCHINDUCTOR_CACHE_DIR="${runtime}/torch-cache" \
  CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    -m route_a.training.shared_farm_train \
    --dataset-dir "${DATASET_DIR}" \
    --config "${ROOT}/frozen/${name}.json" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${ROOT}/${name}" \
    --outer-fold 0 \
    --fit-mode inner-selection \
    --inner-fold 0 \
    --device-index 0 \
    --seed 42 \
    --qlora >"${ROOT}/${name}.log" 2>&1
}

pids=()
for index in 0 1 2 3 4; do
  run_variant "${index}" &
  pids+=("$!")
  echo "${pids[-1]}" >"${ROOT}/${NAMES[$index]}.pid"
done

failed=0
for index in 0 1 2 3 4; do
  if wait "${pids[$index]}"; then
    echo "PASS" >"${ROOT}/${NAMES[$index]}.status"
  else
    echo "FAIL" >"${ROOT}/${NAMES[$index]}.status"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "at least one Shared FaRM pilot variant failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${ROOT}" "${NAMES[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for name in sys.argv[2:]:
    value = json.loads((root / name / "validation_metrics.json").read_text(encoding="utf-8"))
    joint = value["joint"]
    rows.append({
        "name": name,
        "selected_update": joint["selected_update"],
        "correct_v30_nll": joint["validation"]["correct_v30"]["nll"],
        "permuted_v30_nll": joint["validation"]["permuted_v30"]["nll"],
        "permuted_nll_increase": joint["validation"]["permuted_nll_increase"],
        "directional_v30_sensitivity_observed": joint["validation"]["directional_v30_sensitivity_observed"],
        "forced_stop_rate": joint["rollout"]["forced_stop_rate"],
        "unique_sequence_rate": joint["rollout"]["unique_sequence_rate"],
        "top_sequence_share": joint["rollout"]["top_sequence_share"],
    })
semantic_nll = next(row["correct_v30_nll"] for row in rows if row["name"] == "semantic")
eligible = [
    row for row in rows
    if row["name"].startswith("shared_")
    and row["permuted_nll_increase"] >= 0.001
    and row["correct_v30_nll"] <= semantic_nll + 0.05
    and row["forced_stop_rate"] <= 0.1
    and row["top_sequence_share"] < 1.0
]
selected = min(eligible, key=lambda row: row["correct_v30_nll"]) if eligible else None
report = {
    "schema_version": "1.0.0",
    "status": "METHOD_PASS" if selected is not None else "METHOD_GATE_FAIL",
    "q7_gold_used": False,
    "gates": {
        "minimum_permuted_nll_increase": 0.001,
        "maximum_nll_delta_vs_semantic": 0.05,
        "maximum_forced_stop_rate": 0.1,
        "require_not_all_rollouts_identical": True,
    },
    "selected": selected,
    "results": rows,
}
(root / "pilot_summary.json").write_text(
    json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

"${PYTHON_BIN}" - "${ROOT}/pilot_summary.json" "${ROOT}/pilot_result.txt" <<'PY'
import json
import pathlib
import sys
status = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"]
pathlib.Path(sys.argv[2]).write_text(status + "\n", encoding="utf-8")
print(f"V13_2_SHARED_FARM_5GPU_PILOT_{status}")
PY
