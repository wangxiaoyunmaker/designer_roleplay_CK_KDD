# Route A Q7 推理

## V12 calibrated Best-of-N

V12 不使用下文的旧 z-score likelihood reranking。正式选择器由
`generate-v12-calibration-inputs` 和 `freeze-best-of-n` 在 Q2 上冻结：

- 计划与文本 likelihood 使用候选集内 percentile rank；
- BGE plan-answer consistency 只有 participant-level AUC ≥ 0.65 且 margin
  bootstrap 95% CI 下界 > 0 时才有资格启用；
- selection-val 只在 consistency weight `{0, 0.5}` 中选择；
- forced-stop、4-gram 循环、空回答和模板泄漏受排除或惩罚；
- Q7 生成必须提供冻结 calibration；consistency 启用时还必须提供 checksummed
  本地 BGE-M3。

若某折 RFT gate 通过，`generate-q7` 可以读取 `realizer-rft-refit`，但必须同时
提供与 checkpoint checksum 一致的 `--rft-gate-report`。gate 失败的折使用
Stage C `realizer-refit`，不能为了 Q7 结果重新调 RFT reward。

V12 Seg2 固定 `temperature=0.7, top_p=0.8, top_k=20`；Planner 保留一条 greedy
和三条采样计划。`greedy-text0` 是唯一真正的贪婪文本候选，其余文本候选均采样，
因此无 calibration 时的回退语义与名称一致。冻结 calibration 与 Q7 推理必须使用
同一 inference config 和 BGE snapshot，五折/三 seed 合并还会校验代码树 SHA。

## V10/V11 旧接口（仅用于基线复现）

先在本地做不加载模型的安全检查：

```bash
python3 -B -m route_a.inference prepare-q7 \
  --dataset-dir processed_data/route_a_training_v1/draft \
  --training-config route_a/configs/training_v1.json \
  --inference-config route_a/configs/inference_v1.json \
  --instrument raw_data/design_task_instrument_q1_q8.json \
  --outer-fold 0 \
  --allow-draft-uncleaned
```

正式服务器生成使用 `generate-q7`，同时提供该 fold 的 Planner 和 Realizer checkpoint。接口只从 Q7 instrument 白名单提取公共情境、主问题和三个 segment prompt；不读取 `task_responses.csv` 或任何 Q7 gold。

没有 development-only 选择校准目录时，系统强制回退到 greedy。提供校准目录后启用的是模型似然重排（likelihood reranking），不是 RFT 或 reward-model Best-of-N；系统会验证目录 checksum、候选记录重算统计、development 参与者集合、fold、数据/配置及两个 selection checkpoint manifest，并要求 `q7_gold_used=false`。当前实现没有冻结 consistency judge，因此 `consistency_weight` 必须为 0；存在自然 STOP 候选时会排除 forced-stop 候选。

每折先用 selection checkpoint 生成合规校准（只访问 inner validation 和 Q2 公共情境）：

```bash
python3 -B -m route_a.inference calibrate-best-of-n \
  --dataset-dir processed_data/route_a_training_v1/formal \
  --training-config route_a/configs/training_v1.json \
  --inference-config route_a/configs/inference_v1.json \
  --outer-fold 0 \
  --planner-checkpoint outputs/seed42/fold0/planner-selection \
  --realizer-checkpoint outputs/seed42/fold0/realizer-selection \
  --model-path /models/Qwen3-8B \
  --output-dir outputs/q7/calibration/fold0
```

它固定 0.5/0.5 的计划/文本平均 log-prob 权重，计算 development z-score 统计并保存候选审计，不读取 Q7。`scripts/run_5fold_q7.sh` 串起五折校准、refit 模型生成、校验、合并和重新封印。

每折同时输出 `q7_predictions_v1.jsonl`、单文件封印 `q7_predictions_v1.sha256`、候选审计与 bundle checksum。Seg2 含 KC 序列，Seg1/Seg3 的 `kc_states` 固定为 `null`。

五折完成后必须运行：

```bash
python3 -B -m route_a.inference merge-q7-folds \
  --fold-dir outputs/fold0/q7 --fold-dir outputs/fold1/q7 \
  --fold-dir outputs/fold2/q7 --fold-dir outputs/fold3/q7 \
  --fold-dir outputs/fold4/q7 \
  --dataset-dir processed_data/route_a_training_v1/formal \
  --ontology route_a/configs/state_ontology_v2.json \
  --output-dir outputs/q7-merged
```

该命令验证折号 0–4、参与者与冻结 `folds.jsonl` 精确一致、三段完整、选择模式一致、配置/数据/模型快照一致和所有 checksum，再排序、合并并重新封印。标准评测应读取合并目录中的 `q7_predictions_v1.jsonl` 与 `q7_predictions_v1.sha256`。

V10/V11 冻结随机种子为 42/43/44；V12 冻结为 17/42/2026。每个 seed 独立完成五折合并后，再运行 `merge-q7-seeds`；它从所给 training config 读取对应 seed 集合，要求参与者覆盖和 provenance 完全一致，输出可由标准评测器直接读取的多 seed 预测文件。评测器先在参与者内平均 seed，再做 participant-level bootstrap。

teacher-forced Macro-F1 使用独立的 `export-q7-teacher-forced`。它先验证已经存在的预测封印、五折 merge manifest、fold 覆盖，以及当前 Planner manifest SHA 与生成时对应 fold 的 Planner 完全相同，然后才打开 Q7 gold；导出结果记录 gold SHA，以 gold KC 历史输出六状态加 STOP 的完整概率，不会重写预测。五折导出后用 `merge-teacher-forced` 校验步号连续和唯一末尾 STOP，再交给 `route_a.evaluation evaluate-teacher-forced`。

完整后封存辅助评测可用 `scripts/run_5fold_teacher_forced.sh`；该脚本必须在主预测合并并封印完成后运行。
