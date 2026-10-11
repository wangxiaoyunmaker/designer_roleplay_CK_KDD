# Route A Q7 推理

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

没有 development-only 选择校准目录时，系统强制回退到 greedy，不会用未经验证的 Best-of-N 分数。提供校准目录后才启用 Best-of-N；系统会验证目录 checksum、候选记录重算统计、development 参与者集合、fold、数据/配置及两个 selection checkpoint manifest，并要求 `q7_gold_used=false`。当前实现没有冻结 consistency judge，因此 `consistency_weight` 必须为 0。

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

冻结随机种子为 42/43/44。每个 seed 独立完成五折合并后，再运行 `merge-q7-seeds`；它要求三个 seed 的参与者覆盖和 provenance 完全一致，输出可由标准评测器直接读取的多 seed 预测文件。评测器先在参与者内平均 seed，再做 participant-level bootstrap。

teacher-forced Macro-F1 使用独立的 `export-q7-teacher-forced`。它先验证已经存在的预测封印、五折 merge manifest、fold 覆盖，以及当前 Planner manifest SHA 与生成时对应 fold 的 Planner 完全相同，然后才打开 Q7 gold；导出结果记录 gold SHA，以 gold KC 历史输出六状态加 STOP 的完整概率，不会重写预测。五折导出后用 `merge-teacher-forced` 校验步号连续和唯一末尾 STOP，再交给 `route_a.evaluation evaluate-teacher-forced`。

完整后封存辅助评测可用 `scripts/run_5fold_teacher_forced.sh`；该脚本必须在主预测合并并封印完成后运行。
