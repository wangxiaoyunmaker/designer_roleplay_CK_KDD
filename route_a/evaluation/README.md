# Route A 标准化评测器

该目录实现 `docs/specs/2026-07-20-standardized-evaluation-design.md`。核心 KC、文本相似度聚合、盲评聚合和统计检验只依赖 Python 标准库；只有生成正式 BGE-M3 embedding 时需要可选的 `FlagEmbedding` 环境。

## 目录

- `schemas.py`：预测、gold、teacher-forced 严格 schema。
- `state_metrics.py`：`F_KC`、KC-NES、Macro-F1、STOP。
- `text_metrics.py`：embedding cache、own-user margin、冻结 BGE-M3 适配器。
- `judge_metrics.py`：persona 与 pairwise 盲评聚合及一致性。
- `bootstrap.py`：参与者 bootstrap、配对 sign-flip、Holm。
- `cli.py`：统一命令行入口。
- `participant_analysis.py`：训练外的样本描述、人工复核 QC 与预测冻结后群体分析；使用方法见 [PARTICIPANT_ANALYSIS.md](PARTICIPANT_ANALYSIS.md)。

## 自动评测

```bash
python3 -m route_a.evaluation.cli evaluate-automatic \
  --predictions path/q7_predictions_v1.jsonl \
  --prediction-checksum path/q7_predictions_v1.sha256 \
  --gold path/q7_gold_eval_v1.jsonl \
  --predicted-embeddings path/predicted_embeddings.json \
  --gold-embeddings path/gold_embeddings.json \
  --ontology route_a/configs/state_ontology_v2.json \
  --config route_a/configs/evaluation_v1.json \
  --output path/evaluation_output
```

预测文件必须先由 `route_a.inference merge-q7-folds` 合并并封存；checksum 不匹配、预测行 payload hash 不匹配或参与者覆盖不全时，评测会在创建输出目录前失败。Seg1、Seg3 不接受状态序列，也不会生成 KC 指标。

## BGE-M3

正式编码必须使用本地模型快照和逐文件 SHA256 manifest，不允许由评测命令从浮动的远程 `main` 下载：

```bash
python3 -m route_a.evaluation.cli encode-bge \
  --input path/text_records.jsonl \
  --kind predicted \
  --model-path path/frozen_bge_m3 \
  --snapshot-manifest path/frozen_bge_m3.sha256 \
  --output path/predicted_embeddings.json
```

gold 编码将 `--kind` 改为 `gold`，输入字段必须是 `gold_text`。快照 manifest 必须逐一列出模型目录中的全部文件；多出任何未冻结权重都会被拒绝。

teacher-forced 七类概率由 post-seal 导出器产生，合并后运行：

```bash
python3 -m route_a.evaluation.cli evaluate-teacher-forced \
  --input path/teacher-forced-merged/teacher_forced_v1.jsonl \
  --ontology route_a/configs/state_ontology_v2.json \
  --output path/teacher_forced_metrics.json
```

安装可选依赖时建议使用训练服务器上的 Python 3.10/3.11 与匹配 CUDA 的 PyTorch 环境；最终安装版本和模型快照 hash 会写入评测 manifest。

## 测试

```bash
python3 -m unittest discover -s route_a/tests/evaluation -v
python3 -m compileall -q route_a/evaluation route_a/tests/evaluation
```

测试不访问网络、不下载模型，也不读取尚未完成的 Q7 标注。
