# Route A 训练数据构建器

本目录把 V30 与 Q1–Q6 转换为可直接由训练代码读取的 JSONL。Q7、Q8、`users.csv`、选项文字和卡片文字不会进入训练数据。

完整冻结设计见：

```text
docs/specs/2026-07-21-route-a-training-data-design.md
```

## 1. 当前 draft 构建

在项目根目录执行：

```bash
python3 -B -m route_a.data build \
  --root /Users/wy/Desktop/KDD \
  --mode draft
```

输出：

```text
processed_data/route_a_training_v1/
├── cleaning/nl_cleaning_input_v1.csv
├── draft/
└── private/user_id_mapping.json
```

`draft` 的自然语言仍是原始 Q1–Q6 回答，固定标记为 `RAW_UNCLEANED`。它只能用于 loader 联调、smoke test 和探索性实验。

验证 draft 时必须显式确认：

```bash
python3 -B -m route_a.data validate \
  --dataset-dir /Users/wy/Desktop/KDD/processed_data/route_a_training_v1/draft \
  --allow-draft-uncleaned
```

训练入口应复用同一门禁；没有显式开关时拒绝 draft。

## 2. 人工清洗

用 Excel 打开：

```text
processed_data/route_a_training_v1/cleaning/nl_cleaning_input_v1.csv
```

只允许编辑：

```text
cleaning_status
clean_text
editor_id
review_note
```

禁止编辑：

```text
record_id
anonymous_user_id
question_id
raw_text_sha256
raw_text
```

状态规则：

- `KEEP`：`clean_text` 与原文完全相同；
- `EDITED`：`clean_text` 非空且确实发生编辑；
- `DROP_REALIZER_ONLY`：`clean_text` 留空，只删除当前用户当前题的文本监督。

重复构建 draft 不会覆盖已经存在的清洗 CSV。

## 3. formal 构建

清洗全部完成后执行：

```bash
python3 -B -m route_a.data build \
  --root /Users/wy/Desktop/KDD \
  --mode formal \
  --cleaning-file /Users/wy/Desktop/KDD/processed_data/route_a_training_v1/cleaning/nl_cleaning_input_v1.csv
```

只要仍存在 `RAW_UNCLEANED`、只读列被修改、记录缺失/重复、KEEP/EDITED/DROP 契约错误，formal 构建就会失败。

## 4. 训练文件

- `participants.jsonl`：匿名参与者和原始 30 维 V30；折内标准化由训练代码完成。
- `aux_records.jsonl`：Q1、Q3–Q6 的非 V 选择状态。
- `planner_records.jsonl`：Q2 删除 V 后的完整六类 KC 序列。
- `realizer_records.jsonl`：三种条件模式 `CHOICE_STATE`、`NO_KC_STATE`、`PROCESS_SEQUENCE`。
- `folds.jsonl`：仅根据匿名 ID 生成的五折。
- `scenario_registry.json`：只含公共情境、题目和 instructions。
- `dataset_manifest.json`：来源、代码、配置、统计和数据文件 SHA。
- `validation_report.json`：机器可读 PASS/FAIL 与计数。
- `checksums.sha256`：数据包完整性。

Planner loader 使用 `expand_planner_teacher_forcing` 在末尾加入 `STOP`。Realizer 使用 `render_realizer_prompt` 生成三种标记的输入，并用 `iter_balanced_realizer` 保证 Q2 `PROCESS_SEQUENCE` 与其他题按 1:1 抽样。

## 5. 服务器传输

可以上传：

- `route_a/` 代码和冻结配置；
- `processed_data/route_a_training_v1/draft` 或完成后的 `formal`；
- `scenario_registry.json`、manifest 和 checksum。

不要上传到普通代码仓库：

- `private/user_id_mapping.json`；
- `.env` 和 API key；
- 未匿名原始 CSV；
- 模型权重、checkpoint、wandb 和运行产物。

## 6. 测试

```bash
python3 -B -m unittest discover -s route_a/tests/data -p 'test_*.py' -v
```

若本机存在 `/Users/wy/Desktop/KDD/raw_data`，真实数据回归测试还会核对 162/751/972/1,109 等冻结统计；没有私有原始数据的环境会跳过该测试。
