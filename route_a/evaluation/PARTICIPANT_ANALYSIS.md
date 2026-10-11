# 人口信息：样本描述、质量控制与冻结后探索性分析

本模块只在训练之外使用问卷人口信息。主模型、V30 编码器、Planner、Realizer、fold 划分和模型选择均不得读取这些产物；Q7 仍然只输入 V30 与公共 segment 提示。

实现入口：

```bash
python3 -m route_a.evaluation.participant_analysis --help
```

## 1. 生成私有元数据与描述性报告

必须使用数据构建器已经生成的私有匿名 ID 映射，不能创建第二套匿名编号：

```bash
python3 -m route_a.evaluation.participant_analysis prepare \
  --users raw_data/questionnaire_v3/users.csv \
  --id-mapping processed_data/route_a_training_v1/private/user_id_mapping.json \
  --config route_a/configs/participant_analysis_v1.json \
  --output processed_data/participant_analysis_v1/prepared
```

输出目录整体属于私有评测数据，已被 `.gitignore` 排除：

- `participant_metadata.csv`：匿名 ID 和白名单内的粗粒度类别；
- `quality_control.csv`：匿名 ID、总答题时长和人工复核标记；
- `sample_description.csv`：论文样本描述的聚合计数与比例；
- `quality_control_summary.json`：聚合 QC 结果；
- `prepare_manifest.json`：输入/config 哈希和“禁止训练使用”声明；
- `checksums.sha256`：上述文件的封存校验。

程序不会输出微信号、自由文本、精确完成时间、V30 或任何模型输入。源数据中出现未在配置中冻结的新类别时，程序直接失败，不会临时创建新组。

## 2. 质量控制规则

`participant_analysis_v1.json` 当前冻结的人工复核阈值为：

| 总时长 | 过快复核阈值 | 过慢复核阈值 |
|---|---:|---:|
| V30 量表 | 少于 60 秒 | 多于 1800 秒 |
| Q1–Q8 任务 | 少于 300 秒 | 多于 10800 秒 |

这些阈值只产生 `manual_review`，不会自动删除参与者。正式排除必须由研究团队在查看原始答题完整性后另行制定、记录并冻结；不得根据模型效果决定排除谁。

## 3. 冻结后的探索性分析

先完成 Q7 推理和标准化评测，使评测目录包含 `participant_metrics.csv`、`run_manifest.json` 和匹配的 `checksums.sha256`，然后运行：

```bash
python3 -m route_a.evaluation.participant_analysis analyze-frozen \
  --metadata processed_data/participant_analysis_v1/prepared/participant_metadata.csv \
  --participant-metrics results/route_a_full/evaluation/participant_metrics.csv \
  --config route_a/configs/participant_analysis_v1.json \
  --system-id route_a_full \
  --output results/route_a_full/participant_subgroups
```

运行前会同时验证私有元数据 bundle 和标准评测 bundle 的全部 checksum，并要求评测 manifest 存在封存预测的 SHA256。任何文件在封存后被修改，分析都会在创建输出目录前失败。

探索性分析仅使用配置中 `exploratory: true` 的三个粗粒度变量：

- 设计相关经验：不足 1 年、1–3 年、3 年以上；
- 教育背景：设计科班、跨专业；
- 跨学科经历：有、没有。

每个参与者先对多个运行 seed 取均值，再进行群体聚合和 participant bootstrap。输出不包含参与者级记录、不计算 p 值，也不允许用于训练、调参或模型选择。只要某变量存在少于 10 人的群体，该变量全部群体都执行互补抑制，防止根据总人数反推出小组人数。

当前配置输出：

- Seg2 主指标：KC-NES、BGE own-user margin；
- Seg1/Seg3 辅助指标：BGE own-user margin。

## 4. 测试

```bash
python3 -m unittest route_a.tests.evaluation.test_participant_analysis -v
python3 -m compileall -q route_a/evaluation route_a/tests/evaluation
```

测试覆盖直接标识/自由文本/时间戳不出现在产物中、未知类别失败、小样本抑制、seed 先在参与者内聚合、checksum 篡改拒绝以及禁止自动排除。
