# 新版 KC 个体化规则统一采样协议（V1）

## 1. 统一口径

三类规则统一采用 `participant_resampling_v1`：

| 设置 | 冻结值 |
|---|---:|
| 抽样单位 | 参与者 |
| bootstrap | 1,000 |
| participant permutation | 1,000 |
| participant-held-out | 5 折 |
| 随机种子 | 20260727 |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |

脚本在正式运行时 fail closed：采样次数、折数或软件版本不符合冻结值时，不生成
正式产物。所有产物记录协议 ID、软件版本、输入 SHA 和随机种子。

## 2. 统一运行结果

| 规则层 | 规则形式 | 探索性候选 | validated |
|---|---|---:|---:|
| Q2 单维主效应 | `(V pole, current KC) -> next KC` | 12 | 0 |
| Q2 双维交互 | `(V relation, current KC) -> next KC` | 49 | 0 |
| Q3–Q6 情境偏好 | `(question, V pole) -> preferred KC` | 29 | 0 |
| **合计** |  | **90** | **0** |

Q2 规则池为 61 条，其中双维关系模型控制对应的两个单维主效应，所以双维规则
表示额外交互信号，不能替代 12 条单维规则。Q3–Q6 使用推荐的
`c03_balanced` 参数臂，覆盖 16/30 个 V30 维度。

## 3. 合并守卫

统一审计器只在以下条件同时成立时输出合并计数：

1. 三类产物的协议 ID、软件版本、随机种子和采样参数一致；
2. 参与者文件和 V30 特质定义 SHA 一致；
3. Q2 单维与双维使用相同 planner 输入 SHA；
4. 指定的 Q3–Q6 参数臂存在。

复现命令：

```bash
python tools/kc_rule_protocol_audit.py \
  --single-audit artifacts/kc_operational_ontology_v1_uniform/audit.json \
  --pair-audit artifacts/kc_v30_relation_ontology_v1_uniform/audit.json \
  --aux-report artifacts/kc_aux_preference_sweep_v1_uniform/sweep_report.json \
  --aux-arm c03_balanced \
  --output artifacts/kc_uniform_rule_audit_v1.json
```

当前审计结果为 `PASS`。这 90 条仍是开发期候选池；由于多重检验后没有规则
达到 `q<=0.10`，论文中不得表述为已经验证的个体规律。

## 4. 规则库冻结决定

`KC_RULEBANK_V1` 于 2026-07-27 冻结，状态为 `FROZEN_EXPLORATORY`。

- 不再扩大规则池；
- 不再调整提取、筛选或方向编译参数；
- 不允许依据 Q7 结果修改、删除或新增某一条规则；
- 后续实验只允许选择启用哪些完整规则层；
- 若某层无效，则在主实验中停用并作为消融结果报告，不重新挖掘。

冻结清单：`artifacts/kc_rulebank_v1_freeze_manifest.json`。每次实验前应运行：

```bash
python tools/verify_kc_rulebank_freeze.py \
  --manifest artifacts/kc_rulebank_v1_freeze_manifest.json \
  --repo-root .
```

只有输出 `status=PASS` 时，实验结果才可归入 `KC_RULEBANK_V1`。
