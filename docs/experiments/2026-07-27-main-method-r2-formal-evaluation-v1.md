# R2 主方法全量自动评测 v1

状态：`COMPLETE_AUTOMATIC_METRICS / R2 RETAINED AS HISTORICAL RESULT / R3 REDESIGN PLANNED`

## 1. 范围与合同

- 目标：Q7 Seg2，162 位参与者。
- A：既有 Qwen3-8B PP one-call 外部锚点。
- B--F：`q7-main-r2-semantic-flexible-static-dynamic-rule-memory-v13`。
- 金标：`q7_human_gold_adjudicated_v2`，计算 KC 指标前删除 `V` 并保序。
- 统计：参与者级配对 bootstrap，10,000 次，95% percentile CI。
- B--F 使用其他参与者 Q7 ICL，因此结果属于 transductive retrieval，不是 zero-shot。
- 三条运行失败均发生在 Phase 2；其 Phase 1 序列有效，故 KC 指标仍按 162 人计算。BGE 不修复失败文本，只在有效文本交集上比较并报告 `n`。

## 2. 系统结果

| 条件 | KC-NES ↑ | State-JS sim. ↑ | Transition F1 ↑ | BGE own-user margin ↑ | BGE n | Unique chains | Chain diversity ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Existing PP | 0.2773 | 0.6193 | 0.1476 | 0.0000 | 162 | 20 | 0.6111 |
| B Dynamic | 0.2448 | 0.5892 | 0.0869 | -0.0007 | 162 | 40 | 0.7840 |
| C Dual | 0.2445 | 0.5954 | 0.0885 | 0.0039 | 162 | 26 | 0.5432 |
| D Rules | 0.2365 | 0.5940 | 0.0873 | -0.0012 | 160 | 45 | 0.6481 |
| E Near | 0.2350 | 0.5881 | 0.0904 | 0.0020 | 161 | 43 | 0.7037 |
| F Far | 0.2391 | 0.5752 | 0.0889 | -0.0007 | 162 | 39 | 0.6975 |

Chain diversity 只诊断是否坍缩，不证明个体匹配正确。C 的 BGE margin 点估计最高，但不能脱离配对置信区间解释。

## 3. 预注册增量比较

| 比较 | ΔKC-NES [95% CI] | ΔBGE margin [95% CI] | 结论 |
|---|---:|---:|---|
| C - B | -0.0002 [-0.0166, 0.0157] | +0.0046 [-0.0023, 0.0116] | Full30 无稳定增量 |
| D - C | -0.0080 [-0.0260, 0.0103] | -0.0047 [-0.0114, 0.0019] | 规则无稳定增量 |
| E - D | -0.0015 [-0.0205, 0.0164] | +0.0032 [-0.0036, 0.0101] | 近邻 Q2 无稳定增量 |
| F - D | +0.0025 [-0.0145, 0.0194] | +0.0006 [-0.0059, 0.0073] | 远邻 Q2 无稳定增量 |
| E - F | -0.0041 [-0.0205, 0.0122] | +0.0023 [-0.0045, 0.0094] | 无近邻优于远邻证据 |
| B - A | -0.0326 [-0.0589, -0.0068] | -0.0007 [-0.0105, 0.0091] | R2 基础臂 KC 显著低于外部 PP 锚点 |

B 相对 A 的平均长度误差更低，差值为 -0.5617，95% CI [-0.9012, -0.2593]；但 KC-NES、State-JS similarity 和 Transition F1 均显著更低。B 的链多样性高于 A，说明当前结果呈现的是“更多样但不更接近本人金标”的准确性-多样性张力。

F 相对 D 的 State-JS similarity 下降 -0.0189，95% CI [-0.0332, -0.0056]。其余 R2 内部的 KC、状态分布、转移或 BGE 增量均未形成稳定正证据。

## 4. 结论边界

1. 当前自动结果不支持“静态 Full30、冻结规则或近邻 Q2 提高轨迹保真度”的结论。
2. C 和 E 的 BGE own-user margin 点估计为正，但相对其直接对照的 CI 均跨 0，不能宣称稳定个体化。
3. B--F 比 A 产生更多不同链，但 KC-NES 没有同步提高，因此 anti-collapse 不能替代 fidelity。
4. A 与 B 同时改变生成合同、状态本体和上下文包，只能作为外部锚点；`B-A` 不是单因素因果比较。
5. V semantic 和 V30 matched-minus-mismatched 人工指标尚未完成，当前报告不是最终价值忠实度结论。

## 5. BGE 与复现审计

- Encoder：冻结 `BAAI/bge-m3`，1024 维 dense embedding，FP32、L2 normalization。
- Snapshot manifest SHA-256：`b2388b4b62d245d8615d300b943b709af3a9bb026d501e23fd811b72c11d2d3a`。
- 当前 v2 gold 与旧正式缓存文本一致：160/162；当前 162 条 gold 均重新编码。
- AutoModel CLS 与旧正式缓存抽样向量的最小 cosine：1.0。
- 有效预测文本：969；所有六个系统共同有效文本 cohort：159 人。
- 不进行选择性重采样或语义修复。

## 6. 产物

- 汇总：`outputs/main_method_r2_evaluation_v1/summary.json`
- 论文表：`outputs/main_method_r2_evaluation_v1/paper_table.csv`
- 配对比较：`outputs/main_method_r2_evaluation_v1/paired_comparisons.csv`
- 参与者级指标：`outputs/main_method_r2_evaluation_v1/participant_metrics.jsonl`
- Manifest：`outputs/main_method_r2_evaluation_v1/manifest.json`
- 评测入口：`tools/prompt_baselines/evaluate_main_method_r2.py`

远端权威副本位于：

```text
/data/wy/main_method_r2_ablation_v1/evaluation/formal_v13_automatic_v1
```

## 7. 事后机制诊断：为什么模块改变了输出但没有形成个体增益

本节是在正式 R2 结果完成后进行的机制诊断，属于 `POST_HOC_DIAGNOSTIC`，不改变第 2--4 节的预注册结论。

### 7.1 模块摄取不是零，但变化方向不稳定

| 比较 | 序列发生变化 | 变化后更接近金标 | 变化后更远离金标 | 全体平均 ΔKC-NES |
|---|---:|---:|---:|---:|
| C - B：加入静态 Full30 | 92/162 (56.8%) | 38 | 29 | -0.0002 |
| D - C：加入规则 | 108/162 (66.7%) | 29 | 41 | -0.0080 |
| E - D：加入近邻 Q2 | 91/162 (56.2%) | 27 | 32 | -0.0015 |
| F - D：加入远邻 Q2 | 89/162 (54.9%) | 30 | 26 | +0.0025 |

因此，Full30、规则和跨用户 Q2 都足以扰动 Qwen3-8B 的采样结果；问题不是模型完全忽略了新增文本，而是这些变化没有稳定指向目标参与者的 Q7 金标。

### 7.2 规则模块的目标操作摄取率低

- 每位参与者均注入 3 条规则：一条单维转移、一条双维转移和一条情境偏好。
- 条件转移规则共出现 230 次可执行机会，只有 45 次产生目标转移，遵循率为 19.6%。
- 情境偏好的目标状态在 D 中出现率为 100%，但同一目标状态在不含规则的 C 中已经达到 98.8%，实际增量只有 1.2%。

这说明当前规则文本一方面足以改变整条序列，另一方面没有稳定完成它所声明的局部转移。R3 不再把这些规则直接注入生成 Prompt；规则仅在通过独立验证后作为候选轨迹评分特征进入 Shared FaRM。

### 7.3 R2 动态 V30 存在候选池过窄和任务先验错配

R2 每一步从 10 个候选维度中选择 8 个。两个 8 元子集在 10 元全集中的 Jaccard 下界已经是 0.6；实际在相同 KC 转移上下文下，不同参与者的 cue-ID 平均 Jaccard 达到 0.903。参与者内部不同步骤的平均 Jaccard 也达到 0.757。

更重要的是，R2 使用的冻结语义先验文件为 `q2_semantic_value_relevance_v2`，其 `question_id` 明确为 Q2，却被用于 Q7 cue 排序。虽然 cue 的局部文本包含 Q7，但主导选择的 0.72 语义分数来自 Q2 先验。这使“动态”更多表现为公共步骤相关性，而不是 Q7 下的参与者差异。

### 7.4 30 人 matched-vs-shuffled 反事实

| 比较 | 指标差值 | 95% CI | 解释 |
|---|---:|---:|---|
| 正确静态 Full30 - 错配静态 Full30 | KC-NES +0.0109 | [-0.0311, 0.0475] | 无稳定的正确画像优势 |
| 正确静态 Full30 - 错配静态 Full30 | BGE own-user margin -0.0040 | [-0.0274, 0.0174] | 无文本个体优势 |
| 正确动态 cue - 错配动态 cue，冻结同一序列 | BGE own-user margin -0.0028 | [-0.0200, 0.0153] | 动态 cue 未使文本更接近本人 |

动态 cue 错配后 30/30 文本发生变化，但正确 cue 并不优于错配 cue。这表明 R2 V30 主要造成生成扰动，尚未形成可验证的身份条件化。

诊断产物：

- `outputs/diagnostics/main_method_r2_component_uptake_v1/report.md`
- `outputs/diagnostics/main_method_r2_component_uptake_v1/summary.json`
- `outputs/diagnostics/main_method_r2_component_uptake_v1/counterfactual_summary.json`

## 8. 架构重新决策：拆分建模，但不拆成逐步多调用

“Planner 与 Realizer 是否拆开”需要区分两个问题：

1. **建模对象是否拆开**：是否先产生可独立评价和排序的结构化轨迹 (M)，再生成文本 (Y)。
2. **Realizer 是否逐步多次调用**：是否为每个状态重新调用一次模型。

已有 30 人 joint-vs-split pilot 没有发现 prompted split 相对 joint 的显著 KC 增益；三个条件甚至各自坍缩为一条固定序列。因此，不能声称“多一次调用天然提高质量”。但论文主方法仍需要独立的 Planner 输出，因为 Shared FaRM 必须在文本实现前对多个候选轨迹评分，而且轨迹 fidelity 必须能脱离语言表面单独评价。

最终架构决策为：

```text
PI + 本人 Q2 + 静态 Full30 + Q7 Planner-active cues + Q7 情境
                              ↓
                Planner samples candidate trajectories
                              ↓
                   Shared FaRM ranks candidates
                              ↓
                       frozen best trajectory
                              ↓
          Q7 step-specific V30 cues selected from that trajectory
                              ↓
           one-call aligned Realizer emits nl_units[]
                              ↓
                   deterministic concatenation
```

- **保留两阶段建模边界**：Planner 和 Realizer 是两个可训练、可替换、可评价的对象。
- **Realizer 只调用一次**：一次输出与冻结序列等长的 `nl_units[]`；不为每一步建立新会话。
- **不使用 Composer**：程序按数组顺序拼接，不进行生成式改写。
- **joint one-call 保留为效率基线**：它直接输出序列和文本，但不是最终主方法。
- **不把 split 当成贡献点**：贡献是可审计的轨迹瓶颈、个体条件化排序和价值忠实实现，而不是调用次数。

这一决策同时符合两个既有 pilot：split 没有普遍质量增益；在已有冻结序列时，单次 aligned Realizer 比逐步 fresh calls 更稳定、更便宜，并能保留严格的状态对齐接口。

## 9. R3 V30 实现

### 9.1 静态 Full30

- 目标设计师全部 30 维仍以方向性自然语言放入共享上下文，位于 PI 之后、本人 Q2 之前。
- 不显示题项 ID、维度名称、原始分数或强弱标签。
- Full30 在 Phase 1 只出现一次；Phase 2 继承同一对话上下文，不再重复粘贴，从而减少约 400--500 个输入 token 的冗余。
- Full30 是完整、可审计的背景条件；它不再独自承担“告诉 Planner 当前最该关注什么”的责任。

### 9.2 Planner-active V30

R3 使用只读取 Q7 公共情境的冻结语义先验，覆盖全部 30 维，不读取任何参与者 Q7 回答或金标。先验只负责相关性门控，参与者显著性负责候选排序：

\[
r_{ik}^{\mathrm{planner}}
= \tilde r_k(X_{Q7})
\left(0.30 + 0.70\,\frac{|v_{ik}-3.5|}{2.5}\right).
\]

Planner 前只呈现得分最高的 4 条自然语言 cue，同时仍保留静态 Full30。这样既不会丢失完整画像，又能明确告诉 Planner 当前 Q7 下哪些价值冲突对该参与者最显著。

### 9.3 Realizer step-specific V30

对冻结轨迹第 (t) 步，在全 30 维中计算：

\[
r_{ikt}^{\mathrm{realizer}}
= \tilde r_k(X_{Q7},m_{t-1},m_t,m_{t+1})
\left(0.30 + 0.70\,\frac{|v_{ik}-3.5|}{2.5}\right)
-0.22\,c_{ik,t-1},
\]

其中 (c_{ik,t-1}) 是该维度在此前步骤中的使用次数。每一步只选择前 4 条，重复惩罚降低机械复述，但不禁止真正核心的价值在多个步骤中持续出现。

R3 预生成审计结果：

| 审计项 | R2 | R3 |
|---|---:|---:|
| 候选池 | 10 | 30 |
| 每步 cue 数 | 8 | 4 |
| 同转移上下文跨用户 cue Jaccard | 0.903 | 0.279 |
| 同用户跨步骤 cue Jaccard | 0.757 | 0.174 |
| Planner active cue 组合数 | N/A | 89/162 |
| Planner cue 跨用户 Jaccard | N/A | 0.325 |

该审计只证明 selector 的区分度问题已被结构性修复，不证明 R3 生成效果提高。

## 10. R3 最小验证矩阵

先在冻结的 30 人 cohort 上运行四臂架构 pilot；通过门槛后再决定是否跑 162 人：

| 条件 | Planner/Realizer 调用 | Planner V30 | Realizer V30 | 目的 |
|---|---|---|---|---|
| J: joint-global4 | 一次 joint | Q7 active4 + Full30 | 同一 global4 | 效率基线 |
| S: split-global4 | Planner + 一次 Realizer | Q7 active4 + Full30 | 同一 global4 | 单独检验拆分调用 |
| D: split-dynamic4 | Planner + 一次 Realizer | Q7 active4 + Full30 | step-specific active4 | 检验动态加载增量 |
| D-shuffled | 与 D 相同 | 错配 active4/Full30 | 错配 step active4 | 检验真正的 profile matching |

主比较为 `S-J`、`D-S` 和 `D-D-shuffled`。不要用 `D-J` 同时为拆分和动态加载归因。

全量实验的最小模块矩阵随后收缩为：

| 条件 | 说明 |
|---|---|
| Direct baseline | 一次生成结构与文本 |
| Two-stage base | PI + 本人 Q2 + Q7 情境；Planner/Realizer 拆分 |
| + V30 | 静态 Full30 + Planner active4 + Realizer dynamic4 |
| + Shared FaRM | 完整主方法，对候选轨迹进行结构与个体匹配排序 |
| Shuffled V30 | 与完整方法等长的反事实诊断，不作为另一个主方法 |

近邻/远邻 Q2 不进入最终主方法矩阵；未经独立验证的规则不再作为 Prompt 文本模块。

## 11. 指标策略

不能因为 R2 的 KC-NES 或 BGE 结果不利而替换指标。R2 已报告指标必须保留；R3 可增加更贴合科学命题的指标，并在运行 R3 之前冻结定义：

1. **Trajectory fidelity**：KC-NES、State-JS、Transition F1，继续保留。
2. **Value faithfulness**：从输出中盲评核心取舍和不可接受边界是否与目标 V30 一致。
3. **Matched-profile advantage**：正确 V30 输出相对错配 V30 输出的配对优势；这是 R3 的主要个体化指标。
4. **Profile attribution**：给定生成回答和等长候选画像，冻结评估器或盲评者能否识别正确画像；随机水平必须明确。
5. **Anti-collapse with stability**：比较同一画像多次采样的组内差异和不同画像之间的组间差异，避免把随机多样性误当作个体化。
6. **Structural/faithfulness violations**：KC 单元语义、虚构已完成证据、长度和格式合规率。

在已经查看过 Q7 结果后新增的指标必须标记为 exploratory。若要作为 confirmatory 主结果，应在未用于 Prompt 调整的新留出题或新参与者数据上评测。
