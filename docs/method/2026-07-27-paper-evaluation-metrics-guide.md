# 最新版论文评测指标指南

> 更新时间：2026-07-27
> 范围：仅保留最近几天确定的 Q7 Seg2 主实验、Seg1/Seg3 辅助实验，以及近期 V7、V/V30 实验实际使用的指标。
> 不再收录旧版 11 指标体系、过往候选指标或训练损失大全。

## 1. 最终指标体系

本版把 **trajectory fidelity**、**trajectory personalization** 和
**anti-collapse** 分开。KC-NES 只回答预测链是否接近本人 gold；它不能单独排除一条群体高频链对很多人都取得中等分数。因而 Seg2 新增两个直接的序列个体化指标，并增加两个软坍缩诊断。所有新增指标都只使用已经生成的单次推理结果，不要求重复采样。

### 1.1 Seg2 主实验

| 层级 | 指标 | 含义 | 来源 | 方向 |
| --- | --- | --- | --- | --- |
| 轨迹保真度 | **KC-NES** | 生成 KC 序列与本人 gold KC 序列的编辑相似度 | 延续并修正旧 `macro-TED` | 越高越好 |
| 主要个体化指标 | **KC own-user AUC** | 本人 gold 链的相似度胜过随机他人 gold 链的概率，并列计 0.5 | 新增 | 越高越好；0.5 为无身份辨识力 |
| 个体化效应量 | **KC own-user margin** | 与本人 gold 链的 NES 减去与其他人 gold 链的平均 NES | 新增 | 越高越好；正值才支持本人优势 |
| 文本个体化 | **BGE own-user margin** | 与本人 gold 文本的相似度，减去与其他人 gold 的平均相似度 | 延续并改进旧 NL Specificity | 越高越好，正值才支持本人优势 |
| 关键次要指标 1 | **V 语义相似度** | 生成回答表达的价值寓意是否与本人 gold 中的价值寓意一致 | 新任务必须新增 | 越高越好 |
| 关键次要指标 2 | **V30 matched-minus-mismatched 契合度** | 生成的价值表达是否更符合本人 V30，而不是其他人的 V30 | 新任务必须新增 | 越高越好，正值表示本人画像优势 |
| 精确链坍缩诊断 | **Chain diversity** | 1 减去最常见完整 KC 链的占比 | 保留旧论文指标 | 越高表示精确重复越少，但不是准确性证据 |
| 软坍缩诊断 | **Pairwise KC diversity retention** | 预测链的平均跨人距离相对 gold 跨人距离保留多少 | 新增 | 1 表示总体差异幅度相当；0 表示完全坍缩 |
| 结构个体化诊断 | **Trajectory geometry correlation** | 预测链跨人距离结构是否重现 gold 的跨人距离结构 | 新增 | 越高越好；0 表示差异与本人结构无关 |

### 1.2 Seg1 辅助实验

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| **BGE own-user margin** | 生成的直觉反应是否更像对应用户，而不是其他用户 | 越高越好，正值才支持本人优势 |
| **人工直觉匹配度** | 主要担忧、第一反应和态度方向是否与本人 gold 一致 | 越高越好 |

### 1.3 Seg3 辅助实验

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| **BGE own-user margin** | 生成的最终底线是否更像对应用户，而不是其他用户 | 越高越好，正值才支持本人优势 |
| **人工底线匹配度** | 底线、妥协方式和价值方向是否与本人 gold 一致 | 越高越好 |
| **Criterion uniqueness** | 不同用户的最终底线是否具有差异性，防止全部生成同一底线 | 越高表示差异越大，但不是匹配正确性的证据 |

## 2. 主要指标

### 2.1 KC-NES

#### 名称

KC Normalized Edit Similarity，KC 归一化编辑相似度。

#### 计算方法

对人工 gold 序列删除 `V` 并保持其余状态顺序；对 Planner 输出删除控制符 `STOP`。设过滤后的 gold 为 $y_u$，预测序列为 $\hat y_u$：

$$
\operatorname{KC\text{-}NES}_u
=1-\frac{\operatorname{Lev}(\hat y_u,y_u)}
{\max(|\hat y_u|,|y_u|,1)}.
$$

其中 Levenshtein distance 的插入、删除和替换成本均为 1。系统结果为参与者均值：

$$
\overline{\operatorname{KC\text{-}NES}}
=\frac{1}{N}\sum_{u=1}^{N}\operatorname{KC\text{-}NES}_u.
$$

#### 怎么看

- 范围为 $[0,1]$；`1` 表示序列完全一致；
- 同时惩罚状态错误、顺序错误和长度错误；
- 它回答“生成的推理骨架与本人 gold 有多接近”；
- 短序列上一个编辑会造成明显降分；必须同时看 KC own-user AUC/margin，避免把押中群体众数误写成个体化。

#### 与旧指标的关系

旧 `macro-TED` 实际计算的是按首状态类别平衡的 raw Levenshtein count，不是 tree edit distance，也没有归一化。KC-NES 修正了名称和量纲，能够在不同序列长度之间比较。

### 2.2 KC Own-User Margin 与 KC Own-User AUC

设预测链 $\hat y_u$ 与任意参与者 $v$ 的 gold 链之间的相似度为

$$
q^{KC}_{uv}=\operatorname{NES}(\hat y_u,y_v).
$$

序列层 own-user margin 定义为

$$
m_u^{KC}=q^{KC}_{uu}-\frac{1}{N-1}\sum_{v\ne u}q^{KC}_{uv}.
$$

它控制了“这条链是否也同样适合人群中其他人”。群体众数链即使对本人有中等
KC-NES，只要对其他人同样相似，其 margin 仍接近 0。

由于 KC 序列短、状态集合离散，gold 链之间存在大量并列。主个体化指标因此采用带并列校正的 pairwise AUC：

$$
\operatorname{AUC}^{KC}_u=\frac{1}{N-1}\sum_{v\ne u}
\left[
\mathbb I(q^{KC}_{uu}>q^{KC}_{uv})
+\frac12\mathbb I(q^{KC}_{uu}=q^{KC}_{uv})
\right].
$$

系统结果均先得到参与者级分数，再对参与者求均值。

- `KC own-user AUC = 0.5`：本人链不比随机他人链更可辨识；
- `> 0.5`：预测链对本人具有排序优势；
- `< 0.5`：预测链更像其他人的 gold；
- margin 报告优势的绝对大小，AUC 报告不依赖相似度刻度的辨识概率，二者不能互相替代。

无效预测按空链处理：对所有非空 gold 的 NES 均为 0，因此 margin 为 0、AUC 为 0.5；同时由 KC-NES=0 保留生成失败惩罚。这样不会通过删除失败样本虚增个体化。

### 2.3 BGE Own-User Margin

#### 计算方法

设生成文本为 $\hat Y_u$，本人 gold 为 $Y_u$，冻结的 BGE-M3 编码器为 $e(\cdot)$：

$$
s_u^{own}=\cos(e(\hat Y_u),e(Y_u)),
$$

$$
s_u^{other}=\frac{1}{N-1}\sum_{v\ne u}
\cos(e(\hat Y_u),e(Y_v)),
$$

$$
m_u=s_u^{own}-s_u^{other}.
$$

系统结果为 $\bar m=N^{-1}\sum_um_u$。Seg1、Seg2、Seg3 分别在各自 segment 内计算，不能混在一起。

冻结实现使用 `BAAI/bge-m3` dense embedding、1024 维、最大长度 8192、FP32、L2 normalization。

#### 怎么看

- $m>0$：生成文本比其他人的 gold 更接近本人 gold；
- $m\approx0$：文本可能合理，但同样适合很多人，不能证明个体化；
- $m<0$：生成文本平均更像其他人的回答；
- 应同时报告 `own similarity` 和 `other-user mean similarity`，便于区分“整体语义差”与“回答过于通用”。

#### 与旧指标的关系

旧 NL Specificity 使用 `own cosine / other cosine`。新版改用差值，避免分母较小造成比值不稳定，并使用冻结 BGE-M3 替代旧 embedding。

## 3. Seg2 关键次要指标

### 3.1 V 语义相似度

#### 评测对象

这里的 `V` 指完整回答中自然表达出的价值观点或价值寓意，不是 Planner 的 KC 状态。Planner 不预测 `V`，因此不存在所谓“Planner V-F1”。

在预测冻结后，由独立盲标流程从 human gold 和 generated response 中提取 `V` spans。分别按原文顺序拼接为 $\mathcal V_u^{gold}$ 和 $\mathcal V_u^{pred}$。

#### 自动语义分数

只在 gold 与 predicted `V` 都非空时计算：

$$
S_u^V=\cos\left(e(\mathcal V_u^{pred}),e(\mathcal V_u^{gold})\right).
$$

越高表示生成文本与本人 gold 表达了更相近的价值寓意。

#### 必须同时报告的支持项

1. `V-presence` precision、recall、F1；
2. gold/predicted `V` 是否为空的四格样本数；
3. conditional semantic similarity 的有效参与者数；
4. 可选的 own-user V semantic margin；
5. 两名盲评员的 1--5 分价值寓意一致性及评审一致性。

不能把 human 和 generated 都为空记为相似度 1。不同系统的 predicted-`V` 非空 cohort 可能不同，因此 conditional mean 不能直接作公平的系统配对检验。

#### 固定 cohort 的 coverage-adjusted 版本

在 gold-`V` 非空的固定参与者 cohort 上：

$$
CAS_u^V=\mathbb 1[\mathcal V_u^{pred}\ne\varnothing]
\max\left(0,S_u^V\right).
$$

predicted `V` 为空时记 0。跨系统比较优先使用这个固定 cohort 指标；原始 conditional score 只作系统内部描述。

#### 怎么看

- 高 `V-presence recall`、低语义相似度：模型会表达价值，但方向不对；
- 高条件语义相似度、低 coverage：模型只在少数容易样本上表达正确；
- coverage-adjusted score 高：既愿意表达，也与本人 gold 的价值寓意相近。

### 3.2 V30 Matched-Minus-Mismatched 契合度

#### 评测方法

只对 generated `V` 非空的回答进行。盲评员看到公共情境、匿名 generated `V` spans 和一个匿名 V30 profile，不看到系统名、参与者身份、完整 gold 或 KC 序列。

每位参与者有：

- 1 个本人真实 V30 profile，即 matched profile；
- 3 个同一 outer-test fold 内、由固定 seed 产生的无固定点 mismatched profiles；
- 两名评审分别给 1--5 分“价值表达与该 V30 profile 的契合度”。

对参与者 $u$：

$$
\Delta_u^{V30}
=r_u^{matched}-\frac{1}{3}\sum_{k=1}^{3}r_{uk}^{mismatched}.
$$

系统结果为参与者均值，并用 participant bootstrap 给出 95% CI。

#### 怎么看

- $\Delta^{V30}>0$：价值表达更符合本人画像，而不是随机他人画像；
- $\Delta^{V30}\approx0$：表达内容可能符合人群中的常见价值，缺乏本人特异性；
- $\Delta^{V30}<0$：表达与本人画像的契合度反而更差；
- 只有 95% CI 下界大于 0，才构成稳定的本人画像契合证据。

predicted `V` 为空时原始契合度为 N/A，不能伪造为真实的 0 分。可另做 coverage-adjusted 敏感性分析：在固定全 cohort 上把 omission 的 matched-minus-mismatched 差值记为 0，同时保留原始 N/A 和有效 $n$。

### 3.3 V30 序列反事实增益（GPT-4o 消融）

R4 已有 `aligned_full` 与 `aligned_shuffled` 两套单次推理。后者保持目标参与者的情境、Q2 历史、Q7 ICL、解码参数和 record seed 不变，只将 Full30 与 active4 一起替换为无固定点的他人画像 $V_{\pi(u)}$。因此可计算：

$$
\Delta^{CF}_{NES,u}=
\operatorname{NES}(\hat y_u(V_u),y_u)-
\operatorname{NES}(\hat y_u(V_{\pi(u)}),y_u),
$$

并以相同方式计算 $\Delta^{CF}_{margin,u}$ 和 $\Delta^{CF}_{AUC,u}$。
三项均按参与者配对 bootstrap。另报告 sequence change rate，但它只是画像敏感性：只有变化同时带来正的 fidelity 或 personalization 增益，才说明正确 V30 产生了有方向的个体化作用。

这一反事实只在 GPT-4o 消融中存在，不能伪装成六模型共有指标。

## 4. 诊断指标

### 4.1 Chain Diversity

设最终生成的 KC 序列为 $\hat y_u$，最常见序列的参与者占比为

$$
\operatorname{TopShare}
=\max_s\frac{\#\{u:\hat y_u=s\}}{N}.
$$

则

$$
\operatorname{ChainDiversity}=1-\operatorname{TopShare}.
$$

#### 怎么看

- 越接近 0，越说明所有用户得到同一条序列；
- 越高，说明序列输出更有差异；
- 它只能诊断 mode collapse，不能证明生成结果与本人匹配；
- 主表或附表还应给出 unique sequence count，避免只看一个比例。

### 4.2 Pairwise KC Diversity Retention

精确字符串不同不代表推理结构真正不同。定义跨参与者平均 KC 距离：

$$
D_{pred}=\frac{2}{N(N-1)}\sum_{u<v}
\left[1-\operatorname{NES}(\hat y_u,\hat y_v)\right],
$$

并用相同方法计算 $D_{gold}$。软多样性保留率为

$$
\operatorname{KCDR}=\frac{D_{pred}}{D_{gold}}.
$$

- `0` 表示所有预测链相同；
- `1` 表示预测链总体差异幅度与 human gold 相当；
- `>1` 表示比 gold 更分散，不自动代表更好；
- 必须同时报告 $D_{pred}$ 与 $D_{gold}$，不能把比率单独解释为准确性。

### 4.3 Trajectory Geometry Correlation

对所有参与者对 $(u,v)$，分别计算预测距离
$d^{pred}_{uv}=1-\operatorname{NES}(\hat y_u,\hat y_v)$ 与 gold 距离
$d^{gold}_{uv}=1-\operatorname{NES}(y_u,y_v)$。定义：

$$
\operatorname{TGC}=\rho_{Spearman}
\left(\{d^{pred}_{uv}\}_{u<v},\{d^{gold}_{uv}\}_{u<v}\right),
$$

采用 average-rank 处理并列。若预测距离为常数（完全坍缩），TGC 定义为 0。
KCDR 只检查“差异有多大”，TGC 进一步检查“哪些人应该相似、哪些人应该不同”是否与 gold 一致。TGC 是 cohort-level 诊断，参与者对不是独立样本，不能把 $N(N-1)/2$ 当作统计样本量。

### 4.4 Criterion Uniqueness

设参与者 $u$ 的 Seg3 最终底线文本为 $C_u$：

$$
\operatorname{CriterionUniqueness}
=1-\frac{2}{N(N-1)}\sum_{u<v}
\cos(e(C_u),e(C_v)).
$$

#### 怎么看

- 越高表示不同参与者的底线文本差异越大；
- 越低表示大家可能生成了相同或高度模板化的底线；
- 它不检查底线是否正确匹配本人，因此必须与 Seg3 BGE own-user margin 和人工底线匹配度一起看。

## 5. Seg1 与 Seg3 人工评测

### 5.1 人工直觉匹配度

盲评界面展示公共 Seg1 问题、本人 gold Seg1 和匿名生成 Seg1。两名评审以 1--5 分评价：

> 生成回答的第一反应、主要担忧和态度方向与该参与者 gold 的一致程度。

每位参与者先对两名评审取均值：

$$
I_u=\frac{r_{u1}+r_{u2}}{2},
$$

再在参与者之间求均值和 bootstrap CI。越高表示直觉方向越像本人。

### 5.2 人工底线匹配度

盲评界面展示公共 Seg3 问题、本人 gold Seg3 和匿名生成 Seg3。两名评审以 1--5 分评价：

> 生成回答的核心底线、可接受妥协及价值方向与该参与者 gold 的一致程度。

聚合方法与 Seg1 相同。越高表示最终决策底线越像本人。

### 5.3 评审一致性

两个 1--5 分指标均应报告：

- quadratic weighted Cohen's kappa；
- ICC(2,1)，即双向随机效应、absolute-agreement、single-measure ICC；
- 有效参与者数。

一致性指标只说明评审是否稳定，不说明模型表现好坏。

## 6. 近期 V7 与 V/V30 实验盘点

本节用于回答“最近到底跑了哪些指标”。它们不全部属于最终论文主指标。

### 6.1 V7 数据合成与数据质量验证

最新可见产物：

- `synthesis_validation_v7_final_polished_v1.json`：162/162 通过；
- `persona_validation_v7_full_normalized.json`：153 通过、8 失败；
- V13.9 严格有效子集：134 位参与者、967 个 steps、六个 KC 状态全部覆盖；
- `q7_gold_used=false`、`pi_used=false`。

实际统计的项目包括：

| 类型 | 字段/指标 | 当前用途 |
| --- | --- | --- |
| 数据完整性 | validation pass/fail count | 检查合成记录是否满足 schema 和文本合同 |
| 数据覆盖 | participant count、step count | 描述有效训练数据规模 |
| 状态覆盖 | 六状态各自 record count、six-state coverage | 防止某状态没有训练/验证样本 |
| 选择器响应性 | role-responsive participants、transition-responsive participants | 检查 active V30 cue selector 是否随 segment role/KC transition 改变 |
| 选择器多样性 | unique active item sets | 检查所有记录是否使用同一组 V30 cues |
| 泄漏审计 | Q7 gold used、PI used、source anchor visible | 数据合同硬检查 |

这些是数据集质量指标，不是 Seg1/2/3 的模型效果指标。

### 6.2 V13.9 Step Realizer Smoke

已实际运行两条 arm：`V30-only` 与 `V30+aux`。两者最终均为 `METHOD_GATE_FAIL / NOT_PAPER_READY`。

实际报告了 7 个开发指标：

| 指标 | V30-only | V30+aux | 要求 | 解释 |
| --- | ---: | ---: | ---: | --- |
| V30 counterfactual coverage | 1.000 | 1.000 | $\ge0.90$ | 是否能为 held-out step 构造合法他人 V30 对照 |
| State permutation NLL increase | -0.0275 | -0.0300 | $>0$ | 换成错误状态后 target NLL 是否升高；当前为负，说明未学会状态条件 |
| V30 permutation NLL increase | -0.0117 | -0.0188 | $>0$ | 换成他人 V30 后 target NLL 是否升高；当前为负，说明未学会本人 V30 条件 |
| Exact output top-share | 0.1250 | 0.0417 | $<0.25$ | 最常见完整 step 输出占比 |
| CtoC per-state exact-output top-share | 1.000 | 1.000 | $<0.40$ | CtoC 内部输出完全坍缩；该轮还存在旧 file-prefix 采样缺陷 |
| Structural failure rate | 0.375 | 0.417 | $=0$ | 长度、禁用标签或结构规则失败比例 |
| Template risk rate | 0.000 | 0.000 | $\le0.20$ | 是否命中冻结通用模板短语 |

结论：V13.9 smoke 只可作为失败诊断，不进入论文主结果。后续虽然修复了 45-token cap 和六状态 round-robin 评测，但尚未重新训练。

### 6.3 Q7 V 语义与 V30 契合度

最新版论文计划使用：

1. `V-presence` precision/recall/F1；
2. V span BGE semantic similarity；
3. coverage-adjusted V semantic score；
4. 可选 own-user V semantic margin；
5. 人工 V 语义一致性；
6. V30 matched-minus-mismatched 契合度；
7. weighted kappa、ICC(2,1) 和有效 $n$。

**当前状态：尚无可引用的完整正式数值。** 已启动的 Q7 V30-expression formal run 因 probe 输出解析失败而无效：162 位 human 中只成功解析 39 条，generated 只成功 363 条，failure 1704 条；每个系统条件只保留 4--13 个实例。该次 `summary.json` 的分数禁止用于论文。

### 6.4 自动 V30 Expression Probe

近期另外实现了一个自动三分类 probe，把 30 个 V30 维度标成 `agree / disagree / not_expressed`。它定义了 4 组自动指标：

| 指标组 | 含义 | 是否为最终主表指标 |
| --- | --- | --- |
| Human Expression Precision/Recall/F1 | 生成文本与 human Q7 是否表达相同 V-item 和相同极性 | 否；可作 V 语义的补充自动指标 |
| Profile Polarity Precision | 生成 claim 的极性是否与本人二值 V30 一致 | 否；补充诊断 |
| Expression Coverage | 生成文本表达的非中性 V30 维度数除以 30 | 否；描述指标 |
| Own-Profile Advantage | 对本人 V30 的 alignment 减去对其他人 V30 的平均 alignment | 否；补充个体化诊断 |

这 4 组定义和单元测试已经完成，但正式 Q7 run 当前无效，因此不能报告其结果。它们不能替代最终的人工 V30 matched-minus-mismatched 指标。

### 6.5 Q1--Q6 V 表达证据与 Salience 资源

近期 option-grounded API 标注实际完成：

- 667/667 eligible records 完成，failures=0；
- 824 个不满足证据合同的 claims 被拒绝；
- 每条记录保留 0/1/2/3 个证据的数量分别为 454/137/49/27；
- 覆盖 162 位参与者、667 条 annotation；
- 与 observed V30 极性 aligned 188 条、opposing 128 条；
- `q7_gold_used=false`、`q7_text_used=false`、`pi_used=false`。

这些数字描述的是 V-expression evidence 数据资源和过滤质量，不是最终系统在 Q7 上的效果。当前语义被限定为 **history salience**，不能把 aligned/opposing 数直接写成模型的 V30 准确率。

### 6.6 Q2 V30 Prompt Ablation

已完成一个 3 人的小型 prompt ablation，条件为：

- `full_v30`；
- `full_v30_plus_active`；
- `active_only`。

每次选择 8 个 active cues，且不使用 Q7/PI。当前 status 只有运行完成与样本元数据，没有冻结的量化效果 summary，因此不能作为论文指标结果。

## 7. 哪些进入论文，哪些不进入

| 内容 | 论文地位 |
| --- | --- |
| Seg2 KC-NES | 轨迹保真度主指标 |
| Seg2 KC own-user AUC、KC own-user margin | 序列个体化主指标与效应量 |
| Seg2 BGE own-user margin | 文本个体化指标 |
| Seg2 V 语义相似度、V30 matched-minus-mismatched | 两个关键次要指标 |
| Seg2 Chain diversity、KCDR、TGC | 精确坍缩、软坍缩和结构对应诊断 |
| GPT-4o full-minus-shuffled V30 | 画像因果敏感性的消融证据 |
| Seg1 BGE margin、人工直觉匹配度 | 独立辅助实验 |
| Seg3 BGE margin、人工底线匹配度、Criterion uniqueness | 独立辅助实验 |
| V7 数据 pass/coverage/leakage | 数据质量与方法附录 |
| V13.9 Step Realizer 7 个 gate | 失败分析或开发附录，不进入主效果表 |
| 自动 V30 Expression Probe 4 组指标 | evaluator 修复并完成全 cohort 后才可作补充结果 |
| Q1--Q6 salience 统计 | 数据资源说明，不是 Q7 模型效果 |
| Q2 三人 prompt ablation | pilot，不作论文量化结论 |

## 8. 统计报告规则

1. 统计单位始终是参与者，不是句子、step、rating 或 mismatched pair。
2. 主指标和系统差异使用 participant-level paired bootstrap，10,000 次，报告 95% percentile CI。
3. 多个 seed 先在参与者内聚合，不能把同一参与者的多个 seed 当作独立样本。
4. Seg1、Seg2、Seg3 分开报告，不能合并为一个 BGE margin。
5. V 语义指标必须报告有效 $n$ 和 coverage，不能只报条件均值。
6. 人工指标同时报告 weighted kappa、ICC(2,1) 和有效 $n$。
7. Chain diversity 与 Criterion uniqueness 只能说明“没有全部生成一样”，不能单独支持“个体化正确”。
8. KC own-user AUC 的无辨识基线是 0.5，KC own-user margin 的无优势基线是 0；两者都报告 participant bootstrap CI。
9. TGC 和 KCDR 是 cohort-level 统计量，不把参与者对当独立观测；本轮不报告基于重复采样的稳定性指标，因为没有重复推理数据。

## 9. 推荐论文结果表

### 9.1 Seg2 主表

| System | KC-NES ↑ | KC own-user AUC ↑ | KC own-user margin ↑ | BGE own-user margin ↑ | V semantic ↑ | V30 fit Δ ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Seg2 诊断附表：

| System | Unique chains | Top share ↓ | $D_{pred}$ ↑ | KCDR (target 1) | TGC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |

GPT-4o 消融表另报 `full - shuffled V30` 的 $\Delta KC$-NES、$\Delta KC$ margin、$\Delta KC$ AUC 和 sequence change rate。

### 9.2 Seg1 辅助表

| System | BGE own-user margin ↑ | Human intuition match ↑ | Effective n |
| --- | ---: | ---: | ---: |

### 9.3 Seg3 辅助表

| System | BGE own-user margin ↑ | Human criterion match ↑ | Criterion uniqueness ↑ | Effective n |
| --- | ---: | ---: | ---: | ---: |

## 10. 实现与证据位置

| 内容 | 路径 |
| --- | --- |
| KC-NES | `.codex-staging/route-a-v11/route_a/evaluation/state_metrics.py` |
| KC own-user margin、AUC、KCDR、TGC | `tools/prompt_baselines/evaluate_main_method_r2.py` |
| BGE own-user margin | `.codex-staging/route-a-v11/route_a/evaluation/text_metrics.py` |
| 人工评分一致性 | `.codex-staging/route-a-v11/route_a/evaluation/judge_metrics.py` |
| V 语义与 V30 人工协议 | `docs/superpowers/specs/2026-07-19-v30-route-a-complete-technical-design.md` 第 17.7 节 |
| 自动 V30 expression metrics | `.codex-staging/realizer-v13-6/route_a/evaluation/value_expression.py` |
| Q7 V30 evaluator 当前无效状态 | `docs/handoff/2026-07-26-q7-v30-expression-evaluator-handoff.md` |
| V7 / V13.9 smoke 结果 | `outputs/realizer_v13_9/progress/paper_readiness_r5.md` |
| V7 数据 manifest | `outputs/realizer_v13_9/step_data_valid_r2/manifest.json` |
| Q1--Q6 V evidence pilot | `docs/audits/2026-07-26-q1-q6-expression-evidence-pilot.md` |
