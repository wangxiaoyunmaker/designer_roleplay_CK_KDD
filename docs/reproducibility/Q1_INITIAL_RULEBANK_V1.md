# Q1 初始反应个体化规则候选库（V1）

## 1. 范围

`Q1_INIT_RULEBANK_V1_CANDIDATE` 只用于 Q7 Seg1 初始直觉生成。它不属于
已冻结的 90 条 `KC_RULEBANK_V1`，也不得注入 Seg2。

规则语义为：

\[
(\text{V30 profile condition})
\Rightarrow
(\text{preferred initial KC operation}).
\]

Q1 没有当前 KC 状态，因此规则中禁止出现 `current_state -> next_state`。目标只
表示第一反应的组织方式，并且不得要求模型虚构事实、经历或外部证据。

## 2. 数据与状态

输入为162人的V30以及 `choice_state_mapping_v2` 下的Q1选择。148人具有有效
六状态KC标签，14个 `V` 或缺失标签按冻结ontology排除：

| Q1状态 | 人数 |
|---|---:|
| `CtoC` | 66 |
| `CtoK` | 23 |
| `KtoK` | 59 |

Q7/Q8被输入加载器显式禁止，未用于发现、筛选或参数选择。

## 3. 单维主效应规则

固定沿用Q3–Q6推荐参数，不针对Q1结果重新调参：`C=0.3`、selection
frequency至少0.80、sign consistency至少0.90、标准化系数绝对值至少0.10、
目标支持至少15人。候选还须满足五折participant-held-out增量log-loss小于0，
至少3/5折改善，且方向一致率至少0.60。

负系数按低分极性编译，不会被改写为高分端的正向prefer规则。

## 4. 双维交互规则

先仅从V30发现12条稳定维度边，再构造高高、低低、张力、优势与平衡关系谓词。
对每个Q1目标状态，基线模型包含两个单维主效应，关系模型额外加入关系谓词：

\[
\operatorname{logit}P(Y=s)=
\alpha+\beta_jv_j+\beta_kv_k+\theta R_{jk}.
\]

因此双维规则表示超出两个单维分数的增量关系。只有正向残差效应、bootstrap
符号一致率至少0.90、held-out loss改善且至少3/5折改善的候选可编译为prefer。

## 5. 统一采样协议

使用 `participant_resampling_v1`：

- participant bootstrap：1,000次；
- participant permutation：1,000次；
- participant-held-out：5折；
- 随机种子：20260727；
- Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.8.0。

## 6. 正式提取结果

| 层 | 候选 | 保留规则 | 最小BH-FDR q | validated |
|---|---:|---:|---:|---:|
| 单维主效应 | 90 | 6 | 0.98701299 | 0 |
| 双维交互 | 192 | 12 | 0.575424 | 0 |
| **合计** | **282** | **18** | **0.575424** | **0** |

目标状态分布：单维为 `CtoC=1, CtoK=1, KtoK=4`；双维为
`CtoC=2, CtoK=3, KtoK=7`。规则明显偏向 `KtoK`，与Q1标签只覆盖三个状态
以及样本分布有关，不能解释为一般性的认知规律。

在162份V30画像上，单维层和双维层均有至少一条可激活规则，参与者覆盖均为
162/162。限流前每人平均激活3.31条单维规则和5.88条双维规则，因此运行时必须
执行窄路由，不能把所有激活规则同时写入Prompt。

## 7. 使用与评估

规则库当前状态为 `EXPLORATORY_NOT_FROZEN`。默认不注入；Seg1消融必须显式
允许 `stable_exploratory`，并比较：无Q1规则、单维、双维、单双维、shuffled
V30与correct V30。主要报告Seg1 `BGE own-user margin`和人工直觉匹配度。

只有correct V30优于无规则及shuffled V30时，才考虑把该候选库冻结为独立的
Seg1方法组件。不得依据Seg1/Q7结果修改这18条后再在同一数据上报告确认性结果。

## 8. 产物

- 提取器：`tools/q1_initial_rule_ontology.py`
- 规则库：`artifacts/q1_initial_rulebank_v1_candidate/rules.json`
- 审计：`artifacts/q1_initial_rulebank_v1_candidate/audit.json`
- 单维候选：`artifacts/q1_initial_rulebank_v1_candidate/single_candidates.csv`
- 双维候选：`artifacts/q1_initial_rulebank_v1_candidate/double_candidates.jsonl`
