# Q3–Q6 V30→KC 偏好规则参数实验（V1）

日期：2026-07-27  
状态：`DEVELOPMENT / NOT VALIDATED`

## 1. 实验问题

本实验不使用旧 CAPS 四元组，也不使用旧 PKVC 状态。它仅使用最新 `choice_state_mapping_v2` 下的 Q3–Q6 选择标签，研究两个问题：

1. 适度放宽发现阶段参数，能否让更多 V30 维度进入候选规则池；
2. 候选池扩大后，每个情境最多注入两条规则，能否保持短 Prompt 并提高参与者间的规则组合差异。

Q3–Q6 每题只提供一个选择状态，因此这里发现的是“情境条件化状态偏好”，不是相邻状态转移：

\[
(Q_q,\ \text{V30 pole})\Rightarrow\text{preferred KC state}.
\]

目标状态仅允许 `K, C, KtoC, CtoK, KtoK, CtoC`。Q7/Q8 被 fail-closed 禁止进入发现与调参。

## 2. 数据

| 问题 | 可用参与者 | v2 KC 状态 |
|---|---:|---|
| Q3 | 162 | CtoC, CtoK, KtoC |
| Q4 | 162 | CtoC, KtoC, KtoK |
| Q5 | 162 | C, CtoC, CtoK, KtoK |
| Q6 | 137 | CtoC, KtoC |

Q6 的 25 个 `V` 标签按冻结 ontology 过滤，不重写为其他 KC 状态。

## 3. 发现与筛选

每题分别拟合 30 维 V30 的 L1 one-vs-rest logistic 模型。该模块与 Q2
单双维规则共享 `participant_resampling_v1`：对参与者进行 1,000 次
bootstrap、1,000 次 participant permutation，并使用五折
participant-held-out 检验。bootstrap 记录系数、选中率与符号一致率；
每条候选还必须通过单维增量检验：

正式复现环境固定为 Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1 和
scikit-learn 1.8.0；版本不一致时脚本 fail closed。

\[
\Delta\mathcal L=
\mathcal L(\text{intercept + V item})-
\mathcal L(\text{intercept})<0.
\]

至少 3/5 折改善，且 held-out 系数方向一致率至少为 0.60。负系数不会被解释为“负向 prefer”；它被翻转为对应低分端触发目标状态。例如 \(\beta<0\) 编译为 `score<=3` 的低分立场触发。

另对全部“问题×目标状态×V维度”候选进行 1,000 次 participant
permutation，并统一作 BH-FDR 校正。

## 4. 参数臂

| 参数臂 | 正则 C | selection | sign | \(|\beta|\) | held-out 后规则 | V 覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| `cv_strict` | 每题 CV 最优 | .90 | .95 | .15 | 0 | 0/30 |
| `c03_strict` | .30 | .90 | .95 | .15 | 20 | 11/30 |
| **`c03_balanced`** | **.30** | **.80** | **.90** | **.10** | **29** | **16/30** |
| `c03_wide` | .30 | .70 | .85 | .08 | 38 | 19/30 |
| `c10_strict` | 1.00 | .90 | .95 | .15 | 40 | 20/30 |

CV 选择的 C 为 Q3/Q4/Q5=`0.1`、Q6=`0.03`。提高到 C=1.0 虽扩大覆盖，但各题完整模型的 CV log-loss 明显恶化，因此 `c10_strict` 只用于展示过拟合侧的覆盖上限。

`c03_balanced` 覆盖：

`V01, V04, V06, V07, V08, V11, V14, V15, V17, V20, V21, V22, V24, V27, V29, V30`。

其 29 条规则按来源分布为 Q3=3、Q4=5、Q5=13、Q6=8；按目标分布为 C=1、CtoC=7、CtoK=6、KtoC=8、KtoK=7。

## 5. Top-2 窄注入实验

每位参与者在每个来源情境中只保留最多两条、且 V 维度不重复的规则。比较两种二级排序：

- `evidence_first`：held-out 增益优先，本人 V 极性强度次之；
- `profile_strength_first`：本人 V 极性强度优先，held-out 增益次之。

两者都必须先通过候选规则的统计门槛；运行时还需在它们之前增加目标场景语义相关性门控。

| 参数臂 | 排序 | 平均注入数 | 无规则情境 | 实际使用 V 数 | 162 人中的不同组合 |
|---|---|---:|---:|---:|---:|
| c03_strict | evidence first | 1.44 | 15.6% | 9 | 95 |
| c03_strict | profile first | 1.44 | 15.6% | 11 | 149 |
| **c03_balanced** | evidence first | 1.79 | 1.7% | 12 | 124 |
| **c03_balanced** | **profile first** | **1.79** | **1.7%** | **16** | **160** |
| c03_wide | evidence first | 1.98 | 0% | 14 | 118 |
| c03_wide | profile first | 1.98 | 0% | 19 | 162 |
| c10_strict | profile first | 1.89 | 0.8% | 20 | 162 |

扩大候选池后若仍按全局证据排序，强规则会反复占据 top-2，新增 V 维度不会真正进入 Prompt。`profile_strength_first` 才落实了“根据本人 V30 收窄注入”，并且不增加注入条数。

## 6. 统计结论

所有参数臂的最小 BH-FDR 为 `q=0.35964036`，没有规则达到
`q<=0.10`。因此：

- 可以把 `c03_balanced` 的 29 条规则作为开发期候选池；
- 不能称其为已验证 ontology；
- `c03_wide` 与 `c10_strict` 不应仅凭更高覆盖被选为主配置；
- 最终是否采用必须由 Q7 Seg2 的受控注入消融决定。

## 7. 推荐的运行时策略

推荐顺序为：

1. 目标场景与来源冲突语义相关；
2. 参与者的 V30 pole 激活该规则；
3. 按本人极性强度排序；
4. 用 held-out log-loss 增益与 bootstrap 稳定性破同分；
5. 每一步/情境最多注入两条不同 V 维规则。

规则应作为软偏好：

\[
(\text{semantic relevance},\ \text{V30 pole})
\Rightarrow
(\text{preferred KC operation},\ \text{guard}).
\]

Q3–Q6 只有单状态标签，不能凭空补写 `current_state→next_state`。若需逐步路由，应与 Q2 提取的转移规则组合：Q3–Q6 提供情境状态 prior，Q2 提供当前状态条件下的 transition rule。

## 8. 下一项必要实验

在同一批 Q7 Seg2 输入、相同模型和 decoding 下比较：

1. 无 Q3–Q6 preference rule；
2. `c03_balanced + evidence_first + top2`；
3. `c03_balanced + profile_strength_first + top2`；
4. 第 3 组使用 shuffled V30；
5. 可选 top1 与 top2 注入上限消融。

只有 correct V30 同时优于无规则和 shuffled V30，且 `KC-NES`、`BGE own-user margin` 不出现结构-个体化反向交换，才能把该层升级为主方法组件。

## 9. 复现产物

- 脚本：`tools/kc_aux_preference_sweep.py`
- 参数与结果：`artifacts/kc_aux_preference_sweep_v1_uniform/sweep_report.json`
- 各参数臂规则：`artifacts/kc_aux_preference_sweep_v1_uniform/rules_by_arm.json`
- 各正则强度完整候选：`candidates_cv.csv`, `candidates_0p3.csv`, `candidates_1p0.csv`

这些产物记录输入 SHA，且声明 `q7_used=false`。
