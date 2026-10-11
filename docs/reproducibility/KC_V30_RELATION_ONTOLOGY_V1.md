# V30 内部关系驱动的 KC 操作规则提取方法（V1）

## 1. 研究目标

本方法从参与者的 30 维价值画像（V30）及其 Q2 思考轨迹中，提取可直接用于 Q7 推理的个体化软路由规则。规则不是对亚群的事后描述，而是一个可执行映射：

\[
(R_{jk}(\mathbf v_i)=1,\ z_t=s)
\Rightarrow
(z_{t+1}=y,\ \text{semantic operation},\ \text{guard}).
\]

其中，\(R_{jk}\) 是 V30 中两个维度的内部关系，\(z_t\) 仅取六种 KC 状态：

`K, C, KtoC, CtoK, KtoK, CtoC`。

`START` 与 `STOP` 只作为序列边界标记。规则发现不读取 Q7 gold，Q7 场景语义只在推理时用作适用性 guard。

## 2. 输入与分析单位

- V30：每位参与者的 30 个 Likert 分数；
- Q2：每位参与者按时间排序的 KC 状态序列；
- 分析单位：给定当前状态 \(s\) 时的下一状态 \(y\)；
- 划分单位：参与者，而不是单条转移，防止同一参与者同时进入训练折与测试折。

脚本首先将 Q2 序列转换为相邻转移，保留重复状态，并加入 `START`/`STOP` 边界转移。输入加载器采用 fail-closed 策略：一旦发现 Q2 之外的题号、非法 KC 状态、缺失 V30 或重复参与者记录即终止。

## 3. V30 内部关系发现

### 3.1 稳定维度边

对任意维度对 \((j,k)\)，计算 Spearman 相关：

\[
\rho_{jk}=\operatorname{Corr}(\operatorname{rank}(v_j),
                              \operatorname{rank}(v_k)).
\]

以参与者为单位进行 bootstrap。维度边需同时满足：

- \(|\rho_{jk}|\ge 0.20\)；
- bootstrap 符号一致率至少为 0.90；
- 95% bootstrap 区间不跨 0。

按 \(|\rho|\) 排序后仅保留预设数量的边。该步骤只使用 V30，不使用任何 Q2 转移结果。

### 3.2 可解释关系谓词

对每条稳定边构造七类二元谓词：

| 谓词 | 计算定义 |
|---|---|
| `synergy_high` | \(v_j\ge4 \land v_k\ge4\) |
| `synergy_low` | \(v_j\le3 \land v_k\le3\) |
| `tension_left_high` | \(v_j\ge4 \land v_k\le3\) |
| `tension_right_high` | \(v_j\le3 \land v_k\ge4\) |
| `left_dominant` | \(v_j-v_k\ge2\) |
| `right_dominant` | \(v_k-v_j\ge2\) |
| `balanced` | \(|v_j-v_k|\le1\) |

激活组与未激活组均须有至少 20 位参与者。谓词表达的是 V30 内部配置，而不是两个独立单维规则的简单并列。

## 4. 相对单维主效应的增量检验

对每个当前状态 \(s\)、候选下一状态 \(y\) 和关系谓词 \(R_{jk}\)，拟合 one-vs-rest logistic 模型。基线模型已包含轨迹位置和两个单维主效应：

\[
\operatorname{logit}P(Y=y)=
\alpha+\gamma_1u+\gamma_2u^2+\beta_j\tilde v_j+\beta_k\tilde v_k.
\]

关系模型额外加入配置项：

\[
\operatorname{logit}P(Y=y)=
\alpha+\gamma_1u+\gamma_2u^2+\beta_j\tilde v_j+\beta_k\tilde v_k
+\theta R_{jk}.
\]

因此，本检验回答的是“两个维度的关系是否提供超出两个单维分数本身的信息”，而不是重复发现某个单维与转移的相关性。

使用 5 折 participant-held-out 验证，记录：

\[
\Delta\mathcal L=
\mathcal L_{\mathrm{relation}}-\mathcal L_{\mathrm{baseline}}.
\]

\(\Delta\mathcal L<0\) 表示关系项改善未见参与者上的 log-loss。

另以基线模型的 OOF 残差 \(e_i=y_i-\hat p_i\) 计算参与者级关系效应：

\[
E_R=\mathbb E[\bar e_i\mid R_i=1]
-\mathbb E[\bar e_i\mid R_i=0].
\]

只有 \(E_R>0\) 才能编译为“优先进入目标状态”。负效应表示该关系抑制目标状态，不能通过取绝对值后改写为 `prefer`。

## 5. 稳定性、显著性与规则等级

开发阶段的 `stable_exploratory` 规则须同时满足：

- \(E_R\ge0.05\)；
- 参与者 bootstrap 效应符号一致率至少为 0.90；
- 平均 held-out \(\Delta\mathcal L<0\)；
- 至少 3/5 个 held-out folds 改善；
- 各折关系系数与残差效应的方向一致率至少为 0.60。

正式 `validated` 规则还须满足 participant permutation test 经全候选 Benjamini-Hochberg 校正后的 \(q\le0.10\)。正式运行要求关系边 bootstrap、效应 bootstrap 和 permutation 次数均不低于 1,000。开发运行永远不能产生 `validated` 标签。

## 6. 操作规则与 Prompt 注入

输出规则示例：

```json
{
  "trigger": {
    "profile_relation": {
      "relation_type": "left_dominant",
      "natural_language": "数据 vs 直觉维度相对团队依恋维度至少高2分"
    },
    "current_state": "KtoK",
    "context_policy": "activate_only_when_semantically_relevant"
  },
  "action": {
    "next_state": "CtoK",
    "routing": "prefer",
    "semantic_operation": "检验已有方案所依赖的证据、用户影响和可行性",
    "guard": "完成至少一项证据或风险检查前不得结束推理。"
  }
}
```

运行时先按该参与者的 V30 分数判定关系谓词，再检查当前 KC 状态。只有场景与语义操作相关时才作为软偏好注入；规则不得被用于虚构事实。默认 Prompt 渲染器只注入 `validated`，探索规则必须显式指定状态后才能用于消融或开发实验。

## 7. 统一重采样协议

单维主效应与双维关系规则共享 `participant_resampling_v1` 协议：

- 所有 bootstrap 均以参与者为抽样单位，正式运行不少于 1,000 次；
- permutation 均以参与者为置换单位，正式运行不少于 1,000 次；
- held-out 检验统一使用 5 折 participant-grouped 划分；
- 固定随机种子 `20260727`；
- 固定运行环境为 Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1、
  scikit-learn 1.8.0；
- 只有正式运行可以产生 `validated` 标签，开发运行一律为
  `stable_exploratory`；
- 单维与双维规则只有在输入哈希和 `resampling_protocol_id` 一致时才允许
  合并报告。

双维关系模型有两个不同用途的 bootstrap：V30 关系边稳定性和关系增量
效应稳定性。二者都必须达到 1,000 次，不能用其中一次代替另一次。

## 8. 2026-07-27 开发运行结果

配置：200 次关系 bootstrap、200 次效应 bootstrap、199 次 permutation、最多 12 条 V30 边。

| 项目 | 数量/结果 |
|---|---:|
| 参与者 | 162 |
| Q2 KC token | 1,109 |
| 稳定 V30 边 | 12 |
| 可分析关系谓词 | 66 |
| 关系-转移候选 | 1,633 |
| 通过正向稳定性与 held-out 门槛 | 43 |
| `validated` | 0 |
| 最小 BH-FDR q | 1.0 |

43 条规则说明双维配置能产生比单维开发版 12 条更丰富的探索性路由，但多并不等于证据更强。由于没有候选通过多重检验，这些规则只能用于后续消融、预注册筛选或独立样本验证，不能在论文中表述为已证实的个体规律。

### 8.1 统一协议正式重跑

使用冻结的 `participant_resampling_v1`（两个 bootstrap 各 1,000 次、
participant permutation 1,000 次、5 折 participant-held-out）和冻结软件
环境重跑后：

| 项目 | 数量/结果 |
|---|---:|
| 稳定 V30 边 | 12 |
| 可分析关系谓词 | 66 |
| 关系-转移候选 | 1,633 |
| `stable_exploratory` | 49 |
| `validated` | 0 |

当前计数以这 49 条为准；43 条开发运行结果只保留作参数和环境审计，不再进入
统一规则总数。

## 9. 与旧 CAPS 负向规则的区别

旧 legacy CAPS 产物曾将负系数的绝对值用于 top routing，同时把其自然语言写为“抑制目标状态”。在中心化特征下，负系数对应的低分端实际上会提高目标 logit；将“低分端触发”和“负系数抑制”同时写入同一条目标路由会造成方向矛盾。

本方法不复用该编译逻辑：

- 双维关系的正向效应才可生成 `routing=prefer`；
- 负向候选保留在候选审计文件中，但不进入可注入规则库；
- 如未来需要显式抑制规则，应使用独立的 `avoid` 动作及相应 Prompt 语义，不能与 `prefer` 共用编译模板。

## 10. 复现命令与产物

```bash
python tools/kc_v30_relation_ontology.py \
  --participants PATH/participants.jsonl \
  --planner PATH/planner_records.jsonl \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/kc_v30_relation_ontology_v1_uniform \
  --run-kind formal \
  --relation-bootstrap 1000 \
  --effect-bootstrap 1000 \
  --permutations 1000 \
  --folds 5 \
  --max-pairs 12
```

主要产物：

- `relation_edges.json`：V30 内部稳定边及 bootstrap 证据；
- `relation_candidates.jsonl`：所有候选及 held-out、bootstrap、permutation、FDR 统计；
- `rules.json`：可执行规则库；
- `audit.json`：输入哈希、参数、样本量与计数审计。

严格归纳评估可在训练折内重新发现关系边；若边集合在正式实验前冻结，也可将其视为不使用 Q2 结果的预注册 V30 结构。两种方案应在论文中预先声明，不应在观察 Q7 结果后选择。
