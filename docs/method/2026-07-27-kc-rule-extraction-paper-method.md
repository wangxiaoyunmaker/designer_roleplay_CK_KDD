# V30 条件化 KC 规则提取方法：论文写作参考

日期：2026-07-27  
对应版本：`KC_RULEBANK_V1`、`Q1_INIT_RULEBANK_V1_CANDIDATE`  
用途：论文 Method、Implementation Details 和 Reproducibility 部分的写作依据

## 1. 方法目标与当前规则范围

本方法的目标不是从问卷中归纳描述性人格标签，而是从参与者的 V30 价值画像与已标注的 KC 行为之间学习可执行的认知操作偏好。最终规则必须能够转换成推理时的软路由指令，例如：

\[
(\text{画像条件},\ z_t=s)
\Rightarrow
(z_{t+1}=y,\ \text{semantic operation},\ \text{guard}).
\]

当前共有两套需要在论文中分开说明的规则库：

| 使用位置 | 规则层 | 数量 | 当前状态 |
|---|---|---:|---|
| Seg1 | Q1 单维初始状态规则 | 6 | `EXPLORATORY_NOT_FROZEN` |
| Seg1 | Q1 双维初始状态规则 | 12 | `EXPLORATORY_NOT_FROZEN` |
| Seg2 | Q2 单维状态转移规则 | 12 | `FROZEN_EXPLORATORY` |
| Seg2 | Q2 双维关系转移规则 | 49 | `FROZEN_EXPLORATORY` |
| Seg2 | Q3–Q6 情境状态偏好规则 | 29 | `FROZEN_EXPLORATORY` |

因此，Seg2 冻结规则库共有 90 条；若把独立的 Seg1 Q1 候选也计入，目前共有 108 条规则产物。但论文中不能把 108 条统称为同一个冻结库：冻结的是 Seg2 的 90 条，Q1 的 18 条仍是单独的探索性候选。

所有规则均只使用 Q1–Q6 与 V30 提取。Q7/Q8 在输入加载阶段被显式禁止，未参与规则发现、参数筛选或方向编译。

## 2. 统一符号与数据表示

设参与者为 \(i=1,\ldots,N\)，其 V30 画像为

\[
\mathbf v_i=(v_{i1},\ldots,v_{i30}),\qquad v_{ij}\in\{1,2,3,4,5,6\}.
\]

新版 KC 状态空间固定为

\[
\mathcal S=\{K,C,K{\to}C,C{\to}K,K{\to}K,C{\to}C\}.
\]

其中：

| KC 状态 | 操作含义 |
|---|---|
| `K` | 补充可核查的事实、数据、经验或现实约束 |
| `C` | 形成明确的设计概念、原则或候选方案 |
| `KtoC` | 从证据或约束推导新方案 |
| `CtoK` | 检验已有方案的证据、用户影响或可行性 |
| `KtoK` | 推导已知信息之间的后果、限制或冲突 |
| `CtoC` | 细化、调整、组合或替换已有方案 |

`START` 和 `STOP` 只用于 Q2 序列边界，不属于六种认知状态。旧版 PK、VC 或 CAPS 四元组均不进入当前规则提取。

## 3. 统一的重采样与防泄漏协议

四个提取模块共享 `participant_resampling_v1`：

| 项目 | 固定设置 |
|---|---:|
| 重采样单位 | 参与者 |
| Bootstrap | 1,000 次 |
| Participant permutation | 1,000 次 |
| Held-out 验证 | 5 折 GroupKFold |
| 随机种子 | 20260727 |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |

同一参与者的多条 Q2 转移始终位于同一折，避免同一个人的轨迹同时进入训练集和测试集。正式提取脚本采用 fail-closed 策略：采样次数、折数、软件版本、V30 长度、题号或 KC 标签不符合协议时，停止生成正式产物。

“统一协议”指重采样单位、采样次数、held-out 划分和运行环境统一；不同规则层的数据结构不同，因此候选模型与筛选阈值并不完全相同。

## 4. Q2 单维状态转移规则

### 4.1 序列展开

对每位参与者的 Q2 KC 轨迹

\[
(z_{i1},\ldots,z_{iT_i}),
\]

加入边界后展开为

\[
START\to z_{i1}\to\cdots\to z_{iT_i}\to STOP.
\]

重复状态和原始序列长度均被保留。第 \(t\) 条转移的位置为 \(p_{it}=t/T_i\)，并转换为中心化阶段变量

\[
\phi_{it}=2p_{it}-1.
\]

正式数据包括 162 位参与者、162 条 Q2 序列、1,109 个 KC token，以及加入边界后的 1,271 条转移。

### 4.2 稀疏条件主效应模型

针对每个当前状态 \(s\) 单独建模。对每个可能的下一状态 \(y\)，使用 L1 正则 one-vs-rest logistic regression：

\[
\operatorname{logit}P(z_{i,t+1}=y\mid z_{it}=s)
=\alpha_{sy}
+\boldsymbol\beta_{sy}^{\top}\widetilde{\mathbf v}_i
+\gamma_{1,sy}\phi_{it}
+\gamma_{2,sy}\phi_{it}^{2}.
\]

其中 \(\widetilde{\mathbf v}_i\) 是标准化后的 V30。因为 30 个维度同时进入模型，\(\beta_{sy,j}\) 表示控制其他 V30 维度及序列位置后的稀疏条件主效应。

正则强度从

\[
C\in\{0.03,0.1,0.3,1.0,3.0\}
\]

中按参与者分组五折交叉验证的多分类 log-loss，为每个当前状态分别选择。

### 4.3 Bootstrap 稳定性

每次 bootstrap 有放回抽取参与者，并通过样本权重保留该参与者的全部转移。对每个 `(当前状态 s, 下一状态 y, V30 维度 j)` 记录：

- 全数据标准化系数 \(\hat\beta_{sy,j}\)；
- 1,000 次 bootstrap 中系数非零的比例，即 selection frequency；
- 非零系数与全数据方向相同的比例，即 sign consistency；
- bootstrap 百分位 95% 区间；
- 支持该目标转移的参与者数与转移数。

稳定探索候选的门槛为：

\[
n_{support}\ge20,\quad
f_{select}\ge0.90,\quad
f_{sign}\ge0.95,\quad
|\hat\beta|\ge0.15.
\]

### 4.4 Held-out 增量检验与 permutation

在每个参与者 held-out 折中，基线模型只使用序列位置：

\[
M_0:\quad y\sim\phi+\phi^2,
\]

增量模型加入一个候选 V30 维度：

\[
M_1:\quad y\sim\phi+\phi^2+\widetilde v_j.
\]

定义增量损失为

\[
\Delta\mathcal L
=\mathcal L(M_1)-\mathcal L(M_0).
\]

负值表示该 V30 维度提高了未见参与者上的下一状态预测。另将 out-of-fold 残差先在参与者内平均，再进行参与者级 permutation；全部 `(s,y,j)` 候选共同接受 Benjamini–Hochberg FDR 校正。

Q2 单维层采用两级证据制度：稳定性门槛决定是否进入 `stable_exploratory` 规则池；只有同时满足以下条件才能升级为 `validated`：

\[
q\le0.10,\quad
\Delta\mathcal L<0,\quad
\text{improved folds}\ge3/5,\quad
f_{heldout-sign}\ge0.60.
\]

这一点在论文中应准确表述：冻结的 12 条单维规则是稳定探索候选，并非 12 条都通过 FDR 或完整 held-out 验证。正式运行共产生 1,410 个候选，保留 12 条 `stable_exploratory`，`validated=0`。

### 4.5 极性与规则编译

若 \(\hat\beta>0\)，高分端提高目标状态的条件 logit，编译为 `score >= 4`；若 \(\hat\beta<0\)，低分端提高目标状态的条件 logit，编译为 `score <= 3`：

\[
r_{syj}=
\begin{cases}
(v_{ij}\ge4,z_t=s)\Rightarrow z_{t+1}=y,&\hat\beta>0,\\
(v_{ij}\le3,z_t=s)\Rightarrow z_{t+1}=y,&\hat\beta<0.
\end{cases}
\]

这里的负系数不是“禁止进入目标状态”。它表示低分极性对应更高的目标状态倾向，因此仍可编译成低分端触发的正向 `prefer` 规则。

## 5. Q2 双维关系转移规则

### 5.1 先发现 V30 内部稳定关系边

对 30 个 V30 维度的 435 个无序维度对计算 Spearman 相关：

\[
\rho_{jk}=\operatorname{Corr}(\operatorname{rank}(v_j),
\operatorname{rank}(v_k)).
\]

以参与者为单位 bootstrap 1,000 次。维度边必须满足：

\[
|\rho_{jk}|\ge0.20,\quad
f_{sign}\ge0.90,
\]

且 bootstrap 95% 区间不跨 0。再按 \(|\rho|\) 排序，固定保留前 12 条边。此步骤只使用 V30，不使用 Q2/Q7 结果。

### 5.2 从维度边构造关系谓词

对每条稳定边 \((j,k)\) 构造七种离散关系：

| 关系类型 | 激活条件 |
|---|---|
| `synergy_high` | \(v_j\ge4\land v_k\ge4\) |
| `synergy_low` | \(v_j\le3\land v_k\le3\) |
| `tension_left_high` | \(v_j\ge4\land v_k\le3\) |
| `tension_right_high` | \(v_j\le3\land v_k\ge4\) |
| `left_dominant` | \(v_j-v_k\ge2\) |
| `right_dominant` | \(v_k-v_j\ge2\) |
| `balanced` | \(|v_j-v_k|\le1\) |

关系成立组和不成立组均至少有 20 位参与者时，谓词才进入转移候选分析。12 条边最终形成 66 个可分析关系谓词。

### 5.3 控制两个单维主效应后的增量关系模型

对当前状态 \(s\)、目标状态 \(y\) 和关系谓词 \(R_{jk}\)，held-out 基线模型为

\[
M_0:\quad
\operatorname{logit}P(y=1)
=\alpha+\gamma_1\phi+\gamma_2\phi^2
+\beta_j\widetilde v_j+\beta_k\widetilde v_k.
\]

增量模型只增加关系项：

\[
M_1:\quad M_0+\theta R_{jk}(\mathbf v_i).
\]

因此双维规则表示超出两个维度各自主效应之后的额外交互，而不是把两个单维规则简单拼接。

使用基线模型的 out-of-fold 概率得到残差

\[
e_{it}=\mathbb 1(z_{i,t+1}=y)-\widehat P_{M_0}(y=1),
\]

先对同一参与者的残差取平均，再计算关系组与非关系组之差：

\[
\delta_R=\mathbb E(\bar e_i\mid R_i=1)
-\mathbb E(\bar e_i\mid R_i=0).
\]

正的 \(\delta_R\) 表示在控制位置和两个单维分数后，基线仍系统性低估关系组进入目标状态的倾向。

### 5.4 双维候选门槛

候选目标状态至少由 20 位参与者支持；在相应 source-state 数据中，关系成立与不成立组也均至少 20 人。可注入探索规则还必须满足：

\[
\delta_R\ge0.05,\quad
f_{bootstrap-sign}\ge0.90,\quad
\Delta\mathcal L<0,
\]

并且至少 3/5 held-out 折改善、held-out 方向一致率至少 0.60。Permutation p 值经全部关系候选统一 BH-FDR 校正；只有 \(q\le0.10\) 才能升级为 `validated`。

负的 \(\delta_R\) 表示该关系降低目标状态倾向。当前动作语言只有 `prefer target`，因此所有负向关系候选只保留在候选审计中，不进入可注入规则库。这样避免把“降低目标状态概率”错误编译成“优先进入目标状态”。

正式运行得到 1,633 个关系-转移候选，保留 49 条正向 `stable_exploratory` 规则，`validated=0`。

## 6. Q3–Q6 情境状态偏好规则

### 6.1 任务定义

Q3–Q6 每题只提供一个映射后的 KC 选择，而非完整轨迹。因此该层学习的是

\[
(Q_q,\ \text{V30 pole})\Rightarrow\text{preferred KC state},
\]

而不是 `current_state -> next_state`。论文中不应为这类规则补造不存在的当前状态。

可用样本为 Q3=162、Q4=162、Q5=162、Q6=137。Q6 中 25 个 `V` 或缺失标签按冻结的六状态 ontology 排除，不映射为其他 KC 状态。

### 6.2 每题的稀疏多分类模型

每个问题分别拟合 30 维 L1 one-vs-rest logistic model：

\[
\operatorname{logit}P(Y_{iq}=y)
=\alpha_{qy}+\boldsymbol\beta_{qy}^{\top}\widetilde{\mathbf v}_i.
\]

每个 `(问题 q, 目标状态 y, V30 维度 j)` 构成一条候选，共 360 条。Bootstrap 记录 selection frequency、sign consistency 和系数区间。

Held-out 检验使用题内目标状态流行率作为基线，并与只加入一个候选维度的模型比较：

\[
\Delta\mathcal L
=\mathcal L(\text{intercept}+\widetilde v_j)
-\mathcal L(\text{intercept}).
\]

另对全部“问题 × 目标状态 × V30 维度”候选进行参与者级 permutation，并统一作 BH-FDR 校正。

### 6.3 固定的 `c03_balanced` 候选策略

在不读取 Q7 的前提下比较了五个参数臂。最终固定 `c03_balanced`，其设置为：

\[
C=0.3,\quad n_{support}\ge15,\quad
f_{select}\ge0.80,\quad f_{sign}\ge0.90,
\quad |\hat\beta|\ge0.10.
\]

此外要求 \(\Delta\mathcal L<0\)、至少 3/5 held-out 折改善、held-out 符号一致率至少 0.60。

`c03_balanced` 是覆盖率与稳定性的开发期折中，不是根据 Q7 效果选出的最优超参数。其 29 条规则覆盖 16/30 个 V30 维度，来源分布为 Q3=3、Q4=5、Q5=13、Q6=8。

这 29 条规则的最小 FDR 为 \(q=0.3596\)，所以全部标记为 `heldout_supported_exploratory`，没有 validated 规则。FDR 在该层用于报告证据等级，不是进入探索池的门槛。

### 6.4 极性编译

与 Q2 单维相同，\(\beta>0\) 编译为高分端 `score >= 4`，\(\beta<0\) 编译为低分端 `score <= 3`。这是把同一个正向目标偏好定向到相应 V30 极性，不是把负系数候选改写成无条件正向规则。

## 7. Q1 初始反应规则

### 7.1 与 Q2 转移规则的区别

Q1 只表示第一反应的 KC 状态，没有前序状态。因此其规则形式为

\[
(\text{V30 condition})\Rightarrow z_{initial}=y,
\]

而不是 `current_state -> next_state`。

162 位参与者中有 148 个可用 Q1 六状态标签；14 个 `V` 或缺失标签被排除。可用状态分布为 `CtoC=66`、`CtoK=23`、`KtoK=59`。

### 7.2 Q1 单维规则

Q1 单维直接复用 Q3–Q6 的 `c03_balanced` 参数，不根据 Q1 结果重新搜索阈值：

\[
C=0.3,\ n_{support}\ge15,\ f_{select}\ge0.80,
\ f_{sign}\ge0.90,\ |\hat\beta|\ge0.10.
\]

同时要求 held-out \(\Delta\mathcal L<0\)、至少 3/5 折改善、方向一致率至少 0.60。90 个候选中保留 6 条。

### 7.3 Q1 双维规则

先用全部 V30 画像发现同一组 12 条稳定关系边和七类关系谓词，再将每位参与者的 Q1 标签表示成一条 `START -> initial state` 转移。关系模型控制两个单维主效应，只允许正向残差效应编译为 `prefer initial state`。192 个候选中保留 12 条。

Q1 最终共有 18 条探索规则，单维层和双维层在 162 份画像上均达到 162/162 激活覆盖。覆盖率只表示至少有一条规则能被画像阈值触发，不表示规则已经通过外部效度验证。

## 8. 从统计候选到 JSON 规则

统计候选被编译为结构化 JSON，核心字段包括：

```json
{
  "rule_id": "KCOR_xxx",
  "rule_status": "stable_exploratory",
  "trigger": {
    "profile": {
      "trait_index": 5,
      "operator": "score_lte",
      "threshold": 3,
      "stance": "低分极性语义"
    },
    "current_state": "KtoK",
    "context_policy": "activate_only_when_semantically_relevant"
  },
  "action": {
    "next_state": "C",
    "routing": "prefer",
    "semantic_operation": "形成一个明确的设计概念、原则或候选方案",
    "guard": "下一步必须给出具体概念或原则，不能只罗列已知条件。"
  },
  "evidence": {
    "coefficient": -0.51,
    "selection_frequency": 0.99,
    "heldout_delta_log_loss": -0.04,
    "fdr_q": 0.70
  }
}
```

生成时不直接把 V30 分数、系数或统计描述写进 Prompt，而是将规则确定性转换成自然语言操作。例如：

> 当画像条件满足且当前状态为 `KtoK` 时，下一步优先进入 `C`，即形成一个明确的设计概念、原则或候选方案。

规则是软偏好，不是强制状态，也不能要求模型虚构已经完成的调研、访谈、数据或用户经历。

## 9. 运行时窄路由

候选池的规模与单次注入量分离。运行时先按画像激活规则，再收窄注入：

1. 检查目标场景与规则语义是否相关；
2. 检查参与者是否满足高分、低分或双维关系谓词；
3. 优先按该参与者越过阈值的强度排序；
4. 使用 held-out loss 增益和 bootstrap 稳定性破同分；
5. 限制每个激活层的规则数；
6. 同一 source state 下单双维方向冲突时，以双维关系作为对单维主效应的修正。

候选池阶段保留较多规则、推理阶段再根据本人 V30 收窄，是个体化路由的核心：不同参与者可以从同一冻结库激活不同的短规则组合，而不需要把整套规则写入 Prompt。

在 30 人快速消融中，为减少 Prompt 长度混杂，每个激活层最多注入 1 条；这是实验期注入上限，不是新的提取门槛。Q3–Q6 原始窄路由审计同时保留了 top-2 配置。

## 10. 统一审计、冻结与数据隔离

三个 Seg2 层只有在以下条件一致时才能合并计数：

- 重采样协议 ID、采样次数、随机种子和软件版本一致；
- V30 画像文件与特质定义文件 SHA-256 一致；
- Q2 单维和双维层使用相同 Q2 planner 输入；
- Q3–Q6 明确指定 `c03_balanced` 参数臂；
- Q7 未用于发现或选择。

统一审计通过后，Seg2 得到 12 + 49 + 29 = 90 条探索规则，覆盖的 V30 维度并集为 24/30。该库于 2026-07-27 冻结为 `KC_RULEBANK_V1`：不再扩大规则池、不重新调参、不允许根据 Q7 结果修改单条规则；后续实验只允许启用或关闭完整规则层。

## 11. 当前证据等级与论文表述边界

当前 90 条 Seg2 规则和 18 条 Q1 规则的 validated 数均为 0。30 人 correct-vs-shuffled 快速消融也尚未证明正确画像路由稳定优于错配画像路由。因此论文中推荐使用以下表述：

> We mined a frozen exploratory library of V30-conditioned KC routing candidates using participant-level resampling and held-out diagnostics. The rules were treated as soft operational priors rather than validated psychological laws.

不应使用以下表述：

> We discovered 90 validated individualized cognitive laws.

可以报告规则在训练数据中的稳定性、held-out 预测增益、覆盖率和下游消融；但“个体化有效”必须由独立的 correct-profile vs shuffled-profile 对照，以及 KC-NES、BGE own-user margin、V30 matched-minus-mismatched 或人工匹配评价共同支持。

## 12. 可直接改写为论文 Method 的中文版本

> 我们将参与者的 30 维价值画像表示为 \(\mathbf v_i\)，并将设计推理过程表示为六状态 KC 序列。对于 Q2，我们将完整轨迹展开为包含 `START` 和 `STOP` 边界的相邻状态转移，并针对每个当前状态拟合 L1 正则化的 one-vs-rest logistic 模型，以联合估计 V30 条件效应和序列位置效应。所有重采样和数据划分均以参与者为单位：我们执行 1,000 次 bootstrap 估计变量选择频率与方向稳定性，使用五折 participant-held-out log-loss 评估未见参与者上的增量预测能力，并通过 1,000 次 participant permutation 与 Benjamini–Hochberg 校正估计多重检验后的证据等级。
>
> 为建模 V30 维度间的组合关系，我们首先从 V30 本身识别 bootstrap 稳定的 Spearman 关系边，再构造高—高、低—低、张力、相对优势和平衡等离散关系谓词。双维增量模型在控制两个维度的单维主效应和序列位置后加入关系谓词；只有正向、稳定且改善 held-out loss 的残差效应才能编译为 `prefer` 路由，负向效应仅保留用于审计。对于只有单个 KC 选择标签的 Q1 与 Q3–Q6，我们分别学习初始状态偏好和问题条件化状态偏好，而不补造不存在的状态转移。
>
> 最后，我们将统计候选编译为包含画像触发条件、KC 操作、自然语言语义和事实性 guard 的结构化 JSON。推理时仅激活与当前画像和场景相关的少量规则，并将其作为软路由先验注入。规则发现阶段完全不读取目标任务 Q7；规则库在下游实验前冻结，下游结果只能决定启用哪些完整规则层，不能用于修改单条规则。

## 13. 复现入口与产物

| 内容 | 路径 |
|---|---|
| Q2 单维提取器 | `tools/kc_operational_ontology.py` |
| Q2 双维提取器 | `tools/kc_v30_relation_ontology.py` |
| Q3–Q6 提取器 | `tools/kc_aux_preference_sweep.py` |
| Q1 提取器 | `tools/q1_initial_rule_ontology.py` |
| Q2 单维规则 | `artifacts/kc_operational_ontology_v1_uniform/rules.json` |
| Q2 双维规则 | `artifacts/kc_v30_relation_ontology_v1_uniform/rules.json` |
| Q3–Q6 规则 | `artifacts/kc_aux_preference_sweep_v1_uniform/rules_by_arm.json` 中的 `c03_balanced` |
| Q1 规则 | `artifacts/q1_initial_rulebank_v1_candidate/rules.json` |
| 统一协议审计 | `artifacts/kc_uniform_rule_audit_v1.json` |
| Seg2 冻结清单 | `artifacts/kc_rulebank_v1_freeze_manifest.json` |

将下列尖括号路径替换为正式数据路径即可重建各层。Q2 单维：

```bash
python tools/kc_operational_ontology.py mine \
  --participants <participants.jsonl> \
  --planner <q2_planner_records.jsonl> \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/kc_operational_ontology_v1_uniform \
  --run-kind formal \
  --bootstrap 1000 \
  --permutations 1000 \
  --folds 5
```

Q2 双维：

```bash
python tools/kc_v30_relation_ontology.py \
  --participants <participants.jsonl> \
  --planner <q2_planner_records.jsonl> \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/kc_v30_relation_ontology_v1_uniform \
  --run-kind formal \
  --relation-bootstrap 1000 \
  --effect-bootstrap 1000 \
  --permutations 1000 \
  --folds 5 \
  --max-pairs 12
```

Q3–Q6 情境偏好：

```bash
python tools/kc_aux_preference_sweep.py \
  --participants <participants.jsonl> \
  --aux <q1_q6_auxiliary_labels.jsonl> \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/kc_aux_preference_sweep_v1_uniform \
  --bootstrap 1000 \
  --permutations 1000 \
  --folds 5 \
  --max-rules-per-context 2
```

Q1 初始状态：

```bash
python tools/q1_initial_rule_ontology.py \
  --participants <participants.jsonl> \
  --aux <q1_q6_auxiliary_labels.jsonl> \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/q1_initial_rulebank_v1_candidate \
  --bootstrap 1000 \
  --permutations 1000 \
  --folds 5 \
  --single-c 0.3 \
  --max-pairs 12
```

实验前应执行：

```bash
python tools/verify_kc_rulebank_freeze.py \
  --manifest artifacts/kc_rulebank_v1_freeze_manifest.json \
  --repo-root .
```

只有冻结校验输出 `status=PASS` 时，实验结果才属于 `KC_RULEBANK_V1`。
