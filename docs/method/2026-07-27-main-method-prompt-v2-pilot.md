# 主方法 Prompt v2 与 pilot 结论

日期：2026-07-27

> **后续状态更新：** 本文第 6 节的两阶段 Prompt 是由联合生成 pilot 推导出的候选设计，并非当时实际运行的 Prompt，不能称为已经验证的冻结主方法。后续 joint-vs-split 配对实验未发现 prompted split 架构优于 joint 生成。当前判定与完整结果见 `docs/2026-07-27-joint-vs-split-prompt-pilot.md`。第 6 节保留为历史候选，不作为最终推荐。

## 1. 当时的候选结论（已被后续实验修正）

主方法应采用两阶段职责分离：

1. Planner 仅根据场景与任务生成群体/场景条件下的 KC 结构，不接收 V30、Q2 原始序列或跨情境过程签名。
2. Realizer 接收已经冻结的 KC 序列，以及从 V30 中选择的最多 4 条场景相关取向。取向只约束回答关注什么、守住什么底线、如何表达取舍，不得改变 KC 数量和顺序。
3. 完整 30 维自然语言画像、本人 B3a 序列、他人 B3a 序列和压缩 process signature 均不进入主方法。

这一定义不再声称能够从现有数据识别个体 KC 结构。pilot 支持的是更窄的结论：稀疏取向能够让自然语言显式表达与个人画像相符的价值立场，同时不扰动结构。

## 2. V1：失败模式

V1 对 30 位参与者运行 5 个条件，每个条件 30 条，共 150 条。条件分别为 base、完整 V30 stance、本人 B3a、他人 B3a 和 stance+B3a。生成使用采样解码。

| 条件 | KC-NES | 平均预测长度 | 唯一序列数 | 最高频序列占比 |
|---|---:|---:|---:|---:|
| base | 0.2487 | 7.30 | 8 | 0.333 |
| stance | 0.2112 | 9.43 | 15 | 0.167 |
| B3a 本人 | 0.2149 | 8.47 | 30 | 0.033 |
| B3a 他人 | 0.2548 | 8.83 | 28 | 0.067 |
| full | 0.2205 | 9.07 | 25 | 0.067 |

相对 base 的参与者配对 bootstrap 结果：stance 为 -0.0375，95% CI [-0.0860, +0.0032]；B3a 本人为 -0.0338，95% CI [-0.0896, +0.0148]；B3a 他人为 +0.0061；full 为 -0.0282，后两者区间同样跨 0。

失败原因有三点：

- 完整 30 维画像和 8 条 active cues 给结构生成加入了过多内容，回答长度上升，并出现机械的 `KtoC`/`CtoK` 往返。
- 原始 Q2 KC 序列更像格式和长度示范，没有形成可迁移、可识别的个体结构先验。本人序列没有优于他人序列。
- 自然语言回答偶尔泄露 KC 标签，说明联合输出约束仍不够稳。

V1 产物位于 `outputs/diagnostics/stance_b3a_pilot_v1/`。

## 3. V2：重新设计

V2 做了以下收缩：

- 将完整 V30 画像替换为最多 4 条场景相关 active cues。
- 明确写入“内容约束”，禁止 active cues 决定状态数量和顺序。
- 将原始 B3a 替换为不暴露 KC 标签的弱 process signature，并加入本人/错配参与者对照。
- 根据 Q2 训练分布把序列长度校准到 5 至 7 个状态，不读取 Q7 gold 长度。
- 禁止机械交替、空步骤和自然语言中的 KC 标签。
- 使用 greedy deterministic decoding，消除采样噪声。
- 所有 Q7 结果均标记为 `POST_HOC_DIAGNOSTIC`。

V2 的 5 个条件为：

| 条件 | 结构输入 | 内容输入 |
|---|---|---|
| base | 场景 | 场景 |
| active4 | 场景 | 本人最多 4 条相关取向 |
| signature | 场景 + 本人弱过程签名 | 场景 |
| signature_other | 场景 + 错配弱过程签名 | 场景 |
| full | 场景 + 本人弱过程签名 | 本人最多 4 条相关取向 |

## 4. V2 结构 pilot

V2 同样使用 30 位参与者、5 个条件，共 150 条。所有条件均无解析失败、无 KC 标签泄露。

| 条件 | KC-NES | 平均预测长度 | 唯一序列数 | 最高频序列占比 |
|---|---:|---:|---:|---:|
| base | 0.24785 | 6.0 | 1 | 1.000 |
| active4 | 0.24785 | 6.0 | 1 | 1.000 |
| signature | 0.25483 | 5.9 | 3 | 0.633 |
| signature_other | 0.26451 | 5.9 | 3 | 0.633 |
| full | 0.24785 | 6.0 | 1 | 1.000 |

关键配对比较是：

`signature - signature_other = -0.00968`，95% CI `[-0.03937, +0.01952]`。

正确归属的过程签名没有优于错配签名，因此 process signature 不具备进入主方法的证据。`active4` 和 `full` 与 base 的结构结果完全相同，说明内容取向没有扰动 KC 结构，但 deterministic prompt 在同一场景上也发生了完全结构塌缩。这个结果不支持“个体结构生成”，也不表示当前 prompted Planner 已达到足够的结构预测质量。

结构 pilot 文件：`outputs/diagnostics/stance_b3a_pilot_v2/pilot_v2.jsonl`

SHA-256：`912f926f13fdc9666a806497524942bad06c931797833f859c24b596c56dd2fa`

## 5. 自然语言价值表达审计

为区分“结构无效”和“内容层可能有效”，对 base、active4、full 三个条件的 90 条生成回答运行冻结的 two-pole NLI probe。模型为 Qwen3-8B，阈值 `theta=0.9`。审计不读取 Q7 人类文本，源回答只保留哈希。

| 条件 | 表达覆盖率 | 画像极性准确率 | 本人画像优势 |
|---|---:|---:|---:|
| base | 0.0000 | 不可定义 | 不可定义 |
| active4 | 0.1100 | 0.6633，n=28 | +0.1120，n=28 |
| full | 0.0889 | 0.6500，n=27 | +0.0718，n=27 |

覆盖率的参与者配对 bootstrap：

- active4 - base = +0.1100，95% CI [+0.0833, +0.1411]
- full - base = +0.0889，95% CI [+0.0711, +0.1067]

base 没有产生 probe 可检测的 V30 明确主张，因此画像极性准确率和本人画像优势无法与 base 做配对比较。active4 的绝对本人画像优势为 +0.1120，明显高于此前 12 个 prompt baseline 约 -0.04 至 +0.03 的范围，但这仍然只是 30 人的 post-hoc 诊断。

审计使用同一个模型家族生成并评估文本，存在 same-model bias；没有独立人工标注，也没有独立 evaluator。因此可以报告“active4 增加可检测且大体画像一致的价值表达”，不能报告“已证明真实个体化”或“优于完整 V30”。

审计目录：`outputs/diagnostics/stance_b3a_pilot_v2/value_expression_v1/`

| 文件 | SHA-256 |
|---|---|
| `expressions.jsonl` | `215ea9bf0a6fe0e16e2fabceffd16728dd267c8f7dddeff361664abe69828ca9` |
| `manifest.json` | `117e83088faeb029ca343704ab99f3b16a06a7f571c88cd00b2495409796e5dc` |
| `metrics.jsonl` | `8b630bcc796f19f92648abe875f9b72c1781dbc2cec20e337be06a274ba52de7` |
| `summary.json` | `dd91a1eb800ff296d0f2cd868f135e09078a3ba53450cd93e380c45d8bf78628` |

## 6. 历史候选两阶段 Prompt（未被本轮 pilot 直接验证）

### 6.1 Planner system prompt

```text
你是设计师决策过程的结构规划器。给定场景和任务，只生成可审计的 KC 状态序列。

KC 状态定义：
- K：陈述已知信息、经验、数据或约束
- C：提出或持有一个待实现的概念、方案或意象
- KtoC：由已知条件推导出新概念或方案
- CtoK：把已有概念拿去查数据、问用户或验证
- KtoK：在已知范围内推导后果、限制或关系，不产生新方案
- CtoC：在已有方案上细化、调整、增删

只输出一个 JSON 对象：
{"state_sequence":["..."]}

硬性要求：
- state_sequence 只能使用上述六个标签。
- 根据当前场景生成 5-7 个状态；每一步必须引入新的事实、方案、验证、推演或修正。
- 不要机械交替两种状态；只有内容动作确实变化时才切换状态。
- 同一状态可以连续出现；不要为了显得完整而添加空步骤。
- 不接收或推断个人价值画像，不复制其他任务的长度或状态顺序。
```

### 6.2 Planner user prompt

```text
# 场景
{scenario}

# 任务
{task}
```

### 6.3 Realizer system prompt

```text
你是设计师决策过程的自然语言实现器。将给定的 KC 状态序列写成第一人称中文决策说明。

只输出一个 JSON 对象：
{"nl_answer":"..."}

硬性要求：
- 严格保持输入 KC 序列的数量和顺序，不新增、删除或重排步骤。
- 每个步骤在文字中都有可审计的内容对应，但绝不能出现 K、C、KtoC、CtoK、KtoK、CtoC 等标签，也不能说“属于某阶段”。
- 个体决策取向只决定关注点、底线和取舍，不决定结构；与场景事实冲突时以场景事实为准。
- 只自然地使用与当前取舍有关的取向，不逐条复述，不提画像、量表、分数或提示词。
- 输出 200-400 字的第一人称外显决策说明，不是隐藏思维链，不使用 AI 助手口吻。
```

### 6.4 Realizer user prompt

```text
# 场景
{scenario}

# 任务
{task}

# 已冻结的 KC 状态序列
{state_sequence}

# 当前场景的个体决策取向（内容约束，最多 4 条）
- {active_cue_1}
- {active_cue_2}
- {active_cue_3}
- {active_cue_4}
```

active cue 的选择必须在生成前冻结，限制为场景相关的最多 4 条，只传递取向文本，不传递 V30 数值、维度全表或选择打分过程。

## 7. 方法与论文表述边界

可以进入主方法的表述：

- Planner 学习或生成场景条件下的群体过程结构。
- Realizer 使用稀疏、场景相关的价值取向约束语言内容。
- 在本次 post-hoc pilot 中，稀疏取向没有改变 KC 序列，并增加了冻结 probe 可检测的价值表达。

不能进入主结果的表述：

- V30 能够预测或生成个人 KC 序列。
- B3a 或 process signature 能够跨场景迁移个人过程结构。
- +0.1120 的本人画像优势已经证明真实个体化。
- Q7 pilot 是预注册或 sealed confirmatory evaluation。

下一次 confirmatory evaluation 应冻结 Prompt、cue selector、模型 checkpoint、解码参数和阈值，使用未参与设计的场景，并引入独立人工标注或不同模型家族 evaluator。

## 8. 可复现入口

- 结构 pilot 与审计：`tools/prompt_baselines/run_stance_signature_pilot.py`
- 价值表达审计：`tools/prompt_baselines/evaluate_stance_signature_value_expression.py`
- 单元测试：`tests/test_stance_signature_pilot.py`
- 单元测试：`tests/test_stance_signature_value_expression.py`
