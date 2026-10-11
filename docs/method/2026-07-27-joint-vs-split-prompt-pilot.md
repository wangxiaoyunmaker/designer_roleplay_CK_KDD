# Joint 与 Split Prompt 配对实验

日期：2026-07-27

状态：`POST_HOC_DIAGNOSTIC / NO_SIGNIFICANT_ARCHITECTURE_WINNER`

## 1. 结论

在同一批 30 位参与者、同一 Qwen3-8B、同一场景和 greedy decoding 下，没有证据表明 prompted split 生成优于一次 joint 生成。

- 两个 split 臂的 KC-NES 仅比 joint 高约 `+0.009`，参与者配对 95% CI 均跨 0。
- 三个条件都退化为条件内 30 人共享同一条 KC 序列，split 没有解决结构塌缩。
- split-routed 的本人画像优势数值更高，但与 joint 的配对 CI 跨 0；价值表达覆盖率反而数值更低。
- joint 只需一次调用，长度合规率不差，并在事后事实性风险扫描中少于两个 split 臂。

因此，不增加 prompt-only Planner 作为默认主方法。Prompt baseline 暂时保留更简单的 joint active4。若主方法需要独立 Planner，应使用已经单独验证的群体序列生成器，再将冻结序列交给整段 Realizer；本实验不支持用另一次通用 LLM Prompt 充当 Planner。

## 2. 实验问题

用户提出比较：

1. 一次生成 KC 序列和自然语言；
2. 先生成 KC 序列，再生成自然语言。

为了避免把“分开调用”和“V30 是否进入 Planner”混成一个变量，实验设置三臂：

| 条件 | 调用方式 | Planner 是否看 active4 | Realizer 是否看 active4 |
|---|---|---:|---:|
| `joint_active4` | 一次生成序列和回答 | 是，同一次调用 | 是，同一次调用 |
| `split_shared_active4` | Planner、Realizer 两次调用 | 是 | 是 |
| `split_routed_active4` | Planner、Realizer 两次调用 | 否 | 是 |

`split_shared - joint` 主要诊断调用拆分；`split_routed - split_shared` 诊断 V30 cue 路由。三臂使用相同参与者、场景、模型 snapshot、active cue selector 和 greedy decoding。

joint 臂复用已冻结的 V2 `active4` 结果，原文件 SHA-256 为 `912f926f13fdc9666a806497524942bad06c931797833f859c24b596c56dd2fa`。实验不读取 Q7 人类文本；Q7 KC gold 只用于 post-hoc 评价，不能用于训练或模型选择。

## 3. 结构结果

| 条件 | KC-NES | 平均长度 | 唯一序列 | top share | 解析失败 | 标签泄露 |
|---|---:|---:|---:|---:|---:|---:|
| joint | 0.24785 | 6.0 | 1 | 1.0 | 0 | 0 |
| split-shared | 0.25762 | 7.0 | 1 | 1.0 | 0 | 0 |
| split-routed | 0.25699 | 7.0 | 1 | 1.0 | 0 | 0 |

参与者配对 bootstrap：

- split-routed - joint：`+0.00914`，95% CI `[-0.02302,+0.04343]`
- split-shared - joint：`+0.00978`，95% CI `[-0.01984,+0.03860]`
- split-routed - split-shared：`-0.00063`，95% CI `[-0.03532,+0.03476]`

三个条件各自输出的固定序列：

```text
joint:         K > C > KtoC > CtoK > KtoK > CtoC
split-shared:  K > KtoC > CtoK > KtoC > CtoK > KtoK > CtoC
split-routed:  K > KtoC > CtoK > KtoK > CtoC > KtoC > CtoK
```

split 改变了模型偏好的模板序列，但没有产生参与者间结构变化。split-shared 中每个人的 active cues 不同，序列仍完全相同，说明 V30 cues 在这个 prompted Planner 中没有形成可观察的个体结构控制。

## 4. 回答合规与多样性

| 条件 | 平均非空白字符 | 200–400 合规率 | 唯一回答 | top answer share |
|---|---:|---:|---:|---:|
| joint | 352.03 | 0.967 | 29 | 0.067 |
| split-shared | 321.03 | 0.967 | 29 | 0.067 |
| split-routed | 338.77 | 0.900 | 29 | 0.067 |

三臂均无解析失败和标签泄露。split 没有带来回答多样性优势；split-routed 的长度合规率略低。

## 5. 价值表达结果

冻结 two-pole NLI probe 使用 Qwen3-8B、`theta=0.9`，不读取 Q7 人类文本。生成和评估使用同一模型家族，结果只能作诊断。

| 条件 | 覆盖率 | 画像极性准确率 | 本人画像优势 |
|---|---:|---:|---:|
| joint | 0.1100 | 0.6633，n=28 | +0.1120，n=28 |
| split-shared | 0.0933 | 0.6032，n=27 | +0.0733，n=27 |
| split-routed | 0.0800 | 0.7358，n=27 | +0.1999，n=27 |

相对 joint 的参与者配对比较：

| split 条件 | 指标差值 | 95% CI | 配对 n |
|---|---:|---:|---:|
| routed | coverage -0.0300 | [-0.0689,+0.0067] | 30 |
| routed | precision +0.0755 | [-0.1040,+0.2475] | 26 |
| routed | own advantage +0.0897 | [-0.0847,+0.2681] | 26 |
| shared | coverage -0.0167 | [-0.0522,+0.0178] | 30 |
| shared | precision -0.0381 | [-0.1878,+0.1139] | 25 |
| shared | own advantage -0.0224 | [-0.1732,+0.1238] | 25 |

所有 CI 均跨 0。split-routed 呈现“表达更少、表达时更可能对齐本人”的数值趋势，但样本不足以判定它优于 joint。

## 6. 事后事实性风险扫描

配对回答抽查发现，模型会把应当计划执行的 `CtoK` 查证写成已经发生并得到结果，例如“我查阅了相关数据，发现……”。这些结果并未出现在场景输入中。

在看到结果后定义的透明正则启发式扫描得到：

```text
我(?:查阅|调研|访谈|测试|收集|咨询|再次确认)|通过(?:用户访谈|问卷调查|测试|调研)|调研结果|测试结果|数据显示|数据表明
```

| 条件 | 包含疑似已执行证据表述 |
|---|---:|
| joint | 21/30 |
| split-shared | 29/30 |
| split-routed | 28/30 |

该扫描不是预注册指标，也不是人工事实标注，不能用于显著性结论；它只用于定位失败模式。两个 split 固定序列均包含两个 `CtoK`，很可能迫使 Realizer 为验证步骤编造完成态证据。后续 Prompt 应把 `CtoK` 明确实现为待执行的验证计划，不得声称已经观察到结果。

## 7. 方法决策

### 当前采用

- Prompt baseline：一次调用的 `joint_active4`，因为没有较差的结构或价值指标，调用成本更低。
- V30：继续只解释为语言内容约束，不声称控制个人 KC 结构。
- 独立结构主干：需要时使用单独验证的群体序列生成器，而不是本实验中的 prompted Planner。

### 暂不采用

- prompt-only Planner + Realizer 作为默认主方法；
- 因 split-routed 的绝对本人画像优势较高，就宣称 split 更好；
- 将条件内不同的固定序列误解为个体化。

### 下一项有信息量的实验

若继续优化 Realizer，应先只修复“计划查证被写成已完成结果”，冻结其余 Prompt，然后比较 joint 与“外部群体序列 + 整段 Realizer”。不建议继续增加 Step Realizer/Composer 层数，因为现有结果没有显示分层本身带来收益。

## 8. 产物

- 生成与结构汇总：`outputs/diagnostics/joint_vs_split_pilot_v1/`
- 价值表达审计：`outputs/diagnostics/joint_vs_split_pilot_v1/value_expression_v1/`
- 运行脚本：`tools/prompt_baselines/run_joint_vs_split_pilot.py`
- 价值审计器：`tools/prompt_baselines/evaluate_stance_signature_value_expression.py`
- 测试：`tests/test_joint_vs_split_pilot.py`

主要文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `pilot.jsonl` | `954fe99336ebe027d3c95aa67ad9118cb4eeacd8b4bd08e780d0612699c7c072` |
| `manifest.json` | `c7454e077ca96d53cdd0015a4b49a943a2e0c5587c932f08d0dbf2041a59435d` |
| `summary.json` | `ada574b90614669f342e9e4e3cf8f70cd56f90c678b0b1f90925e30b4036c18b` |
| `value_expression_v1/summary.json` | `d0a89dec8ef4c8da95ba7e218e0cda629e43f4e018ec1e6a0c3ec3a10705346d` |
