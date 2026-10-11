# Direct 与 Stepwise 文本实现消融实验

日期：2026-07-27

状态：`POST_HOC_DIAGNOSTIC / PROVISIONAL PROMPT DECISION`

## 1. 结论

本实验不支持“拆成多次调用就会普遍更好”。真正稳定加分的是两项：

1. 给每个 KC 状态写清可检查的动作与证据边界，尤其把 `CtoK` 限定为将来态验证计划；
2. 需要逐步生成时，把此前生成的具体方案显式传给下一步。

推荐分两种用途：

- Prompt-only baseline：采用 `direct_guarded_joint`。它仍然一次生成序列和整段回答，但已把疑似虚构完成态证据从 `9/10` 降至 `0/10`，不需要多调用。
- 已有冻结 Planner 序列的主方法：优先采用一次调用的 `onecall_aligned_steps`，输出 `steps=[{state, utterance}]`，再规则拼接 utterance。它保持一次调用、逐步可审计和 `0/10` 完成态证据风险，并取得本组最高的本人画像优势。
- 若方案跨步连续性是首要目标，可采用 `stepwise_context_concat`：每步都是新调用，但把此前 utterance 显式放入下一步 Prompt。它的事后方案锚点连续率从 one-call aligned 的 `0.367` 提高到 `0.900`，但没有价值指标增益。
- 删除 Composer。它对 `10/10` 输出逐字不改，所有指标增量均为 0，却额外增加一次调用和约 5.45 秒平均运行时间。

这仍不是最终确证结论。样本只有 10 人，没有盲评的人类连贯性/自然度评分，价值探针与生成模型同属 Qwen3-8B，且所有条件仍只有一条固定序列。

## 2. 实验问题与控制

本轮专门比较文本实现，不再把 Planner 差异混入主要 stepwise 对照。每位参与者的逐步条件复用原 joint 条件生成的完全相同序列：

```text
K > C > KtoC > CtoK > KtoK > CtoC
```

10 位参与者、同一 Qwen3-8B snapshot、同一 Q7 公开场景、相同 active4 cue selector、greedy decoding。未读取 Q7 人类回答；全部结果只能作为事后诊断。

六个条件：

| 条件 | 上下文/调用方式 | 调用数 |
|---|---|---:|
| `direct_joint` | 原始一次生成序列和整段 | 1 |
| `direct_guarded_joint` | 一次生成序列和整段，加入逐状态证据边界 | 1 |
| `onecall_aligned_steps` | 输入冻结序列，一次生成完整 `steps[]` | 1 |
| `stepwise_fresh_concat` | 每步新调用，不提供此前 utterance，规则拼接 | 6 |
| `stepwise_context_concat` | 每步新调用，在 Prompt 中显式提供此前 utterance，规则拼接 | 6 |
| `stepwise_context_composed` | context-step 后再调用受限 Composer | 7 |

`stepwise_context` 不是服务端共享会话；每一步都是独立 chat request。所谓“带上下文”，是将已生成文本作为只读字段重新放入下一步 Prompt，因而输入可复现、可审计。

## 3. 结构、格式与事实性

| 条件 | 解析失败 | 200-400 字合规 | 疑似完成态证据 | 逐步格式失败率 | 平均秒数 |
|---|---:|---:|---:|---:|---:|
| original joint | 0/10 | 90% | 9/10 | N/A | 未记录 |
| guarded joint | 0/10 | 100% | 0/10 | N/A | 6.63 |
| one-call aligned | 0/10 | 100% | 0/10 | 20.0% | 9.11 |
| fresh-step concat | 0/10 | 100% | 0/10 | 0% | 6.52 |
| context-step concat | 0/10 | 100% | 0/10 | 0% | 6.66 |
| context-step + Composer | 0/10 | 100% | 0/10 | 0% | 12.11 |

guardrail 相对原 joint 的完成态证据风险差值为 `-0.90`，参与者配对 bootstrap 95% CI `[-1.00,-0.70]`。因此，不能把事实性改善归因于拆步；一次调用加入同样规则已经得到相同改善。

`onecall_aligned_steps` 的逐步格式失败集中在 2 位参与者：JSON 和步骤数量正确，但模型连续六步都省略句末标点，因此共有 `12/60` 步触发严格格式告警。可用确定性标点规范化或一次受限 retry 修复，但本实验没有把修复后结果计入成绩。

所有结构化条件的状态语义告警率均为 `1/6`，集中在 `KtoK`：模型在推演后果后又写了“需要增加人工辅助”等解法，越过了“不产生新方案”的边界。这是当前 Prompt 的剩余失败模式。

## 4. 各组件边际作用

| 组件对比 | 观察结果 | 判断 |
|---|---|---|
| state/evidence guardrail：guarded joint - original joint | 完成态证据 `-0.90`；长度合规 `+0.10` | 明确加分 |
| aligned `steps[]`：one-call aligned - guarded joint | 事实性不变；本人画像优势 `+0.136`，CI `[+0.006,+0.282]` | 倾向加分，需扩大样本 |
| 多调用：context-step - one-call aligned | 价值三指标无可靠增益；事后连续率 `+0.533` | 只对连续性加分 |
| 前文回传：context-step - fresh-step | 连续率 `+0.833`；价值覆盖 `-0.0267` | 连续性必要，但存在表达收窄 |
| Composer：composed - rule concat | 文本、长度、事实性和价值指标全部严格相同 | 明确减分 |

方案连续率是看到 fresh-step 方案跳变后定义的透明事后词汇指标，不是预注册主指标。词表只检查语音、触控、人工支持、标识、大字、多语言、货架/陈列、透明、隐私和分阶段等公开场景方案锚点。数值为：

| 条件 | 事后方案锚点连续率 |
|---|---:|
| fresh-step | 0.067 |
| one-call aligned | 0.367 |
| context-step | 0.900 |

配对文本也呈现相同模式：fresh-step 常在“指引标识”“人工窗口”“自助机测试”“语音通道”之间跳换；context-step 更常持续查证和细化同一个语音/触控方案。

## 5. 价值表达

冻结 two-pole probe，`theta=0.9`：

| 条件 | 覆盖率 | 画像极性准确率 | 本人画像优势 |
|---|---:|---:|---:|
| original joint | 0.0867 | 0.4963, n=9 | -0.0708, n=9 |
| guarded joint | 0.0733 | 0.5167 | -0.1017 |
| one-call aligned | 0.0800 | 0.6417 | +0.0342 |
| fresh-step | 0.0900 | 0.5500 | -0.0613 |
| context-step | 0.0633 | 0.6000 | -0.0156 |
| context-step + Composer | 0.0633 | 0.6000 | -0.0156 |

one-call aligned 相对 original joint：

- coverage `-0.0067`，95% CI `[-0.0333,+0.0200]`；
- precision `+0.2167`，95% CI `[+0.0778,+0.3796]`，配对 n=9；
- own-profile advantage `+0.1781`，95% CI `[+0.0364,+0.3270]`，配对 n=9。

这些结果支持继续测试 aligned 输出，但不能当作确证结论：样本小、探针并非人工金标准，而且只有 one-call aligned 的绝对本人画像优势略为正值。

## 6. 最终 Prompt 决策

当前最合理的主方法接口为：

```json
{
  "steps": [
    {"state": "K", "utterance": "..."},
    {"state": "CtoK", "utterance": "..."}
  ]
}
```

实现规则：

1. Planner 或上游模块冻结 `state_sequence`；V30 active4 只进入 Realizer，控制关注点、底线和取舍，不声称控制状态顺序。
2. 单次 Realizer 调用逐项输出 `steps[]`，数量、标签和值顺序必须与冻结序列完全一致。
3. 六类状态使用不同的句内构成约束；`CtoK` 必须是验证对象 + 计划方法 + 指标，且只能用将来/条件语气。
4. 最终回答由程序按顺序拼接 utterance，不调用 Composer；先做 JSON、数量、标签、单句、长度、标签泄露和完成态证据检查。
5. 对缺少句末标点做确定性规范化；对真正的字段/数量/标签失败才 retry。

如果下一轮盲评确认 one-call aligned 的跨步连贯性不足，再把 `stepwise_context_concat` 作为高质量变体，而不是默认方案。不要采用 fresh-step，也不要保留 Composer。

## 7. 限制与下一步

- 10 人不足以支持论文级主张，应扩大到至少预先冻结的一组验证样本。
- 需要不知道条件标签的人工评审，至少评价事实忠实、KC 对齐、方案连续、自然度和个体化。
- 所有条件仍输出同一条序列，不能据此主张个体 KC 结构控制。
- 需要单独修复并复测 `KtoK` 夹带方案的问题。
- 本实验看过 2 人 smoke 后校准了长度与句式契约；正式 10 人 Prompt 随后冻结，未按正式结果继续调参。

## 8. 产物

- 运行脚本：`tools/prompt_baselines/run_direct_vs_stepwise_realization_pilot.py`
- 单元测试：`tests/test_direct_vs_stepwise_realization_pilot.py`
- 六臂输出：`outputs/diagnostics/direct_vs_stepwise_pilot_v1/full_v2/pilot.jsonl`
- 结构汇总：`outputs/diagnostics/direct_vs_stepwise_pilot_v1/full_v2/summary.json`
- 价值汇总：`outputs/diagnostics/direct_vs_stepwise_pilot_v1/full_v2/value_expression_v1/summary.json`

主要文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `pilot.jsonl` | `6e92c6604e8aea3f625880f8381b2e64051f70175494c8eb3880b52742ba615f` |
| `summary.json` | `90ff68ae0a6e4129b9a692288625ea4612a0e3d2023cc97665abea4aa29ce8ac` |
| `manifest.json` | `da8b12b90ff7c59e66a51784482f731110542e03a2ad96cd907d81bcb740ae45` |
| `value_expression_v1/summary.json` | `c8c2901aba8675f3bbe6eb05f1140b887486f59318ca19a4d70b00b4f94aabf0` |

## 9. Follow-up：标点与 KtoK 契约修复

针对第 3 节记录的两个剩余问题，后续实现了生成后契约层并重新跑了同一规模的 10 人验证。

句末标点采用确定性规范化：只在 utterance 缺少 `。！？!?` 时补 `。`；若末尾是引号或括号，则把句号插入闭合符号之前。它不调用模型，不修改正文语义。中间验证运行中，one-call aligned 自动修复了 `30/30` 个实际缺失句号，修复后逐步格式失败为 `0/60`。

`KtoK` 改成正向、有限输出域：

```text
我判断，如果[场景已知前提]，[后果]可能发生，
这与[三个月期限 / 完全无人化指令 / 银发用户服务可达性 / 商业效率目标]形成冲突。
```

校验器禁止“需要增加/设置/保留/采用”等解决行动，但允许“需要额外开发时间或资源”作为可行性约束。真正越界时只 retry 当前 `KtoK` utterance，一次后仍不合规则保留失败标记，不静默删除内容。

最终 10 人 refined audit：

| 条件 | 格式失败 | KtoK/状态契约告警 | 完成态证据风险 |
|---|---:|---:|---:|
| one-call aligned | 0/60 | 0/60 | 0/10 |
| fresh-step | 0/60 | 0/60 | 0/10 |
| context-step | 0/60 | 0/60 | 0/10 |

fresh-step 的 2 次真实越界 retry 均成功。context-step 运行时曾有 4 条被旧版校验器误报为越界，内容均为“需要额外开发时间”；收窄谓词判定后，最终输出重新审计均合法。两个先前的真实非法冲突对象样本（“优先相信设计判断”“保留操作灵活性”）经定向 repair 后都被替换为“银发用户服务可达性”并通过校验。

修复验证产物：

- `outputs/diagnostics/direct_vs_stepwise_fix_v1/full_v2/pilot.jsonl`
- `outputs/diagnostics/direct_vs_stepwise_fix_v1/full_v2/summary_refined_audit.json`

SHA-256：

| 文件 | SHA-256 |
|---|---|
| `pilot.jsonl` | `0db32cfc25dd857be51ef44097791541bae13b9c118bd4ec9b99417002cb12cf` |
| `summary_refined_audit.json` | `c0ef80f49b3914ca795c7b5fb1655a33d3da9d76e80c6ec4baf9f89510802845` |
