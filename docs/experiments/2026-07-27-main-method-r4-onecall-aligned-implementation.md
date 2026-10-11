# R4 主方法：One-call Aligned C-K Prompting

状态：`IMPLEMENTED / TESTED / FORMAL PRIOR FROZEN / TWO 10-PERSON DEVELOPMENT SMOKES COMPLETE`

权威 runner：`tools/prompt_baselines/run_main_method_r4_aligned.py`

Prompt 版本：`q7-main-r4-onecall-aligned-full30-active4-q2-q7icl-boundary-v2`

## 1. 方法边界

R4 是纯 Prompt Engineering 方法。一次模型调用同时选择 KC 状态并生成与其一一对应的文本：

```json
{"units":[{"state":"KtoK","text":"……。"},{"state":"KtoC","text":"……。"}]}
```

程序只执行两个确定性投影：

```text
state_sequence = units[*].state
nl_answer = concat(units[*].text)
```

R4 不包含独立 Planner、Shared FaRM、逐步 dynamic V30、逐状态模型调用、Composer、规则 Prompt 或近邻/远邻 Q2。

## 2. 完整方法输入

`aligned_full` 按以下顺序组装：

1. 三条其他设计师 Q7 Seg2 人工对齐 ICL；
2. 目标设计师 PI；
3. 目标设计师静态 Full30；
4. 从全部 30 维中选择的 Q7-specific active4；
5. 目标设计师本人完整 Q2 序列和回答；
6. 正式 Q7 公开情境与 Seg2 任务；
7. aligned-unit JSON 输出要求。

Q7 ICL 只使用每个人工文本单元恰好对应一个 KC 状态的记录。目标本人始终排除。当前生成协议因此标记为 `TRANSDUCTIVE_RETRIEVAL`，不得声称 zero-shot 或 inductive generalization。

## 3. 消融条件

| 条件 | PI | 本人 Q2 | Full30 | active4 | V30 来源 |
|---|---:|---:|---:|---:|---|
| `aligned_full` | 是 | 是 | 是 | 是 | 目标本人 |
| `aligned_no_q2` | 是 | 否 | 是 | 是 | 目标本人 |
| `aligned_no_full30` | 是 | 是 | 否 | 是 | 目标本人 |
| `aligned_no_active4` | 是 | 是 | 是 | 否 | 目标本人 |
| `aligned_shuffled` | 是 | 是 | 是 | 是 | 同一个错配 donor |

五个条件使用相同 Q7 ICL、模型、解码参数和 target-level seed。`aligned_shuffled` 的 Full30 与 active4 必须来自同一位错配参与者，且 donor permutation 不得存在 fixed point。

## 4. R4 Q7 Semantic Prior

生成入口：`tools/prompt_baselines/generate_main_method_r4_q7_prior.py`

该入口只读取：

- 正式 instrument 中渲染出的完整 Q7 公开任务；
- V01--V30 长词典；
- all-30 candidate map；
- 六状态定义。

它不接受 PI、参与者 V30、Q2/Q7 回答或 Q7 金标路径。产物必须满足：

- schema 为 `q7_main_r4_public_semantic_prior_v1`；
- `pi_used = false`；
- `q7_gold_used = false`；
- `participant_records_used = false`；
- Q7 task SHA-256 与 runner 的实际渲染文本一致；
- instrument、长词典和 candidate map SHA-256 与运行时资源一致；
- V01--V30 恰好各出现一次。

由于 R4 已删除逐步 dynamic4，prior 每个维度只保存面向完整 Q7 的 `task_relevance` 和简短 `rationale`；不再生成或存储 position、segment、state 或 transition relevance。

旧 `q7_semantic_value_relevance_v2.json` 使用了与正式题目不一致的场景文本，R4 runner 会因 hash 不一致主动拒绝，不能通过改写 metadata 复用。

## 5. 输出与恢复合同

- JSON 根对象只能包含 `units`。
- 每个 unit 只能包含 `state` 和 `text`。
- 状态必须属于六状态集合；units 数为 1--25。
- 缺失句末标点仅做确定性补齐并记录，不调用修复模型。
- 生成 Prompt 不设置长度上下限；字符数与历史 200--400 区间仅保留为诊断字段，不筛样本、不重采样。
- 标签泄露、metadata 泄露、结构前置条件和虚构完成态证据只作为审计项，不筛样本、不重采样。
- 输出文件中任何已有 `(condition, target_id)`，无论成功或失败，续跑均不会再次追加；重复键会直接报错。

## 6. 当前验证

- R4 新增测试：10 项通过。
- R2/R3/R4 联合回归测试：27 项通过。
- Python 语法编译通过。
- `git diff --check` 通过。
- 合成数据端到端 dry-run：6 人、3 条件、18 个 jobs，通过；Q7 ICL target leak 为 0，shuffled fixed point 为 0。
- 真实金标中有 46 位设计师满足 aligned ICL 的单状态单文本与 200--400 字筛选合同。
- 正式 R4 prior 已冻结，SHA-256 为 `58c7764ce96b31cd8660be697a05a53ec26b27f23e77dc0763cd5725f66b6627`。
- 原始 Prompt 与 boundary-v2 Prompt 均完成同一批 10 人 `aligned_full` development smoke；对比见 `docs/experiments/2026-07-27-main-method-r4-boundary-v2-smoke-comparison.md`。
- 这 10 人已经用于 Prompt 诊断，应固定为 development smoke，不再用于选择后续 Prompt 版本或报告无偏主结果。
