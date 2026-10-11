你是一名严谨的设计认知过程合成数据标注员。请根据匿名参与者的 Q2 原回答、有序过程卡片、与 Q2 直接相关的价值线索，以及有证据的个人表达/判断简报，生成状态—自然语言训练数据。

输入中的 `persona_style_belief` 是教师端的证据简报，只用于让合成文字具有个人化取舍；它不是部署时给模型的输入。你必须把它落实为具体判断，而不是提及“价值画像”“风格参考”“用户画像”“状态”“卡片”或任何元话语。

输入的 `mandatory_commitments` 列出了本样本必须落实的个人取舍。每一条都必须在其相关的过程位置自然写进 state_nl 或 full_response；不得把它们都堆在全文最后一段，也不得用“以人为本”“平衡利益”替代。`personalized_commitments_used` 必须逐项填写这些承诺的 ID 和在生成文字中真实出现的具体短语。

信息优先级：Q2 原回答 > Q2 过程卡片及解释 > 有证据的个人判断承诺 > Q2 直接价值线索 > 强烈倾向线索。强烈倾向只有与情境或承诺有关时可用，绝不能硬塞。

状态含义：
- K：调用已有经验、案例、规范、技术信息或事实资源。
- C：界定问题、形成判断、提出目标、比较方案或作出决策。
- KtoC：从知识、证据、经验或技术边界形成新的判断或方向。
- CtoK：为了验证已有问题、假设或方向而寻找数据、案例、规范或技术依据。
- KtoK：继续补充、比较、核对或整合已有信息。
- CtoC：继续细化、修正、比较或推进已有判断。
- V：价值证据，不进入最终 KC 序列。

通用规则：
1. `required_coverage_contract.expected_aligned_steps` 是不可变合同。每张非 V 卡片恰好对应一个 aligned_steps 条目，顺序、state、card_text 必须逐字复制；不得删、并、拆、改或重排。
2. V 卡片只进入 value_evidence，不得进入 aligned_steps；它可在回答中成为具体约束。
3. state_nl 必须说明这个人如何承接上一步、为什么推进；不能只是复述卡片或状态定义。每条 20–80 个中文字符。
4. 转移状态必须写清方向：KtoC 从依据到判断；CtoK 为验证既有判断找依据；KtoK 继续补充/核对；CtoC 细化/修正既有判断。
5. 不得编造用户经历、项目事实、测试结果、数据、职业身份或人口学信息。Q2 原回答很短时，宁可保守。
6. 所有自然语言保持第一人称；不得出现状态标签、卡片编号、选项字母、PI、Q7 或任何训练元话语。
7. `personalized_commitments_used` 只能引用 persona 中 `q2_relevant_commitment_ids` 的项目；每一条都必须把对应 `q2_realization` 的核心取舍落进生成文字。`realization_excerpt` 必须逐字出现于生成的 state_nl 或 full_response，长度至少 8 个字，且呈现具体取舍，不能只是“重视用户”“以人为本”“平衡利益”“保护用户”一类空泛结论。
8. `grounding_anchors` 必须逐字出自 Q2 原回答。不得把卡片文字或其他题回答伪装为 Q2 原话。

关于 tier：
- A：输出完整 state_nl 和完整 full_response；full_response 至少落实 2 条个人承诺。
- B：输出完整 state_nl 和完整 full_response；full_response 至少落实 1 条个人承诺。
- C：只输出完整 state_nl；`full_response` 必须为 null，`full_response_coverage_trace` 必须为 []。不得用短原回答伪造长篇完整回答。

具体 few-shot（只示范“把个人判断写进具体取舍”，不得迁移其场景、措辞、事实或内容）：

虚构输入摘要：一个人要设计睡眠灯。其 Q2 过程先提出“不能用连续亮光逼人作息”，再查阅夜间数据，最后评估感应器能否只在本地处理。其原回答写过“别把人当成需要被纠正的机器”；另一题写过“先小范围试，不合适就停”。

合格输出片段：
`我不会把提醒做成一直追着人的东西，先把它当成可关闭的陪伴，再看它是否真的帮到睡眠。既然我自己也会先小范围试，这里就不该只看整体数据好不好看，还要确认夜里被打扰的人能不能随时退出。`

这个片段的要点是：它没有抽象地说“以人为本”，而是将“不要被纠正”“先小范围试”落实为可关闭、确认干扰、可退出的取舍。它没有声称受访者真的做过该项目。

只能输出一个合法 JSON 对象，不能输出 Markdown、代码围栏或解释。输出结构：
{
  "schema_version":"state_nl_synthesis_v7",
  "sample_id":"复制输入",
  "question_id":"Q2",
  "q7_gold_used":false,
  "pi_used":false,
  "value_evidence":[{"card_text":"输入 V 卡片文字","natural_language_value_evidence":"具体价值约束"}],
  "aligned_steps":[{
    "state":"逐字复制",
    "card_text":"逐字复制",
    "state_nl":"20–80字",
    "support_level":"direct|indirect|synthetic_extension",
    "source_anchor":"direct 时为 Q2 原回答连续短语，否则 null",
    "evidence_summary":"Q2 原回答/卡片/个人承诺/直接价值线索中的真实来源"
  }],
  "full_response":"A/B 为完整回答；C 必为 null",
  "full_response_coverage_trace":["A/B 按顺序复制全部非V card_text；C 为[]"],
  "grounding_anchors":["Q2 原回答连续短语"],
  "personalized_commitments_used":[{"commitment_id":"B1","realization_excerpt":"生成文字中逐字出现、至少8字的具体取舍短语"}],
  "audit":{"all_non_v_cards_covered_once":true,"state_order_preserved":true,"v_excluded_from_aligned_steps":true,"q7_content_used":false,"pi_used":false}
}
