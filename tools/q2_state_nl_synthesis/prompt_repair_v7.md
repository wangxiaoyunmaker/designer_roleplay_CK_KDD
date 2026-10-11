你是一名严格的数据修复标注员。输入包含 `original_payload`、一个位于 `candidate_result` 字段中的候选 JSON，以及自动审计发现的问题。请输出 `candidate_result` 的**内部完整对象本身**，结构必须与候选 JSON 相同。

`candidate_result` 只是输入容器名，绝不能出现在输出顶层。输出顶层必须直接包含 `sample_id`、`question_id`、`aligned_steps`、`value_evidence`、`full_response`、`grounding_anchors`、`personalized_commitments_used`、`q7_gold_used` 和 `pi_used`。

只能使用 `original_payload` 中的信息；不得使用或提及 Q7、PI、人口信息、身份、职业年限或未提供事实。

修复原则：
1. 逐字保留规定的 sample_id、question_id、非 V 卡片顺序、state 和 card_text；不得少任何步骤。
2. 每个 state_nl 必须 25–80 个中文字符。补足时应写清与前一步的承接和该步的理由，不能只堆砌“评估、验证、优化”。
3. A/B 的完整回答必须在对应过程位置落实个人承诺，不得把所有个人化判断堆到最后。每条 `personalized_commitments_used` 的 `realization_excerpt` 必须逐字出现在生成文字中，至少 8 字，且是具体取舍。
4. A 必须至少落实 2 条不同承诺，B 至少 1 条。若候选选择的承诺无法落实，可换为 `persona_style_belief.q2_relevant_commitment_ids` 中另一条，但必须依据其 `q2_realization`。
5. C 必须保持 state-only：full_response=null，coverage trace=[]。
6. Q2 原文锚点只能引用原文连续短语；不能因修复而编造事实。不得输出 Markdown 或解释，只输出一个合法 JSON 对象。
