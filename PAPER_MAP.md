# 最终论文与文件对应

唯一论文依据是 `paper/2125_submitted.pdf`（用户提供的 KDD v2.pdf，SHA256：e5524e62fba5c3436312749cafd3903aba5a9974f9cf41ad93eb3dbbbab2146d）。

| 论文内容 | 必要代码 / 产物 | 核对情况 |
|---|---|---|
| §3–4：30 维画像、Q2 C–K 输入 | `raw_data/*instrument*.json`、`data/route_a_training_v1/formal/`、`route_a/configs/` | 保留单份输入和必需的匿名 ID 对应 |
| §4：规则归纳与冻结 | `tools/kc_*`、三份 `*_uniform` 规则目录、freeze manifest / audit | SHA256 与冻结清单一致：12 + 49 + 29 = 90；validated = 0，保持 exploratory 标记 |
| §4–5：最终单次调用方法 | `run_main_method_r4_aligned.py`，条件 `ours`，版本 `trait-conditioned-aligned-ck-active4-q2-rules-q7icl-v1` | 最终保留 active4，不把 Full30 当主方法 |
| 表 4：Qwen3-8B / Llama-3.1-8B / GPT-4o Ours | 三个 `full162_ours_rules_*_final.jsonl` | 每份 162 人；序列指标重新计算与表格一致 |
| 表 4：SP / PP / ICL | `run_q7_prompt_baselines.py`、v10 `raw/predictions_*` 和 `sequence_personalization_full162_v1/` | 保留最终 full162 结果，不保留 heldout152 对照副本 |
| 表 4：CoSER / RAR-RIA | `external_baseline_inputs/` | 保留已有生成文本；本地缺其完整服务器训练代码、权重及 BGE 向量缓存 |
| 表 5：GPT-4o 五项消融 | `full162_ours_no_trait_rules_*`、`plus_full30_*_final`、`no_q2_*_final`、`no_active4_*_final`、`shuffled_rules_*_final` | 全部 162 人；保留原始总结及重计算指标 |
| §5.3 另外声明的两个 backbone | 两份 `full162_r5_active4_only_*` | 与正文声明对应，但没有列入论文表 4；不作为额外主表结果 |
| 附录表 7 初始回答候选规则 | `artifacts/q1_initial_rulebank_v1_candidate/` | 仅附录候选；不与最终 Seg2 冻结规则混用 |

## 运行与复现范围

`evaluate_final_paper.py` 使用原始 `evaluate_main_method_r2.py` 中的序列度量函数，适配最终 condition / prompt version、完整 D001–D162 cohort。
旧的独立 `evaluate_main_method_r4.py` CLI 强制旧 `aligned_full` / Full30 / 152 人协议，已移除，避免误用。
`run_final_paper.sh` 显式指定所有保留输入和最终条件；`--dry-run` 不调用模型。

表格 BGE 数值没有重新计算：已有最终文本保留，但完整 BGE encoder 快照和金标向量缓存未在本地归档中齐备。
`bge_inputs/all_current_systems.jsonl` 是单份合并文本输入，去除了逐系统的重复副本。
不宣称本仓库可以离线重跑 CoSER / RAR-RIA 训练或 BGE 全流程。

论文正文采用最终单次 aligned generation；附录仍有早期 R2 的两次续写描述。保存的最终输出与正文方法对应，不能将附录早期协议当作最终实现。
少数被最终实现导入的历史命名辅助模块（例如 R2 / R3、rulebank pilot）仍保留；其作用为数据加载、状态定义、trait selector 和规则渲染依赖。
同一源码模块内未被最终条件执行的历史分支未作算法重写，所有独立旧实验产物已移出当前目录。

清理包括：formal152 / aligned_full 的独立输出、R6–R9 投稿后实验、训练器试跑和 smoke、重试分片、旧规则版本、重复数据及结果、开发笔记和旧版论文。
