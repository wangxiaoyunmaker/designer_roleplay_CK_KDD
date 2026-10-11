# KDD — Trait-conditioned C–K Trajectory Generation（私有研究归档）

2026-10-11 整理，对应 Learning Trait-Conditioned Expert Decision Trajectories for LLM Agents under Wicked Problems，submission 2125。

## 投稿版本与后续实验

已将论文 Table 5 的 162 人 GPT-4o 结果对齐到 `results/submission/r4_r5/full162_gpt4o_current_ablation_sequence_summary.json`：Ours 的 KC-NES=0.2637873443、KCDR=0.5904424530、Unique=132、TopShare=0.0308641975。投稿主方法是一次调用对齐生成、active4、本人 Q2、其他参与者 Q7 ICL 与探索性规则；Full30 是消融。

权威 runner 是 `tools/prompt_baselines/run_main_method_r4_aligned.py`，投稿主条件为 `--conditions ours`，版本为 `trait-conditioned-aligned-ck-active4-q2-rules-q7icl-v1`。同一文件含较早 R4 和后续 R6–R9，必须显式指定 conditions，不能把默认所有条件作为投稿复现命令。

R8 的无 Q7 ICL / Q2 行为证据优先 / 无当前规则路线是后续实验，formal152 与 development10 汇总独立放在 `results/post_submission/r6_r9/`。早期 R2 是两阶段协议；论文附录保留了 R2 描述，与一次调用主方法有版本差异。这些后续或早期文件不替代投稿 full162 主表。

## 最终代码与关键产物

| 内容 | 路径 | 用途 |
|---|---|---|
| 主方法与消融 | `tools/prompt_baselines/run_main_method_r4_aligned.py` | ours 与 matched ablations |
| Prompt-only baselines | `tools/prompt_baselines/run_q7_prompt_baselines.py` | SP / PP / ICL |
| 指标与标准化评测 | `tools/prompt_baselines/evaluate_main_method_r2.py`, `evaluate_main_method_r4.py`, `route_a/evaluation/` | sequence、BGE、统计协议 |
| selector 与价值词典 | `route_a/data/build_v7_step_realizer.py`, `route_a/training/value_profile.py`, `route_a/configs/` | runner 实际导入依赖 |
| 规则提取 | `tools/kc_operational_ontology.py`, `tools/kc_v30_relation_ontology.py`, `tools/kc_aux_preference_sweep.py` | 三类规则来源 |
| 冻结规则库 | `artifacts/kc_rulebank_v1_freeze_manifest.json` 及三个 uniform 层 | 12 + 49 + 29 = 90 条，validated=0 |
| Q7 公共语义 prior | `artifacts/q7_main_r4_public_semantic_prior_v1.json` | 不读取逐人 Q7 的任务相关度 |
| 问卷原始数据 | `raw_data/` | 162 人版本、题目、量表及原始 checksums |
| formal 训练中间产物 | `data/route_a_training_v1/formal/` | participants、planner、realizer、folds、manifest、checksums |
| 清洗与 Q7 人工金标 | `processed_data/` | Q7 adjudicated v1/v2，数据处理版与候选版分别保留 |
| Q2 对齐与文本合成 | `intermediate/`, `outputs/q2_state_nl_synthesis/`, `outputs/q1_q6_cleaning/` | 分阶段中间产物与验证记录 |
| 投稿族逐人预测和消融 | `outputs/main_method_r4_aligned_v1/` | final、raw、分片、结果与 value input 的原始记录 |
| 原始 prompt baseline 产物 | `outputs/prompt_baselines/` | 预测、full162 / heldout152 指标 |
| 后续 R6–R9 产物 | `outputs/main_method_r6_pilot_v1/` | development10 与 formal152 的原始输出 |
| 可快速查看的汇总 | `results/submission/`, `results/baselines/`, `results/post_submission/` | 与逐人原始产物分别保留 |
| 主 runner 身份映射 | `processed_data/route_a_training_v1/private/user_id_mapping.json` | --uid-map 使用此 JSON；CSV 私有映射另存在 private/ |
| 投稿 PDF | `paper/2125_submitted.pdf` | 当前本地提交稿参照 |
| 早期 Route A | `archive/route_a_initial/` | 与主 route_a 快照分开保存 |

## 使用与限制

在仓库根目录查看入口：

```bash
python3 -m tools.prompt_baselines.run_main_method_r4_aligned --help
python3 -m tools.prompt_baselines.evaluate_main_method_r4 --help
python3 tools/verify_kc_rulebank_freeze.py --manifest artifacts/kc_rulebank_v1_freeze_manifest.json
```

`route_a/` 来自本地 realizer-v13-6 快照，保留 runner 导入的 selector 和词典支持。训练/推理/评测历史代码均保存，但远端 CoSER/RAR-RIA 完整训练代码、checkpoint 和 GPU 环境未全部同步到本地；本地存在的输入与结果已经保留。此次没有重新训练、调用 LLM 或重新计算全套论文指标。

90 条规则是冻结的探索性规则，不能称为验证过的心理规律。后续诊断见 `docs/experiments/2026-07-27-r5-to-r8-iterative-method-optimization.md`。Table 5 已对齐，其余模型结果请按保存的汇总与预测核对。

原始身份和文本按源文件保留；此前 raw_data README 的“不适合公开”说明仍适用。见 `PUSH.md`，只推 Private 仓库。来源与 hash 见 `SOURCE_MANIFEST.json`、`CHECKSUMS.sha256`。密钥、环境、embedding cache、日志和重复压缩包已排除，原目录均保留。

## 本次归档检查

文件复制 hash、Python 语法、密钥模式扫描和 GitHub 单文件大小检查通过；没有 50 MiB 以上文件。冻结规则 90 条校验通过；主方法 18 项现有测试通过；真实数据 dry-run 已成功组装投稿条件输入，未调用模型。主 runner 与 evaluator 的 --help 正常。`tests/test_build_dual_release.py` 中的模拟密钥已脱敏，该历史发布测试不用于本次归档检查。

GitHub 私有仓库：https://github.com/wangxiaoyunmaker/KDD。后续在本目录提交更新，并运行 `git push origin main`。
