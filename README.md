# KDD：最终论文代码与关键研究产物

整理依据：`paper/2125_submitted.pdf`，与用户提供的 **KDD v2.pdf** 字节完全一致。
本仓库已按最终论文筛选；完整旧归档保存在本地，仓库当前目录移除了旧实验、试跑、重复输出和投稿后探索。

| 内容 | 位置 |
|---|---|
| 最终 active4 + Q2 + frozen rules + Q7 ICL 方法 | `tools/prompt_baselines/run_main_method_r4_aligned.py` |
| 可直接使用的最终配置入口 | `tools/prompt_baselines/run_final_paper.sh` |
| 表 4、表 5 的最终生成与消融结果 | `outputs/main_method_r4_aligned_v1/` |
| SP / PP / ICL 基线及评估 | `outputs/prompt_baselines/evaluation_v10_q2_icl/` |
| 162 人原始问卷、画像、Q2 输入和人工金标 | `raw_data/`、`data/`、`processed_data/` |
| 冻结规则、语义先验及必要配置 | `artifacts/`、`route_a/configs/` |
| 最终序列指标的离线重计算 | `results/paper_sequence_metrics.json` |

从仓库根目录运行：

```bash
python3 -m tools.prompt_baselines.evaluate_final_paper
bash tools/prompt_baselines/run_final_paper.sh --output /tmp/kdd_prompt_check.jsonl --dry-run
```

生成需要模型和运行环境；安装说明及环境要求见 `requirements.txt`。规则重挖掘还须使用源码内冻结的软件版本。
密钥仅通过环境变量提供。数据为私有研究产物。
详细对应关系及未能完整复现的部分见 [PAPER_MAP.md](PAPER_MAP.md)；逐文件保留依据见 `PAPER_FILES.json`。
