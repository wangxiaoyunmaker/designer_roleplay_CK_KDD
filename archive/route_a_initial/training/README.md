# Route A V10 训练代码

## 本地 dry-run

draft 必须显式授权：

```bash
python3 -B -m route_a.training prepare \
  --dataset-dir processed_data/route_a_training_v1/draft \
  --config route_a/configs/training_v1.json \
  --outer-fold 0 \
  --allow-draft-uncleaned
```

它不会加载 Qwen 或 GPU 库，会检查数据 checksum、五折隔离、内层划分、样本数、提示、状态空间和泄漏边界。

## 服务器环境

建议在单独虚拟环境安装：

```bash
python -m pip install -r route_a/training/requirements-training.txt
```

版本下限按 Qwen 官方 Quickstart 设置：`transformers>=4.51.0`、建议 `PyTorch>=2.6`。参考：<https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html>。

主方案是单卡 BF16 LoRA。显存实测不足时才加 `--qlora`。

先把 `Qwen/Qwen3-8B` 下载到服务器的只读本地目录并保存文件清单。正式命令必须传 `--model-path /models/Qwen3-8B`；代码使用 `local_files_only=True`，并校验官方 8B 架构关键字段，防止误传 Qwen3-Base 或其他尺寸。checkpoint 会记录整个快照的 SHA，禁止训练时跟随远程 `main` 漂移。

```bash
python -m route_a.training train-aux \
  --dataset-dir processed_data/route_a_training_v1/formal \
  --config route_a/configs/training_v1.json \
  --model-path /models/Qwen3-8B \
  --outer-fold 0 \
  --output-dir outputs/fold0/aux \
  --device-index 0 --seed 42
```

Planner 和 Realizer 分别使用 `train-planner`、`train-realizer`，并用 `--upstream-checkpoint` 指向上一阶段目录。每个命令都支持 `--dry-run`。

每个 outer fold 必须先运行 `--fit-mode selection` 得到 checkpoint，再将该目录直接交给 refit：

```bash
--fit-mode outer-refit --selection-checkpoint <同阶段selection目录>
```

三个阶段的 upstream checkpoint 必须使用相同 fit mode。`outer-refit` 会从同阶段 selection manifest 自动读取 `best_update`，校验数据、模型快照、随机种子和组件哈希，并记录 selection manifest SHA；不接受手工抄写更新步数。Q7 推理只接受 `outer-refit` checkpoint，避免把内层 validation 数据永久丢出训练。

正式训练不要传 `--allow-draft-uncleaned`。当前 draft 只用于代码联调和服务器 smoke test。
