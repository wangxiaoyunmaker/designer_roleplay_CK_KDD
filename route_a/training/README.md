# Route A V10/V11/V12 训练代码

## V12 正式方法

V10/V11 保留为不可变基线。`route_a/configs/training_v12.json` 是 STOP pilot 的
基础配置；pilot 产出的 `outputs/v12-stop-pilot/frozen/training_v12.json` 才是正式配置。
它把 outer-train 划成 80% development-train
和 20% selection-val，并在 development-train 内运行 participant-level 三折。

Planner 每 300 次更新才执行一次完整 greedy rollout；这使五次 rollout 检查与
`patience=300` 对齐。每次检查仍覆盖完整的 inner validation 轨迹和冻结 STOP bias
候选，不改变训练损失、质量门槛或数据边界。

Realizer 按 schema 显式版本化：V10/V11 保留原始 `[z_pref; V30]` 46 维软前缀条件，
V12 才使用仅含 `z_pref` 的 16 维条件。因此部署 V12 修复不会让已有 V10/V11 基线
checkpoint 因投影层形状变化而失效。checkpoint 继续记录训练代码 SHA；推理会要求同一
Planner/Realizer 来自相同训练版本，但不会仅因之后的推理代码更新要求重训。

先运行 STOP 强度 pilot：

```bash
bash route_a/training/scripts/run_v12_stop_pilot.sh \
  processed_data/route_a_training_v1/formal /models/Qwen3-8B \
  outputs/v12-stop-pilot 0 0
```

pilot 只在 Q2 development fold 比较自然 12.75%、20% 和 25%，选择通过 rollout
与状态 Macro-F1 非退化门槛的最小比例。冻结选中的 config 后，每折运行：

```bash
bash route_a/training/scripts/run_one_v12_fold.sh \
  processed_data/route_a_training_v1/formal /models/Qwen3-8B \
  outputs/v12/seed42/fold0 0 42 <冻结的pilot-config>
```

该脚本依次训练三个 inner selection 模型、三个不读取目标折验证指标的严格 OOF
refit、development provisional 和 outer-refit。outer-refit 的训练步数取三个 inner
best update 的中位数；seed 17/2026 通过最后一个 `METHOD_ROOT` 参数复用 seed 42 的
方法选择结果，不重新查看 selection-val。

Best-of-N 和可选 RFT 使用：

```bash
bash route_a/training/scripts/run_v12_selection_and_rft.sh \
  processed_data/route_a_training_v1/formal /models/Qwen3-8B \
  outputs/v12/seed42/fold0 0 /models/bge-m3 /models/bge-m3.sha256 \
  route_a/configs/inference_v1.json 42 <冻结的pilot-config>
```

脚本生成 Q2-only calibration、三折 OOF 加 provisional RFT 候选、2:1
gold/pseudo Realizer continuation，并在隔离的 selection-val 上决定是否启用 RFT。
RFT 的“一轮”是对冻结训练表无放回遍历一次；每个被接收的 pseudo 精确配两条同记录
gold。方法选择产物固定由 seed 42 生成，最终 RFT checkpoint 则使用当前 final seed。
门槛失败是正常结果，最终模型自动回退 Stage C Realizer。

## 本地 dry-run

在正式训练前，先用一张空闲的 A800 跑 V12 全链路 smoke。它串行执行 3 组
inner selection、3 组严格 OOF、provisional 和 outer-refit，共 24 个一更新
checkpoint；只验证真实 Qwen3-8B 前反向、16 维 Realizer 条件、保存与阶段契约，
不代表模型质量达标：

```bash
bash route_a/training/scripts/run_v12_one_gpu_smoke.sh \
  data/route_a_training_v1/formal /models/Qwen3-8B \
  outputs/v12-gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ) 0
```

脚本要求所选物理卡至少有 60000 MiB 空闲显存，拒绝覆盖已有输出，并在输出根目录
保存 `preflight.txt`、`training.log`、`exit_status.txt` 和最终 `smoke_result.txt`。
smoke 配置关闭 rollout 质量门槛仅用于可运行性验证；正式训练仍必须使用 STOP pilot
冻结且开启门槛的配置。

STOP pilot 成功后，可用单卡、可恢复的全量矩阵启动器依次运行 5 个 outer fold 与
seed 17/42/2026，并在每个训练/RFT 阶段前检查磁盘余量。最后一个参数是最小保留
空间（MiB）；建议至少保留 12 GiB：

```bash
bash route_a/training/scripts/run_v12_full_matrix_sequential.sh \
  data/route_a_training_v1/formal /models/Qwen3-8B outputs/v12 \
  /models/bge-m3 /models/bge-m3.sha256 \
  outputs/v12-stop-pilot/frozen/training_v12.json 0 12288
```

该脚本拒绝 draft、未冻结的 STOP 配置、缺失的 BGE provenance 和磁盘余量不足；
已完成的 checkpoint 由各阶段启动器跳过，可在释放空间后从同一输出目录恢复。

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

当只有部分 GPU 空闲时，可逐折启动，外部用 `CUDA_VISIBLE_DEVICES` 绑定物理卡：

```bash
CUDA_VISIBLE_DEVICES=0 bash route_a/training/scripts/run_one_formal_fold.sh \
  processed_data/route_a_training_v1/formal \
  /models/Qwen3-8B \
  outputs/seed42/fold0 \
  0 42
```

脚本按 `aux selection/refit → planner selection/refit → realizer selection/refit`
串行执行，拒绝覆盖已有 fold 输出目录。多折并行时，每折必须使用不同输出目录和 GPU。
