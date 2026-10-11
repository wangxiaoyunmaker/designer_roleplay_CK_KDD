# Q2 原文约束的逐状态 NL 对齐工具

本目录维护 Q2 状态—文本对齐的数据准备、第三方 API 调用和返回校验。`q2_alignment.py` 始终是纯离线工具；只有显式运行 `run_alignment_api.py` 并确认外部数据传输时才会调用外部 LLM。所有工具都不修改 `raw_data/`，也不读取 Q7。

## 为什么放在这里

`pipeline/` 保存可重复执行的数据处理代码；本任务的脚本、prompt 和测试彼此强耦合，因此放在同一子目录。运行生成的 job、模型返回和审计文件应放在：

```text
processed_data/q2_state_nl_alignment/
```

不要把运行结果写回 `raw_data/`。

## 文件

```text
q2_state_nl_alignment/
├── q2_alignment.py       # prepare / validate CLI
├── run_alignment_api.py  # OpenAI-compatible relay runner
├── prompt_template.md    # 严格原文约束的版本化 system prompt
├── README.md
└── tests/
    ├── test_q2_alignment.py
    └── test_run_alignment_api.py
```

## API 配置与环境

项目根目录使用以下文件：

```text
.env                 # 真实密钥与中转地址，权限 600，不提交
.env.example         # 无密钥模板
requirements-api.txt # API runner 依赖
```

推荐建立项目独立虚拟环境：

```bash
cd /Users/wy/Desktop/KDD
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-api.txt
```

`.env` 变量为：

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://yunwu.ai/v1
OPENAI_MODEL=gpt-4o
```

环境变量优先于 `.env` 中的同名变量。runner 不会输出密钥。

## 状态来源

工具要求显式选择状态来源，避免把旧标签静默当成新 K/C gold。

### `cards`

直接保留 Q2 `answer_value` 中的 cardId、顺序、重复项和卡片文字。输出会写入：

```json
{
  "source": "raw_q2_answer_value_card_sequence",
  "state_space": "legacy_card_id",
  "is_kc_gold": false
}
```

此模式适合检查“Q2 原文能覆盖多少原始卡片步骤”，不代表已经获得新 K/C 金标。

### `annotations`

读取外部 JSONL 中已经确认的状态序列。每行格式：

```json
{"participant_id":"D001","state_sequence":[{"state":"KtoC","subtypes":[]},{"state":"CtoK","subtypes":[]}]}
```

文件必须覆盖当前分析队列中的全部 `D###`。脚本只接受标签，不负责从旧卡片推导标签。

## 准备 prompt jobs

在 `/Users/wy/Desktop/KDD` 下运行：

```bash
mkdir -p processed_data/q2_state_nl_alignment

python3 pipeline/q2_state_nl_alignment/q2_alignment.py prepare \
  --root /Users/wy/Desktop/KDD \
  --state-source cards \
  --prompt pipeline/q2_state_nl_alignment/prompt_template.md \
  --output processed_data/q2_state_nl_alignment/alignment_jobs.jsonl \
  --audit processed_data/q2_state_nl_alignment/prepare_audit.json
```

每个 job 包含：

- 匿名 participant_id；
- Q2 共享任务情境；
- 固定状态序列及来源说明；
- 未经改写的 Q2 `text_supplement`；
- 可逐字回溯的 source units；
- 按 V01-V30 固定顺序排列的 30 个数值；
- 版本化 system prompt。

V30 在合成 prompt 中只是数值元数据，不被翻译成人格描述。最终训练模型应另行实现 `g_V(v30)` 连续前缀编码器。

## 调用第三方中转 API

先进行完全离线的检查；它不需要安装 SDK或读取密钥：

```bash
cd /Users/wy/Desktop/KDD
python3 pipeline/q2_state_nl_alignment/run_alignment_api.py \
  --jobs processed_data/q2_state_nl_alignment/alignment_jobs.jsonl \
  --output processed_data/q2_state_nl_alignment/alignment_results.jsonl \
  --dry-run \
  --limit 3
```

确认研究同意和中转站的数据保留政策允许发送 Q2 原文与 V30 后，建议先真实试跑 3 条：

```bash
.venv/bin/python pipeline/q2_state_nl_alignment/run_alignment_api.py \
  --jobs processed_data/q2_state_nl_alignment/alignment_jobs.jsonl \
  --output processed_data/q2_state_nl_alignment/alignment_results.jsonl \
  --failures processed_data/q2_state_nl_alignment/alignment_failures.jsonl \
  --limit 3 \
  --confirm-external-data-transfer
```

人工查看 3 条返回并运行下方 `validate` 后，删除 `--limit 3` 即可继续全部任务。runner 会读取已完成的 `job_id` 并自动跳过，因此同一命令可断点续跑。成功和失败分别写入两个 JSONL；失败不会污染交给验证器的结果文件。

## 模型返回格式

提供给 `validate` 的 JSONL 每行使用：

```json
{
  "job_id": "q2-align-D001",
  "result": {
    "participant_id": "D001",
    "question_id": "Q2",
    "alignment_status": "partial",
    "steps": [
      {
        "step_index": 1,
        "state": "K1",
        "source_unit_ids": ["U1"],
        "source_quote": "原始回答中的逐字片段",
        "aligned_text": "原始回答中的逐字片段",
        "alignment_type": "extractive",
        "training_eligible_for_text_loss": true
      }
    ]
  }
}
```

无法从原文支持的步骤必须使用：

```json
{
  "source_unit_ids": [],
  "source_quote": null,
  "aligned_text": null,
  "alignment_type": "no_text_support",
  "training_eligible_for_text_loss": false
}
```

## 校验模型返回

```bash
python3 pipeline/q2_state_nl_alignment/q2_alignment.py validate \
  --jobs processed_data/q2_state_nl_alignment/alignment_jobs.jsonl \
  --results processed_data/q2_state_nl_alignment/alignment_results.jsonl \
  --report processed_data/q2_state_nl_alignment/validation_report.json
```

有 error 时命令退出码为 1。验证范围包括：状态顺序、step 数量、source unit、quote 是否为原文子串、`no_text_support` 空值规则、文本损失资格和 source unit 重复使用。

## 测试

```bash
python3 -m unittest discover \
  -s pipeline/q2_state_nl_alignment/tests \
  -v
```

只使用 Python 标准库，无需安装额外依赖。

## 隐私与研究边界

- 输入问卷含私密身份字段，不应发送给外部模型；准备好的 job 不包含源 user_id。
- 尽管 job 已去除源 user_id，Q2 自由文本和 V30 仍属于研究数据；第三方中转调用属于外部数据传输，必须先确认知情同意、数据保留和删除政策。
- job 使用固定分析队列内按源 ID 排序产生的 `D###`，不写出映射表。
- Q2 NL 只从原始问卷读取，不默认使用 GPT-4o 扩写版。
- Q7 的 Seg1、Seg2、Seg3均不进入准备或校验流程。
- `cards` 输出不是新的 K/C gold，论文和数据卡必须保留这一来源说明。
- 外部 LLM生成的 aligned_text 是合成/弱监督数据，最终 Q7 评价仍需使用真实用户 gold。
