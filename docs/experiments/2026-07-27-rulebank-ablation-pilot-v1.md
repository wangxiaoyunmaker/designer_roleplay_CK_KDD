# 30 人冻结规则库快速消融（v1）

## 实验定位

本实验是 `POST_HOC_DIAGNOSTIC` 级快速消融，用于判断冻结规则层是否值得进入后续正式实验，不用于修改、增删或重新调参单条规则。

- 被试：固定 30 人（D002-D031）。
- 模型：Qwen3-8B，greedy decoding，`max_new_tokens=512`。
- 规则：Seg1 Q1 候选库 18 条；Seg2 `KC_RULEBANK_V1` 冻结库 90 条（Q2 单维 12、Q2 双维 49、Q3-Q6 情境 29）。
- 路由：每个激活层最多注入 1 条规则；画像强度优先、held-out gain 次级排序；双维冲突时作为单维修正。
- 防泄漏：330 条回答生成完并封存后才读取 Q7 gold；生成记录全部标记 `q7_gold_used_for_generation=false`。
- 指标：KC-NES、BGE own-user margin、chain diversity、解析失败数；差值置信区间为 10,000 次被试配对 bootstrap 95% CI。

## 完整性检查

- 11 个条件 x 30 人 = 330 条，条件-被试键 330 个且无重复。
- 每个条件均为 30 条，JSON/KC/NL 解析失败为 0。
- 两个 shuffled 条件采用固定无自映射置换，自映射数为 0。
- 冻结库校验仍为 PASS：12 + 49 + 29 = 90，validated rule count 仍为 0。

## 条件均值

| Segment | 条件 | KC-NES | BGE own-user margin | 唯一链数 | Chain diversity |
|---|---|---:|---:|---:|---:|
| Seg1 | none | 0.0828 | 0.0000 | 1 | 0.0000 |
| Seg1 | Q1 single | 0.1206 | -0.0041 | 3 | 0.3667 |
| Seg1 | Q1 double | 0.1939 | 0.0038 | 3 | 0.5000 |
| Seg1 | Q1 both | 0.2328 | -0.0058 | 6 | 0.7000 |
| Seg1 | Q1 shuffled | 0.1672 | -0.0008 | 6 | 0.7000 |
| Seg2 | none | 0.2478 | 0.0000 | 1 | 0.0000 |
| Seg2 | Q2 single | 0.2773 | -0.0002 | 5 | 0.5333 |
| Seg2 | Q2 double | 0.2543 | -0.0035 | 6 | 0.5667 |
| Seg2 | Q2 both | 0.2307 | -0.0006 | 10 | 0.7000 |
| Seg2 | full correct | 0.2432 | -0.0030 | 16 | 0.8333 |
| Seg2 | full shuffled | 0.2321 | 0.0046 | 16 | 0.8333 |

`none` 条件在固定场景、greedy decoding 下为所有人生成同一 KC 链，因此其 chain diversity 为 0，BGE margin 数值上接近 0；这不代表具备个体化能力。

## 预注册关键负对照

| 对比（correct - shuffled） | 指标 | 差值与 95% CI | 判定 |
|---|---:|---:|---|
| Seg1 Q1 both - Q1 shuffled | KC-NES | +0.0656 [-0.0344, 0.1617] | 未通过 |
| Seg1 Q1 both - Q1 shuffled | BGE margin | -0.0049 [-0.0131, 0.0031] | 未通过 |
| Seg2 full correct - full shuffled | KC-NES | +0.0111 [-0.0537, 0.0836] | 未通过 |
| Seg2 full correct - full shuffled | BGE margin | -0.0077 [-0.0203, 0.0047] | 未通过，点估计方向相反 |

四个置信区间均跨 0。正确画像路由没有稳定优于错配画像路由，因此本次快速消融没有验证规则的个体特异性。

## 分层观察

Seg1 中，Q1 double 相对 none 的 KC-NES 提升为 +0.1111，95% CI [0.0333, 0.1883]；Q1 both 相对 none 提升为 +0.1500，95% CI [0.0711, 0.2294]。这说明 Q1 规则能改变并部分改善 KC 链结构，但其 correct-vs-shuffled 对照未通过，BGE margin 也未改善，所以只能视为结构控制证据，不能视为个体化有效性证据。

Seg2 中，Q2 single 的平均 KC-NES 最高（0.2773），相对 none 为 +0.0295，95% CI [-0.0158, 0.0732]，尚不稳定。Q2 double、Q2 both 和 full correct 均未在 KC-NES 或 BGE margin 上显示可靠优势。full correct 将 chain diversity 从 0 提高到 0.8333，但 full shuffled 同样为 0.8333，说明多样性来自规则条件化本身，不能证明规则与对应用户匹配。

## 决策

1. 不把任何规则或规则层标记为 validated，冻结的 90 条规则保持原样。
2. 当前不支持将 Seg2 full 三层规则作为主实验默认配置。
3. Q2 single 可作为后续正式规模验证中唯一较有希望的 Seg2 探索层，但本次结果不足以启用它作为已验证方法。
4. Q1 double / both 可保留为 Seg1 结构控制候选；在人工直觉匹配或更强的 matched-minus-mismatched 评价通过前，不作个体化主张。
5. 下一步应扩大被试规模并补跑预注册的人评（Seg1 直觉匹配、Seg2 V30 matched-minus-mismatched）；仍只允许启用或禁用完整层，不得回看 Q7 后修改 90 条规则。

## 局限

- BGE-M3 在本次快速诊断中由 Transformers 直接加载并使用归一化 CLS dense pooling，未使用 FlagEmbedding 封装。
- 尚未运行人工直觉匹配、人工语义判断或 V30 matched-minus-mismatched 评分。
- 30 人样本的置信区间较宽，本实验适合筛除明显无效方案，不足以支持论文中的最终有效性结论。

原始生成、逐人指标、汇总结果及 SHA-256 封存清单位于 `outputs/diagnostics/rulebank_ablation_pilot_v1/`。
