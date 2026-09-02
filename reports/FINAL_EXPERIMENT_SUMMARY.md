# BLIP-2 / InstructBLIP VQA 项目实验汇总

更新时间：2026-08-22（Australia/Sydney）

## 1. 项目目标与研究问题

本项目研究 BLIP-2 与 InstructBLIP 在 VQAv2 上的视觉问答能力，并使用参数高效微调（PEFT/LoRA）适配跨模态推理任务。除标准 VQA 准确率外，项目还评估视觉依赖、yes/no 幻觉、VQAv2 complementary pairs 和 COCO 物体 grounding。

实验围绕四个研究问题组织：

- **RQ1：** BLIP-2、InstructBLIP 和不同 LoRA 作用范围在标准 VQA 上如何比较？
- **RQ2：** 标准微调和显式对比目标能否增强视觉依赖并减少幻觉？
- **RQ3：** COCO grounding、错误驱动 grounding 和知识保持式蒸馏能否同时提高 grounding 与 VQA？
- **RQ4：** 1,000 条验证子集上的结论能否推广到完整 VQAv2 validation？
- **RQ5：** 标准 VQA 上的改善能否推广到 POPE、CHAIR 和 HallusionBench
  等外部幻觉协议？

## 2. 统一实验协议

- 数据集：VQAv2 train（443,757 问题）与 validation（214,354 问题）。
- 主干模型：Salesforce BLIP-2 Flan-T5-XL、InstructBLIP Flan-T5-XL。
- 推理：4-bit 量化、确定性生成、`max_new_tokens=10`。
- 标准对比子集：VQAv2 validation 中 seed=42 的固定 1,000 条样本。
- 最终评测：E6 在全部 214,354 条 validation 问题上评测。
- 主指标：VQA soft accuracy；细分 Number、Other、Yes/No。
- 幻觉代理：false-yes、false-no、无效 yes/no 输出。
- 视觉依赖：正常图像与错配图、灰图、噪声图的准确率下降和答案变化率。
- Complementary：正确图像相对交换图像的 token log-probability margin 和配对偏好率。
- Grounding：存在性正例、存在性负例和计数三类，共 1,000 条 COCO 派生验证样本。
- Multiple-choice：5,000 条 VQAv2-derived MCQ；正确答案来自官方
  `multiple_choice_answer`，干扰项从相同 question/answer type 确定性采样。
- 外部幻觉：官方 COCO POPE random/popular/adversarial 各 3,000 题；固定
  seed=42 的 500 张 COCO val2014 图像 CHAIR 开放描述；完整 HallusionBench
  1,129 题。

注意：E0–E14 的标准 VQA 表使用同一 1,000 条子集；E15 使用完整 validation，不能将其细分类别数字与 1,000 条结果当作同规模重复实验。

## 3. RQ1：模型、提示词与 PEFT 选择

### 3.1 固定 1,000 条验证子集

| Run | 模型/方法 | 训练样本 | Overall | Number | Other | Yes/No | False-yes | False-no |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | BLIP-2 zero-shot | 0 | 59.76 | 44.26 | 45.75 | 82.68 | 19.59 | 34.04 |
| E1 | BLIP-2 Q-Former LoRA r8 | 1k | 61.98 | 51.83 | 49.30 | 81.56 | 17.53 | 38.83 |
| E2 | BLIP-2 LLM LoRA r8 | 1k | 65.17 | 55.48 | 53.87 | 82.81 | 15.38 | 38.30 |
| E4 | BLIP-2 Dual LoRA r8 | 1k | 65.15 | 54.96 | 54.23 | 82.45 | 15.38 | 38.83 |
| E3 | InstructBLIP zero-shot，默认提示 | 0 | 67.51 | 59.74 | 56.49 | 84.22 | 18.68 | 21.55 |
| E3b | InstructBLIP zero-shot，短答案提示 | 0 | 70.36 | 63.83 | 61.28 | 84.17 | 16.49 | 28.49 |
| E5 | InstructBLIP LLM LoRA r8 | 1k | 70.54 | **65.30** | 61.54 | 83.85 | **15.26** | 31.07 |
| **E6** | **InstructBLIP LLM LoRA r8** | **10k** | **71.29** | 62.52 | **63.19** | **84.48** | 18.85 | **26.26** |

E1 仅训练 473,088 个参数（0.0202%）；E2/E5/E6 训练 4,718,592 个参数（约 0.201%）；Dual LoRA 训练 5,191,680 个参数（0.221%）。所有主 PEFT 实验均使用 rank 8、alpha 16、dropout 0.05、1 epoch。

**Finding 1.** InstructBLIP 的指令预训练比在 BLIP-2 上扩大 LoRA 作用范围更重要。短提示的 InstructBLIP zero-shot 达到 70.36，比 BLIP-2 最佳 1k LoRA 结果 65.17 高 5.19 点；BLIP-2 Dual LoRA 比 LLM-only 还低 0.02 点。

**Finding 2.** 输出约束本身是强基线。将 InstructBLIP 的默认提示改为短答案提示，使准确率从 67.51 提高到 70.36（+2.85），其增益明显大于从 zero-shot 到 1k LoRA 的 +0.18。

**Finding 3.** 将 InstructBLIP LLM LoRA 从 1k 扩大至 10k，使 Overall 从 70.54 提高到 71.29（+0.75），主要增益来自 Other 和 Yes/No；Number 从 65.30 降至 62.52，说明扩大普通 VQA 数据并未稳定改善计数推理。

## 4. RQ4：完整 VQAv2 validation

| 模型 | Eval N | Overall | Number | Other | Yes/No | False-yes | False-no | Invalid yes/no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E6（1k 子集） | 1,000 | 71.29 | 62.52 | 63.19 | 84.48 | 18.85 | 26.26 | 3.65 |
| E15：E6 直接生成 | 214,354 | 70.61 | 52.32 | 63.64 | 86.14 | 19.89 | 20.61 | 3.37 |
| **E19：E6 + 短答案重排** | **214,354** | **71.62** | **52.32** | **63.62** | **88.86** | **20.21** | **20.57** | **0.22** |

全量验证结果文件位于：

`/root/autodl-tmp/vision-language/outputs/E15_E6_instructblip_full_vqav2_val214354/metrics.json`

**Finding 4.** 1,000 条子集将 E6 的 Overall 高估了 0.68 点，并将 Number 高估了 10.20 点；Other 与 Yes/No 的全量结果反而更高。E6 adapter 的直接生成结果应报告 70.61，而 71.29 只能作为统一消融子集上的可比结果；加入 E19 后的完整系统结果另行报告为 71.62。

E19 只重排被 yes/no router 识别且生成不合规的答案，不改变 E6 adapter。
它处理 2,570 条候选，使 Overall 提高 1.011 点；配对 bootstrap 的 95% CI
为 +0.968 到 +1.054 点。增益几乎全部来自 Yes/No 和输出合规性，Number
没有实质变化，false-yes 还略升。因此 E19 是有效的系统级解码改进，不能
表述为视觉事实理解或训练后对齐能力的提升。

### 4.1 Multiple-choice 补充评测（E16）

VQAv2 v2 官方问题包不包含候选选项，因此本项目构建了明确标注为
**VQAv2-derived MCQ** 的补充评测，而不将其称为官方VQAv2多选榜单。评测集包含
5,000题：1,869道Yes/No两选题、647道Number四选题和2,484道Other四选题。
两选题与四选题的加权随机基线为34.345%。

| Run | 模型 | Overall | Number | Other | Yes/No | 2-option | 4-option | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E16a | InstructBLIP zero-shot | 83.46 | 70.17 | 91.06 | 77.96 | 77.96 | 86.75 | 0.00 |
| **E16b** | **E6 LLM LoRA** | **85.12** | **75.89** | **92.15** | **78.97** | **78.97** | **88.79** | **0.00** |

同题配对结果为：4,130题两者都正确、126题仅E6正确、43题仅zero-shot正确、
701题两者都错误。连续性校正McNemar检验为 $\chi^2=39.79$，
$p=2.83\times10^{-10}$。

**Finding 9.** E6在派生多选任务上达到85.12%，比zero-shot提高1.66点，且两组均无无效选项输出；最大提升来自Number（+5.72）。这为系统能够处理multiple-choice问题提供了直接证据，但由于干扰项是自动构造的，不能将该成绩与官方多选基准直接比较。

## 5. RQ2：视觉依赖、对比学习与幻觉

### 5.1 E6 视觉依赖诊断

| 条件 | 准确率下降 | 答案变化率 |
|---|---:|---:|
| 错配图像 | 34.75 | 65.9 |
| 灰图 | 34.46 | 66.5 |
| 噪声图 | 36.77 | 68.9 |

E6 在正常图像上的诊断准确率为 71.49，错配图像为 36.74。这证明模型确实使用视觉信息；但错配后仍有 34.1% 的答案不变，视觉依赖并不充分。

### 5.2 显式对齐消融

| Run | 方法 | VQA | 错配下降 | 错配答案变化 | Complementary margin | 配对偏好率 |
|---|---|---:|---:|---:|---:|---:|
| E6 | 基线 | 71.29 | **34.75** | **65.9** | **1.695** | 92.7 |
| E8a | VQA continuation 控制 | 71.46 | 34.33 | 63.4 | — | — |
| E8b | 随机错配图损失 | 70.75 | 34.12 | 64.4 | — | — |
| E9a | Complementary 控制 | 71.10 | 33.65 | 66.1 | — | — |
| E9b | Complementary log-prob loss | 71.31 | 34.13 | 65.9 | — | — |
| E10a | Hard-pair 控制 | **71.43** | 34.57 | 65.0 | 1.472 | **92.9** |
| E10b | Hard-pair log-prob loss | 71.24 | 34.48 | 65.2 | 1.458 | 92.8 |
| E11a | 70/30 混合 Q-Former 控制，best | 71.29 | 34.75 | 65.9 | 1.695 | 92.7 |
| E11b | 70/30 Q-Former 对比，best | 71.29 | 34.75 | 65.9 | 1.695 | 92.7 |

E11 两组的 best checkpoint 都是 step 0，因此表中的 E11 指标等同于 E6，而不是训练后获得的新提升。

**Finding 5.** 当前标量错配损失、complementary log-probability loss 和 hard-pair loss 均未同时提高标准 VQA 与视觉依赖。E8b 相对控制组损失 0.71 VQA 点；E10b 相对控制组损失 0.19 点且 margin 更低；E11 的选择器回退到 step 0。该系列应作为有控制组支持的负结果报告。

**Finding 6.** False-yes 的降低可能只是回答偏置迁移。E2 的 false-yes 最低（15.38），但 false-no 高达 38.30；E6 的两者更平衡（18.85/26.26），全量为 19.89/20.61。因此不能仅凭 false-yes 宣称“幻觉减少”。

### 5.3 外部幻觉基准（H1–H3）

#### 官方 COCO POPE

| 模型 | Macro accuracy | Precision | Recall | F1 | Yes ratio | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| InstructBLIP zero-shot | 83.94 | **91.28** | 75.31 | 82.47 | 41.37 | 0.00 |
| **E6 LLM LoRA** | **84.93** | 89.91 | **78.96** | **84.02** | 44.02 | 0.00 |

E6 在合并 9,000 题上比 zero-shot 高 0.989 点，配对 bootstrap 95% CI
为 +0.644 到 +1.333，McNemar $p=2.15\times10^{-8}$。提升主要来自 recall，
同时 precision 降低、yes ratio 上升。pair-logprob 解码的 macro accuracy
为 84.79，反而比直接生成低 0.14 点，说明 E19 风格重排不能自动推广成
POPE 幻觉改进。

#### CHAIR 开放描述（固定 COCO val2014 500 图）

| 模型 | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ | Mean object mentions |
|---|---:|---:|---:|---:|
| **InstructBLIP zero-shot** | **31.80** | **10.93** | 62.16 | 5.38 |
| E6 LLM LoRA | 43.40 | 14.53 | **66.98** | 5.41 |

E6 的 CHAIRs 恶化 11.60 点（95% CI +7.40 到 +16.00），CHAIRi 恶化
3.60 点（+2.04 到 +5.19），但 object recall 提高 4.82 点（+3.35 到
+6.38）。该结果说明 E6 会提到更多真实物体，同时也引入更多不存在物体。
这里采用原 CHAIR BSD 规则的 Python 3 端口；为适配当前环境，以确定性
regex/singularization 替代原 Python 2 的 pattern/nltk tokenizer，因此报告
时应注明实现差异。

#### HallusionBench（完整 1,129 题）

| 模型 | Question | Question-pair | Figure | Easy pair | Hard pair | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **InstructBLIP zero-shot** | **54.30** | **18.68** | **24.86** | 52.97 | **46.98** | **39.22** | 54.34 |
| E6 LLM LoRA | 52.70 | 16.70 | 23.12 | **56.26** | 41.86 | 56.28 | **35.33** |

E6 的 question accuracy 变化为 −1.594 点，95% CI −4.163 到 +0.974，
McNemar $p=0.239$，没有显著改善。其 yes ratio 从 41.98% 上升到 59.88%，
FPR 明显升高而 FNR 降低。因为 InstructBLIP 接口要求图像，本项目对
`visual_input=0` 的文本控制题输入白图；这是明确记录的适配约定，不应与
能够真正省略视觉输入的模型结果无条件横比。

**Finding 10.** 外部基准给出的结论不是“E6 减少了所有幻觉”，而是
“E6 提高了视觉对象召回和 POPE 封闭式存在性判断，但增加了 affirmative/
object-mention 倾向，开放描述 CHAIR 明显恶化，HallusionBench 没有改善”。
这验证了联合评估封闭式与开放式幻觉的必要性。

## 6. RQ3：Grounding 与知识保持

### 6.1 单种子 grounding 实验

| Run | 方法 | VQA | Grounding | Positive | Negative | Count | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| E6 | 基线 | 71.29 | 85.90 | 83.75 | 99.25 | 63.50 | 0.510 |
| E12a | Q-Former VQA 控制 | 71.38 | 86.00 | 83.75 | 99.25 | 64.00 | 0.505 |
| E12b | 30% 通用 COCO grounding | 70.89 | **86.80** | 83.75 | 99.00 | **68.50** | **0.455** |
| E13a | 精炼控制 | 71.32 | 85.90 | 83.75 | 99.25 | 63.50 | 0.510 |
| E13b | 10% 错误驱动 grounding | 70.85 | 85.50 | 83.25 | 99.00 | 63.00 | 0.515 |
| E13c | 20% 错误驱动 grounding | 70.74 | 85.40 | 83.25 | 99.00 | 62.50 | 0.520 |

E12/E13 更新 Q-Former rank-8 LoRA，共 473,088 个参数（0.0202%）。

**Finding 7.** 通用 COCO grounding 确实能改善计数，但存在标准 VQA 代价。E12b 相对 E12a 将 Count 从 64.0 提升到 68.5、Grounding 从 86.0 提升到 86.8，但 VQA 从 71.38 降到 70.89。错误驱动的 VQAv2 风格 grounding 没有复现该收益，10% 和 20% 配比都同时降低 VQA 与 Grounding。

### 6.2 E14 三随机种子知识保持实验

E14 使用完整 1,000 条 VQA rehearsal，不再用 grounding 替换普通样本；加入 E6 teacher KL（权重 0.5、温度 2），Q-Former LoRA 降为 rank 4（236,544 个可训练参数），学习率 1e-5。

| 组别 | VQA mean ± std | Grounding mean ± std | Count mean ± std |
|---|---:|---:|---:|
| 控制组 | 71.310 ± 0.035 | 85.900 ± 0.173 | 63.500 ± 0.866 |
| Grounding | 71.317 ± 0.031 | 85.967 ± 0.058 | 63.833 ± 0.289 |
| 平均变化 | +0.007 | +0.067 | +0.333 |

逐 seed 的 Grounding 变化为 +0.30、0.00、-0.10；Count 变化为 +1.50、0.00、-0.50。

**Finding 8.** 蒸馏与 rehearsal 成功避免了 E12/E13 式明显遗忘，但 grounding 增益没有跨随机种子稳定出现。E14 可以支持“防遗忘有效、额外 grounding 收益有限”的结论，不能作为优于 E6 的新主模型。

## 7. 最终模型与结论

- 最终主 adapter：**E6 InstructBLIP Flan-T5-XL + LLM LoRA r8，10k VQAv2**。
- 最终 VQA 系统：**E6 + E19 短答案重排**；全量 VQAv2 validation
  **71.62**。E6 直接生成结果为 70.61。
- VQAv2-derived MCQ：**85.12**，相对zero-shot提升1.66点。
- PEFT 参数量：4,718,592，仅占总参数约 **0.201%**。
- 最强 grounding 单次结果：E12b Grounding 86.80 / Count 68.50，但标准 VQA 有 0.49 点代价。
- 最稳妥结论：E14 防止遗忘，但未带来稳定显著的 grounding 提升。
- 对齐结论：E6 明显依赖图像，但现有显式对比目标没有稳定优于 E6。
- 幻觉结论：POPE 有显著但有偏置权衡的提升；CHAIR 显著恶化；
  HallusionBench 无显著改善。因此只能说“完成了多协议幻觉研究”，不能说
  “解决了幻觉”。

## 8. 证据边界与待补项

1. 只有 E6 完成了 214,354 条全量验证；BLIP-2/InstructBLIP 基线仍是固定 1,000 条子集，因此不能声称在全量集上统计显著地优于所有基线。
2. 除 E14 外，大多数训练实验只有一个随机种子；小于约 0.5 点的差异应表述为趋势，而非稳定提升。
3. Grounding 数据是 COCO 标注派生诊断集，不等同于 VQAv2 官方指标。
4. 已增加官方 POPE、CHAIR-compatible 开放描述和 HallusionBench；但没有
   运行需要额外评审模型与 COCO2017 5,000 图像的完整官方 THRONE。
5. E16是自动构造的VQAv2-derived MCQ，不是官方多选候选集；部分干扰项较容易，
   因此85.12应作为系统能力补充证据，而不是与外部MCQ榜单直接比较。
6. 尚未有人类评测，也没有与 GPT-4o/Gemini 等闭源模型做同协议对照。
7. CHAIR 是 Python 3 兼容端口，HallusionBench 文本控制题使用白图占位；
   两项适配约定必须随结果一起报告。

## 9. 推荐最终实验表结构

- **Table 1：** BLIP-2 / InstructBLIP / LoRA 主结果（固定 1k）。
- **Table 2：** E6 完整 VQAv2 validation 主结果。
- **Table 3：** 对齐、complementary 和 hard-pair 消融。
- **Table 4：** Grounding 与知识保持消融。
- **Table 5：** Multiple-choice zero-shot与E6对照。
- **Table 6：** POPE、CHAIR、HallusionBench 的 zero-shot/E6 外部幻觉对照。
- **Figure 1：** 模型与训练流水线。
- **Figure 2：** Overall、Number、Other、Yes/No 分组柱状图。
- **Figure 3：** 正常图/错配图/灰图/噪声图视觉依赖图。
- **Figure 4：** VQA–Grounding 权衡散点图。

机器可读结果位于同目录：

- `core_vqa_results.csv`
- `alignment_ablation_results.csv`
- `grounding_results.csv`
- `multiple_choice_results.csv`
- `hallucination_results.csv`
- `E20_LITE_PROTOCOL.md`
