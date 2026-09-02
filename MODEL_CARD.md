# Model Card: E6 Grounded VQA Adapter and E19 Decoding System

Updated: 2026-08-22

## Model description

- Base: Salesforce InstructBLIP Flan-T5-XL.
- Adapter: E6 LLM LoRA, rank 8, alpha 16, dropout 0.05.
- Trainable parameters: 4,718,592 (about 0.201% of the model).
- Training data: a deterministic random 10,000-example subset of VQAv2 train,
  one epoch.
- Final VQA system: the unchanged E6 adapter plus E19 short-answer reranking for
  routed yes/no questions.

E19 is a decoding component, not a separately trained checkpoint. External
POPE, CHAIR, and HallusionBench results use direct deterministic generation
unless the result explicitly says pair-logprob.

## Intended use

This research model is intended for short-answer visual question answering,
controlled PEFT experiments, and study of visual-language alignment and
hallucination trade-offs. It is not intended for safety-critical decisions or
unverified real-world factual claims.

## Main results

| Evaluation | Result |
|---|---:|
| Full VQAv2, E6 direct | 70.609 |
| Full VQAv2, E6 + E19 | **71.620** |
| VQAv2-derived MCQ | 85.12 |
| POPE macro accuracy, E6 | 84.933 |
| POPE macro accuracy, zero-shot | 83.944 |
| CHAIRs, E6 / zero-shot (lower is better) | 43.40 / **31.80** |
| CHAIRi, E6 / zero-shot (lower is better) | 14.53 / **10.93** |
| CHAIR object recall, E6 / zero-shot | **66.98** / 62.16 |
| HallusionBench question accuracy, E6 / zero-shot | 52.70 / **54.30** |

E6 significantly improves POPE accuracy by 0.989 points, but does so by raising
object recall and affirmative answers while lowering precision. On open-form
CHAIR, E6 significantly raises both object recall and hallucination. On
HallusionBench, its −1.594 point accuracy change is not significant and its
false-positive rate increases. The supported conclusion is a measurable
recall–hallucination trade-off, not universal hallucination mitigation.

## Alignment evidence

On the fixed VQAv2 diagnostic subset, replacing the correct image with a
mismatched image lowers accuracy by 34.75 points and changes 65.9% of answers.
This establishes that the model uses visual input. Mismatch, complementary,
hard-pair, Q-Former, and grounding objectives did not consistently improve both
VQA and alignment, and are retained as controlled negative or trade-off
results.

## Limitations

- Only the E6/E19 system has a complete 214,354-question VQAv2 validation run;
  many training ablations use a fixed 1,000- or 5,000-question development set.
- Most training ablations have one seed. E14 is the main three-seed exception.
- CHAIR uses a documented Python 3 port of the original object rules with a
  deterministic tokenizer instead of the original Python 2 pattern/nltk stack.
- HallusionBench text-only rows use a white image because InstructBLIP requires
  an image input.
- Full official THRONE, human evaluation, and closed-model comparisons have not
  been run.
- The VQAv2-derived MCQ set has automatically sampled distractors and is not an
  official multiple-choice leaderboard.

## Reproducibility artifacts

Server project root: `/root/autodl-tmp/vision-language`

- Adapter: `outputs/E6_instructblip_llm_lora_r8_train10k_random/final-adapter`
- Full direct VQA: `outputs/E15_E6_instructblip_full_vqav2_val214354`
- E19 full VQA: `outputs/E19_E6_short_answer_rerank_full_val`
- POPE: `outputs/H1_E6_official_COCO_POPE`
- CHAIR: `outputs/H2_E6_CHAIR500`
- HallusionBench: `outputs/H3_E6_HallusionBench`
- Consolidated report: `reports/FINAL_EXPERIMENT_SUMMARY.md`
