# E20-Lite: Fixed-10k Negative Grounding Protocol

Status: training started 2026-08-22 on `screen e20-lite`.

## Motivation

E6 improves POPE recall and CHAIR object recall, but also raises affirmative
bias, HallusionBench false positives, CHAIRs, and CHAIRi. E20-Lite therefore
targets unsupported object assertions rather than additional object recall.

## Leakage control

Training negatives use only COCO train2014 instance annotations and reference
captions. POPE, CHAIR evaluation images, HallusionBench, and VQAv2 validation
answers are not used to construct or rank training examples.

## Fixed compute and data

- 9,000 ordinary VQAv2 train examples.
- 1,000 COCO train hard-negative existence questions.
- Total: 10,000 examples, one epoch; no data scaling beyond E6.
- Hard negatives are absent from both instance labels and reference-caption
  object mentions, then ranked by category co-occurrence with present objects.
- Student: E6 LLM LoRA initialized from the final E6 adapter.
- Teacher: a frozen copy of the E6 adapter in the same base model.
- Learning rate: `1e-5`; negative loss weight: `1.0`.
- Teacher KL: weight `0.5`, temperature `2`, applied every fourth ordinary
  rehearsal step.

## Promotion gates

E20-Lite is promoted only if the same checkpoint meets all of these conditions:

1. Fixed 5k VQAv2 dev is no worse than E6 by more than 0.30 points after the
   same E19 reranker.
2. POPE precision or F1 improves without an accuracy drop larger than 0.30.
3. CHAIRs and CHAIRi do not worsen; a useful result should lower at least one
   with a paired 95% confidence interval excluding zero.
4. CHAIR object recall may not fall by more than 2 points.
5. HallusionBench false-positive rate falls without merely moving all errors to
   false negatives; question accuracy must remain within 1 point of E6.

Only a checkpoint passing the development and external hallucination gates is
eligible for a new full 214,354-question VQAv2 validation run. Otherwise E6 +
E19 remains the final VQA system and E20-Lite is reported as an ablation.

## Artifacts

- Training data: `data/coco/grounding/e20_hard_negative1000_seed60.jsonl`
- Output: `outputs/E20_lite_e6_llm_hardnegative10pct_distill`
- Launcher: `scripts/run_e20_lite.sh`
- Log pointer: `logs/e20_lite_latest.path`
