# Experiment Log

Last updated: 2026-08-22 (Australia/Sydney)

> The consolidated E0-E15 results, full VQAv2 validation, alignment ablations,
> grounding experiments, three-seed E14 statistics, and final findings are in
> [`reports/FINAL_EXPERIMENT_SUMMARY.md`](reports/FINAL_EXPERIMENT_SUMMARY.md).

## Server and storage

- Compute: RTX 5090 32 GB, Ubuntu 22.04, Python 3.12.
- Project root: `/root/autodl-tmp/vision-language` (data disk).
- System disk policy: no datasets, model weights, virtual environments, caches,
  temporary training files, or checkpoints.
- The environment is activated through `scripts/server_env.sh`, which redirects
  Hugging Face, Torch, pip, temporary, and XDG caches to the data disk.

## Verified assets

- VQAv2 train: 443,757 questions, 82,783 COCO train2014 images.
- VQAv2 validation: 214,354 questions, 40,504 COCO val2014 images.
- BLIP-2 Flan-T5-XL: both weight shards verified by SHA-256.
- Dataset audits: 1,000 train joins and 500 validation joins passed.

## Pilot results

All rows below use the same deterministic 1,000-example VQAv2 validation subset
(`seed=42`) and official VQA soft accuracy.

| Run | Training | Overall | Number | Other | Yes/no | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| E0 | BLIP-2 zero-shot | 59.76 | 44.26 | 45.75 | 82.68 | 3.04 GiB |
| E1 | Q-Former LoRA, 1k train examples, 1 epoch | 61.98 | 51.83 | 49.30 | 81.56 | 3.04 GiB (evaluation) |
| E2 | LLM LoRA, 1k train examples, 1 epoch | 65.17 | 55.48 | 53.87 | 82.81 | 3.04 GiB (evaluation) |
| E3 | InstructBLIP zero-shot | 67.51 | 59.74 | 56.49 | 84.22 | 2.96 GiB |
| E3b | InstructBLIP zero-shot, short-answer prompt | 70.36 | 63.83 | 61.28 | 84.17 | 2.96 GiB |
| E4 | Dual LoRA, 1k train examples, 1 epoch | 65.15 | 54.96 | 54.23 | 82.45 | 3.04 GiB (evaluation) |
| E5 | InstructBLIP LLM LoRA, 1k train examples, short prompt | 70.54 | 65.30 | 61.54 | 83.85 | 2.96 GiB (evaluation) |
| E6 | InstructBLIP LLM LoRA, 10k random train examples | **71.29** | **62.52** | **63.19** | **84.48** | 2.96 GiB (evaluation) |
| E8a | E6 + 1k VQA-only continuation control | 71.46 | 61.30 | 63.05 | 85.47 | 2.96 GiB (evaluation) |
| E8b | E6 + 1k mismatch continuation (`lambda=0.1`, margin `0.2`) | 70.75 | 60.96 | 62.63 | 84.27 | 2.96 GiB (evaluation) |

E1 updates 473,088 parameters (0.0202% of 2.343B) and produces a 1.9 MB
adapter. Relative to E0, overall accuracy improves by 2.22 points.
E2 updates 4,718,592 parameters (0.2010% of 2.347B), produces an 18.9 MB
adapter, and improves overall accuracy by 5.41 points over E0.

### Yes/no hallucination proxy

| Metric | E0 | E1 | Change |
|---|---:|---:|---:|
| Predicted yes rate | 42.19 | 38.80 | -3.39 |
| Invalid-answer rate | 0.52 | 0.52 | 0.00 |
| False-yes rate | 19.59 | 17.53 | -2.06 |
| False-no rate | 34.04 | 38.83 | +4.79 |

For E2, predicted-yes, invalid-answer, false-yes, and false-no rates are 38.02,
0.26, 15.38, and 38.30 respectively. LLM LoRA has the strongest accuracy and
false-yes result so far, but its elevated false-no rate still requires explicit
grounding controls.

E3's yes/no proxy has a 45.83 predicted-yes rate, 18.68 false-yes rate, and
21.55 false-no rate. Its 5.47% invalid-answer rate is much higher than BLIP-2,
largely because instruction-following generations can be full sentences rather
than VQA-style short answers. Answer extraction/prompt-format ablations are
therefore required before attributing the gap entirely to visual reasoning.

E3b reduces E3's invalid-answer rate from 5.47% to 4.43% and raises overall
accuracy by 2.85 points. False-yes falls from 18.68% to 16.49%, while false-no
rises from 21.55% to 28.49%; prompt formatting helps but does not remove answer
bias.

E4 updates 5,191,680 parameters (0.2212% of 2.348B) and produces a 20.8 MB
adapter. It does not improve on E2 despite using more trainable parameters, so
LLM-only is the preferred BLIP-2 LoRA scope for the next scale-up gate.

E5 updates 4,718,592 parameters (0.2011% of 2.347B) and produces an 18.9 MB
adapter. It improves E3b by only 0.18 overall points: number improves by 1.48,
other by 0.26, and yes/no falls by 0.31. False-yes improves to 15.26%, while
false-no worsens to 31.07%. This is a promising pilot, not yet evidence of a
statistically reliable gain.

E6 is the current primary checkpoint. Scaling to 10k examples improves E5 by
0.75 overall points, but number accuracy falls while other and yes/no improve.

## Visual-dependence diagnostics

The same 1,000 validation examples are evaluated with the normal image, a
deterministically mismatched image, a gray image, and uniform RGB noise.

| Run | Normal | Mismatched | Accuracy drop | Answer-change rate | Unchanged rate | Normal-advantage rate |
|---|---:|---:|---:|---:|---:|---:|
| E6 | 71.49 | 36.74 | 34.75 | 65.9 | 34.1 | 44.7 |
| E8a control | 71.43 | 37.10 | 34.33 | 63.4 | 36.6 | 43.9 |
| E8b mismatch | 70.99 | 36.87 | 34.12 | 64.4 | 35.6 | 43.9 |

E6 clearly uses visual input, but 34.1% of normalized answers remain unchanged
after image mismatch and 49.4% of examples have equal VQA soft scores. VQA-only
continuation makes these diagnostics worse. The first conservative mismatch
pilot recovers 1.0 point of answer-change rate relative to its continuation
control, but loses 0.71 standard VQA accuracy and does not improve the
normal-versus-mismatch accuracy gap. E8b is therefore a documented negative
result, not a replacement for E6.

The Q-Former pilot reduces false-positive visual assertions but increases
false-negative answers. This is evidence of a response-bias shift, not yet a
general hallucination reduction. The next controls are LLM-only LoRA,
dual-module LoRA, mismatched/blank-image sensitivity, and grounded negative
examples.

## Final experimental status

1. E6 remains the primary checkpoint.
2. E15 completed all 214,354 VQAv2 validation questions: 70.61 overall,
   52.32 Number, 63.64 Other, and 86.14 Yes/No.
3. E8-E11 mismatch, complementary, hard-pair, and mixed Q-Former objectives did
   not jointly improve VQA and visual dependence.
4. E12 generic COCO grounding improved counting with a VQA trade-off; E13
   error-driven grounding did not improve either objective.
5. E14 three-seed distillation/rehearsal avoided forgetting, but its grounding
   gain was too small and inconsistent to replace E6.
6. E16 evaluated 5,000 VQAv2-derived multiple-choice questions. E6 reached
   85.12% versus 83.46% zero-shot and a 34.345% weighted random baseline, with
   zero invalid option outputs in both model conditions.
7. E19 applies short-answer reranking to the same E6 adapter and raises full
   VQAv2 validation from 70.609 to 71.620 (+1.011; paired bootstrap 95% CI
   +0.968 to +1.054). This is primarily a yes/no format correction, not a new
   visual representation.
8. H1 official COCO POPE (9,000 questions) gives E6 84.933 macro accuracy
   versus 83.944 zero-shot (+0.989; 95% CI +0.644 to +1.333). Recall improves,
   while precision falls and the yes ratio rises.
9. H2 CHAIR on the same deterministic 500 COCO val2014 images shows a Type-I
   trade-off: E6 improves object recall 62.155→66.980, but worsens CHAIRs
   31.800→43.400 and CHAIRi 10.929→14.529. All paired CIs exclude zero.
10. H3 full HallusionBench gives E6 52.702 question accuracy versus 54.296
    zero-shot. The −1.594 point change is not significant (95% CI −4.163 to
    +0.974); E6 shifts errors from false negatives toward false positives.
