# VQAv2 Execution Plan

This document supersedes the earlier GQA dataset choice for implementation.
The research survey remains useful for method selection, but the deployed
project uses **VQAv2 as the only primary training dataset** to match Project 019.

## Fixed scope

- Primary dataset: VQAv2 train/val with MS COCO 2014 images.
- Backbones: BLIP-2 Flan-T5-XL and InstructBLIP Flan-T5-XL.
- PEFT comparison: Q-Former LoRA, LLM LoRA, and dual-module LoRA.
- Standard metrics: official VQA soft accuracy overall and by answer type.
- Hallucination focus: object existence, counting, complementary-pair visual
  dependence, mismatched-image sensitivity, and answer coverage.
- Inference-time method: Visual Contrastive Decoding after stable SFT baselines.
- Optional external evaluation: POPE/HallusionBench only after the primary
  VQAv2 result is reproducible.

## Storage contract

Every large or rapidly growing path lives under:

```text
/root/autodl-tmp/vision-language
```

The system disk must not contain datasets, environments, model snapshots,
Hugging Face caches, pip caches, temporary training files, or checkpoints.
Run `source scripts/server_env.sh` before any project command.

## Experiment gates

### Gate 0: environment

- PyTorch recognizes RTX 5090 and can allocate a CUDA tensor.
- All unit tests pass.
- System disk use remains effectively unchanged after dependency installation.

### Gate 1: model inference

- BLIP-2 and InstructBLIP load in 4-bit mode.
- Both answer the same fixed COCO smoke-test question.
- Record peak VRAM, latency, package versions, prompt, and decoding parameters.

### Gate 2: data and evaluator

- Download val questions, annotations, and val2014 images first.
- Verify every ZIP before extraction.
- Validate 100 random question/image/annotation joins.
- Confirm normalization and leave-one-annotator-out scoring with tests.

### Gate 3: zero-shot baselines

- Evaluate a deterministic validation subset first, then full validation.
- Report overall, yes/no, number, and other accuracy.
- Add blank-image and mismatched-image diagnostics.

### Gate 4: PEFT smoke training

- Overfit 32 examples with Q-Former LoRA.
- Repeat with LLM-only and dual LoRA.
- Verify only intended adapter parameters receive gradients.
- Save and reload adapters; predictions must be stable.

### Gate 5: full PEFT comparison

- Use matched data order, seed, effective batch size, and decoding.
- Compare accuracy, trainable parameters, VRAM, throughput, and adapter size.
- Select the best BLIP-2 adapter scope before training InstructBLIP.

### Gate 6: hallucination-aware stage

- Use VQAv2 complementary image pairs as visual-dependence supervision.
- Use COCO instance annotations to generate verified absent-object and count
  negatives.
- Compare SFT with SFT plus hallucination-aware contrastive training.
- Reject improvements caused by answer shortening or excessive `no` responses.

### Gate 7: decoding and final evaluation

- Compare greedy decoding and VCD with identical maximum output length.
- Run three seeds for final variants.
- Produce prediction files, metrics, bootstrap intervals, error categories,
  model card, and reproducible commands.

## Initial experiment IDs

| ID | Backbone | Adapter | Hallucination loss | Decoding |
|---|---|---|---|---|
| E0 | BLIP-2 | none | none | greedy |
| E1 | InstructBLIP | none | none | greedy |
| E2 | BLIP-2 | Q-Former LoRA | none | greedy |
| E3 | BLIP-2 | LLM LoRA | none | greedy |
| E4 | BLIP-2 | dual LoRA | none | greedy |
| E5 | InstructBLIP | best matched LoRA | none | greedy |
| E6 | BLIP-2 | best LoRA | grounded contrastive | greedy |
| E7 | InstructBLIP | best LoRA | grounded contrastive | greedy |
| E8 | best E6/E7 | unchanged | unchanged | VCD |

## Immediate server sequence

1. Install CUDA-capable PyTorch and project dependencies in the data-disk venv.
2. Run unit tests and CUDA capability check.
3. Download model processors/configs and run BLIP-2 smoke inference.
4. Run InstructBLIP smoke inference.
5. Download VQAv2 validation metadata and images in a detached `screen` job.
6. Run a 100-sample data-integrity audit.
7. Produce E0/E1 predictions on a small deterministic subset.

