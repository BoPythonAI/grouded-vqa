from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from grounded_vqa.data.answers import consensus_answer, sampled_reference_answer
from grounded_vqa.data.vqav2 import VQASample
from grounded_vqa.models.loading import ModelKind, PromptStyle, format_prompt


@dataclass
class VQAGenerativeCollator:
    processor: Any
    model_kind: ModelKind
    prompt_style: PromptStyle = "default"
    max_question_tokens: int = 64
    max_answer_tokens: int = 16
    answer_target: str = "consensus"
    answer_seed: int = 42

    def __call__(self, samples: list[VQASample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompts: list[str] = []
        answers: list[str] = []
        metadata: list[dict[str, Any]] = []
        for sample in samples:
            with Image.open(sample.image_path) as image:
                images.append(image.convert("RGB"))
            prompts.append(
                format_prompt(self.model_kind, sample.question, self.prompt_style)
            )
            if self.answer_target == "consensus":
                answer = consensus_answer(sample.answers)
            elif self.answer_target == "frequency":
                answer = sampled_reference_answer(
                    sample.answers,
                    seed=self.answer_seed,
                    sample_key=sample.question_id,
                )
            else:
                raise ValueError(f"Unsupported answer target: {self.answer_target}")
            answers.append(answer)
            metadata.append(
                {
                    "question_id": sample.question_id,
                    "answer_type": sample.answer_type,
                    "question_type": sample.question_type,
                }
            )

        model_inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self.max_question_tokens,
            return_tensors="pt",
        )
        label_batch = self.processor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_answer_tokens,
            return_tensors="pt",
        )
        labels = label_batch.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        model_inputs["metadata"] = metadata
        return model_inputs


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = batch.pop("metadata")
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key == "pixel_values":
                value = value.to(dtype=torch.bfloat16)
        moved[key] = value
    return moved, metadata
