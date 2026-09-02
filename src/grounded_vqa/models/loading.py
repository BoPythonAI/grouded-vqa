from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ModelKind = Literal["blip2", "instructblip"]
Quantization = Literal["none", "4bit", "8bit"]
PromptStyle = Literal["default", "short"]


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    processor: object
    kind: ModelKind


def format_prompt(
    kind: ModelKind,
    question: str,
    style: PromptStyle = "default",
) -> str:
    question = question.strip()
    if style == "short":
        if kind == "blip2":
            return f"Question: {question} Answer with a short phrase:"
        return (
            f"{question}\nAnswer using only a short phrase or a single number. "
            "Do not use a full sentence."
        )
    if kind == "blip2":
        return f"Question: {question} Answer:"
    return question


def load_model(
    model_id: str,
    kind: ModelKind,
    quantization: Quantization = "4bit",
) -> LoadedModel:
    from transformers import (
        BitsAndBytesConfig,
        Blip2ForConditionalGeneration,
        Blip2Processor,
        InstructBlipForConditionalGeneration,
        InstructBlipProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selected BLIP model")

    model_class: type[torch.nn.Module]
    processor_class: type
    if kind == "blip2":
        model_class = Blip2ForConditionalGeneration
        processor_class = Blip2Processor
    elif kind == "instructblip":
        model_class = InstructBlipForConditionalGeneration
        processor_class = InstructBlipProcessor
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    kwargs: dict[str, object] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if quantization != "none":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    processor = processor_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id, **kwargs)
    model.eval()
    return LoadedModel(model=model, processor=processor, kind=kind)


def model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Unable to determine model input device")
