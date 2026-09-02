from __future__ import annotations

import json
import re
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch

AdapterScope = Literal["qformer", "llm", "dual"]

_PATTERNS = {
    "qformer": re.compile(r".*qformer.*\.(query|value)$"),
    "llm": re.compile(r".*language_model.*\.(q|v)$"),
    "dual": re.compile(
        r".*(?:qformer.*\.(?:query|value)|language_model.*\.(?:q|v))$"
    ),
}


def discover_target_modules(
    model: torch.nn.Module,
    scope: AdapterScope,
) -> list[str]:
    pattern = _PATTERNS[scope]
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and pattern.fullmatch(name)
    ]
    if not targets:
        examples = [name for name, _ in list(model.named_modules())[:100]]
        raise RuntimeError(
            f"No LoRA targets found for scope={scope}. Module examples: {examples[-20:]}"
        )
    return targets


def attach_lora(
    model: torch.nn.Module,
    scope: AdapterScope,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    targets = discover_target_modules(model, scope)
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )
    return get_peft_model(model, config)


def load_trainable_lora(model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    from peft import PeftModel, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)


def add_trainable_qformer_to_frozen_adapter(
    model: torch.nn.Module,
    adapter_path: str,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> torch.nn.Module:
    """Extend an existing LoRA with Q-Former targets while preserving its weights."""

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    targets = discover_target_modules(model, "qformer")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig.from_pretrained(adapter_path)
    existing_targets = set(config.target_modules or [])
    config.target_modules = sorted(existing_targets | set(targets))
    config.inference_mode = False
    config.lora_dropout = dropout
    config.rank_pattern = {**(config.rank_pattern or {}), **dict.fromkeys(targets, rank)}
    config.alpha_pattern = {**(config.alpha_pattern or {}), **dict.fromkeys(targets, alpha)}
    extended = get_peft_model(model, config)
    existing_weights = load_peft_weights(adapter_path, device="cpu")
    load_result = set_peft_model_state_dict(extended, existing_weights)
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys while loading existing adapter: {load_result.unexpected_keys}"
        )
    for name, parameter in extended.named_parameters():
        parameter.requires_grad = "qformer" in name and ".lora_" in name
    if not any(parameter.requires_grad for parameter in extended.parameters()):
        raise RuntimeError("Q-Former adapter did not expose any trainable parameters")
    return extended


def save_adapter_stack(model: torch.nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


@contextmanager
def qformer_lora_disabled(model: torch.nn.Module):
    """Temporarily recover the frozen pre-Q-Former adapter as an in-place teacher."""

    from peft.tuners.lora.layer import LoraLayer

    disabled: list[LoraLayer] = []
    for name, module in model.named_modules():
        if (
            "qformer" in name.lower()
            and isinstance(module, LoraLayer)
            and not module.disable_adapters
        ):
            module.enable_adapters(False)
            disabled.append(module)
    if not disabled:
        raise RuntimeError("No active Q-Former LoRA layers were found to disable")
    try:
        yield
    finally:
        for module in disabled:
            module.enable_adapters(True)


def load_inference_lora(model: torch.nn.Module, adapter_path: str | Path) -> torch.nn.Module:
    from peft import PeftMixedModel, PeftModel

    adapter_path = Path(adapter_path)
    manifest_path = adapter_path / "adapter_stack.json"
    if not manifest_path.is_file():
        return PeftModel.from_pretrained(model, adapter_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active = list(manifest["active_adapters"])
    adapter_paths = dict(manifest["adapters"])
    first = active[0]
    mixed = PeftMixedModel.from_pretrained(
        model,
        adapter_path / adapter_paths[first],
        adapter_name=first,
        is_trainable=False,
    )
    for name in active[1:]:
        mixed.load_adapter(
            adapter_path / adapter_paths[name],
            adapter_name=name,
            is_trainable=False,
        )
    mixed.set_adapter(active, inference_mode=True)
    return mixed


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, float | int]:
    parameters: Iterable[torch.nn.Parameter] = model.parameters()
    total = 0
    trainable = 0
    for parameter in parameters:
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }
