from __future__ import annotations

import argparse
import json

from grounded_vqa.models.loading import load_model
from grounded_vqa.models.lora import discover_target_modules


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LoRA targets")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    args = parser.parse_args()

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    report = {
        scope: discover_target_modules(loaded.model, scope)
        for scope in ("qformer", "llm", "dual")
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

