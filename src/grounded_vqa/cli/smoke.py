from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

from grounded_vqa.models.loading import format_prompt, load_model, model_input_device

DEFAULT_IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one BLIP VQA inference")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Use a local image instead of downloading --image-url.",
    )
    parser.add_argument("--question", default="How many cats are in the image?")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    else:
        response = requests.get(args.image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    prompt = format_prompt(args.model_kind, args.question)
    inputs = loaded.processor(images=image, text=prompt, return_tensors="pt")
    device = model_input_device(loaded.model)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    answer = loaded.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(
        json.dumps(
            {
                "model": args.model_id,
                "kind": args.model_kind,
                "question": args.question,
                "answer": answer,
                "gpu": torch.cuda.get_device_name(0),
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
