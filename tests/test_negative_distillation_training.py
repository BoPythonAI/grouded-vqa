from __future__ import annotations

import torch

from grounded_vqa.cli.train_negative_distillation import (
    per_sample_cross_entropy,
    select_batch_rows,
)


def test_per_sample_cross_entropy_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [100.0, -100.0]],
            [[0.0, 4.0], [4.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, -100], [1, 0]])

    losses = per_sample_cross_entropy(logits, labels)

    expected = -torch.log_softmax(torch.tensor([4.0, 0.0]), dim=0)[0]
    assert losses.shape == (2,)
    assert torch.allclose(losses, torch.tensor([expected, expected]))


def test_select_batch_rows_only_slices_batched_tensors() -> None:
    mask = torch.tensor([True, False, True])
    batch: dict[str, object] = {
        "input_ids": torch.arange(6).reshape(3, 2),
        "scalar": torch.tensor(2.0),
        "config": "keep",
    }

    selected = select_batch_rows(batch, mask)

    assert torch.equal(selected["input_ids"], torch.tensor([[0, 1], [4, 5]]))
    assert selected["scalar"] is batch["scalar"]
    assert selected["config"] == "keep"
