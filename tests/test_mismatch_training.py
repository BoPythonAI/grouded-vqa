import torch

from grounded_vqa.cli.train_mismatch import mismatch_margin_loss


def test_mismatch_margin_loss_is_zero_when_margin_is_satisfied() -> None:
    loss = mismatch_margin_loss(torch.tensor(0.5), torch.tensor(0.8), margin=0.2)
    assert loss.item() == 0.0


def test_mismatch_margin_loss_penalizes_wrong_image_confidence() -> None:
    loss = mismatch_margin_loss(torch.tensor(0.8), torch.tensor(0.5), margin=0.2)
    assert torch.isclose(loss, torch.tensor(0.5))
