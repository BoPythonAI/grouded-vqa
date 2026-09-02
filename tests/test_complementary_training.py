import torch

from grounded_vqa.cli.train_complementary import (
    pairwise_logistic_loss,
    sequence_token_nll,
)


def test_sequence_token_nll_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [0.0, 0.0]],
            [[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[0, 1, -100], [1, 0, 1]])
    losses = sequence_token_nll(logits, labels)
    assert losses.shape == (2,)
    assert torch.all(losses < 0.02)


def test_pairwise_logistic_prefers_lower_positive_nll() -> None:
    good = pairwise_logistic_loss(torch.tensor([0.2]), torch.tensor([1.2]), 0.5)
    bad = pairwise_logistic_loss(torch.tensor([1.2]), torch.tensor([0.2]), 0.5)
    assert good < bad
    assert good > 0
