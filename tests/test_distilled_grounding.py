import torch

from grounded_vqa.cli.train_distilled_grounding import distillation_kl


def test_distillation_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]])
    labels = torch.tensor([[1, -100]])
    loss = distillation_kl(logits, logits, labels, temperature=2.0)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_distillation_kl_is_positive_for_different_logits() -> None:
    teacher = torch.tensor([[[3.0, 1.0, 0.0]]])
    student = torch.tensor([[[0.0, 1.0, 3.0]]], requires_grad=True)
    labels = torch.tensor([[1]])
    loss = distillation_kl(student, teacher, labels, temperature=2.0)
    assert loss.item() > 0
    loss.backward()
    assert student.grad is not None
