"""Training objectives.

Merge points for the other two extensions:
  * Limitation 2 (class-balanced / focal): add a branch to `build_task_loss`.
  * Limitation 1 (confusion-aware contrastive): implement `contrastive_loss(fused, labels)` on out['fused'].
"""
import torch
from torch import nn


class Loss(nn.Module):
    """Original ECERC loss (focal form with gamma=0 -> class-weighted NLL)."""

    def __init__(self, gamma=0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, log_prob, target):
        logpt = log_prob.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.detach().exp()
        if self.alpha is not None:
            logpt = logpt * self.alpha.to(log_prob).gather(0, target)
        return (-1 * (1 - pt) ** self.gamma * logpt).mean()


def build_task_loss(args, class_weights):
    if args.task_loss == 'ce':
        return Loss(alpha=class_weights if args.class_weight else None)
    raise ValueError(args.task_loss)


def contrastive_loss(fused, labels):
    return fused.new_zeros(())
