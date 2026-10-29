"""Training objectives.

Merge points for the other two extensions:
  * Limitation 2 (class-balanced / focal): `build_task_loss` ('release', 'cb').
  * Limitation 1 (confusion-aware contrastive): implement `contrastive_loss(fused, labels)` on out['fused'].
"""
import numpy as np
import torch
from torch import nn


class Loss(nn.Module):
    """Original ECERC loss: -(1 - p_t)^gamma * alpha_t * log p_t.
    The release uses gamma=1 on MELD (MELD/loss.py) and gamma=0 with class weights on IEMOCAP (IEMOCAP/loss.py)."""

    def __init__(self, gamma=0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, log_prob, target):
        logpt = log_prob.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.detach().exp()
        if self.alpha is not None:
            logpt = logpt * self.alpha.to(log_prob).gather(0, target)
        return (-1 * (1 - pt) ** self.gamma * logpt).mean()


def class_balanced_weights(counts, beta):
    """Effective-number weights (Cui et al., 2019), rescaled so the mean weight per training utterance is 1.
    Same as Limitation 2's train_imbalance.py; beta=0.999 on MELD gives [0.725, 1.026, 3.055, 1.452, 0.871, 3.026, 1.072]."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return weights * counts.sum() / (counts * weights).sum()


def release_loss(args, cfg, class_weights):
    """The authors' objective for the dataset; also used for validation loss (early stopping) in every variant."""
    return Loss(gamma=cfg['gamma'], alpha=class_weights if args.class_weight else None)


def build_task_loss(args, cfg, class_weights, train_counts):
    """'release': the authors' objective. 'cb': class-balanced weights as alpha inside the same loss (Limitation 2).
    'ce': gamma=0, the earlier ext/ default (for comparison with runs made before the merge)."""
    if args.task_loss == 'release':
        return release_loss(args, cfg, class_weights)
    if args.task_loss == 'cb':
        alpha = torch.tensor(class_balanced_weights(train_counts, args.cb_beta), dtype=torch.float32)
        return Loss(gamma=cfg['gamma'], alpha=alpha)
    if args.task_loss == 'ce':
        return Loss(gamma=0, alpha=class_weights if args.class_weight else None)
    raise ValueError(args.task_loss)


def contrastive_loss(fused, labels):
    return fused.new_zeros(())
