"""Training objectives.

Merged extensions:
  * Limitation 1 (Divyansh): soft labels from annotator votes (`soft_label_nll`) and an auxiliary VAD regression
    head on out['fused'] (`vad_loss`); IEMOCAP only, the targets come from prepare_extras.py.
  * Limitation 2 (Keshav): the zero-shot LLM fusion is post hoc (final_report.py), not a training loss.
`--focal_gamma` is the release's own exponent: 1 on MELD, 0 on IEMOCAP.
"""
import torch.nn.functional as F
from torch import nn


class Loss(nn.Module):
    """Original ECERC loss: -(1 - p_t)^gamma * alpha_t * log p_t (release: gamma=1 on MELD, 0 on IEMOCAP)."""

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
        return Loss(gamma=args.focal_gamma, alpha=class_weights if args.class_weight else None)
    raise ValueError(args.task_loss)


def soft_label_nll(log_prob, labels, votes, beta, class_weights=None):
    """NLL against (1 - beta) * onehot(y) + beta * annotator votes, weighted by the hard label's class weight."""
    target = (1 - beta) * F.one_hot(labels, log_prob.size(1)).float() + beta * votes
    w = class_weights[labels] if class_weights is not None else 1.
    return (w * -(target * log_prob).sum(1)).mean()


def vad_loss(pred, vad_z, vmask):
    """MSE between predicted and z-scored (valence, activation, dominance), on annotated utterances only."""
    m = vmask > 0
    return ((pred[m] - vad_z[m]) ** 2).mean() if m.any() else pred.sum() * 0
