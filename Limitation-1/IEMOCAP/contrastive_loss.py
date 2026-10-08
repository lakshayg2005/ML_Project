import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionAwareContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with extra push-away weight on confusable class pairs,
    plus a prototype cosine-distance margin between confusable pairs.

    IEMOCAP labels: hap0 sad1 neu2 ang3 exc4 fru5
    """

    def __init__(self, confusable_pairs, n_classes=6, temperature=0.1,
                 conf_weight=2.0, margin=0.5, proto_weight=0.5):
        super().__init__()
        self.pairs = confusable_pairs
        self.T = temperature
        self.margin = margin
        self.proto_weight = proto_weight
        W = torch.ones(n_classes, n_classes)
        for a, b in confusable_pairs:
            W[a, b] = conf_weight
            W[b, a] = conf_weight
        self.register_buffer('class_w', W)

    def forward(self, feats, labels):
        """
        feats:  (N, D) fused utterance features (before classifier)
        labels: (N,)
        """
        N = feats.size(0)
        if N < 2:
            return feats.sum() * 0.0

        z = F.normalize(feats, dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)

        logits = z @ z.t() / self.T
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        # Up-weight negatives belonging to confusable class pairs
        Wm = self.class_w[labels][:, labels]                     # (N, N)
        denom_logits = (logits + torch.log(Wm)).masked_fill(eye, float('-inf'))
        log_denom = torch.logsumexp(denom_logits, dim=1, keepdim=True)
        log_prob = (logits - log_denom).masked_fill(eye, 0.0)

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
        pos_cnt = pos_mask.sum(1)
        valid = pos_cnt > 0
        if valid.sum() == 0:
            supcon = feats.sum() * 0.0
        else:
            per_anchor = -(log_prob * pos_mask).sum(1) / pos_cnt.clamp(min=1)
            supcon = per_anchor[valid].mean()

        # Prototype margin between confusable pairs
        proto_terms = []
        for a, b in self.pairs:
            ma, mb = labels == a, labels == b
            if ma.any() and mb.any():
                mu_a = F.normalize(z[ma].mean(0), dim=0)
                mu_b = F.normalize(z[mb].mean(0), dim=0)
                cos = (mu_a * mu_b).sum()
                # want (1 - cos) >= margin
                proto_terms.append(F.relu(self.margin - (1.0 - cos)))
        proto = torch.stack(proto_terms).mean() if proto_terms else feats.sum() * 0.0

        return supcon + self.proto_weight * proto