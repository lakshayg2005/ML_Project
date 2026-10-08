import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionAwareContrastiveLossV2(nn.Module):
    """
    SupCon on a PROJECTED feature (separate MLP head, training-only),
    with extra weight on confusable negatives + prototype margin.
    IEMOCAP labels: hap0 sad1 neu2 ang3 exc4 fru5
    """

    def __init__(self, in_dim, confusable_pairs, n_classes=6, proj_dim=128, temperature=0.07,
                 conf_weight=4.0, margin=0.5, proto_weight=0.5, only_confusable=False):
        super().__init__()
        self.pairs = confusable_pairs
        self.T = temperature
        self.margin = margin
        self.proto_weight = proto_weight
        self.only_confusable = only_confusable
        self.proj = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, proj_dim))

        W = torch.ones(n_classes, n_classes)
        for a, b in confusable_pairs:
            W[a, b] = conf_weight
            W[b, a] = conf_weight
        self.register_buffer('class_w', W)

        keep = torch.zeros(n_classes, dtype=torch.bool)
        for a, b in confusable_pairs:
            keep[a] = True
            keep[b] = True
        self.register_buffer('keep_cls', keep)
        self.proto_active = 0.0   # fraction of pairs where the margin term was active (for logging)

    def forward(self, feats, labels):
        if self.only_confusable:
            m = self.keep_cls[labels]
            feats, labels = feats[m], labels[m]
        N = feats.size(0)
        if N < 2:
            return feats.sum() * 0.0

        z = F.normalize(self.proj(feats), dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)

        logits = z @ z.t() / self.T
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        Wm = self.class_w[labels][:, labels]
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

        proto_terms, active = [], 0
        for a, b in self.pairs:
            ma, mb = labels == a, labels == b
            if ma.any() and mb.any():
                mu_a = F.normalize(z[ma].mean(0), dim=0)
                mu_b = F.normalize(z[mb].mean(0), dim=0)
                cos = (mu_a * mu_b).sum()
                t = F.relu(self.margin - (1.0 - cos))
                active += int(t.item() > 0)
                proto_terms.append(t)
        self.proto_active = active / max(len(self.pairs), 1)
        proto = torch.stack(proto_terms).mean() if proto_terms else feats.sum() * 0.0

        return supcon + self.proto_weight * proto


def pair_confusion_penalty(log_prob, labels, partner):
    """
    Penalize probability mass put on the confusable partner class.
    partner: LongTensor [n_classes], partner[c] = confusable class of c, or -1 if none.
    loss = mean over samples with a partner of  -log(1 - p(partner))
    """
    p = partner[labels]
    m = p >= 0
    if m.sum() == 0:
        return log_prob.sum() * 0.0
    p_partner = log_prob[m].gather(1, p[m].unsqueeze(1)).squeeze(1).exp()
    return -torch.log((1.0 - p_partner).clamp(min=1e-6)).mean()