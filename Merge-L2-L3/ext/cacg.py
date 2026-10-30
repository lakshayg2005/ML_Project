"""Confidence-Aware Cause Gating (CACG) -- Limitation 3 (owner: Lakshay Gupta).

Sits on top of ECERC's Feature Gating stage. The four cause-conditioned features
f_k (k in {self-contagion, cross-emotion, self-event, cross-event}) are re-weighted
by a cause-level distribution

    tau_i = tau_min + (tau_max - tau_min) * sigmoid(MLP(h_i))              (SOP Eq. 1)
    g_i   = softmax( (W_g [f_1; f_2; f_3; f_4] + b_g) / tau_i  + log m_i )  (SOP Eq. 2, + availability mask m_i)
    fused = [K g_i1 f_1 ; ... ; K g_iK f_K]       (K g = 1 for uniform g -> reduces exactly to ECERC)

Gate supervision (`gate_loss`):
  * 'entropy'  -- SOP Eq. 3: (H(g_i) - a_i * H_max)^2 with a heuristic ambiguity proxy a_i.
  * 'utility'  -- proposed replacement: each cause gets an auxiliary classifier p_k(y|f_k);
                  the target is the mixture-of-experts posterior responsibility
                  q_ik = p_k(y_i|f_ik)^(1/T) / sum_j p_j(y_i|f_ij)^(1/T)   (detached)
                  and L_gate = KL(q_i || g_i). The target is sharp when one cause clearly explains
                  the gold emotion and flat when several do equally -- i.e. the "decisive vs.
                  blended" behaviour the SOP wants, but grounded in the label rather than a
                  hand-made proxy that tau_i could trivially copy from h_i.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

CAUSE_NAMES = ['self_contagion', 'cross_emotion', 'self_event', 'cross_event']
DESC_DIM = 3 + 3 + 6  # modality gates + context counts + pairwise cause cosines


def build_descriptor(modal_gates, n_self, n_cross, feats):
    """Evidence-quality descriptor h_i (SOP Sec. III-C).
    modal_gates: (B,L,3); n_self/n_cross: (B,L) prior same/other-speaker utterances; feats: list of K (B,L,D)."""
    ctx = torch.stack([torch.log1p(n_self + n_cross), torch.log1p(n_self), torch.log1p(n_cross)], dim=-1)
    normed = [F.normalize(f, dim=-1) for f in feats]
    cos = [(normed[a] * normed[b]).sum(-1) for a in range(len(feats)) for b in range(a + 1, len(feats))]
    return torch.cat([modal_gates, ctx, torch.stack(cos, dim=-1)], dim=-1).detach()


class CauseGate(nn.Module):
    """mode: 'softmax' (fixed tau) or 'cacg' (input-conditioned tau)."""

    def __init__(self, feat_dim, n_classes, n_causes=4, mode='cacg', tau_min=0.25, tau_max=4.0,
                 fixed_tau=1.0, mask_unavailable=True, aux_heads=False):
        super().__init__()
        self.mode, self.K = mode, n_causes
        self.tau_min, self.tau_max, self.fixed_tau = tau_min, tau_max, fixed_tau
        self.mask_unavailable = mask_unavailable
        self.scorer = nn.Linear(feat_dim * n_causes, n_causes)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)  # uniform gate at init == original ECERC
        if mode == 'cacg':
            self.tau_net = nn.Sequential(nn.LayerNorm(DESC_DIM), nn.Linear(DESC_DIM, 32), nn.ReLU(), nn.Linear(32, 1))
            # start at tau = fixed_tau
            p0 = (fixed_tau - tau_min) / (tau_max - tau_min)
            nn.init.zeros_(self.tau_net[-1].weight)
            nn.init.constant_(self.tau_net[-1].bias, math.log(p0 / (1 - p0)))
        self.aux = nn.ModuleList([nn.Linear(feat_dim, n_classes) for _ in range(n_causes)]) if aux_heads else None

    def forward(self, feats, desc, avail):
        """feats: list of K (B,L,D); desc: (B,L,DESC_DIM); avail: (B,L,K) bool."""
        logits = self.scorer(torch.cat(feats, dim=-1))  # (B,L,K)
        if self.mode == 'cacg':
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_net(desc))
        else:
            tau = torch.full_like(logits[..., :1], self.fixed_tau)
        scaled = logits / tau
        if self.mask_unavailable:
            scaled = scaled.masked_fill(~avail, -1e4)
        g = F.softmax(scaled, dim=-1)
        fused = torch.cat([self.K * g[..., k:k + 1] * f for k, f in enumerate(feats)], dim=-1)
        aux_logits = torch.stack([h(f) for h, f in zip(self.aux, feats)], dim=-2) if self.aux is not None else None
        return fused, {'g': g, 'tau': tau.squeeze(-1), 'avail': avail, 'desc': desc, 'aux_logits': aux_logits}


def _entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


def ambiguity_proxy(desc, avail, ctx_saturation=10.0):
    """SOP's self-supervised ambiguity a_i in [0,1]: high when context is short or causes look alike."""
    n_ctx = torch.expm1(desc[..., 3])
    short_ctx = 1.0 - (n_ctx / ctx_saturation).clamp(max=1.0)
    sim = ((desc[..., 6:].mean(-1) + 1) / 2).clamp(0, 1)
    return 0.5 * short_ctx + 0.5 * sim


def gate_losses(info, labels_flat, flat_idx, gate_loss='utility', utility_T=1.0):
    """Returns (L_gate, L_aux) averaged over valid (non-padded) utterances.
    flat_idx: (B,L) bool mask of valid utterances, in the same order as labels_flat."""
    g = info['g'][flat_idx]                 # (N,K)
    avail = info['avail'][flat_idx]         # (N,K)
    zero = g.new_zeros(())
    l_aux = zero
    if info['aux_logits'] is not None:
        aux_lp = F.log_softmax(info['aux_logits'][flat_idx], dim=-1)          # (N,K,C)
        ll = aux_lp.gather(-1, labels_flat.view(-1, 1, 1).expand(-1, aux_lp.size(1), 1)).squeeze(-1)  # (N,K)
        l_aux = -(ll * avail).sum() / avail.sum().clamp_min(1)
    if gate_loss == 'none':
        return zero, l_aux
    if gate_loss == 'entropy':
        a = ambiguity_proxy(info['desc'][flat_idx], avail)
        h_max = torch.log(avail.sum(-1).float().clamp_min(1))
        return ((_entropy(g) - a * h_max) ** 2).mean(), l_aux
    if gate_loss == 'utility':
        assert info['aux_logits'] is not None, "utility gate loss needs aux heads"
        q = F.softmax((ll.detach() / utility_T).masked_fill(~avail, -1e4), dim=-1)
        kl = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(g.clamp_min(1e-12)))).sum(-1)
        return kl.mean(), l_aux
    raise ValueError(gate_loss)
