"""Classification, calibration and gate diagnostics."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def ece_score(probs, labels, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return ece


def reliability_bins(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append((float((lo + hi) / 2), float(acc[m].mean()) if m.any() else None,
                    float(conf[m].mean()) if m.any() else None, int(m.sum())))
    return out


def fit_temperature(logits, labels):
    """Post-hoc temperature scaling (Guo et al., 2017), fitted on validation logits."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def classification_metrics(log_probs, labels, target_names):
    probs = np.exp(log_probs)
    preds = probs.argmax(1)
    conf = probs.max(1)
    wrong = preds != labels
    per_class = f1_score(labels, preds, average=None, labels=list(range(len(target_names))))
    return {
        'acc': 100 * accuracy_score(labels, preds),
        'wf1': 100 * f1_score(labels, preds, average='weighted'),
        'mf1': 100 * f1_score(labels, preds, average='macro'),
        'per_class_f1': {n: 100 * float(v) for n, v in zip(target_names, per_class)},
        'nll': float(-log_probs[np.arange(len(labels)), labels].mean()),
        'brier': float(((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1).mean()),
        'ece': 100 * ece_score(probs, labels),
        'overconf_err': 100 * float((wrong & (conf > 0.9)).mean()),  # share of all utterances that are confident (>0.9) errors
        'confusion': confusion_matrix(labels, preds, labels=list(range(len(target_names)))).tolist(),
    }


def gate_metrics(g, tau, n_ctx, avail):
    """g: (N,K); tau: (N,); n_ctx: (N,); avail: (N,K) bool."""
    ent = -(g * np.log(np.clip(g, 1e-12, None))).sum(1)
    n_av = avail.sum(1)
    h_norm = ent / np.log(np.maximum(n_av, 2))
    out = {
        'gate_mean': g.mean(0).tolist(),
        'gate_entropy_mean': float(ent.mean()),
        'gate_entropy_std': float(ent.std()),
        'gate_entropy_norm_mean': float(h_norm.mean()),
        'tau_mean': float(tau.mean()), 'tau_std': float(tau.std()),
    }
    if tau.std() > 1e-8:
        out['corr_tau_ctx'] = float(np.corrcoef(tau, np.log1p(n_ctx))[0, 1])
    if ent.std() > 1e-8:
        out['corr_entropy_ctx'] = float(np.corrcoef(ent, np.log1p(n_ctx))[0, 1])
    return out
