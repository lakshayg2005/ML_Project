# Merging Limitation 3 (CACG) into your solution

Owner: Lakshay Gupta. Full code: [`ECERC/ext/`](ECERC/ext/) · Full results: [`ECERC/runs/RESULTS.md`](ECERC/runs/RESULTS.md), [`ECERC/runs/RESULTS_ensemble_coverage.md`](ECERC/runs/RESULTS_ensemble_coverage.md)

**In one line:** CACG adds a cause-level softmax gate (with a per-utterance temperature and a "no-history" mask) after ECERC's Feature Gating, plus temperature scaling after training. It keeps accuracy at baseline level and cuts calibration error (ECE) by 75–78%.

**Recommended setting for each of you:**

| Who | Dataset | Use | Why |
|---|---|---|---|
| Divyansh (L1) | IEMOCAP | CACG with dynamic τ, **no gate loss** + temperature scaling | Best IEMOCAP wF1 (70.65) and best calibration (ECE 2.95) |
| Keshav (L2) | MELD | CACG with dynamic τ + **utility gate loss** + temperature scaling | Best MELD wF1 (65.59) and macro-F1 (49.33), most stable across seeds (±0.04) |

---

## 1. What I did

1. **Reproduced the baseline.** Rewrote ECERC into one package for both datasets. `test_equivalence.py` shows its output equals the authors' `model.py` exactly (max difference 0.0). The released checkpoints give IEMOCAP 71.60 Acc / 71.77 wF1 and MELD 67.55 / 66.58 (paper: 71.60 / 71.78 and 67.32 / 66.46).
2. **Found three issues in the released code** (worth one line each in the report):
   - **There is no softmax over the four causes.** Paper Eq. 14 and the code use an independent sigmoid gate per cause. The SOP's "g = softmax(W_g[f1..f4] + b_g)" is not in the paper or the code.
   - **Empty-history bug.** When a cause has no allowed earlier turn (e.g. a speaker's first turn), every attention score is masked, so softmax spreads attention evenly over the whole padded dialogue, future turns included. This affects 29% of MELD test utterances (no earlier same-speaker turn) and about 4% of IEMOCAP.
   - **MELD modality split.** The MELD loader puts the 342-d visual features in the audio slot while the model splits at d_a = 300, so Evidence Gating's audio and visual gates see mixed features. Left unchanged so the released checkpoint still loads.
3. **Built CACG** (Section 2) and **tested it in an ablation**: 7 variants × 3 seeds × 2 datasets = 42 runs, release hyper-parameters, model selected by best validation weighted F1.
4. **Post-hoc analyses** (no retraining): temperature scaling, cause-aware temperature scaling, 3-seed ensembles, and selective prediction.

## 2. The method (equations)

Notation follows our README: f₁ self-contagion, f₂ cross-emotion, f₃ self-event, f₄ cross-event. In code these are `R_lis[0..3]`, and after the original sigmoid Feature Gating they are f̃₁…f̃₄ = `wR_lis[0..3]`. To avoid clashing with the paper's g^{s-eve} notation, call the cause gate **π** in the report.

```
C1  availability   m_i = [ n_self>0 , n_cross>0 , 1 , n_cross>0 ]
                   n_self  = # earlier turns by the same speaker,  n_cross = # earlier turns by other speakers

C2  descriptor     h_i = [ mean(w_t), mean(w_a), mean(w_v),                      evidence-gate strength
                           log(1+n_self+n_cross), log(1+n_self), log(1+n_cross),  amount of history
                           6 cosine similarities between f̃1..f̃4 ]               how alike the causes are   (12 numbers)

C3  temperature    τ_i = 0.25 + 3.75 · σ( MLP(h_i) )        MLP = LayerNorm(12) → Linear(12,32) → ReLU → Linear(32,1)

C4  cause scores   z_i = W_g [f̃1; f̃2; f̃3; f̃4] + b_g          W_g, b_g initialised to 0

C5  cause gate     π_i = softmax( z_i / τ_i ), with unavailable causes set to −∞      ← τ is applied only here

C6  fusion         R_i = [ 4·π_i1·f̃1 ; 4·π_i2·f̃2 ; 4·π_i3·f̃3 ; 4·π_i4·f̃4 ]
                   At initialisation π is uniform over the available causes. With the mask off that is 1/4 each,
                   so 4π = 1 and the model equals the original ECERC exactly (verified: max difference 0.0).

Utility gate loss (optional, recommended on MELD):
    p_k(y | f̃k)  = Linear(f̃k)                                     one helper classifier per cause, training only
    L_aux        = cross-entropy of the helper heads (available causes only)
    q_ik         = softmax_k( log p_k(y_true | f̃k) / T ) over available causes, T = 1, gradient stopped
    L_gate       = KL( q_i ‖ π_i )
    L_total      = L_task + 1.0 · L_gate + 0.5 · L_aux             (L_task = your own loss, unchanged)

Temperature scaling (after training, always):
    T* = argmin_T  NLL( softmax(logits_val / T), y_val )          one scalar, validation set only
    p  = softmax( logits_test / T* )                               same argmax → accuracy unchanged
```

The SOP's entropy loss `(H(π) − a·log 4)²` was tested and **removed**: it is the worst variant on both datasets.

## 3. Results

> **Correction (2026-10-30).** The MELD numbers below were trained with γ = 0, but the release's `MELD/loss.py` uses focal γ = 1. The true retrained MELD baseline is 65.90 wF1 / 49.78 mF1 with raw ECE 3.62, and with that loss CACG does not beat it (65.51 / 49.25). IEMOCAP is unaffected. See [`FINAL_MODEL.md`](../FINAL_MODEL.md).

Test set, mean ± std over 3 seeds. ECE in %, raw → after temperature scaling (TS). Lower ECE is better.

| Variant | IEMOCAP wF1 | IEMOCAP ECE | MELD wF1 | MELD mF1 | MELD ECE |
|---|---|---|---|---|---|
| ECERC baseline (retrained) | 70.11 ± 0.46 | 13.64 → 4.21 | 65.39 ± 0.34 | 48.96 | 9.48 → 3.01 |
| Bug fix alone (self-attend on empty history) | 68.99 ± 1.36 | 14.82 → 5.66 | 65.36 ± 0.22 | 47.73 | 9.14 → 3.18 |
| Cause softmax + mask, fixed τ | 69.78 ± 1.11 | 13.30 → 3.87 | 65.32 ± 0.15 | 48.10 | 8.97 → **2.41** |
| **CACG, dynamic τ** | **70.65 ± 0.86** | 13.54 → **2.95** | 65.34 ± 0.16 | 48.63 | 9.12 → 2.42 |
| CACG + SOP entropy loss | 69.36 ± 1.29 | 14.49 → 4.26 | 65.20 ± 0.18 | 47.22 | 8.19 → 2.54 |
| **CACG + utility loss** | 69.44 ± 1.03 | 13.48 → 4.22 | **65.59 ± 0.04** | **49.33** | 8.67 → 3.18 |

**3-seed ensembles** (average the probabilities of the 3 trained seeds; no retraining):

| Variant | IEMOCAP wF1: single → ensemble | IEMOCAP ECE + TS | MELD wF1: single → ensemble | MELD mF1 | MELD ECE + TS |
|---|---|---|---|---|---|
| ECERC baseline | 70.11 → 70.99 | 1.88 | 65.39 → 65.62 | 49.44 | 3.25 |
| CACG, dynamic τ | 70.65 → **71.30** | 2.34 | 65.34 → **65.98** | 49.65 | 2.52 |
| CACG + utility loss | 69.44 → 70.84 | 2.26 | 65.59 → 65.54 | **49.86** | 3.25 |

**Selective prediction** (answer only the most confident X% of utterances, 3-seed ensemble + TS): accuracy rises from about 71% at 100% coverage to about 85% at 50% coverage on both datasets. Baseline and CACG rank their confidence almost equally (IEMOCAP AURC 15.45 vs 15.31; MELD 16.92 vs 16.98). Figure: [`ECERC/runs/figures/selective_accuracy.png`](ECERC/runs/figures/selective_accuracy.png).

**What to claim:**
- **Accuracy:** CACG does not change it beyond seed noise (±0.5–1.3 F1 on IEMOCAP).
- **Calibration:** CACG + TS gives the lowest ECE, 20–30% below the baseline after the same temperature scaling, and 75–78% below the raw baseline.
- **Interpretability:** π shows which cause drove each prediction. Emotion causes dominate (self-contagion ≈ 0.5, cross-emotion ≈ 0.4, events < 0.1), and self-contagion grows with dialogue length.
- **Correctness:** π removes the noisy empty-history vectors.
- **Ensembling** is the easiest accuracy gain for every variant. The best single number is the CACG dynamic-τ ensemble: IEMOCAP 71.30 wF1, MELD 65.98 wF1.

## 4. How to merge it into your code

These steps patch the **authors' original `model.py`** (your `model_con.py` / `MELD/model.py` use the same variable names). About 30 lines.

### Step 1: copy the module

Copy [`ECERC/ext/cacg.py`](ECERC/ext/cacg.py) next to your `model.py`. It has no dependencies beyond PyTorch.

### Step 2: make Evidence Gating also return the gate strengths

In `class ModalFilterV2.forward`, replace the `return` line:

```python
gates = torch.stack([w_t.mean(-1), w_a.mean(-1), w_v.mean(-1)], dim=-1)   # (B, L, 3)
return torch.cat([w_t * (t), w_a * (a), w_v * (v)], dim=-1), gates
```

### Step 3: create the gate in `ECERC.__init__`

```python
from cacg import CauseGate, build_descriptor

# at the end of __init__ (input_size here is hidden_size * 3 = 384)
self.cause_gate = CauseGate(input_size, n_classes, mode='cacg', tau_min=0.25, tau_max=4.0,
                            mask_unavailable=True, aux_heads=use_utility_loss)   # True only for the utility loss
self.gate_info = None
```

### Step 4: use it in `ECERC.forward`

```python
# Evidence Gating now returns two values
evidence_enc, modal_gates = self.modal_filtering(U_e_[:, :, :self.d_t], U_e_[:, :, self.d_t:(self.d_t + self.d_a)],
                                                 U_e_[:, :, (self.d_t + self.d_a):])

# ... the four attentions and wR_lis exactly as before ...

# replace:   R = torch.cat(wR_lis, dim=-1)
n_self = (smask & (~imask)).sum(-1).float()        # earlier same-speaker turns
n_cross = cmask.sum(-1).float()                     # earlier other-speaker turns
avail = torch.stack([n_self > 0, n_cross > 0, torch.ones_like(n_self, dtype=torch.bool), n_cross > 0], -1)
desc = build_descriptor(modal_gates, n_self, n_cross, wR_lis)
R, self.gate_info = self.cause_gate(wR_lis, desc, avail)
# R keeps its shape (B, L, 1536): the classifier, your contrastive head and your loss need no change
```

The order of `wR_lis` must stay `[contagion, cross_emotion, self_event, cross_event]`, which is the authors' order.

### Step 5 (utility loss only): add the gate loss in your training loop

```python
from cacg import gate_losses

log_prob = model(...)                                     # unchanged
loss = your_loss(log_prob, label)                         # your L1 / L2 objective, unchanged
l_gate, l_aux = gate_losses(model.gate_info, label, umask.bool(), gate_loss='utility', utility_T=1.0)
loss = loss + 1.0 * l_gate + 0.5 * l_aux
```

`umask.bool()` selects the real (non-padded) utterances in the same batch-major order the original code uses to flatten `label`. For the **dynamic-τ variant without a gate loss** (recommended on IEMOCAP), skip this step and set `aux_heads=False`.

### Step 6: temperature scaling and ECE at evaluation

Copy `fit_temperature` and `ece_score` from [`ECERC/ext/metrics.py`](ECERC/ext/metrics.py). After training, with the best checkpoint loaded:

```python
from metrics import fit_temperature, ece_score

T = fit_temperature(valid_log_probs, valid_labels)        # numpy arrays: (N, C) log-probs, (N,) labels
p_test = softmax(test_log_probs / T)                        # predictions identical, confidence calibrated
print('ECE raw', 100 * ece_score(np.exp(test_log_probs), test_labels), ' ECE+TS', 100 * ece_score(p_test, test_labels))
```

### Checklist

- [ ] Sanity check: build the merged model with `mask_unavailable=False`, load the original weights (`strict=False`) and run one batch. Before training, its outputs must equal the original model's exactly. With the mask on they differ on purpose: on utterances with missing history, the available causes get 4/3 or 4 instead of 1, and the noisy unavailable ones get 0. I ran this check on the authors' `IEMOCAP/model.py` patched with Steps 1–5: difference 0.0 with the mask off, and the utility loss and backward pass run.
- [ ] Report ECE raw and after TS for **both** your baseline and your merged model, so the comparison is fair.
- [ ] Keep model selection on the validation set only.
- [ ] Use 3 or more seeds and report mean ± std. Differences inside the std are noise.

## 5. Settings used

| Setting | IEMOCAP | MELD |
|---|---|---|
| Epochs / patience | 200 / 50 | 40 / 20 |
| Learning rate / L2 | 1e-4 / 2e-4 | 1e-5 / 2e-4 |
| Batch size | 64 | 32 |
| Task loss | class-weighted NLL (release) | unweighted NLL (release) |
| Seeds | 2007, 1, 2 | 0, 1, 2 |
| CACG | τ ∈ [0.25, 4], mask on, λ_gate = 1.0, λ_aux = 0.5, utility T = 1 | same |
| Hardware | GeForce 940MX 2 GB: ~2 s/epoch | ~6 s/epoch |
