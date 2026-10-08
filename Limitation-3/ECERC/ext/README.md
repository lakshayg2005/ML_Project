# ECERC Revisited — extension code (`ext/`)

Unified, mergeable re-implementation of the official ECERC release plus **Limitation 3:
Confidence-Aware Cause Gating (CACG)** (owner: Lakshay Gupta). The original `IEMOCAP/` and `MELD/`
folders are left untouched.

```
ext/
  layers.py           transformer blocks copied verbatim from the release (same parameter names)
  model.py            ECERC for both datasets; gate='baseline' == official model (bit-exact, see test)
  cacg.py             CACG module + gate losses (self-contained)
  losses.py           task loss + MERGE POINTS for Limitation 1 and 2
  data.py             loaders, identical splits to the release (reads compact cache)
  prepare_data.py     one-time conversion of the released pickles -> float32 cache
  train.py            train / evaluate (one entry point, release hyper-parameters per dataset)
  calibrate.py        post-hoc global TS and cause-aware TS (fit on validation only)
  aggregate.py        tables (mean ± std over seeds) + figures -> runs/RESULTS.md, runs/figures/
  run_grid.sh         the ablation grid
  test_equivalence.py checks ext model == official model (max abs diff 0.0)
```

## 1. Setup

```bash
# features (Google Drive folder linked in the official README) -> ECERC/data/{iemocap,meld}/*.pkl
python ext/prepare_data.py          # ~5 min once; the raw pickles need >8 GB RAM to load together
python ext/test_equivalence.py      # sanity check: prints max diff 0.00e+00
```
Requirements: PyTorch ≥ 2.0, scikit-learn, pandas, matplotlib. A 2 GB GPU is enough
(IEMOCAP ≈ 2 s/epoch, MELD ≈ 6 s/epoch on a GeForce 940MX); a full grid fits on Colab's free T4 as well.

## 2. Baseline reproduction

| | IEMOCAP Acc / wF1 | MELD Acc / wF1 |
|---|---|---|
| Paper (avg of 3 runs) | 71.60 / 71.78 | 67.32 / 66.46 |
| Released checkpoint, evaluated with `ext/` | **71.60 / 71.77** | **67.55 / 66.58** |
| Retrained with release settings, 3 seeds | see results table | see results table |

```bash
python ext/train.py --dataset IEMOCAP --eval_ckpt IEMOCAP/ECERC_MODEL.pkl
python ext/train.py --dataset MELD    --eval_ckpt MELD/ECERC_MODEL.pkl
```
Retraining lands ~1 point below the released checkpoints (the authors note exact reproduction needs their
environment). The official MELD schedule (lr 1e-5, 40 epochs) usually selects the last epoch, i.e. it is
slightly under-trained; we keep it unchanged so that all comparisons are like-for-like.

## 3. What we found in the released code (relevant to the report)

1. **There is no softmax over the four cause types.** The SOP describes Feature Gating as
   `g = softmax(W_g[f1;f2;f3;f4] + b_g)`; the released code instead applies an *element-wise sigmoid*
   gate to each cause vector (`sigmoid(W f_k) ⊙ f_k`) and concatenates the four. There is therefore no
   "fixed temperature" and no gating entropy in the original model. The SOP sentence claiming we measured
   constant gating entropy on the released checkpoints must be rewritten — CACG *introduces* the
   cause-level distribution.
2. **Empty-history attention bug.** When a cause has no valid history (e.g. no earlier utterance by the
   same speaker), every key is masked with −1e9 and the softmax becomes *uniform over the whole padded
   dialogue*, including future turns and padding. The resulting cause vector is noise that depends on the
   batch. This affects 3–4 % of IEMOCAP test utterances but **28.6 %** (no prior same-speaker turn) /
   **15.8 %** (no prior other-speaker turn) of MELD test utterances.
3. **MELD modality split is misaligned.** The MELD loader puts the 342-d visual features in the audio slot
   while the model splits at `d_a = 300`, so Evidence Gating's "audio" gate sees 300 visual dims and its
   "visual" gate sees 42 visual + 300 audio dims. Kept as-is for checkpoint compatibility.

## 4. CACG — final formulation

Applied on top of the release's element-wise Feature Gating (which is kept):

```
h_i   = [modality-gate means (3), log(1+#prior), log(1+#prior same-spk), log(1+#prior other-spk),
         6 pairwise cosines between cause vectors]                         (evidence-quality descriptor)
tau_i = tau_min + (tau_max - tau_min) * sigmoid(MLP(h_i))                  (SOP Eq. 1, tau in [0.25, 4])
g_i   = softmax( (W_g [f1;f2;f3;f4] + b_g) / tau_i ,  masked to available causes )   (SOP Eq. 2 + mask)
fused = [4 g_i1 f1 ; 4 g_i2 f2 ; 4 g_i3 f3 ; 4 g_i4 f4]
```
* `W_g, b_g` are zero-initialised, so at initialisation CACG is *exactly* the original ECERC (verified).
* **Availability mask**: a cause with no valid history gets weight 0, which removes the noisy
  vectors from bug 2 instead of feeding them to the classifier.

**Gate loss — changed from the SOP (and why).** SOP Eq. 3 pulls `H(g_i)` toward `a_i·log 4`, where `a_i`
is a hand-made function of the same descriptor `h_i` that `tau_i` is computed from. The temperature can
satisfy it by copying the heuristic, and nothing ties it to whether the cause weighting is *right*.
We replace it with a label-grounded target:

```
p_k(y | f_k)  = auxiliary classifier per cause                  (trained with L_aux, deep supervision)
q_ik          = p_k(y_i|f_ik)^(1/T) / sum_j p_j(y_i|f_ij)^(1/T)  (posterior "which cause explains y", detached)
L_gate        = KL(q_i || g_i)
L_total       = L_task + λ_gate L_gate + λ_aux L_aux             (λ_gate = 1, λ_aux = 0.5)
```
`q_i` is sharp when one cause clearly explains the gold emotion and flat when several do. That is the
"decisive vs. blended" behaviour the SOP asks for, grounded in the label. Both losses are implemented
(`--gate_loss entropy|utility`) and compared in the ablation.

**Output calibration.** The analysis showed that output over-confidence is roughly uniform across
context lengths (baseline: mean confidence 0.81 vs accuracy 0.69 at every context bucket), so it cannot be
fixed through the gate alone. We therefore report ECE raw, after global temperature scaling (TS,
fitted on validation), and after *cause-aware* TS (CA-TS: per-utterance temperature predicted from
`h_i`, `H(g_i)`, `tau_i`, fitted on validation). CA-TS did **not** beat global TS on held-out data,
which is reported as a negative result.

## 5. Running the ablation

```bash
cd ext
bash run_grid.sh IEMOCAP "2007 1 2"     # ~7.5 min per run
bash run_grid.sh MELD "0 1 2"           # ~5 min per run
python calibrate.py --dataset IEMOCAP && python calibrate.py --dataset MELD
python aggregate.py                     # -> runs/RESULTS.md and runs/figures/*.png
```
Model selection follows the release: best validation weighted-F1 epoch, test metrics reported at that epoch.

## 6. Merge guide for Limitation 1 and 2

`model(...)` returns a dict:

| key | shape | use |
|---|---|---|
| `log_prob` | (N, C) | class-balanced / focal loss — **Limitation 2**: add a branch in `losses.build_task_loss` (`--task_loss cb` / `focal`) |
| `fused` | (N, 4·384) | representation before the classifier — **Limitation 1**: implement `losses.contrastive_loss(fused, labels)`, enable with `--lambda_contrast` |
| `gate` | dict | CACG internals (used by `cacg.gate_losses`) |

`train.run_epoch` already computes `L_task + λ_gate L_gate + λ_aux L_aux + λ_contrast L_contrast`
(SOP Eq. 4), so each extension only touches its own function. Oversampling for MELD goes into
`data.get_loaders` (a `WeightedRandomSampler` over dialogues).

## 7. Results

Full tables: `runs/RESULTS.md`; figures: `runs/figures/`. 3 seeds per row; test metrics at the best-validation epoch.

| | IEMOCAP wF1 | IEMOCAP ECE raw → TS | MELD wF1 | MELD mF1 | MELD ECE raw → TS |
|---|---|---|---|---|---|
| ECERC (reproduced) | 70.11 ± 0.46 | 13.6 → 4.2 | 65.39 ± 0.34 | 48.96 | 9.5 → 3.0 |
| + cause softmax, fixed τ | 69.78 ± 1.11 | 13.3 → 3.9 | 65.32 ± 0.15 | 48.10 | 9.0 → 2.4 |
| + dynamic τ, no gate loss | **70.65 ± 0.86** | 13.5 → **3.0** | 65.34 ± 0.16 | 48.63 | 9.1 → 2.4 |
| + dynamic τ + SOP Eq.3 entropy loss | 69.36 ± 1.29 | 14.5 → 4.3 | 65.20 ± 0.18 | 47.22 | 8.2 → 2.5 |
| + dynamic τ + utility loss (CACG, ours) | 69.44 ± 1.03 | 13.5 → 4.2 | **65.59 ± 0.04** | **49.33** | 8.7 → 3.2 |

Takeaways:
* No CACG variant changes weighted F1 beyond seed noise on IEMOCAP (std ≈ 0.5–1.3). On MELD the utility loss
  gives a small, low-variance gain (+0.2 wF1, +0.4 mF1, Fear F1 21.1 → 23.2, Sad 38.5 → 39.5).
* The SOP's entropy loss (Eq. 3) is the worst variant on both datasets (MELD Disgust F1 30.2 → 21.0).
* Without a gate loss, τ collapses to a near-constant (std ≈ 0.01); with the SOP loss it tracks context
  (corr(τ, ctx) = −0.85 on MELD) but this does not help accuracy or calibration.
* Output over-confidence is the main calibration problem. Global temperature scaling removes ~70 % of
  ECE (IEMOCAP 13.6 → 4.2, MELD 9.5 → 3.0) at zero accuracy cost; cause-aware TS does not beat it.
* The gate is interpretable: emotion causes dominate (self-contagion ≈ 0.5, cross-emotion ≈ 0.4, events < 0.1),
  and the self-contagion share grows with dialogue history (`runs/figures/*_gate.png`).
