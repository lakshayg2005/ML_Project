# ECERC merged: Limitation 1 + Limitation 3 (IEMOCAP)

- **Limitation 1** (Divyansh): WavLM-large audio instead of IS10, plus an auxiliary VAD regression loss (λ = 1.0) on the fused feature R.
- **Limitation 3** (Lakshay): Confidence-Aware Cause Gating (CACG, dynamic τ, availability mask, no gate loss) after Feature Gating, plus temperature scaling fitted on validation.

```
L = w_y · CE(p, y)  +  1.0 · MSE(Linear(R), zscore(VAD))          R = CACG-gated [f1; f2; f3; f4]  (B, L, 1536)
after training:  T* = argmin_T NLL(valid) ,  p_test = softmax(logits / T*)
```

## Files

| File | From | What it does |
|---|---|---|
| `model.py`, `layers.py`, `cacg.py`, `metrics.py` | Lakshay `ext/`, unchanged | ECERC with pluggable cause gate; CACG module; ECE and temperature scaling |
| `dataloader_aux.py` | Divyansh, unchanged | IEMOCAP loader with VAD targets, annotator votes and optional WavLM audio |
| `train_merged.py` | Divyansh `train_aux.py` + CACG | One run: training, model selection on validation wF1, temperature scaling, results |
| `run_sweep.py` | Divyansh | All configs × seeds over the GPUs |
| `summarize_merged.py` | new | Mean ± std, paired bootstrap vs a reference, pair errors split by annotator agreement |
| `test_merge.py` | new | Sanity checks (see below) |
| `kaggle_merged.ipynb` | new | The Kaggle notebook |

## Configs (`run_sweep.py`)

| Config | Gate | Audio | VAD loss | Role |
|---|---|---|---|---|
| `CTRL` | baseline | IS10 | – | original ECERC |
| `W_VAD` | baseline | WavLM | 1.0 | Limitation 1 alone |
| `CACG` | CACG, no gate loss | IS10 | – | Limitation 3 alone |
| **`CACG_W_VAD`** | CACG, no gate loss | WavLM | 1.0 | **merged model** |
| `W_CTRL`, `CACG_W`, `CACG_UTIL_W_VAD` | | | | optional ablations |

## Run

Locally (CPU is slow but works):
```bash
python test_merge.py                    # should end with OK
python train_merged.py --tag CACG_W_VAD --seed 1 --gate cacg --gate_loss none --lambda_vad 1.0 \
       --audio_feat ../ECERC/data/iemocap/iemocap_wavlm.pkl
```

On Kaggle (GPU T4 x2, Internet off is fine):
1. Add `iemocap_wavlm.pkl` to the `iemocap-ecerc-data` dataset (New Version), so it sits next to the IEMOCAP pickles and `iemocap_aux_labels.pkl`.
2. Upload the contents of this folder as a new private dataset, `ecerc-merged-code`.
3. Import `kaggle_merged.ipynb`, attach both datasets, and run **Save Version → Save & Run All**. 80 runs take about 45 min.
4. Download `merged_run.zip` from the Output tab. The final checkpoint, the `CACG_W_VAD` seed with the best validation wF1, is in `merged_run/final/`.

## Checks (`test_merge.py`, all pass)

1. Built with the same seed, the merged model (`gate='baseline'`) matches the authors' model (`model_con.py`). The initial weights are identical, and so are the train-mode outputs, dropout included (max difference 0.0). So `CTRL` and `W_VAD` in this code should reproduce the earlier `train_aux.py` runs on the same hardware.
2. CACG with the mask off equals the baseline at initialisation (max difference 0.0).
3. CACG + WavLM (audio dim 1024) + VAD trains: the cause gate, the τ network and the VAD head all get gradients.

A 2-epoch smoke run of `CTRL` and `CACG_W_VAD` through `run_sweep.py` and `summarize_merged.py` also works.

## Merge issues and decisions

No blocking conflicts. CACG keeps R at (B, L, 1536), so the VAD head plugs in unchanged, and WavLM only changes the audio width that Evidence Gating splits on. Things to know:

1. **Two training pipelines existed.** Lakshay's `ext/train.py` reads a `cache.pkl` built by `prepare_data.py`, uses seeds 2007/1/2, and has the audio width fixed at 1582. The merged code uses Divyansh's pipeline, because only it loads WavLM features, VAD targets and annotator votes. The data split, class weights, optimiser, early stopping and selection rule are the same in both.
2. **Lakshay's earlier numbers are not directly comparable.** They come from 3 seeds in his pipeline. That is why `CACG` alone is rerun here with the same 20 seeds as the other configs.
3. **The Limitation 1 merge point in `ext/losses.py` is not used.** It is an empty `contrastive_loss` stub from the SOP's original plan. The final Limitation 1 method is the VAD loss in `train_merged.py`, which reads `out['fused']`.
4. **ECE definition.** All results use Lakshay's `metrics.ece_score` (15 bins). Some ECE values in the earlier Limitation 1 report used a 10-bin version, so they can differ by a few tenths.
5. **No soft labels.** Annotator soft labels only improved calibration, and CACG + temperature scaling already does that better. `--soft_beta` still works if needed.
6. **Empty-history attention bug.** It is handled by CACG's availability mask for the cross-speaker and self-contagion causes. Lakshay's separate attention fix (`--fix_empty_attention`) stays off, as in his recommended IEMOCAP setting.
7. **Checkpoints.** A `CACG_W_VAD` checkpoint has extra `cause_gate.*` weights and expects 1024-d WavLM audio. Load it with this `model.py` (`gate='cacg'`, `d_a=1024`), not with the authors' `inference.py`. The VAD head is saved separately (`VADhead_*.pkl`) and is not needed for prediction.
8. **The τ network starts with zero gradient.** CACG's cause scores start at 0, so τ has no effect until the scorer moves, after the first step. This is by design: at initialisation the merged model equals the original.
