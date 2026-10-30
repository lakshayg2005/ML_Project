# Merge of Limitations 2 and 3: ECERC + class-imbalance work + CACG

One codebase that produces every row of the L2 + L3 ablation with the same data pipeline, model, losses and
selection rule. It is a copy of Limitation 3's unified package (`Limitation-3/ECERC/ext`, commit `4bbcefc`) with the
Limitation 2 parts added; `Limitation-*/` folders stay unchanged as the record of each person's own experiments.
The L1 + L3 merge and the merge of all three are done separately.

| Limitation | What is merged | Where |
|---|---|---|
| 2 Class imbalance (Keshav) | Class-balanced loss (`--task_loss cb`, ablation row) and zero-shot LLM fusion (MELD) | `ext/losses.py`, `ext/fuse_llm.py` |
| 3 Cause-gating calibration (Lakshay) | CACG with dynamic τ, availability mask, temperature scaling | `ext/cacg.py`, `ext/model.py` |

## Changes relative to `Limitation-3/ECERC/ext`

1. **Release objective per dataset** (`losses.py`, `train.py`). The default task loss is now `--task_loss release`:
   focal γ = 1 without class weights on MELD (upstream `MELD/loss.py`), γ = 0 with class weights on IEMOCAP (upstream
   `IEMOCAP/loss.py`). The earlier `ext/` default trained MELD with γ = 0; it is still available as `--task_loss ce`.
2. **Class-balanced loss** (`--task_loss cb --cb_beta 0.999`): effective-number weights from training-split counts,
   rescaled to mean 1 per training utterance, as α inside the release loss. Identical to Limitation 2's study; on MELD
   the weights are `[0.725, 1.026, 3.055, 1.452, 0.871, 3.026, 1.072]`.
3. **Validation loss always uses the release objective**, so early stopping compares like with like across variants.
4. **Dev outputs are saved** (`*_valid_outputs.npz` next to `*_test_outputs.npz`), so post-hoc steps are chosen on dev.
5. **`fuse_llm.py`** (new): fuses any MELD variant with the cached zero-shot LLM scores in
   `Limitation-2/results/run_kaggle_zeroshot_1/outputs_zeroshot/` (the LLM is not re-run). Two stages:
   - `freeze` reads only dev outputs, chooses the fusion weight w per variant and a temperature per run, and writes
     `runs/MELD/fusion_freeze.json`;
   - `test` reads that file, scores test once and refuses to run again.
6. **`run_ablation.sh`** (new): every training run of the ablation, then the freeze and the test stage.

## Run

```bash
cd Merge-L2-L3/ext
python prepare_data.py                 # data in Merge-L2-L3/data/{meld,iemocap}/ as described in ext/README.md
bash run_ablation.sh train             # MELD: baseline, cb0.999, cacg_none x seeds 0,1,2; IEMOCAP: baseline, cacg_none x 2007,1,2
bash run_ablation.sh freeze            # dev only
bash run_ablation.sh test              # once; writes runs/MELD/fusion_results.{json,md}
python aggregate.py                    # Limitation 3's tables for the trained variants (both datasets)
```

## Rows produced

MELD (mean ± std over seeds 0, 1, 2; `fusion_results.md`): `baseline`, `cb0.999`, `cacg_none`, and each of them
`+ LLM`. The full system is `cacg_none + LLM`. IEMOCAP (seeds 2007, 1, 2; `aggregate.py`): `baseline`, `cacg_none`.

## Results

The full write-up, including every Limitation 2 method and the figures, is in [`REPORT.md`](REPORT.md).

Full tables: `runs/MELD/fusion_results.md` and `runs/RESULTS.md`. Every w and T was fixed on dev before test was read.

**MELD test, mean ± std over seeds 0, 1, 2:**

| Row | Acc | wF1 | Macro-F1 | Fear F1 | Disgust F1 | Joy F1 | ECE after TS | w (dev) | Dev macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 66.83 ± 0.46 | 65.72 ± 0.38 | 49.57 ± 0.54 | 23.2 | 30.5 | 64.2 | 3.49 | – | 54.07 |
| cb0.999 | 65.80 ± 0.33 | 65.68 ± 0.28 | 49.50 ± 0.38 | 22.2 | 30.0 | 63.5 | 2.56 | – | 54.18 |
| cacg_none | 67.09 ± 0.18 | 65.96 ± 0.24 | 49.67 ± 0.50 | 22.8 | 29.9 | 64.4 | 3.05 | – | 53.45 |
| baseline + LLM | 68.11 ± 0.04 | 67.01 ± 0.04 | 53.59 ± 0.06 | 32.3 | 38.9 | 61.6 | 3.38 | 0.20 | **59.96** |
| cb0.999 + LLM | 68.43 ± 0.04 | 67.66 ± 0.01 | 54.45 ± 0.08 | 34.9 | 39.6 | 63.0 | 3.58 | 0.15 | 59.38 |
| **cacg_none + LLM (merged system)** | 68.17 ± 0.06 | 67.11 ± 0.07 | 53.77 ± 0.07 | 32.9 | 39.1 | 62.1 | 3.18 | 0.20 | 59.51 |

**Macro-F1 differences on test** (mean over seeds, 95% dialogue bootstrap at the frozen w, 10,000 resamples):

| Comparison | Δ macro-F1 | 95% CI |
|---|---|---|
| cacg_none + LLM vs baseline (merged system) | +4.20 | [+1.76, +6.61] |
| baseline + LLM vs baseline (LLM alone) | +4.02 | [+1.55, +6.45] |
| cacg_none + LLM vs baseline + LLM (CACG, given the LLM) | +0.18 | [−0.24, +0.62] |
| cb0.999 + LLM vs baseline + LLM (CB, given the LLM) | +0.86 | [−0.24, +2.08] |
| cacg_none vs baseline | +0.10 | [−0.87, +1.07] |
| cb0.999 vs baseline | −0.07 | [−1.36, +1.25] |

**IEMOCAP test, mean ± std over seeds 2007, 1, 2** (`aggregate.py`, population std; the LLM was not run on IEMOCAP):

| Row | Acc | wF1 | Macro-F1 | ECE after TS |
|---|---|---|---|---|
| baseline | 69.64 ± 0.72 | 69.88 ± 0.78 | 68.86 ± 0.97 | 4.09 ± 0.91 |
| cacg_none | 70.14 ± 0.80 | 70.40 ± 0.84 | 69.65 ± 0.64 | 3.91 ± 0.50 |

**Reading the results:**

- **The gain comes from the LLM.** Fusion raises MELD macro-F1 by about 4 points, led by Fear and Disgust. Neither CACG nor CB
  has an effect on macro-F1 that is separable from zero, alone or on top of the LLM.
- **`cb0.999 + LLM` has the best test numbers but not the best dev numbers,** so it is not the selected system. It ranked
  third on dev, and its test lead over `baseline + LLM` is within the interval above.
- **On dev, `baseline + LLM` ranks first** (59.96 against 59.51 for the merged system). `cacg_none + LLM` is named as the
  merged system because it combines both limitations, not because dev preferred it.
- **The small ± on `+ LLM` rows is not the uncertainty of the result.** The LLM scores are one fixed run and their
  log-probability spread is about 5 times the model's (median 25 against 4.5), so fused predictions depend less on the
  seed. Use the bootstrap intervals.
- **The gain is about 3.2 to 4.2 points, depending on the baseline.** This retraining's baseline is 0.83 below Limitation
  2's (see the check below). Against Limitation 2's 50.40 the same fused rows gain about 3.2 to 3.4, in line with the
  +3.38 on the released checkpoints.
- **CACG on IEMOCAP** moves weighted F1 by +0.52, inside the seed spread, as in Limitation 3's own runs.

## Checks done before the first real run

- **The fusion code reproduces Limitation 2's result exactly.** With the three released checkpoints' saved outputs
  standing in for three seeds, `fuse_llm.py` chooses w = 0.2 on dev (macro-F1 54.04 → 59.98) and gives test
  67.32 / 66.46 / 50.87 → 68.24 / 67.26 / 54.25 (accuracy / weighted F1 / macro-F1), as recorded in Limitation 2.
- **Every training path runs** on synthetic data (CPU, 2 epochs): MELD baseline, cb0.999 and cacg_none, IEMOCAP
  cacg_none; each writes its JSON and its dev and test outputs.
- **Equivalence with Limitation 2's retraining: not exact.** Same objective and selection rule, but different runs.
  Seed 0 gave 67.36 / 66.15 / 50.15 at epoch 39, against 67.51 / 66.54 / 51.37 at epoch 35. Over three seeds:
  66.83 / 65.72 / 49.57 against 66.87 / 66.00 / 50.40. Macro-F1 is lower on all three seeds, while dev macro-F1 matches
  (54.07 against 54.18). The cause was not established; the two runs used different code paths. About half
  of the gap is Fear (F1 23.2 against 26.2, on 50 test utterances).

## Notes for the report

- **Report ECE after temperature scaling for fused rows.** Fusion leaves the scores over-confident (`baseline + LLM`
  seed 0: mean confidence 0.84 at accuracy 0.68; ECE about 15 raw, about 3.4 after scaling with T ≈ 1.65).
- **The fused system is ECERC + a 32B LLM at inference.** Joy F1 falls about 3 points with fusion.
- **Test-set use.** Limitation 2's experiments evaluated the MELD test set three times; the results above are the fourth
  look. Do not re-run the test stage or choose among these rows by their test numbers.
- **For the merge of all three.** Limitation 3's hook for Limitation 1 is kept unchanged: `contrastive_loss(fused,
  labels)` in `ext/losses.py`, weighted by `--lambda_contrast` (0 here, so it has no effect).
