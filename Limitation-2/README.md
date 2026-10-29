# Limitation 2: Class imbalance on MELD, solved by fusing ECERC with a zero-shot LLM

Owner: Keshav Mishra.

ECERC reaches about 67% accuracy on MELD, but its rare emotions stay low: Fear 26 and Disgust 31 F1 (mean of the three
released checkpoints). Fusing ECERC's predictions with a **zero-shot** Qwen2.5-32B-Instruct, a model that has never been
trained on MELD's skewed labels, raises test macro-F1 by **+3.38 points (95% CI +0.64 to +5.73)**, with the largest
gains on exactly the classes Limitation 2 is about.

## Result (MELD test, 2,610 utterances; mean over the three released ECERC checkpoints)

| System | Accuracy | Weighted F1 | Macro-F1 | Fear F1 | Sadness F1 | Disgust F1 | Joy F1 |
|---|---|---|---|---|---|---|---|
| ECERC alone (paper's released models) | 67.32 | 66.46 | 50.87 | 26.12 | 40.95 | 31.43 | 64.95 |
| **ECERC + zero-shot LLM (w = 0.2)** | **68.24** | **67.26** | **54.25** | **33.77** | **46.69** | **40.22** | 61.67 |
| Zero-shot LLM alone | 63.10 | 63.14 | 51.05 | 31.88 | 45.38 | 37.58 | 55.57 |

- **Macro-F1 +3.38 [+0.64, +5.73]**, positive with every checkpoint (+2.83, +3.65, +3.65).
- **Per class (Δ F1):** Disgust +8.79, Fear +7.64, Sadness +5.74, Surprise +3.23, Anger +1.72, Neutral −0.21, Joy −3.28.
- **Accuracy +0.92 and weighted F1 +0.80.** The rare-class gain is not bought with overall performance; both are above
  the ECERC paper's Table 3 (67.32 / 66.46).
- **Cost:** Joy F1 falls 3.28 points, beyond the pre-registered −2.0 per-class guardrail, so the pre-registered verdict
  is **TRADE-OFF** rather than SUCCESS.

The baseline row reproduces the ECERC paper exactly: the mean of the three released checkpoints gives the paper's
Table 3 (accuracy 67.32, weighted F1 66.46) with identical per-class F1.

## Why it works

Standard imbalance fixes did not help ECERC. Class-balanced loss weights, a stronger focal exponent, minority
oversampling and post-hoc logit adjustment all lowered test macro-F1, and two experts fine-tuned on MELD
(RoBERTa-large, Qwen2.5-7B) traded one class for another. Their extra minority predictions were mostly wrong: the
errors come from overlap between related emotions in the evidence, not from a decision threshold.

Any model trained on MELD inherits its label skew. A zero-shot LLM does not. On its own it matches ECERC on macro-F1
(51.05 vs 50.87) with the **opposite error profile**: much better on Disgust, Fear and Sadness, much worse on
Neutral and Joy. Fusion keeps the strengths of both.

## Method

1. **ECERC:** the three checkpoints released by the authors (C1–C3), unchanged. The four upstream files are embedded
   verbatim (`ecerc_original/`) and checked by SHA-256 at run time.
2. **Zero-shot LLM:** `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (4-bit copy of Qwen2.5-32B-Instruct, pinned revision),
   frozen, on Kaggle 2× T4. One chat per utterance with the dialogue so far (past context only, ECERC's setting) and the
   question "Which emotion does Speaker k express in the last line? Answer with one word: …". The score for each class
   is the next-token probability of its label word.
3. **Label-free calibration:** p̃(c) ∝ p(c) / q(c) · π_train(c), where q is the LLM's mean prediction on the dev inputs
   (no labels) and π_train the training label prior. It mattered for a 3B pilot model; for the 32B model it changes
   the gain by only 0.1 (+3.27 without it).
4. **Fusion:** score = (1 − w)·log p_ECERC + w·log p̃_LLM, with w chosen on dev from {0, 0.05, …, 1}. The chosen
   w = 0.2 was stable: picked in 4 of 5 dev folds and in 75% of test bootstrap replicates.

## Protocol

- **Pre-registered** ([`PROTOCOL.md`](PROTOCOL.md)): rules, gate and success criteria were committed before the run.
- **Dev-only decisions:** the calibration, w and a cross-fitted dev gate (+5.63 against a +0.5 threshold) were frozen
  to `frozen_config_zeroshot.json` before the test set was read.
- **One test pass.** The 95% CI is a dialogue-level bootstrap that re-selects w in every replicate (B = 10,000).
- **All 18 integrity checks passed** (`checks.json`): data and checkpoint hashes, label-token checks, freeze before
  test.

## Caveats

- The improved system is **ECERC + a 32B LLM at inference**; ECERC itself is unchanged.
- The LLM's pretraining data may include Friends transcripts or the public MELD files. Alone it is weaker than ECERC
  overall (weighted F1 63.1 vs 66.5), which argues against memorised labels.
- The MELD test set had been evaluated twice before in this study; a Bonferroni-adjusted (third-look) 98.33% interval
  is [+0.00, +6.28].
- Not tested on IEMOCAP.

## Reproduce (Kaggle, about 3 hours; the recorded run took 2 h 45 min)

1. Sign in at <https://www.kaggle.com> (phone verification is needed once for GPUs and internet).
2. **Create → New Notebook → File → Import Notebook**, upload `notebooks/ECERC_L2_zeroshot_llm_fusion.ipynb`.
3. **Session options:** Accelerator **GPU T4 x2** (the 4-bit 32B model needs both GPUs), **Internet On**.
4. **Save Version → Save & Run All (Commit)**. The notebook downloads the MELD features, the three checkpoints and the
   LLM (19 GB), and checks every hash.
5. When it finishes, download `ecerc_l2/ecerc_l2_zeroshot_results.zip` from **Output**.

The notebook is self-contained: it embeds `ecerc_original/` and `src/`. If the gate had failed, it would have stopped
before the test set.

## Folder layout

| Path | Contents |
|---|---|
| `notebooks/ECERC_L2_zeroshot_llm_fusion.ipynb` | The experiment, end to end |
| `src/l2_zeroshot.py` | LLM scoring, calibration, freeze and test stages |
| `src/l2_expert.py`, `src/l2_posthoc.py` | Shared library: fusion grid, cross-fitted gate, bootstrap, data and checkpoint loading, metrics |
| `ecerc_original/` | Verbatim copies of upstream `MELD/model.py`, `dataloader.py`, `inference.py`, `loss.py` |
| `PROTOCOL.md` | The pre-registration |
| `results/run_kaggle_zeroshot_1/outputs_zeroshot/` | The recorded run: `SUMMARY_zeroshot.md`, `test_results_zeroshot.json`, frozen config, checks, and the cached dev/test LLM scores in `dumps/` |
| `results/pilot_local_zeroshot_3b/` | Dev-only 3B pilot that fixed the prompt and the calibration step (its comparison inputs, the checkpoints' dev predictions, are not included) |
| `MERGING.md` | How to fuse the cached LLM scores with another ECERC variant (e.g. CACG) |

## References

- T. Zhang and Z. Tan. *ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation.* ACL 2025.
- Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024.
- S. Poria et al. *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.* ACL 2019.
