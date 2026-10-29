# Pre-registration: Experiment 7 (zero-shot LLM fusion)

Verbatim excerpt of the addendum in the full protocol, committed on 2026-10-19 (UTC 19:01), three minutes before the recorded run. "Experiment 4" refers to an earlier text-expert fusion experiment whose fusion procedure this one reuses; the verdict thresholds it cites are written out in `results/run_kaggle_zeroshot_1/outputs_zeroshot/frozen_config_zeroshot.json` (`criteria`).

## Addendum (2026-10-19, fixed before running): Experiment 7, a zero-shot LLM fused with ECERC

## Motivation
- **Experiments 4 and 5:** experts fine-tuned on MELD inherit its label skew. Experiment 5's LLM scored 22–23 dev
  Disgust F1, and fusion traded one class for another.
- **Experiment 6:** better non-text evidence shifted the boundaries between related emotions (Sadness up, Joy down)
  without separating them.
- **A zero-shot LLM is never trained on MELD's labels.** Published zero-shot results on the MELD test set
  (arXiv 2609.05806) show the error profile that ECERC lacks:
  - LLaMA-70B: Fear 33.3, Disgust 39.8, Sadness 47.3.
  - Qwen-32B: Fear 26.8, Disgust 38.9, Sadness 47.1.
  - ECERC: 26.1, 31.4 and 41.0.
  - Both LLMs are weaker than ECERC on Neutral and Joy.
- **Dev-only pilot** (2026-10-19, `results/pilot_local_zeroshot_3b/`; the test split was not read). Zero-shot
  Qwen2.5-3B-Instruct with the prompt below, against the released checkpoints' saved dev predictions:
  - The LLM alone scored 35.6 dev macro-F1.
  - Plain fusion had a cross-fitted dev gain of −0.91.
  - After label-free prior matching (below), the cross-fitted gain was +0.76.
  - The pilot fixed the prompt and the calibration step. Because it used dev, the dev gate below is not independent
    of it.

## LLM
- **Model:** `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` at revision `aa79e347…`. It is a pre-quantized 4-bit (NF4) copy of
  `Qwen/Qwen2.5-32B-Instruct` (Apache-2.0), loaded frozen across two T4 GPUs.
- **Input:** one chat per utterance.
  - System message: "You are an expert annotator of emotions in TV-show dialogue."
  - User message: "Here is a dialogue, in order:", then every turn up to and including the target as "Speaker k: text"
    (generic speaker labels, Experiment 4's text clean-up), then "Which emotion does Speaker k express in the last
    line? Answer with one word: neutral, surprise, fear, sadness, joy, disgust, or anger."
  - The model's chat template is applied, with the generation prompt.
  - Past context only, ECERC's task setting.
- **Scoring:**
  - For each class, take the next-token log-probabilities of the first token of its label word, written four ways
    (lower or capitalised, with or without a leading space), and combine them by log-sum-exp.
  - Renormalise over the seven classes.
  - The first tokens must be distinct across classes, or the run stops.
  - The key/value cache of the shared prefix is reused across turns; cached and full computation agree to about
    1e-7.
- **Calibration (label-free prior matching):**
  - p̃(c) ∝ p(c) / q(c) · π_train(c), where q is the LLM's mean prediction over the dev inputs.
  - q is frozen before the test and applied unchanged to the test inputs.

## Fusion, selection and gate
These are identical to Experiment 4.
- **Pairs:** the three released checkpoints C1–C3, each with the one LLM.
- **Score:** (1 − w)·log p_ECERC + w·log p̃_LLM, with w ∈ {0.00, 0.05, …, 1.00}.
- **Selection:** w\* has the highest mean dev macro-F1 over the three pairs; ties go to the smaller w.
- **Gate:** the test runs only if the cross-fitted dev gain (5 dialogue folds, seed 12345) is at least +0.5.

## Freeze
q, w\*, the model revision, the label token ids, the prompt's SHA-256 and the criteria are written to
`frozen_config_zeroshot.json` before any test access.

## Test
Run once.
- **Primary:** the mean over the three pairs of test macro-F1(fused at w\*) − macro-F1(ECERC alone).
- **95% CI:** a two-sample dialogue bootstrap that re-selects w in every replicate (B = 10,000, seed 20261019).
- **Secondary, descriptive:**
  - fusion without calibration at its own dev-chosen w, with a test-only CI;
  - the LLM alone;
  - weighted F1, accuracy and per-class changes.

## Verdict
The thresholds are those of experiments 1 and 3 to 6. Consistency means Δ > 0 in at least 70% of pairs, which for
three pairs means all three.

## Multiplicity
This would be the third test look, after experiments 1 and 4. The verdict uses the pre-registered 95% rule. A
Bonferroni-adjusted 98.33% interval is also reported, with a flag for whether its lower bound is above 0.

## Disclosure
- The result is reported as "ECERC + zero-shot LLM"; ECERC itself is unchanged.
- The LLM's pretraining data may include Friends transcripts or the public MELD files with their labels; this cannot be
  excluded. Generic speaker labels avoid giving character names.
- The pilot used a 3B model on an Apple laptop (MLX). Experiment 7 uses a 32B model on Kaggle (transformers,
  bitsandbytes), so its LLM numbers are not expected to match the pilot's.

---
