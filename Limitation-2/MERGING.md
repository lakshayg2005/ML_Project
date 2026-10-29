# Merging the zero-shot LLM fusion into another ECERC variant

The fusion works on any model's output, so it can be added on top of CACG or the contrastive loss without touching the
model. **You do not need to run the LLM:** its scores for every MELD dev and test utterance are saved.

## 1. The three files

All under `results/run_kaggle_zeroshot_1/outputs_zeroshot/`:

| File | Contents |
|---|---|
| `dumps/dev_llm.npz` | `logp` (1109, 7): LLM log-probabilities for every dev utterance; `y` (1109,): labels |
| `dumps/test_llm.npz` | `logp` (2610, 7) and `y` (2610,) for test |
| `frozen_config_zeroshot.json` | `log_q_dev`, `log_prior_train` (label-free calibration constants), `w_grid`, `w_star` = 0.2 |

- **Class order:** the release's `neu, sur, fea, sad, joy, dis, ang`, the same as `Limitation-3/ECERC/ext/train.py`'s
  `target_names`.
- **Row order:** dialogues in the order of the data pickle's `valid_ids` / `test_ids`, utterances in turn order.
  `ext/data.py` (unshuffled valid/test loaders, labels flattened with `umask`) produces this order; the code below
  checks it by comparing labels.

## 2. Code

```python
import json
import numpy as np

Z = 'Limitation-2/results/run_kaggle_zeroshot_1/outputs_zeroshot'
F = json.load(open(f'{Z}/frozen_config_zeroshot.json'))
log_q, log_prior = np.array(F['log_q_dev']), np.array(F['log_prior_train'])


def llm_scores(split):
    """Calibrated LLM log-probs: divide out the LLM's mean dev prediction, multiply in the train prior."""
    z = np.load(f'{Z}/dumps/{split}_llm.npz')
    s = z['logp'] - log_q + log_prior
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True), z['y']


def fuse(model_logp, llm_logp, w):
    return (1 - w) * model_logp + w * llm_logp


def macro_f1(y, scores, C=7):
    p = scores.argmax(1)
    return 100 * np.mean([2 * ((p == c) & (y == c)).sum() / max(1, (p == c).sum() + (y == c).sum()) for c in range(C)])


# model_dev / model_test: (N, 7) log-softmax outputs of your trained model (e.g. va_res['lp'] / te_res['lp'])
llm_dev, y_dev = llm_scores('dev')
llm_test, y_test = llm_scores('test')
assert (y_dev == dev_labels).all() and (y_test == test_labels).all(), 'utterance order differs from the release'

w = max(F['w_grid'], key=lambda w: macro_f1(y_dev, fuse(model_dev, llm_dev, w)))   # choose w on dev only
fused_test = fuse(model_test, llm_test, w)
```

- **Temperature scaling / ECE:** fit `T` on the fused dev scores and apply it to `fused_test`. Because `w` is
  re-chosen on dev, the order of temperature scaling and fusion does not change the predictions.
- **Dev outputs are required.** `ext/train.py` currently saves only `*_test_outputs.npz`; save `va_res` the same way.

## 3. Notes

- **Task loss (MELD only).** `ext/losses.Loss` defaults to `gamma=0` (plain NLL). The released `MELD/train.py` uses
  `Loss(gamma=1)`, a focal loss (see upstream `MELD/loss.py`); IEMOCAP's release does use `gamma=0` with class weights,
  so `ext/` is correct there. Retrained with γ = 1 on seeds
  0–2, ECERC gives MELD test 66.00 wF1 / 50.40 mF1, against 65.39 / 48.96 for the `ext/` baseline. With γ = 1,
  `train.py --dataset MELD --gate baseline --seed 0` should give test 67.51 acc / 66.54 wF1 / 51.37 mF1 (epoch 35) if
  the two code paths are equivalent.
- **It carries over to retrained models.** Fusing these cached scores at w = 0.2 with ECERC retrained on seeds 0–2
  (release recipe) raised dev macro-F1 from 54.18 to 59.79 (+5.61), matching the released checkpoints (+5.63). The
  test set was not used for this check.
- **Test-set use.** The MELD test set has been evaluated three times in Limitation 2's experiments; a combined run is
  a further look. Fix every choice (variant, `w`, `T`) on dev first and score test once.
- **For the report:** the system is ECERC + a 32B LLM at inference, Joy falls about 3 F1, and it has not been tested
  on IEMOCAP. See [`README.md`](README.md).
