# ECERC Revisited

Course project extending **ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in
Conversation** (Zhang & Tan, ACL 2025; official code: https://github.com/TAN-OpenLab/ECERC).
The project plan is in [ML_SOP.pdf](ML_SOP.pdf).

Each team member works on one limitation in their own folder:

| Folder | Limitation | Owner |
|---|---|---|
| [Limitation-1](Limitation-1/) | Category confusion: confusion-aware contrastive regularizer | Divyansh Dubey |
| [Limitation-2](Limitation-2/) | Class imbalance: class-balanced / focal loss + oversampling | Keshav Mishra |
| [Limitation-3](Limitation-3/) | Cause-gating calibration: Confidence-Aware Cause Gating (CACG) | Lakshay Gupta |

## How to add your code

```bash
git clone https://github.com/lakshayg2005/ML_Project.git
cd ML_Project
# put your code inside your own folder only (Limitation-1 or Limitation-2)
git add Limitation-1
git commit -m "Limitation 1: <what you added>"
git pull --rebase
git push
```

Guidelines:
- Commit only inside your own folder, so the three parts never conflict.
- Do not commit datasets, model weights or logs. `.gitignore` already excludes `data/`, `*.pt`, `*.npz`
  and `runs/logs/`. GitHub rejects files over 100 MB, and the features are copyrighted.
- If you copy the ECERC code into your folder, delete its `.git` folder first, otherwise git treats it
  as a separate repository and won't upload its files.
- Report results in the same format (test metrics at the best-validation epoch, mean ± std over seeds) so
  the combined ablation can be put together. `Limitation-3/ECERC/ext/` already has a unified training
  script with merge points for Limitation 1 and 2; see its README, section 6.
