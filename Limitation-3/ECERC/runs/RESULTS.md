### IEMOCAP (test, mean ± std over seeds)

| Model | n | valid wF1 | Acc | wF1 | mF1 | ECE↓ | ECE+TS↓ | NLL↓ | ConfErr↓ | ECE+CA-TS↓ | H(g) | corr(τ, ctx) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECERC (reproduced) | 3 | 74.67 ± 0.23 | 69.95 ± 0.58 | 70.11 ± 0.46 | 69.27 ± 0.58 | 13.64 ± 1.17 | 4.21 ± 0.87 | 0.864 ± 0.035 | 7.85 ± 1.18 | 3.91 ± 1.19 | – | – |
| ECERC + empty-history attention fix | 3 | 75.16 ± 0.55 | 68.97 ± 1.28 | 68.99 ± 1.36 | 68.63 ± 1.29 | 14.82 ± 1.62 | 5.66 ± 2.16 | 0.922 ± 0.043 | 8.67 ± 0.89 | 5.63 ± 1.87 | – | – |
| + cause softmax (masked), fixed τ | 3 | 74.84 ± 0.28 | 69.58 ± 1.07 | 69.78 ± 1.11 | 69.16 ± 0.92 | 13.30 ± 0.60 | 3.87 ± 0.98 | 0.906 ± 0.010 | 8.05 ± 1.02 | 3.96 ± 1.26 | 1.025 ± 0.059 | – |
| + dynamic τ (no L_gate) | 3 | 74.84 ± 0.30 | 70.47 ± 0.87 | 70.65 ± 0.86 | 69.84 ± 0.63 | 13.54 ± 0.18 | 2.95 ± 0.77 | 0.885 ± 0.019 | 8.09 ± 0.15 | 4.11 ± 0.75 | 0.989 ± 0.038 | -0.64 ± 0.13 |
| + dynamic τ + L_gate (SOP Eq.3 entropy) | 3 | 75.13 ± 0.40 | 69.21 ± 1.31 | 69.36 ± 1.29 | 68.69 ± 1.00 | 14.49 ± 1.10 | 4.26 ± 0.27 | 0.918 ± 0.032 | 8.48 ± 0.62 | 4.32 ± 0.85 | 0.607 ± 0.031 | -0.63 ± 0.07 |
| + dynamic τ + L_gate (utility, T=1) | 3 | 74.74 ± 0.11 | 69.11 ± 0.99 | 69.44 ± 1.03 | 68.75 ± 0.93 | 13.48 ± 1.75 | 4.22 ± 0.65 | 0.888 ± 0.041 | 7.46 ± 1.85 | 4.27 ± 0.14 | 1.226 ± 0.016 | -0.48 ± 0.23 |
| + dynamic τ + L_gate (utility, T=0.25) | 3 | 75.08 ± 0.24 | 69.36 ± 0.29 | 69.52 ± 0.32 | 68.79 ± 0.11 | 15.90 ± 0.57 | 3.55 ± 0.50 | 0.951 ± 0.008 | 9.32 ± 0.41 | 4.13 ± 0.48 | 0.905 ± 0.021 | -0.62 ± 0.15 |

Per-class F1 (IEMOCAP):

| Model | hap | sad | neu | ang | exc | fru |
|---|---|---|---|---|---|---|
| ECERC (reproduced) | 60.1 ± 3.8 | 78.1 ± 0.7 | 71.1 ± 0.5 | 65.1 ± 1.9 | 75.3 ± 1.4 | 66.0 ± 0.6 |
| ECERC + empty-history attention fix | 63.4 ± 1.1 | 77.6 ± 1.1 | 70.1 ± 0.5 | 65.8 ± 1.4 | 69.2 ± 4.6 | 65.7 ± 1.8 |
| + cause softmax (masked), fixed τ | 61.4 ± 0.8 | 78.1 ± 0.6 | 69.8 ± 1.4 | 66.1 ± 1.7 | 72.3 ± 2.9 | 67.2 ± 1.5 |
| + dynamic τ (no L_gate) | 61.7 ± 3.1 | 76.7 ± 1.2 | 70.8 ± 0.9 | 65.6 ± 0.5 | 76.7 ± 2.4 | 67.4 ± 1.1 |
| + dynamic τ + L_gate (SOP Eq.3 entropy) | 61.6 ± 1.5 | 76.2 ± 0.3 | 70.6 ± 1.0 | 65.4 ± 0.8 | 72.1 ± 4.2 | 66.2 ± 1.3 |
| + dynamic τ + L_gate (utility, T=1) | 59.5 ± 1.6 | 77.5 ± 0.9 | 68.8 ± 2.3 | 66.0 ± 1.2 | 74.2 ± 2.2 | 66.4 ± 1.0 |
| + dynamic τ + L_gate (utility, T=0.25) | 61.0 ± 2.0 | 76.3 ± 0.7 | 70.5 ± 0.6 | 65.3 ± 0.3 | 73.5 ± 0.7 | 66.2 ± 1.2 |

IEMOCAP cacg_entropy_sop (seed 1st): entropy_correct=0.652, entropy_wrong=0.645, tau_correct=0.892, tau_wrong=0.894, corr_tau_conf=-0.213

IEMOCAP cacg_noloss (seed 1st): entropy_correct=0.953, entropy_wrong=0.937, tau_correct=0.918, tau_wrong=0.919, corr_tau_conf=-0.253

IEMOCAP cacg_utility_T025 (seed 1st): entropy_correct=0.892, entropy_wrong=0.883, tau_correct=0.890, tau_wrong=0.892, corr_tau_conf=-0.208

IEMOCAP cacg_utility (seed 1st): entropy_correct=1.222, entropy_wrong=1.200, tau_correct=0.955, tau_wrong=0.955, corr_tau_conf=-0.221

### MELD (test, mean ± std over seeds)

| Model | n | valid wF1 | Acc | wF1 | mF1 | ECE↓ | ECE+TS↓ | NLL↓ | ConfErr↓ | ECE+CA-TS↓ | H(g) | corr(τ, ctx) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECERC (reproduced) | 3 | 65.73 ± 0.10 | 66.65 ± 0.34 | 65.39 ± 0.34 | 48.96 ± 0.29 | 9.48 ± 0.24 | 3.01 ± 0.18 | 1.002 ± 0.002 | 4.07 ± 0.24 | 3.65 ± 0.24 | – | – |
| ECERC + empty-history attention fix | 3 | 65.31 ± 0.16 | 66.58 ± 0.14 | 65.36 ± 0.22 | 47.73 ± 1.16 | 9.14 ± 0.16 | 3.18 ± 0.19 | 1.004 ± 0.002 | 3.91 ± 0.39 | 3.50 ± 0.26 | – | – |
| + cause softmax (masked), fixed τ | 3 | 65.57 ± 0.22 | 66.72 ± 0.09 | 65.32 ± 0.15 | 48.10 ± 0.86 | 8.97 ± 0.74 | 2.41 ± 0.26 | 1.011 ± 0.008 | 3.86 ± 0.37 | 2.97 ± 0.16 | 0.913 ± 0.039 | – |
| + dynamic τ (no L_gate) | 3 | 65.52 ± 0.13 | 66.67 ± 0.05 | 65.34 ± 0.16 | 48.63 ± 0.52 | 9.12 ± 0.71 | 2.42 ± 0.22 | 1.013 ± 0.003 | 4.01 ± 0.46 | 2.85 ± 0.21 | 0.895 ± 0.018 | 0.23 ± 0.12 |
| + dynamic τ + L_gate (SOP Eq.3 entropy) | 3 | 65.40 ± 0.31 | 66.59 ± 0.25 | 65.20 ± 0.18 | 47.22 ± 0.96 | 8.19 ± 0.35 | 2.54 ± 0.41 | 1.013 ± 0.011 | 3.47 ± 0.22 | 3.16 ± 0.41 | 0.612 ± 0.009 | -0.85 ± 0.02 |
| + dynamic τ + L_gate (utility, T=1) | 3 | 65.36 ± 0.33 | 66.69 ± 0.11 | 65.59 ± 0.04 | 49.33 ± 0.49 | 8.67 ± 0.57 | 3.18 ± 0.33 | 1.003 ± 0.004 | 3.46 ± 0.02 | 3.15 ± 0.39 | 1.142 ± 0.001 | 0.78 ± 0.08 |
| + dynamic τ + L_gate (utility, T=0.25) | 3 | 65.21 ± 0.26 | 66.67 ± 0.25 | 65.50 ± 0.13 | 49.08 ± 0.30 | 8.92 ± 0.38 | 2.91 ± 0.37 | 1.007 ± 0.006 | 3.63 ± 0.11 | 2.98 ± 0.07 | 1.086 ± 0.006 | 0.70 ± 0.11 |

Per-class F1 (MELD):

| Model | neu | sur | fea | sad | joy | dis | ang |
|---|---|---|---|---|---|---|---|
| ECERC (reproduced) | 79.4 ± 0.1 | 57.6 ± 0.2 | 21.1 ± 1.8 | 38.5 ± 1.7 | 63.9 ± 0.4 | 30.2 ± 0.6 | 52.0 ± 1.2 |
| ECERC + empty-history attention fix | 79.8 ± 0.1 | 57.8 ± 0.6 | 15.9 ± 5.4 | 38.3 ± 1.4 | 64.0 ± 0.4 | 26.7 ± 1.5 | 51.6 ± 0.4 |
| + cause softmax (masked), fixed τ | 79.6 ± 0.1 | 57.0 ± 0.5 | 19.9 ± 3.0 | 37.2 ± 1.3 | 64.2 ± 0.0 | 26.0 ± 2.9 | 52.9 ± 0.5 |
| + dynamic τ (no L_gate) | 79.4 ± 0.2 | 57.7 ± 0.5 | 21.2 ± 1.7 | 37.6 ± 1.4 | 64.2 ± 0.3 | 28.2 ± 3.2 | 52.0 ± 0.9 |
| + dynamic τ + L_gate (SOP Eq.3 entropy) | 79.4 ± 0.3 | 58.0 ± 1.0 | 17.2 ± 5.8 | 38.4 ± 1.8 | 64.3 ± 0.8 | 21.0 ± 1.8 | 52.2 ± 1.3 |
| + dynamic τ + L_gate (utility, T=1) | 79.5 ± 0.0 | 57.4 ± 0.4 | 23.2 ± 4.2 | 39.5 ± 0.3 | 64.5 ± 0.3 | 29.1 ± 1.2 | 52.1 ± 0.3 |
| + dynamic τ + L_gate (utility, T=0.25) | 79.6 ± 0.1 | 56.9 ± 0.6 | 23.0 ± 4.3 | 39.4 ± 0.7 | 64.4 ± 0.4 | 28.5 ± 0.9 | 51.8 ± 0.9 |

MELD cacg_entropy_sop (seed 1st): entropy_correct=0.601, entropy_wrong=0.598, tau_correct=0.976, tau_wrong=0.975, corr_tau_conf=-0.059

MELD cacg_noloss (seed 1st): entropy_correct=0.906, entropy_wrong=0.915, tau_correct=0.969, tau_wrong=0.969, corr_tau_conf=-0.049

MELD cacg_utility_T025 (seed 1st): entropy_correct=1.082, entropy_wrong=1.093, tau_correct=0.973, tau_wrong=0.972, corr_tau_conf=0.198

MELD cacg_utility (seed 1st): entropy_correct=1.134, entropy_wrong=1.156, tau_correct=0.983, tau_wrong=0.982, corr_tau_conf=0.157
