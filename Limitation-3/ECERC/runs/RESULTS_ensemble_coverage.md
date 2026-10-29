# Seed ensemble and selective prediction (test set, no retraining)

## IEMOCAP

### 1. Seed ensemble (3 seeds averaged) vs single model (mean of 3 seeds)

| Variant | Model | Acc | wF1 | mF1 | ECE raw | ECE + TS | NLL + TS |
|---|---|---|---|---|---|---|---|
| ECERC baseline | single | 69.95 | 70.11 | 69.27 | 13.64 | 4.21 | 0.760 |
| | **ensemble** | **70.86** | **70.99** | **70.34** | 10.39 | 1.88 | 0.734 |
| Cause softmax, fixed τ | single | 69.58 | 69.78 | 69.16 | 13.30 | 3.87 | 0.774 |
| | **ensemble** | **70.30** | **70.59** | **69.98** | 11.24 | 2.64 | 0.750 |
| CACG, dynamic τ | single | 70.47 | 70.65 | 69.84 | 13.54 | 2.95 | 0.751 |
| | **ensemble** | **71.10** | **71.30** | **70.52** | 10.86 | 2.34 | 0.728 |
| CACG + SOP entropy loss | single | 69.21 | 69.36 | 68.69 | 14.49 | 4.26 | 0.770 |
| | **ensemble** | **70.79** | **70.99** | **70.33** | 10.82 | 3.13 | 0.743 |
| CACG + utility loss | single | 69.11 | 69.44 | 68.75 | 13.48 | 4.22 | 0.773 |
| | **ensemble** | **70.43** | **70.84** | **70.11** | 10.47 | 2.26 | 0.747 |

### 2. Selective prediction: accuracy on the most confident X% of utterances

| Variant | Model | 100% | 90% | 80% | 70% | 50% | AURC ↓ |
|---|---|---|---|---|---|---|---|
| ECERC baseline | single | 70.0 | 72.7 | 75.4 | 78.1 | 83.3 | 16.58 |
| | ensemble | 71.0 | 73.6 | 77.0 | 79.1 | 84.6 | 15.45 |
| Cause softmax, fixed τ | single | 69.6 | 72.6 | 74.8 | 77.3 | 83.0 | 16.95 |
| | ensemble | 70.4 | 73.4 | 76.0 | 78.5 | 83.9 | 16.01 |
| CACG, dynamic τ | single | 70.5 | 73.3 | 76.0 | 78.6 | 83.6 | 16.18 |
| | ensemble | 71.0 | 74.1 | 77.1 | 80.0 | 84.6 | 15.31 |
| CACG + SOP entropy loss | single | 69.2 | 71.9 | 74.8 | 77.5 | 82.8 | 16.83 |
| | ensemble | 70.9 | 73.7 | 76.2 | 78.2 | 83.0 | 15.86 |
| CACG + utility loss | single | 69.1 | 72.1 | 74.7 | 77.5 | 82.8 | 17.41 |
| | ensemble | 70.4 | 73.4 | 76.1 | 78.1 | 84.1 | 16.31 |

## MELD

### 1. Seed ensemble (3 seeds averaged) vs single model (mean of 3 seeds)

| Variant | Model | Acc | wF1 | mF1 | ECE raw | ECE + TS | NLL + TS |
|---|---|---|---|---|---|---|---|
| ECERC baseline | single | 66.65 | 65.39 | 48.96 | 9.48 | 3.01 | 0.974 |
| | **ensemble** | **66.86** | **65.62** | **49.44** | 9.21 | 3.25 | 0.968 |
| Cause softmax, fixed τ | single | 66.72 | 65.32 | 48.10 | 8.97 | 2.41 | 0.984 |
| | **ensemble** | **67.20** | **65.84** | **49.48** | 8.16 | 2.39 | 0.974 |
| CACG, dynamic τ | single | 66.67 | 65.34 | 48.63 | 9.12 | 2.42 | 0.985 |
| | **ensemble** | **67.28** | **65.98** | **49.65** | 8.56 | 2.52 | 0.975 |
| CACG + SOP entropy loss | single | 66.59 | 65.20 | 47.22 | 8.19 | 2.54 | 0.991 |
| | **ensemble** | **66.78** | **65.37** | **47.51** | 7.45 | 3.00 | 0.976 |
| CACG + utility loss | single | 66.69 | 65.59 | 49.33 | 8.67 | 3.18 | 0.984 |
| | **ensemble** | **66.63** | **65.54** | **49.86** | 8.89 | 3.25 | 0.978 |

### 2. Selective prediction: accuracy on the most confident X% of utterances

| Variant | Model | 100% | 90% | 80% | 70% | 50% | AURC ↓ |
|---|---|---|---|---|---|---|---|
| ECERC baseline | single | 66.7 | 70.1 | 73.4 | 77.0 | 84.0 | 17.05 |
| | ensemble | 66.8 | 70.2 | 73.6 | 77.0 | 84.1 | 16.92 |
| Cause softmax, fixed τ | single | 66.7 | 70.1 | 73.3 | 76.6 | 84.0 | 17.44 |
| | ensemble | 67.2 | 70.4 | 73.4 | 76.6 | 84.1 | 17.17 |
| CACG, dynamic τ | single | 66.7 | 70.0 | 73.4 | 76.2 | 83.7 | 17.46 |
| | ensemble | 67.3 | 70.4 | 73.4 | 76.6 | 84.1 | 17.20 |
| CACG + SOP entropy loss | single | 66.6 | 70.2 | 73.2 | 76.6 | 83.2 | 17.56 |
| | ensemble | 66.8 | 70.3 | 73.4 | 76.7 | 83.8 | 17.07 |
| CACG + utility loss | single | 66.7 | 70.1 | 73.3 | 76.7 | 83.7 | 17.13 |
| | ensemble | 66.6 | 69.9 | 73.4 | 77.0 | 84.3 | 16.98 |
