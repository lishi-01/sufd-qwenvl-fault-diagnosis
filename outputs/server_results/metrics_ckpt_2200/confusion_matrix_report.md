# Confusion Matrix Analysis

- Total samples: 1030
- Correct predictions: 500
- Accuracy: 0.4854

## Count Matrix

| True \ Pred | N 正常 | BF 滚动体 | IF 内圈 | OF 外圈 | CF 混合 | Recall |
|---|---:|---:|---:|---:|---:|---:|
| N 正常状态 | 139 | 32 | 9 | 10 | 16 | 0.6748 |
| BF 滚动体故障 | 64 | 72 | 18 | 20 | 32 | 0.3495 |
| IF 内圈故障 | 25 | 13 | 88 | 29 | 51 | 0.4272 |
| OF 外圈故障 | 35 | 17 | 19 | 94 | 41 | 0.4563 |
| CF 混合故障 | 27 | 22 | 32 | 18 | 107 | 0.5194 |

## Row-Normalized Matrix

| True \ Pred | N 正常 | BF 滚动体 | IF 内圈 | OF 外圈 | CF 混合 |
|---|---:|---:|---:|---:|---:|
| N 正常状态 | 67.5% | 15.5% | 4.4% | 4.9% | 7.8% |
| BF 滚动体故障 | 31.1% | 35.0% | 8.7% | 9.7% | 15.5% |
| IF 内圈故障 | 12.1% | 6.3% | 42.7% | 14.1% | 24.8% |
| OF 外圈故障 | 17.0% | 8.3% | 9.2% | 45.6% | 19.9% |
| CF 混合故障 | 13.1% | 10.7% | 15.5% | 8.7% | 51.9% |

## Class Metrics From Matrix

| Class | Precision | Recall |
|---|---:|---:|
| N 正常状态 | 0.4793 | 0.6748 |
| BF 滚动体故障 | 0.4615 | 0.3495 |
| IF 内圈故障 | 0.5301 | 0.4272 |
| OF 外圈故障 | 0.5497 | 0.4563 |
| CF 混合故障 | 0.4332 | 0.5194 |

## Key Observations

- The best recognized class is N/正常状态 by recall, while BF/滚动体故障 remains the weakest class.
- CF/混合故障 has relatively high recall, but its precision is limited because IF/OF/BF samples are sometimes predicted as CF.
- BF is often confused with N and CF, which is the main bottleneck for further improvement.
