# DPO 偏好优化实验结果

## 实验目的

本实验在 SFT 最优模型 checkpoint-2200 的基础上继续进行 DPO 偏好优化。

DPO 数据采用完整诊断报告偏好对：

- chosen：正确故障类型 + 诊断依据 + 维护建议
- rejected：错误故障类型 + 诊断依据 + 维护建议

目标是让模型学习工业故障诊断报告中“正确诊断回答”相对于“错误诊断回答”的偏好关系。

## 实验结果

| 模型 | 训练方式 | Checkpoint | Accuracy | Macro-F1 | 说明 |
|---|---|---:|---:|---:|---|
| Qwen2.5-VL-3B-Instruct + LoRA | SFT class_only | checkpoint-2200 | 0.4854 | 0.4805 | 当前最佳分类模型 |
| Qwen2.5-VL-3B-Instruct + LoRA | DPO full report | checkpoint-400 | 0.4524 | 0.4383 | DPO 最优 checkpoint，但分类指标低于 SFT best |

## DPO checkpoint-400 混淆矩阵

| True Label | 中文类别 | pred_N | pred_BF | pred_IF | pred_OF | pred_CF |
|---|---|---:|---:|---:|---:|---:|
| N | 正常状态 | 141 | 16 | 7 | 14 | 28 |
| BF | 滚动体故障 | 74 | 46 | 20 | 19 | 47 |
| IF | 内圈故障 | 20 | 11 | 67 | 43 | 65 |
| OF | 外圈故障 | 26 | 4 | 13 | 94 | 69 |
| CF | 混合故障 | 33 | 8 | 24 | 23 | 118 |

## 结果分析

DPO 训练过程中，训练 loss 持续下降，rewards/accuracies 持续上升，说明模型能够学习 chosen/rejected 之间的偏好关系。

但是，验证 loss 随训练逐步上升，且 DPO 最优 checkpoint-400 的 Accuracy 和 Macro-F1 均低于 SFT checkpoint-2200。

这说明当前 DPO 偏好优化虽然增强了模型对完整诊断报告偏好关系的学习，但没有进一步提升故障分类性能。主要原因可能包括：

1. 当前 DPO 数据规模较小，偏好对主要由模板构造，回答多样性不足。
2. DPO 优化目标关注完整回答偏好，而最终分类指标只评价故障类型是否正确。
3. rejected 样本虽然故障类型错误，但诊断依据和维护建议也由模板生成，难以覆盖真实模型幻觉回答。
4. DPO 继续训练可能导致模型偏向特定回答模式，从而影响原有分类能力。

## 最终采用方案

最终系统仍采用 SFT class_only checkpoint-2200 作为故障分类模型。

DPO 实验作为强化学习偏好优化阶段的探索，用于说明：

- 项目已经完成 chosen/rejected 偏好数据构造；
- 已经跑通 DPO 训练流程；
- 已经完成 DPO checkpoint 评估；
- 当前 DPO 在分类指标上未超过 SFT best，因此未作为最终部署模型。
