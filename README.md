# 基于 Qwen2.5-VL 的 SUFD 工业设备故障诊断多模态微调系统

本项目将 SUFD 齿轮箱/轴承振动信号转换为多模态指令微调数据，使用 Qwen2.5-VL + LoRA 完成故障类型识别，并通过模板扩展生成中文诊断报告。项目重点不是传统 CNN 分类，而是构建一条从工业原始信号到多模态大模型诊断输出的完整工程流程。

当前最佳结果：

| 模型 | 数据格式 | Best Checkpoint | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct + LoRA | 仅故障类型 class_only | checkpoint-2200 | **0.4854** | **0.4805** |

最终系统采用两步输出策略：

1. 使用最佳 class_only 模型预测故障类型。
2. 根据预测标签和工况参数生成完整诊断报告，包括故障类型、诊断依据和维护建议。

## 项目目标

输入：

- 由振动信号生成的三轴 RGB 包络时频图
- 工况参数：转速、负载/电压、采样率
- 中文故障诊断指令

输出：

- 故障类型：正常状态、滚动体故障、内圈故障、外圈故障、混合故障
- 诊断依据：结合时频图和工况给出简短说明
- 维护建议：给出检查或维护建议

## 数据集与标签

原始数据为 `.mat` 文件，每个变量 shape 为：

```text
1048560 × 8
```

8 个通道含义：

| 通道 | 含义 | 本项目是否使用 |
|---:|---|---|
| 1 | 电机振动 | 否 |
| 2 | 行星齿轮箱 X 方向振动 | 是 |
| 3 | 行星齿轮箱 Y 方向振动 | 是 |
| 4 | 行星齿轮箱 Z 方向振动 | 是 |
| 5 | 电机转矩 | 否 |
| 6 | 并联齿轮箱 X 方向振动 | 否 |
| 7 | 并联齿轮箱 Y 方向振动 | 否 |
| 8 | 并联齿轮箱 Z 方向振动 | 否 |

标签映射：

| 文件前缀 | 标签 | 中文类别 |
|---|---|---|
| `health` | `N` | 正常状态 |
| `ball` | `BF` | 滚动体故障 |
| `inner` | `IF` | 内圈故障 |
| `outer` | `OF` | 外圈故障 |
| `comb` | `CF` | 混合故障 |

工况：

| 后缀 | 转速 | 负载/电压 |
|---|---:|---:|
| `20_0` | 20 Hz | 0 V |
| `30_2` | 30 Hz | 2 V |

## 方法流程

```text
SUFD .mat 原始信号
        ↓
提取第 2/3/4 通道：行星齿轮箱 X/Y/Z 振动
        ↓
按 4096 点切片，hop=2048
        ↓
简化谱峭度频带搜索
        ↓
带通滤波
        ↓
Hilbert 包络分析
        ↓
Envelope-STFT 时频图
        ↓
X/Y/Z 三方向融合为 RGB 图像
        ↓
构造 ShareGPT 多模态 SFT 数据
        ↓
Qwen2.5-VL-3B-Instruct + LoRA 微调
        ↓
测试集推理与评估
        ↓
class_only 预测 + 模板扩展生成最终诊断报告
```

## 目录结构

```text
.
├── data/
│   └── processed/
│       ├── images/all/                  # 生成的 RGB 包络 STFT 图像
│       ├── metadata/                    # all/train/gap/test metadata
│       └── json/                        # ShareGPT SFT JSON
├── outputs/
│   └── server_results/
│       ├── experiment_summary/          # 实验结果表
│       ├── metrics_ckpt_2200/           # 最佳 checkpoint 评估结果
│       └── final_reports_ckpt_2200/     # 最终诊断报告
├── scripts/
│   ├── 01_inspect_mat.py
│   ├── 02_generate_envelope_stft_images.py
│   ├── 03_split_dataset.py
│   ├── 04_build_sharegpt_json.py
│   ├── 05_eval_predictions.py
│   ├── 06_build_final_reports.py
│   └── 07_improve_confusion_matrix.py
└── README.md
```

## 脚本说明

| 脚本 | 功能 |
|---|---|
| `01_inspect_mat.py` | 检查 `.mat` 字段、shape、dtype、有效通道统计 |
| `02_generate_envelope_stft_images.py` | 生成三轴 RGB 包络 STFT 图像和 `all_metadata.csv` |
| `03_split_dataset.py` | 按时间顺序划分 train/gap/test，默认 75%/5%/20% |
| `04_build_sharegpt_json.py` | 生成 Qwen-VL ShareGPT 多模态 SFT JSON，支持 `full` 和 `class_only` |
| `05_eval_predictions.py` | 从模型输出抽取标签，计算 Accuracy、Macro-F1 和混淆矩阵 |
| `06_build_final_reports.py` | 将 class_only 预测扩展为完整中文诊断报告 |
| `07_improve_confusion_matrix.py` | 生成增强版混淆矩阵 CSV、Markdown 和 SVG |

## 数据预处理

检查原始 `.mat`：

```bash
python scripts/01_inspect_mat.py
```

生成包络 STFT 图像：

```bash
python scripts/02_generate_envelope_stft_images.py --overwrite
```

输出：

```text
data/processed/images/all/
data/processed/metadata/all_metadata.csv
```

划分数据集：

```bash
python scripts/03_split_dataset.py
```

默认划分策略：

```text
前 75%：train.csv
中间 5%：gap.csv，不参与训练和测试
后 20%：test.csv
```

这样做是为了减少滑窗切片带来的相邻样本泄漏。

## 构造 SFT 数据

生成完整回答版本：

```bash
python scripts/04_build_sharegpt_json.py
```

生成分类优先版本：

```bash
python scripts/04_build_sharegpt_json.py --answer-mode class_only
```

输出：

```text
data/processed/json/train_sft.json
data/processed/json/test_sft.json
data/processed/json/train_sft_class_only.json
data/processed/json/test_sft_class_only.json
```

class_only 样本只要求模型输出：

```text
故障类型：<滚动体故障/内圈故障/外圈故障/混合故障/正常状态>
```

## LLaMA-Factory 训练

训练环境：

```text
AutoDL / Linux
Python 3.11
LLaMA-Factory
Qwen2.5-VL-3B-Instruct
LoRA
bf16
```

核心模型：

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

LLaMA-Factory 数据集注册示例：

```json
{
  "sufd_fault_train_class_only": {
    "file_name": "sufd/train_sft_class_only.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  },
  "sufd_fault_test_class_only": {
    "file_name": "sufd/test_sft_class_only.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

最佳训练配置：

```yaml
model_name_or_path: /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
dataset: sufd_fault_train_class_only
template: qwen2_vl
cutoff_len: 1024
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 8.0e-5
num_train_epochs: 5.0
bf16: true
output_dir: saves/qwen2_5vl_3b/lora/sufd_class_only_r16_ep5
```

训练命令：

```bash
llamafactory-cli train configs/sufd/qwen2_5vl_3b_lora_sufd_class_only_r16_ep5.yaml
```

最佳 checkpoint：

```text
saves/qwen2_5vl_3b/lora/sufd_class_only_r16_ep5/checkpoint-2200
```

## 实验结果

| 实验 | 数据格式 | 训练设置 | Accuracy | Macro-F1 | 结论 |
|---|---|---|---:|---:|---|
| E1 完整答案版 | 故障类型+诊断依据+维护建议 | r8, lr=1e-4, epoch=3 | 0.2660 | 0.2621 | 长答案生成干扰分类学习 |
| E2 分类版 baseline | 仅故障类型 | r8, lr=1e-4, epoch=3 | 0.3379 | 0.3184 | 分类任务收敛更好 |
| E3 分类版 r16 ep5 | 仅故障类型 | r16, lr=8e-5, epoch=5 | 0.4738 | 0.4683 | LoRA 容量和训练轮数提升有效 |
| E4 最佳 checkpoint | 仅故障类型 | r16, lr=8e-5, epoch=5, ckpt-2200 | **0.4854** | **0.4805** | 当前最佳分类模型 |
| E5 BF 过采样 2x | 仅故障类型 | r16, lr=6e-5, epoch=5 | 0.3767 | 0.3713 | 简单过采样破坏类别边界 |
| E6 二阶段完整回答 | 完整诊断报告 | 从 E4 继续 SFT, lr=3e-5, epoch=2 | 0.4233 | 0.4096 | 生成能力增强但分类下降 |

实验结论：

- 将输出约束为 `class_only` 后，模型更专注于五分类任务。
- 提高 LoRA rank 到 16 并训练 5 epoch 后，性能显著提升。
- 简单复制 BF 样本会破坏整体类别边界，不适合作为当前优化策略。
- 从最佳分类模型继续训练完整回答会冲掉一部分分类能力，因此最终采用“分类模型 + 模板扩展”方案。

## 最佳模型类别级指标

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 正常状态 | 0.4793 | 0.6748 | 0.5605 | 206 |
| 滚动体故障 | 0.4615 | 0.3495 | 0.3978 | 206 |
| 内圈故障 | 0.5301 | 0.4272 | 0.4731 | 206 |
| 外圈故障 | 0.5497 | 0.4563 | 0.4987 | 206 |
| 混合故障 | 0.4332 | 0.5194 | 0.4724 | 206 |

## 混淆矩阵分析

![增强版混淆矩阵](outputs/server_results/metrics_ckpt_2200/confusion_matrix_enhanced.svg)

关键观察：

- 正常状态召回最高：67.48%。
- 混合故障召回较好：51.94%。
- 滚动体故障最弱：34.95%，主要被误判为正常状态和混合故障。
- IF、OF、CF 之间仍存在明显混淆，说明不同故障类型的包络时频图特征边界仍不够清晰。

增强版混淆矩阵相关文件：

```text
outputs/server_results/metrics_ckpt_2200/confusion_matrix_enhanced.svg
outputs/server_results/metrics_ckpt_2200/confusion_matrix_enhanced.csv
outputs/server_results/metrics_ckpt_2200/confusion_matrix_percent.csv
outputs/server_results/metrics_ckpt_2200/confusion_matrix_report.md
```

## 最终诊断报告

由于完整回答训练会降低分类准确率，本项目最终采用：

```text
最佳 class_only 分类模型
        ↓
抽取预测故障类型
        ↓
根据类别和工况参数生成完整诊断报告
```

示例输出：

```text
故障类型：滚动体故障
诊断依据：模型判断该包络时频图中存在局部冲击和调制特征，结合行星齿轮箱三方向振动信息，更符合滚动体故障表现。当前工况为 20 Hz、0 V。
维护建议：建议重点检查滚动体表面是否存在磨损、点蚀或剥落，并结合包络谱或现场巡检进一步确认故障程度。
```

生成最终报告：

```bash
python scripts/06_build_final_reports.py \
  --test-csv data/processed/metadata/test.csv \
  --pred-path saves/qwen2_5vl_3b/lora/predict_ckpt_2200/generated_predictions.jsonl \
  --output-dir outputs/final_reports_ckpt_2200
```

服务器结果已保存到：

```text
outputs/server_results/final_reports_ckpt_2200/final_reports.jsonl
outputs/server_results/final_reports_ckpt_2200/final_reports.csv
```

## Gradio Demo

项目提供了一个 Gradio Demo，用于演示端到端故障诊断流程：

```text
上传包络 STFT 图像
        ↓
输入转速、负载/电压、采样率
        ↓
Qwen2.5-VL + LoRA best checkpoint 输出故障类型
        ↓
模板扩展生成完整中文诊断报告
```

Demo 脚本：

```text
demo/app_gradio.py
```

在 AutoDL / Linux 训练环境中运行：

```bash
conda activate llamafactory311
cd /root/autodl-tmp/sufd-qwenvl-fault-diagnosis

python demo/app_gradio.py --server-name 0.0.0.0 --server-port 6008
```

如果 AutoDL 只映射固定端口，需要将 Gradio 端口设置为平台映射的端口。例如本次实验中使用：

```text
容器端口：http://127.0.0.1:6008
公网地址：https://uu775756-n0jj-7542c192.westd.seetacloud.com:8443
```

Demo 默认模型路径：

```text
Base model:
/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct

LoRA adapter:
/root/autodl-tmp/LLaMA-Factory/saves/qwen2_5vl_3b/lora/sufd_class_only_r16_ep5/checkpoint-2200
```

也可以通过启动参数或环境变量指定模型路径：

```bash
python demo/app_gradio.py \
  --base-model /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path /path/to/checkpoint-2200 \
  --server-name 0.0.0.0 \
  --server-port 6008
```

Demo 输入：

```text
包络 STFT 图像
转速 Hz
负载/电压 V
采样率 Hz
```

Demo 输出：

```text
故障类型
模型原始输出
最终诊断报告
```

## 评估方式

模型输出中优先匹配：

```text
故障类型：滚动体故障 -> BF
故障类型：内圈故障 -> IF
故障类型：外圈故障 -> OF
故障类型：混合故障 -> CF
故障类型：正常状态 -> N
```

若无法抽取故障类型，则记为 `INVALID`，并计入格式错误。

当前最佳模型：

```text
有效预测数：1030
无效预测数：0
格式正确率：1.0000
Accuracy：0.4854
Macro-F1：0.4805
```

## 复现实验流程

本地数据处理：

```bash
python scripts/01_inspect_mat.py
python scripts/02_generate_envelope_stft_images.py --overwrite
python scripts/03_split_dataset.py
python scripts/04_build_sharegpt_json.py --answer-mode class_only
```

LLaMA-Factory 训练：

```bash
llamafactory-cli train configs/sufd/qwen2_5vl_3b_lora_sufd_class_only_r16_ep5.yaml
```

测试集推理：

```bash
llamafactory-cli train configs/sufd/predict_ckpt_2200.yaml
```

评估：

```bash
python scripts/05_eval_predictions.py \
  --test-csv data/processed/metadata/test.csv \
  --pred-path saves/qwen2_5vl_3b/lora/predict_ckpt_2200/generated_predictions.jsonl \
  --output-dir outputs/metrics_ckpt_2200
```

生成增强版混淆矩阵：

```bash
python scripts/07_improve_confusion_matrix.py \
  --input outputs/metrics_ckpt_2200/confusion_matrix.csv \
  --output-dir outputs/metrics_ckpt_2200
```

## 当前不足

- 最佳 Accuracy 仍为 0.4854，说明 Qwen2.5-VL-3B 对该类包络时频图的故障边界学习仍有限。
- BF 滚动体故障召回较低，是当前主要瓶颈。
- 当前维护建议为类别模板扩展，不是基于每张图像的细粒度严重程度估计。
- 当前仅使用 Qwen2.5-VL-3B，尚未进行 7B 或其他视觉模型对比。

## 后续优化方向

- 尝试 Qwen2.5-VL-7B + QLoRA，验证更大模型是否提升故障区分能力。
- 对比原始 STFT、包络谱图、CWT、小波包等图像表示方式。
- 引入训练集准确率评估，区分欠拟合和泛化不足。
- 增加验证集并自动选择最佳 checkpoint。
- 结合传统故障特征，如包络谱峰值、峭度、RMS、频带能量，构造图文联合输入。
- 在现有 Gradio Demo 基础上增加样例库、批量诊断和历史记录导出功能。

## 说明

本项目中的大模型权重、LoRA checkpoint、原始 `.mat` 数据和大规模图像文件体积较大，建议在 GitHub 中只保留脚本、配置、实验结果摘要和少量示例图像，大文件通过网盘、对象存储或实验服务器保存。
