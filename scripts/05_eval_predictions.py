"""
评估 Qwen-VL 在 SUFD 测试集上的故障分类结果。

输入：
    1. test.csv
       包含真实标签 label、sample_id、image_path 等字段。

    2. generated_predictions.jsonl
       LLaMA-Factory 推理输出文件。常见字段包括：
           predict / prediction / response / generated_text

输出：
    outputs/metrics/eval_metrics.json
    outputs/metrics/prediction_details.csv
    outputs/metrics/confusion_matrix.csv
    outputs/metrics/confusion_matrix.png

评估逻辑：
    1. 从模型生成文本中抽取“故障类型”
    2. 将中文故障类型映射为 BF / IF / OF / CF / N
    3. 与 test.csv 中的 label 对齐比较
    4. 计算 Accuracy、Macro-F1、Precision、Recall、Confusion Matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"
DEFAULT_PRED_PATH = PROJECT_ROOT / "saves" / "qwen2_5vl_3b" / "lora" / "sufd_predict" / "generated_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"

LABEL_ORDER = ["N", "BF", "IF", "OF", "CF"]
LABEL_NAME_MAP = {
    "N": "正常状态",
    "BF": "滚动体故障",
    "IF": "内圈故障",
    "OF": "外圈故障",
    "CF": "混合故障",
}

FAULT_TEXT_TO_LABEL = {
    "正常状态": "N",
    "正常": "N",
    "健康": "N",
    "无故障": "N",
    "滚动体故障": "BF",
    "滚动体": "BF",
    "滚珠故障": "BF",
    "球故障": "BF",
    "内圈故障": "IF",
    "内圈": "IF",
    "内环故障": "IF",
    "外圈故障": "OF",
    "外圈": "OF",
    "外环故障": "OF",
    "混合故障": "CF",
    "复合故障": "CF",
    "组合故障": "CF",
}

PRED_TEXT_KEYS = ["predict", "prediction", "response", "generated_text", "output", "text"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 SUFD Qwen-VL 故障诊断预测结果")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="测试集 CSV 路径")
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_PRED_PATH, help="generated_predictions.jsonl 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="评估结果输出目录")
    return parser.parse_args()


def read_test_csv(path: Path) -> list[dict[str, str]]:
    """读取测试集 metadata。"""
    if not path.exists():
        raise FileNotFoundError(f"测试集 CSV 不存在：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"测试集 CSV 为空：{path}")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 预测文件。"""
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在：{path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{line[:120]}") from exc

    if not records:
        raise ValueError(f"预测文件为空：{path}")
    return records


def get_prediction_text(record: dict) -> str:
    """从 LLaMA-Factory 预测记录中提取模型回答文本。"""
    for key in PRED_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # 有些版本可能把生成内容放在列表或嵌套结构中，这里做一个保守兜底。
    for value in record.values():
        if isinstance(value, str) and "故障" in value:
            return value.strip()

    return json.dumps(record, ensure_ascii=False)


def extract_fault_label(text: str) -> str:
    """从模型回答中抽取故障标签。抽取失败则返回 INVALID。"""
    normalized = text.replace(" ", "").replace("\t", "")

    # 优先抽取“故障类型：xxx”这一行，减少维护建议里的词干扰判断。
    match = re.search(r"故障类型[:：]?\s*([^\n，,。；;]+)", text)
    if match:
        fault_type_text = match.group(1).strip()
        for keyword, label in FAULT_TEXT_TO_LABEL.items():
            if keyword in fault_type_text:
                return label

    # 兜底：在全文中查找关键词。
    # 先匹配更具体的故障，再匹配“正常”，避免文本里出现“异常”等词时误判。
    for keyword in ["滚动体故障", "滚动体", "滚珠故障", "球故障"]:
        if keyword in normalized:
            return "BF"
    for keyword in ["内圈故障", "内圈", "内环故障"]:
        if keyword in normalized:
            return "IF"
    for keyword in ["外圈故障", "外圈", "外环故障"]:
        if keyword in normalized:
            return "OF"
    for keyword in ["混合故障", "复合故障", "组合故障"]:
        if keyword in normalized:
            return "CF"
    for keyword in ["正常状态", "无故障", "健康", "正常"]:
        if keyword in normalized:
            return "N"

    return "INVALID"


def write_json(data: dict, path: Path) -> None:
    """写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """写入 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """绘制并保存混淆矩阵图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [f"{label}\n{LABEL_NAME_MAP[label]}" for label in LABEL_ORDER]
    ax.set_xticks(np.arange(len(LABEL_ORDER)), labels=tick_labels)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), labels=tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("SUFD Fault Diagnosis Confusion Matrix")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """主函数。"""
    args = parse_args()
    test_rows = read_test_csv(args.test_csv.resolve())
    pred_records = read_jsonl(args.pred_path.resolve())

    if len(test_rows) != len(pred_records):
        raise ValueError(
            f"测试集样本数和预测结果数不一致：test_csv={len(test_rows)}, predictions={len(pred_records)}。"
            "请确认 generated_predictions.jsonl 对应的是同一个 test.csv。"
        )

    details = []
    y_true = []
    y_pred = []

    for row, record in zip(test_rows, pred_records):
        true_label = row["label"]
        pred_text = get_prediction_text(record)
        pred_label = extract_fault_label(pred_text)

        y_true.append(true_label)
        y_pred.append(pred_label)
        details.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "true_label": true_label,
                "true_label_name": LABEL_NAME_MAP.get(true_label, true_label),
                "pred_label": pred_label,
                "pred_label_name": LABEL_NAME_MAP.get(pred_label, pred_label),
                "is_correct": int(true_label == pred_label),
                "prediction_text": pred_text,
            }
        )

    valid_true = []
    valid_pred = []
    invalid_count = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == "INVALID":
            invalid_count += 1
            continue
        valid_true.append(true_label)
        valid_pred.append(pred_label)

    if not valid_pred:
        raise ValueError("所有预测都无法抽取故障类型，无法计算分类指标。")

    total_count = len(y_true)
    valid_count = len(valid_pred)
    format_accuracy = valid_count / total_count

    metrics = {
        "total_count": total_count,
        "valid_prediction_count": valid_count,
        "invalid_prediction_count": invalid_count,
        "format_accuracy": format_accuracy,
        "accuracy": accuracy_score(valid_true, valid_pred),
        "macro_f1": f1_score(valid_true, valid_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "classification_report": classification_report(
            valid_true,
            valid_pred,
            labels=LABEL_ORDER,
            target_names=[LABEL_NAME_MAP[label] for label in LABEL_ORDER],
            output_dict=True,
            zero_division=0,
        ),
    }

    cm = confusion_matrix(valid_true, valid_pred, labels=LABEL_ORDER)
    cm_rows = []
    for true_idx, true_label in enumerate(LABEL_ORDER):
        row = {"true_label": true_label, "true_label_name": LABEL_NAME_MAP[true_label]}
        for pred_idx, pred_label in enumerate(LABEL_ORDER):
            row[f"pred_{pred_label}"] = int(cm[true_idx, pred_idx])
        cm_rows.append(row)

    output_dir = args.output_dir.resolve()
    write_json(metrics, output_dir / "eval_metrics.json")
    write_csv(
        details,
        output_dir / "prediction_details.csv",
        [
            "sample_id",
            "image_path",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "is_correct",
            "prediction_text",
        ],
    )
    write_csv(
        cm_rows,
        output_dir / "confusion_matrix.csv",
        ["true_label", "true_label_name"] + [f"pred_{label}" for label in LABEL_ORDER],
    )
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    print("=" * 80)
    print(f"总样本数：{total_count}")
    print(f"有效预测数：{valid_count}")
    print(f"无效预测数：{invalid_count}")
    print(f"格式正确率：{format_accuracy:.4f}")
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"评估结果目录：{output_dir}")
    print(f"详细预测：{output_dir / 'prediction_details.csv'}")
    print(f"混淆矩阵图片：{output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
