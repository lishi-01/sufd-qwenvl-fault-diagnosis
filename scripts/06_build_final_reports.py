"""
将 class_only 模型预测结果扩展为完整故障诊断报告。

用途：
    当前最佳分类模型只输出：
        故障类型：xxx

    本脚本将其扩展为：
        故障类型：xxx
        诊断依据：xxx
        维护建议：xxx

输入：
    1. test.csv
    2. class_only 模型生成的 generated_predictions.jsonl

输出：
    outputs/final_reports/final_reports.jsonl
    outputs/final_reports/final_reports.csv

说明：
    这种方式保留 class_only 模型的分类能力，同时用确定性模板生成稳定的
    诊断依据和维护建议，避免第二阶段 full-answer 训练冲掉分类能力。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"
DEFAULT_PRED_PATH = PROJECT_ROOT / "saves" / "qwen2_5vl_3b" / "lora" / "predict_ckpt_2200" / "generated_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_reports"

LABEL_NAME_MAP = {
    "N": "正常状态",
    "BF": "滚动体故障",
    "IF": "内圈故障",
    "OF": "外圈故障",
    "CF": "混合故障",
    "INVALID": "无法识别",
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

REPORT_TEMPLATE = {
    "N": {
        "basis": "模型判断该包络时频图整体能量分布较平稳，未呈现明显的周期性冲击或异常能量集中，结合当前工况更符合正常运行状态。",
        "advice": "建议保持常规监测，持续记录振动趋势；若后续能量分布或冲击特征发生明显变化，应及时复检。",
    },
    "BF": {
        "basis": "模型判断该包络时频图中存在局部冲击和调制特征，结合行星齿轮箱三方向振动信息，更符合滚动体故障表现。",
        "advice": "建议重点检查滚动体表面是否存在磨损、点蚀或剥落，并结合包络谱或现场巡检进一步确认故障程度。",
    },
    "IF": {
        "basis": "模型判断该包络时频图中存在较明显的周期性冲击和调制成分，结合当前转速与负载工况，更符合内圈故障特征。",
        "advice": "建议重点检查轴承内圈表面损伤情况，并结合历史振动趋势评估故障发展速度。",
    },
    "OF": {
        "basis": "模型判断该包络时频图中存在相对稳定的冲击响应和局部频带能量增强，整体更符合外圈局部损伤特征。",
        "advice": "建议检查轴承外圈固定区域是否存在剥落、裂纹或异常磨损，必要时安排停机复检。",
    },
    "CF": {
        "basis": "模型判断该包络时频图中呈现多组冲击成分和较复杂的能量分布，可能对应多部位共同损伤引起的复合振动特征。",
        "advice": "建议同时检查轴承内圈和外圈，并优先安排停机检测，避免复合故障进一步扩展。",
    },
    "INVALID": {
        "basis": "模型输出中未能稳定抽取有效故障类型，因此无法给出可靠的故障依据。",
        "advice": "建议重新进行推理，或由人工结合时频图、包络谱和现场工况进行复核。",
    },
}

PRED_TEXT_KEYS = ["predict", "prediction", "response", "generated_text", "output", "text"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将 class_only 预测扩展为完整诊断报告")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="测试集 CSV 路径")
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_PRED_PATH, help="class_only 预测 JSONL 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录")
    return parser.parse_args()


def read_test_csv(path: Path) -> list[dict[str, str]]:
    """读取测试集 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"测试集 CSV 不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"测试集 CSV 为空：{path}")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    """读取预测 JSONL。"""
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
    """从预测记录中提取模型回答文本。"""
    for key in PRED_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in record.values():
        if isinstance(value, str) and value.strip():
            return value.strip()

    return json.dumps(record, ensure_ascii=False)


def extract_fault_label(text: str) -> str:
    """从模型输出中抽取故障标签。"""
    normalized = text.replace(" ", "").replace("\t", "")

    match = re.search(r"故障类型[:：]?\s*([^\n，,。；;]+)", text)
    if match:
        fault_type_text = match.group(1).strip()
        for keyword, label in FAULT_TEXT_TO_LABEL.items():
            if keyword in fault_type_text:
                return label

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


def build_report(pred_label: str, row: dict[str, str]) -> str:
    """根据预测标签和工况信息生成完整中文诊断报告。"""
    fault_type = LABEL_NAME_MAP[pred_label]
    template = REPORT_TEMPLATE[pred_label]
    speed_hz = row.get("speed_hz", "")
    load_voltage = row.get("load_voltage", "")

    condition_text = ""
    if speed_hz and load_voltage:
        condition_text = f"当前工况为 {speed_hz} Hz、{load_voltage} V。"

    basis = template["basis"]
    if condition_text:
        basis = f"{basis}{condition_text}"

    return (
        f"故障类型：{fault_type}\n"
        f"诊断依据：{basis}\n"
        f"维护建议：{template['advice']}"
    )


def write_jsonl(rows: list[dict], path: Path) -> None:
    """写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """写入 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """主函数。"""
    args = parse_args()
    test_rows = read_test_csv(args.test_csv.resolve())
    pred_records = read_jsonl(args.pred_path.resolve())

    if len(test_rows) != len(pred_records):
        raise ValueError(
            f"测试样本数和预测数不一致：test_csv={len(test_rows)}, predictions={len(pred_records)}"
        )

    reports = []
    for row, record in zip(test_rows, pred_records):
        pred_text = get_prediction_text(record)
        pred_label = extract_fault_label(pred_text)
        true_label = row.get("label", "")
        report = build_report(pred_label, row)

        reports.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "true_label": true_label,
                "true_label_name": LABEL_NAME_MAP.get(true_label, true_label),
                "pred_label": pred_label,
                "pred_label_name": LABEL_NAME_MAP[pred_label],
                "is_correct": int(true_label == pred_label),
                "raw_prediction": pred_text,
                "final_report": report,
            }
        )

    output_dir = args.output_dir.resolve()
    write_jsonl(reports, output_dir / "final_reports.jsonl")
    write_csv(
        reports,
        output_dir / "final_reports.csv",
        [
            "sample_id",
            "image_path",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "is_correct",
            "raw_prediction",
            "final_report",
        ],
    )

    correct_count = sum(item["is_correct"] for item in reports)
    print("=" * 80)
    print(f"报告数量：{len(reports)}")
    print(f"分类正确数：{correct_count}")
    print(f"分类准确率：{correct_count / len(reports):.4f}")
    print(f"JSONL 已保存：{output_dir / 'final_reports.jsonl'}")
    print(f"CSV 已保存：{output_dir / 'final_reports.csv'}")


if __name__ == "__main__":
    main()
