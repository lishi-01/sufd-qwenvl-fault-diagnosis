"""
检查并美化 SUFD 故障诊断混淆矩阵。

输入：
    outputs/server_results/metrics_ckpt_2200/confusion_matrix.csv

输出：
    confusion_matrix_enhanced.csv      数量 + 行归一化百分比
    confusion_matrix_percent.csv       仅行归一化百分比
    confusion_matrix_report.md         可直接放入 README/实验报告的 Markdown 表
    confusion_matrix_enhanced.svg      高清矢量图，适合报告展示

说明：
    原始混淆矩阵图片中中文类别名可能因为字体缺失显示为方块。
    本脚本生成 SVG，并设置中文字体候选，通常在 Windows 浏览器或文档中可以正常显示。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "server_results" / "metrics_ckpt_2200" / "confusion_matrix.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "server_results" / "metrics_ckpt_2200"

LABEL_ORDER = ["N", "BF", "IF", "OF", "CF"]
LABEL_NAME = {
    "N": "正常状态",
    "BF": "滚动体故障",
    "IF": "内圈故障",
    "OF": "外圈故障",
    "CF": "混合故障",
}
LABEL_EN = {
    "N": "Normal",
    "BF": "Ball fault",
    "IF": "Inner race",
    "OF": "Outer race",
    "CF": "Compound",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="检查并美化混淆矩阵")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="原始 confusion_matrix.csv 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    return parser.parse_args()


def read_confusion_matrix(path: Path) -> list[list[int]]:
    """读取原始混淆矩阵 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"混淆矩阵文件不存在：{path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([int(row[f"pred_{label}"]) for label in LABEL_ORDER])

    if len(rows) != len(LABEL_ORDER):
        raise ValueError(f"混淆矩阵行数应为 {len(LABEL_ORDER)}，实际为 {len(rows)}")
    return rows


def row_percentages(matrix: list[list[int]]) -> list[list[float]]:
    """计算行归一化百分比，即每个真实类别内部的预测占比。"""
    percentages = []
    for row in matrix:
        total = sum(row)
        if total == 0:
            percentages.append([0.0 for _ in row])
        else:
            percentages.append([value / total * 100 for value in row])
    return percentages


def column_totals(matrix: list[list[int]]) -> list[int]:
    """计算每个预测类别的总数。"""
    return [sum(row[col] for row in matrix) for col in range(len(LABEL_ORDER))]


def write_enhanced_csv(matrix: list[list[int]], percentages: list[list[float]], output_dir: Path) -> None:
    """输出数量 + 百分比的增强版 CSV。"""
    path = output_dir / "confusion_matrix_enhanced.csv"
    fieldnames = ["true_label", "true_label_name", "support"]
    for label in LABEL_ORDER:
        fieldnames.append(f"pred_{label}_count")
        fieldnames.append(f"pred_{label}_percent")

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row, percent_row in zip(LABEL_ORDER, matrix, percentages):
            item = {
                "true_label": label,
                "true_label_name": LABEL_NAME[label],
                "support": sum(row),
            }
            for pred_label, count, percent in zip(LABEL_ORDER, row, percent_row):
                item[f"pred_{pred_label}_count"] = count
                item[f"pred_{pred_label}_percent"] = f"{percent:.2f}%"
            writer.writerow(item)


def write_percent_csv(percentages: list[list[float]], output_dir: Path) -> None:
    """输出仅包含百分比的混淆矩阵 CSV。"""
    path = output_dir / "confusion_matrix_percent.csv"
    fieldnames = ["true_label", "true_label_name"] + [f"pred_{label}" for label in LABEL_ORDER]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, percent_row in zip(LABEL_ORDER, percentages):
            item = {"true_label": label, "true_label_name": LABEL_NAME[label]}
            for pred_label, percent in zip(LABEL_ORDER, percent_row):
                item[f"pred_{pred_label}"] = f"{percent:.2f}%"
            writer.writerow(item)


def write_markdown_report(matrix: list[list[int]], percentages: list[list[float]], output_dir: Path) -> None:
    """输出 Markdown 表和简短结论。"""
    path = output_dir / "confusion_matrix_report.md"
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(LABEL_ORDER)))
    accuracy = correct / total if total else 0.0

    recalls = {
        label: (matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0.0)
        for i, label in enumerate(LABEL_ORDER)
    }
    pred_totals = column_totals(matrix)
    precisions = {
        label: (matrix[i][i] / pred_totals[i] if pred_totals[i] else 0.0)
        for i, label in enumerate(LABEL_ORDER)
    }

    lines = [
        "# Confusion Matrix Analysis",
        "",
        f"- Total samples: {total}",
        f"- Correct predictions: {correct}",
        f"- Accuracy: {accuracy:.4f}",
        "",
        "## Count Matrix",
        "",
        "| True \\ Pred | N 正常 | BF 滚动体 | IF 内圈 | OF 外圈 | CF 混合 | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for i, label in enumerate(LABEL_ORDER):
        row = matrix[i]
        lines.append(
            f"| {label} {LABEL_NAME[label]} | "
            + " | ".join(str(value) for value in row)
            + f" | {recalls[label]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Row-Normalized Matrix",
            "",
            "| True \\ Pred | N 正常 | BF 滚动体 | IF 内圈 | OF 外圈 | CF 混合 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, percent_row in zip(LABEL_ORDER, percentages):
        lines.append(
            f"| {label} {LABEL_NAME[label]} | "
            + " | ".join(f"{percent:.1f}%" for percent in percent_row)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Class Metrics From Matrix",
            "",
            "| Class | Precision | Recall |",
            "|---|---:|---:|",
        ]
    )
    for label in LABEL_ORDER:
        lines.append(f"| {label} {LABEL_NAME[label]} | {precisions[label]:.4f} | {recalls[label]:.4f} |")

    lines.extend(
        [
            "",
            "## Key Observations",
            "",
            "- The best recognized class is N/正常状态 by recall, while BF/滚动体故障 remains the weakest class.",
            "- CF/混合故障 has relatively high recall, but its precision is limited because IF/OF/BF samples are sometimes predicted as CF.",
            "- BF is often confused with N and CF, which is the main bottleneck for further improvement.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def color_for_percent(percent: float) -> str:
    """根据百分比生成蓝色系背景。"""
    # 浅色到深色蓝，避免过暗影响文字可读性。
    t = max(0.0, min(percent / 100.0, 1.0))
    r = int(239 - 190 * t)
    g = int(246 - 130 * t)
    b = int(255 - 70 * t)
    return f"rgb({r},{g},{b})"


def text_color_for_percent(percent: float) -> str:
    """深色格子使用白字，浅色格子使用黑字。"""
    return "#ffffff" if percent >= 50 else "#172033"


def svg_text(x: float, y: float, text: str, size: int = 20, weight: str = "400", anchor: str = "middle") -> str:
    """生成 SVG 文本。"""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="#172033">{escape(text)}</text>'
    )


def write_svg(matrix: list[list[int]], percentages: list[list[float]], output_dir: Path) -> None:
    """生成增强版 SVG 混淆矩阵。"""
    path = output_dir / "confusion_matrix_enhanced.svg"
    cell = 120
    left = 190
    top = 150
    width = 900
    height = 850
    matrix_size = cell * len(LABEL_ORDER)

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(LABEL_ORDER)))
    accuracy = correct / total if total else 0.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        "text { font-family: 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', Arial, sans-serif; }",
        ".small { fill: #42526e; }",
        "</style>",
        svg_text(width / 2, 48, "SUFD 故障诊断混淆矩阵（最佳 checkpoint-2200）", 26, "700"),
        svg_text(width / 2, 82, f"Accuracy = {accuracy:.4f}，单元格显示：数量 / 真实类别内占比", 18),
    ]

    # 列标签
    for j, label in enumerate(LABEL_ORDER):
        x = left + j * cell + cell / 2
        parts.append(svg_text(x, top - 42, label, 20, "700"))
        parts.append(svg_text(x, top - 15, LABEL_NAME[label], 17))

    # 行标签
    for i, label in enumerate(LABEL_ORDER):
        y = top + i * cell + cell / 2
        parts.append(svg_text(left - 58, y - 7, label, 20, "700"))
        parts.append(svg_text(left - 58, y + 20, LABEL_NAME[label], 17))

    # 单元格
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            percent = percentages[i][j]
            x = left + j * cell
            y = top + i * cell
            bg = color_for_percent(percent)
            text_color = text_color_for_percent(percent)
            stroke = "#2f5f8f" if i == j else "#d5dde8"
            stroke_width = 2.5 if i == j else 1
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{bg}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 - 8}" text-anchor="middle" '
                f'font-size="25" font-weight="700" fill="{text_color}">{count}</text>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 26}" text-anchor="middle" '
                f'font-size="18" fill="{text_color}">{percent:.1f}%</text>'
            )

    # 坐标轴标题
    parts.append(svg_text(left + matrix_size / 2, top + matrix_size + 70, "预测类别 Predicted Label", 20, "700"))
    parts.append(
        f'<text x="58" y="{top + matrix_size / 2}" text-anchor="middle" font-size="20" '
        f'font-weight="700" fill="#172033" transform="rotate(-90 58 {top + matrix_size / 2})">'
        "真实类别 True Label</text>"
    )

    # 右侧简要发现
    note_x = left + matrix_size + 50
    note_y = top + 20
    notes = [
        "关键观察",
        "N 召回最高：67.5%",
        "BF 最弱：35.0%",
        "BF 易误判为 N/CF",
        "CF 召回：51.9%",
    ]
    parts.append(f'<rect x="{note_x - 20}" y="{note_y - 35}" width="210" height="180" fill="#f6f8fb" stroke="#d5dde8"/>')
    for idx, note in enumerate(notes):
        size = 18 if idx == 0 else 15
        weight = "700" if idx == 0 else "400"
        parts.append(
            f'<text x="{note_x}" y="{note_y + idx * 32}" font-size="{size}" '
            f'font-weight="{weight}" fill="#172033">{escape(note)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    """主函数。"""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = read_confusion_matrix(args.input.resolve())
    percentages = row_percentages(matrix)

    write_enhanced_csv(matrix, percentages, output_dir)
    write_percent_csv(percentages, output_dir)
    write_markdown_report(matrix, percentages, output_dir)
    write_svg(matrix, percentages, output_dir)

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(LABEL_ORDER)))
    print("=" * 80)
    print(f"总样本数：{total}")
    print(f"正确数：{correct}")
    print(f"Accuracy：{correct / total:.4f}")
    print(f"增强 CSV：{output_dir / 'confusion_matrix_enhanced.csv'}")
    print(f"百分比 CSV：{output_dir / 'confusion_matrix_percent.csv'}")
    print(f"Markdown 报告：{output_dir / 'confusion_matrix_report.md'}")
    print(f"增强 SVG：{output_dir / 'confusion_matrix_enhanced.svg'}")


if __name__ == "__main__":
    main()
