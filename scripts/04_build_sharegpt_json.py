"""
将 train.csv / test.csv 转换为 Qwen-VL 微调所需的 ShareGPT 多模态 JSON。

输入：
    data/processed/metadata/train.csv
    data/processed/metadata/test.csv

输出：
    data/processed/json/train_sft.json
    data/processed/json/test_sft.json
    或者在 --answer-mode class_only 时输出：
    data/processed/json/train_sft_class_only.json
    data/processed/json/test_sft_class_only.json

每条样本格式：
    {
      "conversations": [
        {"from": "human", "value": "<image>\\n...中文诊断指令..."},
        {"from": "gpt", "value": "故障类型：...\\n诊断依据：...\\n维护建议：..."}
      ],
      "images": ["data/processed/images/all/sample_xxx.png"]
    }
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "json"


LABEL_ANSWER_MAP = {
    "N": {
        "fault_type": "正常状态",
        "basis": "包络时频图整体能量分布较平稳，未出现明显周期性冲击或异常高频能量集中，符合正常运行状态特征。",
        "advice": "建议保持常规状态监测，持续记录振动趋势，用于后续健康状态评估。",
    },
    "BF": {
        "fault_type": "滚动体故障",
        "basis": "包络时频图中存在局部能量增强和冲击调制特征，符合滚动体损伤可能引起的非平稳振动表现。",
        "advice": "建议检查滚动体表面是否存在磨损、点蚀或剥落，并结合包络谱进一步确认故障程度。",
    },
    "IF": {
        "fault_type": "内圈故障",
        "basis": "包络时频图中存在较明显的周期性冲击和调制成分，符合内圈故障在旋转过程中产生的振动特征。",
        "advice": "建议重点检查轴承内圈表面损伤情况，并结合历史振动趋势评估故障发展速度。",
    },
    "OF": {
        "fault_type": "外圈故障",
        "basis": "包络时频图中存在相对稳定的冲击响应和局部频带能量增强，符合外圈局部损伤导致的周期性激励特征。",
        "advice": "建议检查轴承外圈固定区域是否存在剥落、裂纹或异常磨损，并安排必要的停机复检。",
    },
    "CF": {
        "fault_type": "混合故障",
        "basis": "包络时频图中同时呈现多组冲击成分和复杂能量分布，可能对应内圈与外圈共同损伤引起的复合振动特征。",
        "advice": "建议同时检查轴承内圈和外圈，并优先安排停机检测，避免复合故障进一步扩展。",
    },
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构造 Qwen-VL 多模态 ShareGPT SFT 数据")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV, help="训练集 CSV 路径")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="测试集 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出 JSON 目录")
    parser.add_argument(
        "--answer-mode",
        choices=["full", "class_only"],
        default="full",
        help="答案模式：full 输出故障类型、诊断依据和维护建议；class_only 只输出故障类型",
    )
    parser.add_argument(
        "--use-absolute-image-path",
        action="store_true",
        help="是否在 images 字段中使用图片绝对路径；默认使用项目相对路径",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 metadata CSV。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV 文件为空：{csv_path}")
    return rows


def normalize_image_path(image_path: str, use_absolute: bool) -> str:
    """标准化图片路径，保证 JSON 中使用正斜杠。"""
    path = Path(image_path)
    if use_absolute:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve()).replace("\\", "/")
    return image_path.replace("\\", "/")


def get_required_value(row: dict[str, str], key: str) -> str:
    """读取必需字段。"""
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"metadata 行缺少必需字段：{key}，sample_id={row.get('sample_id', 'UNKNOWN')}")
    return value


def build_prompt(row: dict[str, str], answer_mode: str) -> str:
    """根据 metadata 构造中文诊断指令。"""
    speed_hz = get_required_value(row, "speed_hz")
    load_voltage = get_required_value(row, "load_voltage")
    sampling_rate = get_required_value(row, "sampling_rate")

    if answer_mode == "class_only":
        output_instruction = (
            "请只输出故障类型，不要输出诊断依据或维护建议。\n"
            "请严格按照以下格式输出：\n"
            "故障类型：<滚动体故障/内圈故障/外圈故障/混合故障/正常状态>"
        )
    else:
        output_instruction = (
            "请严格按照以下格式输出：\n"
            "故障类型：<滚动体故障/内圈故障/外圈故障/混合故障/正常状态>\n"
            "诊断依据：<结合包络时频图和工况给出简短说明>\n"
            "维护建议：<给出简短检查或维护建议>"
        )

    return (
        "<image>\n"
        "你是一名工业设备故障诊断助手。请根据给定的轴承包络时频图和工况参数，判断当前轴承状态。\n"
        "该图像由行星齿轮箱 X、Y、Z 三方向振动信号生成：先通过简化谱峭度方法选择冲击显著频带，"
        "再进行带通滤波、Hilbert 包络提取和 STFT 时频分析，最后融合为 RGB 图像。\n"
        "工况参数：\n"
        "- 数据集：SUFD 齿轮箱/轴承故障数据集\n"
        f"- 采样率：{sampling_rate} Hz\n"
        f"- 转速工况：{speed_hz} Hz\n"
        f"- 负载/电压工况：{load_voltage} V\n"
        f"{output_instruction}"
    )


def build_answer(row: dict[str, str], answer_mode: str) -> str:
    """根据标签构造标准中文答案。"""
    label = get_required_value(row, "label")
    if label not in LABEL_ANSWER_MAP:
        raise ValueError(f"未知标签：{label}，sample_id={row.get('sample_id', 'UNKNOWN')}")

    answer = LABEL_ANSWER_MAP[label]
    if answer_mode == "class_only":
        return f"故障类型：{answer['fault_type']}"

    return (
        f"故障类型：{answer['fault_type']}\n"
        f"诊断依据：{answer['basis']}\n"
        f"维护建议：{answer['advice']}"
    )


def build_sharegpt_sample(row: dict[str, str], use_absolute_image_path: bool, answer_mode: str) -> dict:
    """将单行 metadata 转换为一条 ShareGPT 多模态样本。"""
    image_path = normalize_image_path(get_required_value(row, "image_path"), use_absolute_image_path)

    return {
        "conversations": [
            {
                "from": "human",
                "value": build_prompt(row, answer_mode),
            },
            {
                "from": "gpt",
                "value": build_answer(row, answer_mode),
            },
        ],
        "images": [image_path],
    }


def build_dataset(rows: list[dict[str, str]], use_absolute_image_path: bool, answer_mode: str) -> list[dict]:
    """批量构造 ShareGPT 数据集。"""
    return [build_sharegpt_sample(row, use_absolute_image_path, answer_mode) for row in rows]


def write_json(data: list[dict], json_path: Path) -> None:
    """写入 JSON 文件。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize(rows: list[dict[str, str]], name: str) -> None:
    """打印数据集标签统计。"""
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("label", "UNKNOWN")
        counts[label] = counts.get(label, 0) + 1
    print(f"{name} 样本数：{len(rows)}")
    print(f"{name} 标签分布：{dict(sorted(counts.items()))}")


def convert_csv_to_json(
    csv_path: Path,
    json_path: Path,
    use_absolute_image_path: bool,
    answer_mode: str,
    name: str,
) -> None:
    """读取一个 CSV 并转换为 ShareGPT JSON。"""
    rows = read_csv_rows(csv_path)
    data = build_dataset(rows, use_absolute_image_path, answer_mode)
    write_json(data, json_path)
    summarize(rows, name)
    print(f"{name} JSON 已保存：{json_path}")


def main() -> None:
    """主函数。"""
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.answer_mode == "class_only":
        train_json = output_dir / "train_sft_class_only.json"
        test_json = output_dir / "test_sft_class_only.json"
    else:
        train_json = output_dir / "train_sft.json"
        test_json = output_dir / "test_sft.json"

    print("开始构造 Qwen-VL ShareGPT 多模态 SFT 数据")
    print(f"训练集 CSV：{args.train_csv.resolve()}")
    print(f"测试集 CSV：{args.test_csv.resolve()}")
    print(f"输出目录：{output_dir}")
    print(f"答案模式：{args.answer_mode}")
    print(f"图片路径格式：{'绝对路径' if args.use_absolute_image_path else '项目相对路径'}")

    convert_csv_to_json(args.train_csv.resolve(), train_json, args.use_absolute_image_path, args.answer_mode, "训练集")
    convert_csv_to_json(args.test_csv.resolve(), test_json, args.use_absolute_image_path, args.answer_mode, "测试集")

    print("=" * 80)
    print("转换完成。下一步可以配置 LLaMA-Factory dataset_info.json 和 Qwen2.5-VL LoRA 训练参数。")


if __name__ == "__main__":
    main()
