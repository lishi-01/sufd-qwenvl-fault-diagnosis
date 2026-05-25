"""
构造 SUFD-QwenVL 强化学习阶段的偏好数据。

本脚本用于生成 DPO / 奖励模型训练所需的 chosen-rejected 数据：

    chosen   = 真实标签对应的高质量完整诊断报告
    rejected = 错误标签对应的低质量诊断报告

输出格式为 LLaMA-Factory 支持的 ShareGPT preference 格式：

{
  "conversations": [
    {"from": "human", "value": "<image>\\n...诊断指令..."}
  ],
  "chosen": {
    "from": "gpt",
    "value": "故障类型：滚动体故障\\n诊断依据：...\\n维护建议：..."
  },
  "rejected": {
    "from": "gpt",
    "value": "故障类型：正常状态\\n诊断依据：...\\n维护建议：..."
  },
  "images": [
    "data/processed/images/all/sample_BF_20_0_000409.png"
  ]
}

第一版设计目标：
    1. 快速跑通 DPO 数据构造流程
    2. 使用强偏好信号：正确类别 vs 错误类别
    3. 保持图像路径、prompt 和现有 SFT 数据一致
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rlhf"

LABEL_ORDER = ["N", "BF", "IF", "OF", "CF"]

LABEL_NAME_MAP = {
    "N": "正常状态",
    "BF": "滚动体故障",
    "IF": "内圈故障",
    "OF": "外圈故障",
    "CF": "混合故障",
}

REPORT_TEMPLATE = {
    "N": {
        "basis": "包络时频图整体能量分布较平稳，未出现明显周期性冲击或异常高频能量集中，符合正常运行状态特征。",
        "advice": "建议保持常规状态监测，持续记录振动趋势，用于后续健康状态评估。",
    },
    "BF": {
        "basis": "包络时频图中存在局部能量增强和冲击调制特征，符合滚动体损伤可能引起的非平稳振动表现。",
        "advice": "建议检查滚动体表面是否存在磨损、点蚀或剥落，并结合包络谱进一步确认故障程度。",
    },
    "IF": {
        "basis": "包络时频图中存在较明显的周期性冲击和调制成分，符合内圈故障在旋转过程中产生的振动特征。",
        "advice": "建议重点检查轴承内圈表面损伤情况，并结合历史振动趋势评估故障发展速度。",
    },
    "OF": {
        "basis": "包络时频图中存在相对稳定的冲击响应和局部频带能量增强，符合外圈局部损伤导致的周期性激励特征。",
        "advice": "建议检查轴承外圈固定区域是否存在剥落、裂纹或异常磨损，并安排必要的停机复检。",
    },
    "CF": {
        "basis": "包络时频图中同时呈现多组冲击成分和复杂能量分布，可能对应内圈与外圈共同损伤引起的复合振动特征。",
        "advice": "建议同时检查轴承内圈和外圈，并优先安排停机检测，避免复合故障进一步扩展。",
    },
}

# 根据当前最佳 checkpoint 的混淆矩阵设计 hard negative。
# 例如 BF 容易被误判为 N/CF，因此优先把 N/CF 作为 rejected。
HARD_NEGATIVE_MAP = {
    "N": ["BF", "CF", "OF", "IF"],
    "BF": ["N", "CF", "OF", "IF"],
    "IF": ["CF", "OF", "N", "BF"],
    "OF": ["CF", "N", "IF", "BF"],
    "CF": ["IF", "N", "BF", "OF"],
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构造 SUFD DPO/RM 偏好数据")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV, help="训练集 metadata CSV")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="测试集 metadata CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument(
        "--negative-strategy",
        choices=["hard", "random", "cycle"],
        default="hard",
        help="rejected 错误类别选择策略：hard 使用混淆矩阵近邻，random 随机，cycle 固定轮转",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=1,
        help="每条样本构造多少个 rejected 偏好对，默认 1",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--use-absolute-image-path",
        action="store_true",
        help="images 字段是否使用绝对路径；默认使用项目相对路径",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="是否在 JSON 中保留 sample_id、label 等辅助字段；LLaMA-Factory 训练不需要这些字段",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 metadata CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV 文件为空：{path}")
    return rows


def required(row: dict[str, str], key: str) -> str:
    """读取必需字段。"""
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"metadata 行缺少字段 {key}，sample_id={row.get('sample_id', 'UNKNOWN')}")
    return value


def normalize_image_path(image_path: str, use_absolute: bool) -> str:
    """标准化图片路径。"""
    path = Path(image_path)
    if use_absolute:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve()).replace("\\", "/")
    return image_path.replace("\\", "/")


def build_prompt(row: dict[str, str]) -> str:
    """构造完整回答偏好训练用 prompt。"""
    speed_hz = required(row, "speed_hz")
    load_voltage = required(row, "load_voltage")
    sampling_rate = required(row, "sampling_rate")

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
        "请严格按照以下格式输出：\n"
        "故障类型：<滚动体故障/内圈故障/外圈故障/混合故障/正常状态>\n"
        "诊断依据：<结合包络时频图和工况给出简短说明>\n"
        "维护建议：<给出简短检查或维护建议>"
    )


def build_report(label: str, row: dict[str, str]) -> str:
    """根据标签和工况生成完整诊断报告。"""
    if label not in REPORT_TEMPLATE:
        raise ValueError(f"未知标签：{label}")

    speed_hz = required(row, "speed_hz")
    load_voltage = required(row, "load_voltage")
    fault_type = LABEL_NAME_MAP[label]
    template = REPORT_TEMPLATE[label]
    basis = f"{template['basis']}当前工况为 {speed_hz} Hz、{load_voltage} V。"

    return (
        f"故障类型：{fault_type}\n"
        f"诊断依据：{basis}\n"
        f"维护建议：{template['advice']}"
    )


def choose_negative_labels(true_label: str, strategy: str, num_negatives: int, rng: random.Random) -> list[str]:
    """为单条样本选择一个或多个错误标签。"""
    candidates = [label for label in LABEL_ORDER if label != true_label]
    if num_negatives <= 0:
        raise ValueError("num-negatives 必须大于 0")

    if strategy == "hard":
        ordered = [label for label in HARD_NEGATIVE_MAP[true_label] if label != true_label]
        labels = ordered[:num_negatives]
        if len(labels) < num_negatives:
            labels.extend([label for label in candidates if label not in labels])
        return labels[:num_negatives]

    if strategy == "random":
        if num_negatives <= len(candidates):
            return rng.sample(candidates, num_negatives)
        labels = []
        while len(labels) < num_negatives:
            labels.extend(rng.sample(candidates, len(candidates)))
        return labels[:num_negatives]

    if strategy == "cycle":
        start = LABEL_ORDER.index(true_label)
        labels = []
        offset = 1
        while len(labels) < num_negatives:
            label = LABEL_ORDER[(start + offset) % len(LABEL_ORDER)]
            if label != true_label:
                labels.append(label)
            offset += 1
        return labels

    raise ValueError(f"不支持的 negative strategy：{strategy}")


def build_preference_samples(
    rows: list[dict[str, str]],
    strategy: str,
    num_negatives: int,
    rng: random.Random,
    use_absolute_image_path: bool,
    include_meta: bool,
) -> list[dict]:
    """批量构造偏好样本。"""
    samples = []
    for row in rows:
        true_label = required(row, "label")
        image_path = normalize_image_path(required(row, "image_path"), use_absolute_image_path)
        negative_labels = choose_negative_labels(true_label, strategy, num_negatives, rng)

        for pair_index, negative_label in enumerate(negative_labels):
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": build_prompt(row),
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": build_report(true_label, row),
                },
                "rejected": {
                    "from": "gpt",
                    "value": build_report(negative_label, row),
                },
                "images": [image_path],
            }

            if include_meta:
                item.update(
                    {
                        "sample_id": row.get("sample_id", ""),
                        "true_label": true_label,
                        "rejected_label": negative_label,
                        "pair_index": pair_index,
                        "source_field": row.get("source_field", ""),
                        "condition": row.get("condition", ""),
                    }
                )

            samples.append(item)
    return samples


def write_json(data: list[dict], path: Path) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize(samples: list[dict], name: str) -> None:
    """打印偏好数据统计。"""
    chosen_counter = Counter()
    rejected_counter = Counter()

    for item in samples:
        chosen_text = item["chosen"]["value"]
        rejected_text = item["rejected"]["value"]
        for label, name_cn in LABEL_NAME_MAP.items():
            if f"故障类型：{name_cn}" in chosen_text:
                chosen_counter[label] += 1
            if f"故障类型：{name_cn}" in rejected_text:
                rejected_counter[label] += 1

    print(f"{name} 偏好样本数：{len(samples)}")
    print(f"{name} chosen 标签分布：{dict(sorted(chosen_counter.items()))}")
    print(f"{name} rejected 标签分布：{dict(sorted(rejected_counter.items()))}")


def validate_image_paths(samples: list[dict]) -> tuple[int, int]:
    """检查图片路径是否存在，返回存在数量和缺失数量。"""
    exists_count = 0
    missing_count = 0
    for item in samples:
        image_path = item["images"][0]
        path = Path(image_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            exists_count += 1
        else:
            missing_count += 1
    return exists_count, missing_count


def convert_split(
    csv_path: Path,
    output_path: Path,
    split_name: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    """转换单个 split。"""
    rows = read_csv_rows(csv_path)
    samples = build_preference_samples(
        rows=rows,
        strategy=args.negative_strategy,
        num_negatives=args.num_negatives,
        rng=rng,
        use_absolute_image_path=args.use_absolute_image_path,
        include_meta=args.include_meta,
    )
    write_json(samples, output_path)
    summarize(samples, split_name)
    exists_count, missing_count = validate_image_paths(samples)
    print(f"{split_name} 图片路径存在：{exists_count}，缺失：{missing_count}")
    print(f"{split_name} 已保存：{output_path}")


def main() -> None:
    """主函数。"""
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()

    print("开始构造 SUFD DPO/RM 偏好数据")
    print(f"negative strategy：{args.negative_strategy}")
    print(f"num negatives：{args.num_negatives}")
    print(f"输出目录：{output_dir}")

    convert_split(
        csv_path=args.train_csv.resolve(),
        output_path=output_dir / "train_dpo.json",
        split_name="训练集",
        args=args,
        rng=rng,
    )
    convert_split(
        csv_path=args.test_csv.resolve(),
        output_path=output_dir / "test_dpo.json",
        split_name="测试集",
        args=args,
        rng=rng,
    )

    print("=" * 80)
    print("偏好数据构造完成。下一步：上传 train_dpo.json/test_dpo.json 到 LLaMA-Factory data/sufd/ 并注册 dataset_info.json。")


if __name__ == "__main__":
    main()
