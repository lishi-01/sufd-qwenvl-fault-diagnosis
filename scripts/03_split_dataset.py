"""
将 all_metadata.csv 划分为训练集、gap 和测试集。

默认策略：
    1. 读取 data/processed/metadata/all_metadata.csv
    2. 按 source_field 分组，例如 ball_20_0、inner_30_2
    3. 每个 source_field 内部按 segment_index 从小到大排序
    4. 前 75% 样本写入 train.csv
    5. 中间 5% 样本写入 gap.csv，仅作为隔离带，不参与训练和测试
    6. 后 20% 样本写入 test.csv

这样做的原因：
    当前样本来自长时间序列切片，且相邻切片之间存在 overlap。
    如果完全随机划分，训练集和测试集可能出现高度相似的相邻片段，
    容易造成测试指标虚高。按时间顺序切分并加入 gap 更稳妥。
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "all_metadata.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "train.csv"
DEFAULT_GAP_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "gap.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按 75%/5%/20% 划分 SUFD metadata 为训练集、gap 和测试集")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH, help="输入 all_metadata.csv 路径")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH, help="输出 train.csv 路径")
    parser.add_argument("--gap-path", type=Path, default=DEFAULT_GAP_PATH, help="输出 gap.csv 路径")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="输出 test.csv 路径")
    parser.add_argument("--train-ratio", type=float, default=0.75, help="训练集比例，默认 0.75")
    parser.add_argument("--gap-ratio", type=float, default=0.05, help="gap 隔离带比例，默认 0.05")
    parser.add_argument(
        "--group-key",
        type=str,
        default="source_field",
        help="分组字段，默认 source_field；用于保证每个类别/工况都按比例划分",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否在每个分组内随机打乱后再划分；默认保持时间顺序",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，仅在 --shuffle 时生效")
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """读取 CSV，并返回行数据和表头。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata 文件不存在：{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(f"metadata 文件为空：{csv_path}")
    if not fieldnames:
        raise ValueError(f"metadata 文件缺少表头：{csv_path}")

    return rows, fieldnames


def to_int(value: str, default: int = 0) -> int:
    """将字符串转为整数；失败时使用默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def group_rows(rows: list[dict[str, str]], group_key: str) -> dict[str, list[dict[str, str]]]:
    """按指定字段对 metadata 分组。"""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if group_key not in row:
            raise KeyError(f"metadata 中不存在分组字段：{group_key}")
        groups[row[group_key]].append(row)
    return groups


def split_group(
    rows: list[dict[str, str]],
    train_ratio: float,
    gap_ratio: float,
    shuffle: bool,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """对单个分组做训练/gap/测试划分。"""
    rows = list(rows)

    if shuffle:
        rng.shuffle(rows)
    else:
        rows.sort(key=lambda item: to_int(item.get("segment_index", "0")))

    train_count = int(len(rows) * train_ratio)
    gap_count = int(len(rows) * gap_ratio)

    # 极小分组的兜底处理：只要一个分组至少有 3 条数据，就尽量保证三部分都有样本。
    if len(rows) >= 3:
        train_count = min(max(train_count, 1), len(rows) - 2)
        gap_count = min(max(gap_count, 1), len(rows) - train_count - 1)
    else:
        train_count = len(rows)
        gap_count = 0

    test_start = train_count + gap_count
    return rows[:train_count], rows[train_count:test_start], rows[test_start:]


def write_csv(rows: list[dict[str, str]], csv_path: Path, fieldnames: list[str]) -> None:
    """写入 CSV 文件。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], name: str) -> None:
    """打印数据集数量统计。"""
    label_count: dict[str, int] = defaultdict(int)
    condition_count: dict[str, int] = defaultdict(int)

    for row in rows:
        label_count[row.get("label", "UNKNOWN")] += 1
        condition_count[row.get("condition", "UNKNOWN")] += 1

    print(f"{name} 样本数：{len(rows)}")
    print(f"{name} 类别分布：{dict(sorted(label_count.items()))}")
    print(f"{name} 工况分布：{dict(sorted(condition_count.items()))}")


def main() -> None:
    """主函数。"""
    args = parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train-ratio 必须在 0 和 1 之间，当前值：{args.train_ratio}")
    if not 0 <= args.gap_ratio < 1:
        raise ValueError(f"gap-ratio 必须在 0 和 1 之间，当前值：{args.gap_ratio}")
    if args.train_ratio + args.gap_ratio >= 1:
        raise ValueError(
            f"train-ratio + gap-ratio 必须小于 1，当前值：{args.train_ratio + args.gap_ratio}"
        )

    rng = random.Random(args.seed)
    rows, fieldnames = read_csv_rows(args.metadata_path.resolve())
    groups = group_rows(rows, args.group_key)

    train_rows: list[dict[str, str]] = []
    gap_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    print(f"输入 metadata：{args.metadata_path.resolve()}")
    print(f"总样本数：{len(rows)}")
    print(f"分组字段：{args.group_key}")
    print(f"分组数量：{len(groups)}")
    print(f"训练集比例：{args.train_ratio:.2f}")
    print(f"gap 比例：{args.gap_ratio:.2f}")
    print(f"测试集比例：{1 - args.train_ratio - args.gap_ratio:.2f}")
    print(f"划分方式：{'组内随机划分' if args.shuffle else '组内按时间顺序划分'}")

    for group_name in sorted(groups):
        group_train, group_gap, group_test = split_group(
            rows=groups[group_name],
            train_ratio=args.train_ratio,
            gap_ratio=args.gap_ratio,
            shuffle=args.shuffle,
            rng=rng,
        )
        train_rows.extend(group_train)
        gap_rows.extend(group_gap)
        test_rows.extend(group_test)
        print(f"  {group_name}: train={len(group_train)}, gap={len(group_gap)}, test={len(group_test)}")

    write_csv(train_rows, args.train_path.resolve(), fieldnames)
    write_csv(gap_rows, args.gap_path.resolve(), fieldnames)
    write_csv(test_rows, args.test_path.resolve(), fieldnames)

    print("=" * 80)
    summarize(train_rows, "训练集")
    summarize(gap_rows, "gap")
    summarize(test_rows, "测试集")
    print(f"训练集已保存：{args.train_path.resolve()}")
    print(f"gap 已保存：{args.gap_path.resolve()}")
    print(f"测试集已保存：{args.test_path.resolve()}")


if __name__ == "__main__":
    main()
