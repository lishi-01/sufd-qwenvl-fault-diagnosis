"""
检查 SUFD .mat 原始数据文件。

本脚本用于项目第一步：确认每个 .mat 文件中的变量名、数据维度、
数据类型，以及重点有效通道（第 2、3、4 通道）的基本统计信息。

根据当前数据说明：
    第 1 通道：电机振动
    第 2 通道：行星齿轮箱 X 方向振动，有效
    第 3 通道：行星齿轮箱 Y 方向振动，有效
    第 4 通道：行星齿轮箱 Z 方向振动，有效
    第 5 通道：电机转矩
    第 6 通道：并联齿轮箱 X 方向振动
    第 7 通道：并联齿轮箱 Y 方向振动
    第 8 通道：并联齿轮箱 Z 方向振动

注意：
    MATLAB 中常说的“第 2、3、4 行信号”，在当前文件实际 shape 为
    [时间点数, 8] 的情况下，对应 Python 数组中的第 2、3、4 列，
    即 data[:, 1:4]。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


# 项目根目录：scripts/01_inspect_mat.py 的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 当前原始数据所在目录；后续整理项目时也可以改成 data/raw_mat
DEFAULT_RAW_DIR = PROJECT_ROOT / "bearingset mat"

# 只使用行星齿轮箱 X/Y/Z 三个方向的振动信号
# Python 是 0 起始索引，因此第 2、3、4 通道对应列索引 1、2、3
EFFECTIVE_CHANNELS = {
    1: "行星齿轮箱 X 方向振动",
    2: "行星齿轮箱 Y 方向振动",
    3: "行星齿轮箱 Z 方向振动",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="检查 SUFD .mat 文件结构和有效通道统计信息")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="原始 .mat 文件目录，默认读取项目下的 bearingset mat",
    )
    return parser.parse_args()


def find_data_keys(mat_dict: dict) -> list[str]:
    """过滤掉 scipy 读取 .mat 时自动生成的元信息字段。"""
    return [key for key in mat_dict.keys() if not key.startswith("__")]


def safe_stat(values: np.ndarray, stat_name: str) -> float:
    """计算统计量，并自动忽略 NaN。"""
    if stat_name == "min":
        return float(np.nanmin(values))
    if stat_name == "max":
        return float(np.nanmax(values))
    if stat_name == "mean":
        return float(np.nanmean(values))
    if stat_name == "std":
        return float(np.nanstd(values))
    raise ValueError(f"不支持的统计量：{stat_name}")


def print_array_summary(name: str, data: np.ndarray) -> None:
    """打印单个变量的整体信息和有效通道统计。"""
    print(f"  变量名：{name}")
    print(f"  shape：{data.shape}")
    print(f"  dtype：{data.dtype}")

    if data.ndim != 2:
        print("  [警告] 当前变量不是二维数组，请后续人工确认数据结构。")
        return

    if data.shape[1] < 4:
        print("  [警告] 当前变量列数不足 4，无法提取第 2、3、4 通道。")
        return

    print("  有效通道统计：")
    for col_idx, channel_name in EFFECTIVE_CHANNELS.items():
        signal = data[:, col_idx]
        nan_count = int(np.isnan(signal).sum())
        finite_count = int(np.isfinite(signal).sum())

        print(f"    第 {col_idx + 1} 通道：{channel_name}")
        print(f"      length：{signal.shape[0]}")
        print(f"      min：{safe_stat(signal, 'min'):.6f}")
        print(f"      max：{safe_stat(signal, 'max'):.6f}")
        print(f"      mean：{safe_stat(signal, 'mean'):.6f}")
        print(f"      std：{safe_stat(signal, 'std'):.6f}")
        print(f"      NaN 数量：{nan_count}")
        print(f"      有效数值数量：{finite_count}")


def inspect_mat_file(mat_path: Path) -> None:
    """读取并检查单个 .mat 文件。"""
    print("=" * 80)
    print(f"文件：{mat_path.name}")

    mat_dict = sio.loadmat(mat_path)
    data_keys = find_data_keys(mat_dict)

    if not data_keys:
        print("  [警告] 没有找到有效变量。")
        return

    print(f"  有效变量数量：{len(data_keys)}")
    for key in data_keys:
        data = np.asarray(mat_dict[key])
        print_array_summary(key, data)


def main() -> None:
    """主函数：批量检查目录下所有 .mat 文件。"""
    args = parse_args()
    raw_dir = args.raw_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_dir}")

    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"目录中没有找到 .mat 文件：{raw_dir}")

    print(f"原始数据目录：{raw_dir}")
    print(f".mat 文件数量：{len(mat_files)}")
    print("重点检查通道：第 2、3、4 通道（行星齿轮箱 X/Y/Z 方向振动）")

    for mat_path in mat_files:
        inspect_mat_file(mat_path)

    print("=" * 80)
    print("检查完成。下一步可以基于第 2、3、4 通道生成 RGB-STFT 时频图。")


if __name__ == "__main__":
    main()
