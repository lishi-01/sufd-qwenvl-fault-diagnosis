"""
生成基于简化谱峭度频带搜索的包络 STFT 图像。

处理流程：
    1. 读取 SUFD .mat 文件
    2. 提取第 2、3、4 通道，即行星齿轮箱 X/Y/Z 三方向振动信号
    3. 将长信号按固定窗口切片
    4. 对每个切片、每个通道执行简化谱峭度频带搜索
    5. 使用最佳频带进行带通滤波
    6. 通过 Hilbert 变换提取包络信号
    7. 对包络信号做 STFT
    8. 将 X/Y/Z 三方向的包络 STFT 分别作为 RGB 三个通道保存为图像
    9. 保存 metadata.csv，记录标签、工况、切片位置和最佳频带

说明：
    这里的“简化谱峭度”不是完整 Fast Kurtogram，而是工程上更容易复现的
    候选频带搜索：对多个候选频带做带通滤波，计算滤波后信号峭度，
    选择峭度最大的频带作为冲击最显著频带。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from scipy.signal import butter, hilbert, sosfiltfilt, stft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "bearingset mat"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "all"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "all_metadata.csv"

SAMPLING_RATE = 4096
SEGMENT_LEN = 4096
HOP_LEN = 2048
IMAGE_SIZE = 224

# Python 中第 2、3、4 通道对应列索引 1、2、3
EFFECTIVE_CHANNELS = {
    "x": 1,
    "y": 2,
    "z": 3,
}

LABEL_MAP = {
    "health": ("N", "正常状态"),
    "ball": ("BF", "滚动体故障"),
    "inner": ("IF", "内圈故障"),
    "outer": ("OF", "外圈故障"),
    "comb": ("CF", "混合故障"),
}

# 候选频带上限不能超过 Nyquist 频率，即采样率的一半 2048 Hz
# 第一版先使用较粗的频带划分，保证可解释、可复现、运行速度可接受
CANDIDATE_BANDS = [
    (80, 200),
    (200, 400),
    (400, 700),
    (700, 1000),
    (1000, 1400),
    (1400, 1800),
    (1800, 2000),
]


@dataclass
class BandSearchResult:
    """单个通道的最佳频带搜索结果。"""

    band_low: float
    band_high: float
    kurtosis: float
    filtered_signal: np.ndarray


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成 SUFD 包络 STFT RGB 图像")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="原始 .mat 文件目录")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="输出图像目录")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH, help="metadata.csv 输出路径")
    parser.add_argument("--sampling-rate", type=int, default=SAMPLING_RATE, help="采样率，默认 4096 Hz")
    parser.add_argument("--segment-len", type=int, default=SEGMENT_LEN, help="切片长度，默认 4096 点")
    parser.add_argument("--hop-len", type=int, default=HOP_LEN, help="切片步长，默认 2048 点")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="输出图像尺寸，默认 224")
    parser.add_argument(
        "--max-segments-per-file",
        type=int,
        default=None,
        help="每个 .mat 文件最多生成多少个切片；调试时可设为 5 或 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果图像已存在，是否重新生成并覆盖",
    )
    return parser.parse_args()


def find_data_keys(mat_dict: dict) -> list[str]:
    """过滤 scipy 自动生成的 .mat 元信息字段。"""
    return [key for key in mat_dict.keys() if not key.startswith("__")]


def parse_field_name(field_name: str) -> dict[str, str | int]:
    """从变量名中解析故障类别和工况。

    示例：
        ball_20_0 -> ball, 20 Hz, 0 V
        inner_30_2 -> inner, 30 Hz, 2 V
    """
    parts = field_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"变量名格式不符合预期：{field_name}")

    fault_prefix = parts[0]
    if fault_prefix not in LABEL_MAP:
        raise ValueError(f"未知故障前缀：{fault_prefix}")

    label, label_name = LABEL_MAP[fault_prefix]
    speed_hz = int(parts[1])
    load_voltage = int(parts[2])

    return {
        "fault_prefix": fault_prefix,
        "label": label,
        "label_name": label_name,
        "condition": f"{speed_hz}_{load_voltage}",
        "speed_hz": speed_hz,
        "load_voltage": load_voltage,
    }


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对单个信号片段做 z-score 标准化。"""
    signal = np.asarray(signal, dtype=np.float64)
    return (signal - np.mean(signal)) / (np.std(signal) + eps)


def calculate_kurtosis(signal: np.ndarray, eps: float = 1e-12) -> float:
    """计算峭度；冲击越明显，峭度通常越高。"""
    centered = signal - np.mean(signal)
    std = np.std(centered)
    if std < eps:
        return 0.0
    normalized = centered / std
    return float(np.mean(normalized**4))


def bandpass_filter(signal: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Butterworth 带通滤波。"""
    nyquist = fs / 2
    if low <= 0 or high >= nyquist:
        raise ValueError(f"非法频带：{low}-{high} Hz，Nyquist={nyquist} Hz")

    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def search_best_kurtosis_band(signal: np.ndarray, fs: int) -> BandSearchResult:
    """在候选频带中选择滤波后峭度最大的频带。"""
    best_result: BandSearchResult | None = None

    for low, high in CANDIDATE_BANDS:
        filtered = bandpass_filter(signal, fs, low, high)
        kurt = calculate_kurtosis(filtered)

        if best_result is None or kurt > best_result.kurtosis:
            best_result = BandSearchResult(
                band_low=low,
                band_high=high,
                kurtosis=kurt,
                filtered_signal=filtered,
            )

    if best_result is None:
        raise RuntimeError("未能找到最佳谱峭度频带。")
    return best_result


def envelope_signal(filtered_signal: np.ndarray) -> np.ndarray:
    """通过 Hilbert 变换提取包络信号。"""
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    return zscore(envelope)


def envelope_stft_image(envelope: np.ndarray, fs: int, image_size: int) -> np.ndarray:
    """将包络信号转换为归一化 STFT 灰度图。"""
    _, _, spectrum = stft(
        envelope,
        fs=fs,
        window="hann",
        nperseg=256,
        noverlap=128,
        nfft=512,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(spectrum)
    log_magnitude = np.log1p(magnitude)

    # 将不同样本的图像值域统一到 0~1，避免某个通道幅值过大主导 RGB 图像
    min_value = np.min(log_magnitude)
    max_value = np.max(log_magnitude)
    normalized = (log_magnitude - min_value) / (max_value - min_value + 1e-8)

    # 频率轴通常希望低频在下、高频在上；图像数组第 0 行在上，因此这里上下翻转
    normalized = np.flipud(normalized)

    zoom_h = image_size / normalized.shape[0]
    zoom_w = image_size / normalized.shape[1]
    resized = zoom(normalized, (zoom_h, zoom_w), order=1)

    return np.clip(resized, 0.0, 1.0)


def build_rgb_envelope_stft(segment_xyz: np.ndarray, fs: int, image_size: int) -> tuple[np.ndarray, dict[str, float]]:
    """对 X/Y/Z 三方向分别处理，并融合为 RGB 图像。"""
    rgb_channels = []
    band_info: dict[str, float] = {}

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        signal = zscore(segment_xyz[:, axis_idx])
        search_result = search_best_kurtosis_band(signal, fs)
        envelope = envelope_signal(search_result.filtered_signal)
        image_channel = envelope_stft_image(envelope, fs, image_size)
        rgb_channels.append(image_channel)

        band_info[f"{axis_name}_band_low"] = search_result.band_low
        band_info[f"{axis_name}_band_high"] = search_result.band_high
        band_info[f"{axis_name}_kurtosis"] = search_result.kurtosis

    rgb_image = np.stack(rgb_channels, axis=-1)
    return rgb_image, band_info


def iter_segments(data: np.ndarray, segment_len: int, hop_len: int, max_segments: int | None):
    """按固定窗口遍历信号片段。"""
    count = 0
    for start in range(0, data.shape[0] - segment_len + 1, hop_len):
        end = start + segment_len
        yield start, end, data[start:end, :]
        count += 1
        if max_segments is not None and count >= max_segments:
            break


def save_rgb_image(image: np.ndarray, image_path: Path, overwrite: bool) -> None:
    """保存 RGB 图像。"""
    if image_path.exists() and not overwrite:
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(image_path, image)


def process_mat_file(
    mat_path: Path,
    image_dir: Path,
    fs: int,
    segment_len: int,
    hop_len: int,
    image_size: int,
    max_segments_per_file: int | None,
    overwrite: bool,
) -> list[dict[str, str | int | float]]:
    """处理单个 .mat 文件，并返回对应 metadata 记录。"""
    mat_dict = sio.loadmat(mat_path)
    data_keys = find_data_keys(mat_dict)
    if len(data_keys) != 1:
        raise ValueError(f"{mat_path.name} 中有效变量数量不是 1：{data_keys}")

    field_name = data_keys[0]
    field_info = parse_field_name(field_name)
    data = np.asarray(mat_dict[field_name], dtype=np.float64)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{mat_path.name} 数据维度不符合预期，实际 shape={data.shape}")

    # 提取第 2、3、4 通道，即行星齿轮箱 X/Y/Z 三方向振动
    selected = data[:, [EFFECTIVE_CHANNELS["x"], EFFECTIVE_CHANNELS["y"], EFFECTIVE_CHANNELS["z"]]]

    rows = []
    for segment_index, (start, end, segment_xyz) in enumerate(
        iter_segments(selected, segment_len, hop_len, max_segments_per_file)
    ):
        sample_id = f"{field_name}_{segment_index:06d}"
        image_name = f"sample_{field_info['label']}_{field_info['condition']}_{segment_index:06d}.png"
        image_path = image_dir / image_name

        if image_path.exists() and not overwrite:
            # 图像已经存在时跳过耗时的滤波和 STFT 计算，只补齐 metadata。
            # 这种情况通常发生在图像生成成功、但 metadata.csv 被占用导致写入失败之后。
            band_info = {
                "x_band_low": "",
                "x_band_high": "",
                "x_kurtosis": "",
                "y_band_low": "",
                "y_band_high": "",
                "y_kurtosis": "",
                "z_band_low": "",
                "z_band_high": "",
                "z_kurtosis": "",
            }
        else:
            rgb_image, band_info = build_rgb_envelope_stft(segment_xyz, fs, image_size)
            save_rgb_image(rgb_image, image_path, overwrite=overwrite)

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "label": field_info["label"],
                "label_name": field_info["label_name"],
                "fault_prefix": field_info["fault_prefix"],
                "condition": field_info["condition"],
                "speed_hz": field_info["speed_hz"],
                "load_voltage": field_info["load_voltage"],
                "sampling_rate": fs,
                "source_file": mat_path.name,
                "source_field": field_name,
                "segment_index": segment_index,
                "segment_start": start,
                "segment_end": end,
                "x_band_low": band_info["x_band_low"],
                "x_band_high": band_info["x_band_high"],
                "x_kurtosis": band_info["x_kurtosis"],
                "y_band_low": band_info["y_band_low"],
                "y_band_high": band_info["y_band_high"],
                "y_kurtosis": band_info["y_kurtosis"],
                "z_band_low": band_info["z_band_low"],
                "z_band_high": band_info["z_band_high"],
                "z_kurtosis": band_info["z_kurtosis"],
            }
        )

    return rows


def write_metadata_file(rows: list[dict[str, str | int | float]], metadata_path: Path) -> Path:
    """将 metadata 记录写入指定 CSV 文件。"""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("没有生成任何 metadata 记录。")

    fieldnames = list(rows[0].keys())
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def write_metadata(rows: list[dict[str, str | int | float]], metadata_path: Path) -> Path:
    """写入 metadata.csv；如果目标文件被占用，则自动写入备用文件。"""
    try:
        return write_metadata_file(rows, metadata_path)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = metadata_path.with_name(f"{metadata_path.stem}_{timestamp}{metadata_path.suffix}")
        print(f"[警告] 无法写入 metadata 文件，可能正在被 Excel/WPS/编辑器占用：{metadata_path}")
        print(f"[警告] 将改写到备用文件：{fallback_path}")
        return write_metadata_file(rows, fallback_path)


def main() -> None:
    """主函数：批量生成包络 STFT 图像和 metadata。"""
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    image_dir = args.image_dir.resolve()
    metadata_path = args.metadata_path.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_dir}")

    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"目录中没有找到 .mat 文件：{raw_dir}")

    print(f"原始数据目录：{raw_dir}")
    print(f".mat 文件数量：{len(mat_files)}")
    print(f"图像输出目录：{image_dir}")
    print(f"metadata 输出路径：{metadata_path}")
    print("处理方法：简化谱峭度频带搜索 -> 带通滤波 -> Hilbert 包络 -> STFT -> RGB 融合")

    all_rows = []
    for mat_path in mat_files:
        print("=" * 80)
        print(f"开始处理：{mat_path.name}")
        rows = process_mat_file(
            mat_path=mat_path,
            image_dir=image_dir,
            fs=args.sampling_rate,
            segment_len=args.segment_len,
            hop_len=args.hop_len,
            image_size=args.image_size,
            max_segments_per_file=args.max_segments_per_file,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        print(f"完成：{mat_path.name}，生成样本数：{len(rows)}")

    final_metadata_path = write_metadata(all_rows, metadata_path)
    print("=" * 80)
    print(f"全部完成，样本总数：{len(all_rows)}")
    print(f"metadata 已保存：{final_metadata_path}")


if __name__ == "__main__":
    main()
