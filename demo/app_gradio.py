"""
SUFD-QwenVL 故障诊断 Gradio Demo。

功能：
    1. 上传包络 STFT 图像
    2. 输入工况参数
    3. 使用 Qwen2.5-VL-3B + LoRA best checkpoint 预测故障类型
    4. 根据预测类型生成完整中文诊断报告

推荐在 AutoDL / Linux 训练环境中运行：
    conda activate llamafactory311
    cd /root/autodl-tmp/sufd-qwenvl-fault-diagnosis
    python demo/app_gradio.py --server-name 0.0.0.0 --server-port 6008
"""

from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL = os.getenv("SUFD_BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
DEFAULT_ADAPTER_PATH = os.getenv(
    "SUFD_ADAPTER_PATH",
    "/root/autodl-tmp/LLaMA-Factory/saves/qwen2_5vl_3b/lora/sufd_class_only_r16_ep5/checkpoint-2200",
)

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

APP_ARGS: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    """解析启动参数。"""
    parser = argparse.ArgumentParser(description="SUFD-QwenVL Gradio Demo")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Qwen2.5-VL 基座模型路径")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH, help="LoRA adapter 路径")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Gradio 监听地址")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio 监听端口")
    parser.add_argument("--share", action="store_true", help="是否创建 Gradio share 链接")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="模型生成最大 token 数")
    return parser.parse_args()


def build_prompt(speed_hz: float, load_voltage: float, sampling_rate: int) -> str:
    """构造与 class_only 训练阶段一致的诊断提示。"""
    return (
        "你是一名工业设备故障诊断助手。请根据给定的轴承包络时频图和工况参数，判断当前轴承状态。\n"
        "该图像由行星齿轮箱 X、Y、Z 三方向振动信号生成：先通过简化谱峭度方法选择冲击显著频带，"
        "再进行带通滤波、Hilbert 包络提取和 STFT 时频分析，最后融合为 RGB 图像。\n"
        "工况参数：\n"
        "- 数据集：SUFD 齿轮箱/轴承故障数据集\n"
        f"- 采样率：{sampling_rate} Hz\n"
        f"- 转速工况：{speed_hz:g} Hz\n"
        f"- 负载/电压工况：{load_voltage:g} V\n"
        "请只输出故障类型，不要输出诊断依据或维护建议。\n"
        "请严格按照以下格式输出：\n"
        "故障类型：<滚动体故障/内圈故障/外圈故障/混合故障/正常状态>"
    )


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


def build_report(pred_label: str, speed_hz: float, load_voltage: float) -> str:
    """根据预测标签和工况生成完整诊断报告。"""
    fault_type = LABEL_NAME_MAP[pred_label]
    template = REPORT_TEMPLATE[pred_label]
    basis = f"{template['basis']}当前工况为 {speed_hz:g} Hz、{load_voltage:g} V。"

    return (
        f"故障类型：{fault_type}\n"
        f"诊断依据：{basis}\n"
        f"维护建议：{template['advice']}"
    )


@lru_cache(maxsize=1)
def load_model():
    """延迟加载模型，首次点击诊断按钮时执行。"""
    if APP_ARGS is None:
        raise RuntimeError("APP_ARGS 尚未初始化。")

    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "缺少推理依赖，请在训练环境中安装 torch、transformers、peft、qwen-vl-utils 和 gradio。"
        ) from exc

    base_model = APP_ARGS.base_model
    adapter_path = APP_ARGS.adapter_path

    if not Path(base_model).exists():
        raise FileNotFoundError(f"基座模型路径不存在：{base_model}")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在：{adapter_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return model, processor, process_vision_info, torch, device


def run_model(image_path: str, speed_hz: float, load_voltage: float, sampling_rate: int) -> str:
    """调用 Qwen2.5-VL + LoRA 进行 class_only 推理。"""
    model, processor, process_vision_info, torch, device = load_model()
    prompt = build_prompt(speed_hz, load_voltage, sampling_rate)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=APP_ARGS.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


def diagnose(image_path: str | None, speed_hz: float, load_voltage: float, sampling_rate: int):
    """Gradio 点击按钮后的诊断函数。"""
    if not image_path:
        return "未上传图像", "", "请先上传包络 STFT 图像。"

    try:
        raw_prediction = run_model(image_path, speed_hz, load_voltage, int(sampling_rate))
        pred_label = extract_fault_label(raw_prediction)
        report = build_report(pred_label, speed_hz, load_voltage)
        fault_type = LABEL_NAME_MAP[pred_label]
        return fault_type, raw_prediction, report
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        return "推理失败", error_text, error_text


def find_examples() -> list[list[str]]:
    """自动查找少量本地示例图像。"""
    image_dir = PROJECT_ROOT / "data" / "processed" / "images" / "all"
    if not image_dir.exists():
        return []

    examples = []
    for path in sorted(image_dir.glob("*.png"))[:5]:
        examples.append([str(path), 20, 0, 4096])
    return examples


def create_demo() -> gr.Blocks:
    """创建 Gradio 界面。"""
    with gr.Blocks(title="SUFD-QwenVL Fault Diagnosis") as demo:
        gr.Markdown("# SUFD-QwenVL 工业设备故障诊断")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="包络 STFT 图像")
                with gr.Row():
                    speed_input = gr.Number(value=20, label="转速 Hz", precision=0)
                    load_input = gr.Number(value=0, label="负载/电压 V", precision=0)
                    sampling_input = gr.Number(value=4096, label="采样率 Hz", precision=0)
                diagnose_button = gr.Button("开始诊断", variant="primary")

            with gr.Column(scale=1):
                fault_output = gr.Textbox(label="故障类型", lines=1)
                raw_output = gr.Textbox(label="模型原始输出", lines=2)
                report_output = gr.Textbox(label="最终诊断报告", lines=8)

        examples = find_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[image_input, speed_input, load_input, sampling_input],
                label="示例",
            )

        diagnose_button.click(
            fn=diagnose,
            inputs=[image_input, speed_input, load_input, sampling_input],
            outputs=[fault_output, raw_output, report_output],
        )

    return demo


def main() -> None:
    """启动 Gradio。"""
    global APP_ARGS
    APP_ARGS = parse_args()
    demo = create_demo()
    demo.queue()
    demo.launch(server_name=APP_ARGS.server_name, server_port=APP_ARGS.server_port, share=APP_ARGS.share)


if __name__ == "__main__":
    main()
