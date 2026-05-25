"""
使用 TRL GRPO 对 SUFD-QwenVL 进行强化学习微调。

本脚本不依赖 LLaMA-Factory 的 stage 配置，而是直接使用 Hugging Face TRL：

1. 加载 Qwen2.5-VL-3B-Instruct 基座模型。
2. 加载已经训练好的 SFT LoRA checkpoint-2200。
3. 读取 SUFD train.csv 中的图像路径和故障标签。
4. 让模型生成完整诊断报告。
5. 使用规则奖励函数评估：
   - 故障类型是否正确；
   - 是否包含“故障类型 / 诊断依据 / 维护建议”三段结构；
   - 报告是否较为简洁、且没有同时输出多个互相冲突的故障类型。

注意：
    GRPO 是在线生成式训练，比 SFT/DPO 更吃显存。建议先用 --max-steps 20 跑通，
    再扩大训练步数。
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, Image as DatasetImage
from peft import LoraConfig, PeftModel
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer


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

DEFAULT_PROJECT_ROOT = Path("/root/autodl-tmp/sufd-qwenvl-fault-diagnosis")
DEFAULT_TRAIN_CSV = DEFAULT_PROJECT_ROOT / "data" / "processed" / "metadata" / "train.csv"
DEFAULT_EVAL_CSV = DEFAULT_PROJECT_ROOT / "data" / "processed" / "metadata" / "test.csv"
DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
DEFAULT_SFT_ADAPTER = (
    Path("/root/autodl-tmp/LLaMA-Factory")
    / "saves"
    / "qwen2_5vl_3b"
    / "lora"
    / "sufd_class_only_r16_ep5"
    / "checkpoint-2200"
)
DEFAULT_OUTPUT_DIR = (
    Path("/root/autodl-tmp/LLaMA-Factory")
    / "saves"
    / "qwen2_5vl_3b"
    / "lora"
    / "sufd_grpo_from_ckpt2200"
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SUFD Qwen2.5-VL GRPO training script")

    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft-adapter-path", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和奖励函数，不启动训练")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.0, help="GRPO KL 系数；默认 0 与 TRL 默认一致")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument(
        "--device-map",
        choices=["auto", "none"],
        default="auto",
        help="单卡 AutoDL 推荐 auto；如 accelerate 报 device_map 问题可改 none",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj")

    return parser.parse_args()


def validate_grpo_batch_args(args: argparse.Namespace) -> None:
    """检查 GRPO batch size 与 num_generations 是否兼容。"""
    train_global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if train_global_batch % args.num_generations != 0:
        raise ValueError(
            "GRPO 训练 batch 参数不兼容："
            f"per_device_train_batch_size({args.per_device_train_batch_size}) * "
            f"gradient_accumulation_steps({args.gradient_accumulation_steps}) = "
            f"{train_global_batch}，必须能被 num_generations({args.num_generations}) 整除。"
        )

    if args.max_eval_samples != 0 and args.per_device_eval_batch_size % args.num_generations != 0:
        raise ValueError(
            "GRPO 评估 batch 参数不兼容："
            f"per_device_eval_batch_size({args.per_device_eval_batch_size}) "
            f"必须能被 num_generations({args.num_generations}) 整除。"
            "可设置 --per-device-eval-batch-size 2，或用 --max-eval-samples 0 关闭评估。"
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 metadata CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"CSV 不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    return rows


def build_prompt(row: dict[str, str]) -> list[dict[str, str]]:
    """构造 GRPO conversational prompt。图像由 dataset 的 image 字段提供。"""
    speed_hz = row.get("speed_hz", "")
    load_voltage = row.get("load_voltage", "")
    sampling_rate = row.get("sampling_rate", "")

    system_prompt = (
        "你是工业设备故障诊断助手。请根据行星齿轮箱 X/Y/Z 三方向振动信号生成的"
        "RGB 包络 STFT 图像，判断设备故障类型，并给出简短诊断依据和维护建议。"
    )
    user_prompt = (
        "请诊断这张包络时频图对应的设备状态。\n"
        "已知信息：\n"
        "- 信号来源：行星齿轮箱 X/Y/Z 三方向振动\n"
        f"- 转速工况：{speed_hz} Hz\n"
        f"- 负载/电压工况：{load_voltage} V\n"
        f"- 采样率：{sampling_rate} Hz\n"
        "请严格按照以下格式输出：\n"
        "故障类型：<正常状态/滚动体故障/内圈故障/外圈故障/混合故障>\n"
        "诊断依据：<结合包络时频图和工况给出简短说明>\n"
        "维护建议：<给出简短检查或维护建议>"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def resolve_image_path(project_root: Path, image_path: str) -> str:
    """把 CSV 中的相对图像路径转换为绝对路径。"""
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"图像不存在：{path}")
    return str(path)


def build_dataset(
    csv_path: Path,
    project_root: Path,
    max_samples: int,
    seed: int,
) -> Dataset:
    """从 metadata CSV 构造 TRL VLM 数据集。"""
    rows = read_csv_rows(csv_path)
    if max_samples > 0 and max_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)

    records: list[dict[str, Any]] = []
    for row in rows:
        label = row["label"]
        if label not in LABEL_NAME_MAP:
            raise ValueError(f"未知标签：{label}")

        records.append(
            {
                "prompt": build_prompt(row),
                "image": resolve_image_path(project_root, row["image_path"]),
                "label": label,
                "label_name": LABEL_NAME_MAP[label],
                "sample_id": row.get("sample_id", ""),
                "speed_hz": row.get("speed_hz", ""),
                "load_voltage": row.get("load_voltage", ""),
                "source_field": row.get("source_field", ""),
            }
        )

    dataset = Dataset.from_list(records)
    return dataset.cast_column("image", DatasetImage(decode=True))


def completion_to_text(completion: Any) -> str:
    """兼容 TRL 返回的字符串或 conversational message 格式。"""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    if isinstance(completion, list):
        parts = []
        for item in completion:
            parts.append(completion_to_text(item))
        return "\n".join(part for part in parts if part)

    return str(completion)


def extract_fault_label(text: str) -> str:
    """从模型输出中抽取故障标签。"""
    normalized = text.replace(" ", "").replace("\t", "")

    match = re.search(r"故障类型\s*[:：]\s*([^\n。；;]+)", text)
    if match:
        fault_type_text = match.group(1).strip()
        for keyword, label in FAULT_TEXT_TO_LABEL.items():
            if keyword in fault_type_text:
                return label

    for keyword, label in FAULT_TEXT_TO_LABEL.items():
        if keyword in normalized:
            return label

    return "INVALID"


def label_correctness_reward(completions: list[Any], label: list[str], **kwargs: Any) -> list[float]:
    """主奖励：故障类型正确性。"""
    rewards = []
    for completion, gold_label in zip(completions, label, strict=True):
        pred_label = extract_fault_label(completion_to_text(completion))
        if pred_label == gold_label:
            rewards.append(2.0)
        elif pred_label == "INVALID":
            rewards.append(-1.0)
        else:
            rewards.append(-1.5)
    return rewards


def report_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """格式奖励：鼓励三段式完整诊断报告。"""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        score = 0.0
        if "故障类型" in text:
            score += 0.25
        if "诊断依据" in text:
            score += 0.25
        if "维护建议" in text:
            score += 0.25
        if re.search(r"故障类型\s*[:：]", text):
            score += 0.15
        if text.count("\n") >= 2:
            score += 0.10
        rewards.append(score)
    return rewards


def report_quality_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """轻量质量奖励：鼓励简洁且避免多类别互相冲突。"""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        labels_mentioned = {
            label for keyword, label in FAULT_TEXT_TO_LABEL.items() if keyword in text
        }

        score = 0.0
        if 30 <= len(text) <= 260:
            score += 0.25
        elif len(text) > 420:
            score -= 0.20

        if len(labels_mentioned) <= 1:
            score += 0.20
        else:
            score -= 0.30

        if "无法判断" in text or "不确定" in text:
            score -= 0.20

        rewards.append(score)
    return rewards


def load_vlm_model(args: argparse.Namespace):
    """加载 Qwen2.5-VL 基座模型，并可选加载 SFT LoRA adapter。"""
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    model = model_cls.from_pretrained(str(args.model_name_or_path), **model_kwargs)
    model.config.use_cache = False

    if args.sft_adapter_path and args.sft_adapter_path.exists():
        print(f"加载 SFT LoRA adapter：{args.sft_adapter_path}")
        model = PeftModel.from_pretrained(model, str(args.sft_adapter_path), is_trainable=True)
        model.print_trainable_parameters()
        return model, None

    print("未找到 SFT adapter，将从基座模型新建 LoRA adapter。")
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return model, peft_config


def build_processor(model_path: Path) -> AutoProcessor:
    """加载 Qwen2.5-VL processor。"""
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def run_dry_check(train_dataset: Dataset) -> None:
    """启动训练前检查一条样本和奖励函数。"""
    first = train_dataset[0]
    fake_good = [
        {
            "role": "assistant",
            "content": (
                f"故障类型：{first['label_name']}\n"
                "诊断依据：包络时频图中存在与该类别一致的冲击和调制特征。\n"
                "维护建议：建议结合现场巡检和包络谱进一步确认故障程度。"
            ),
        }
    ]
    fake_bad = [
        {
            "role": "assistant",
            "content": (
                "故障类型：正常状态\n"
                "诊断依据：未发现明显异常。\n"
                "维护建议：保持常规监测。"
            ),
        }
    ]
    completions = [fake_good, fake_bad]
    labels = [first["label"], first["label"]]

    print("样本检查：")
    print(f"sample_id: {first['sample_id']}")
    print(f"label: {first['label']} / {first['label_name']}")
    print(f"image: {first['image']}")
    print("label_correctness_reward:", label_correctness_reward(completions, labels))
    print("report_format_reward:", report_format_reward(completions))
    print("report_quality_reward:", report_quality_reward(completions))


def main() -> None:
    """主入口。"""
    args = parse_args()
    validate_grpo_batch_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = build_dataset(
        csv_path=args.train_csv,
        project_root=args.project_root,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    eval_dataset = None
    if args.eval_csv.exists() and args.max_eval_samples != 0:
        eval_dataset = build_dataset(
            csv_path=args.eval_csv,
            project_root=args.project_root,
            max_samples=args.max_eval_samples,
            seed=args.seed,
        )

    print(f"训练样本数：{len(train_dataset)}")
    print(f"评估样本数：{len(eval_dataset) if eval_dataset is not None else 0}")
    run_dry_check(train_dataset)

    if args.dry_run:
        print("dry-run 完成，未启动 GRPO 训练。")
        return

    model, peft_config = load_vlm_model(args)
    processor = build_processor(args.model_name_or_path)

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[
            label_correctness_reward,
            report_format_reward,
            report_quality_reward,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"GRPO 训练完成，模型已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
