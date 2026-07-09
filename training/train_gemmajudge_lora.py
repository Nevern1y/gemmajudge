"""LoRA/QLoRA SFT entrypoint for a GemmaJudge judge adapter on AMD ROCm.

This script intentionally avoids TRL-specific trainer APIs so the pipeline is less
sensitive to TRL version churn. It trains only on the Gemma `model` turn in canonical
GemmaJudge JSONL produced by `scripts/build_judge_dataset.py`.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

RESPONSE_MARKER = "<start_of_turn>model\n"


class JudgeSFTDataset:
    def __init__(self, path: Path, tokenizer: Any, max_seq_len: int) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - setup error
            raise SystemExit("Install torch before running training") from exc

        self._torch = torch
        self.items: list[dict[str, list[int]]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                marker_idx = text.rfind(RESPONSE_MARKER)
                if marker_idx == -1:
                    raise ValueError(f"Gemma response marker not found in {path}")
                label_start = marker_idx + len(RESPONSE_MARKER)
                encoded = tokenizer(
                    text,
                    max_length=max_seq_len,
                    truncation=True,
                    padding=False,
                    add_special_tokens=False,
                )
                prefix_ids = tokenizer(
                    text[:label_start], add_special_tokens=False, truncation=False
                )["input_ids"]
                labels = list(encoded["input_ids"])
                mask_until = min(len(prefix_ids), len(labels))
                labels[:mask_until] = [-100] * mask_until
                if all(label == -100 for label in labels):
                    continue
                self.items.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {key: self._torch.tensor(value) for key, value in item.items()}


class DataCollator:
    def __init__(self, tokenizer: Any) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - setup error
            raise SystemExit("Install torch before running training") from exc
        self._torch = torch
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in features
        ]
        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = []
        for item in features:
            label = item["labels"]
            pad_len = max_len - label.shape[0]
            labels.append(
                self._torch.cat(
                    [label, self._torch.full((pad_len,), -100, dtype=label.dtype)]
                )
            )
        batch["labels"] = self._torch.stack(labels)
        return batch


def build_training_args(args: argparse.Namespace) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(args.out),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "bf16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps
    return TrainingArguments(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GemmaJudge LoRA adapter on judge JSONL.")
    parser.add_argument("--model_id", default="google/gemma-2-9b-it")
    parser.add_argument("--train", type=Path, default=Path("data/judge_train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/judge_val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/gemmajudge-lora"))
    parser.add_argument("--merge_out", type=Path)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=25)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
    except ImportError as exc:  # pragma: no cover - setup error
        raise SystemExit(
            "Install training deps first: torch transformers peft accelerate "
            "and optionally bitsandbytes-rocm for --load_in_4bit"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=args.attn_implementation,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = JudgeSFTDataset(args.train, tokenizer, args.max_seq_len)
    eval_dataset = JudgeSFTDataset(args.validation, tokenizer, args.max_seq_len)
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise SystemExit("Train and validation datasets must both contain label tokens")

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(args.out)

    if args.merge_out:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_implementation,
        )
        merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        args.merge_out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merge_out)
        tokenizer.save_pretrained(args.merge_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
