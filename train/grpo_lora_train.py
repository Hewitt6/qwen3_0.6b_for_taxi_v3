import os
import argparse
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from .reward_functions import build_reward_mapping, make_reward_fn


def get_args():
    p = argparse.ArgumentParser(description="GRPO + LoRA finetuning for Qwen using Taxi-v3 JSONL data")
    p.add_argument("--cleaned_jsonl", type=str, default=os.path.join(os.path.dirname(__file__), "cleaned_grpo_data_fixed.jsonl"),
                   help="Path to cleaned_grpo_data_fixed.jsonl")
    p.add_argument("--eval_jsonl", type=str, default=os.path.join(os.path.dirname(__file__), "grpo_evaluation_complete.jsonl"),
                   help="Path to grpo_evaluation_complete.jsonl")
    p.add_argument("--model_name", type=str, default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.8B-Instruct"),
                   help="Base model id on HF hub (set to the exact Qwen3 1.7B if available)")
    p.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"))
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=4, help="Generations per prompt for GRPO grouping")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_completion_len", type=int, default=256)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build reward mapping and reward function
    mapping = build_reward_mapping(args.cleaned_jsonl, args.eval_jsonl)
    reward_fn = make_reward_fn(mapping)

    # Load data
    prompts: List[str] = list(mapping.keys())
    train_ds = Dataset.from_dict({"prompt": prompts})

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16,
        device_map="auto"
    )

    # LoRA config targeting Qwen-style modules
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # GRPO trainer config
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=200,
        bf16=args.bf16,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_cfg,
        reward_funcs=[reward_fn],
        train_dataset=train_ds,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
