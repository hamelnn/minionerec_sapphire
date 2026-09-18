# -*- coding: utf-8 -*-
"""LoRA DPO 训练入口（方案第 6、7 节）。

- 底座：现有全参数 SFT checkpoint（冻结）；参考策略 = 禁用 adapter 后的同一底座
  （ref_model=None + PEFT，TRL 官方支持路径），不加载第二份模型。
- 新建 DPO LoRA adapter（r=16/alpha=32，七个投影层）；首版不设 modules_to_save，
  embedding/lm_head 冻结，保证参考策略固定。
- precompute_ref_log_probs=True：训练前一次性预计算参考 log-prob。
- 梯度检查点默认关闭（PEFT 冻结底座 + reentrant checkpoint 会断梯度，见
  report/复现实验记录.md 3.1 节；如需开启先单独验证 use_reentrant 行为）。
- 保存的是训练结束时的 adapter，不等于最佳推荐 checkpoint；选模见
  scripts_win/eval_dpo_checkpoints.py。
"""
import json
import os
import subprocess
import time

import fire
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main(sft_path,
         train_jsonl,
         valid_jsonl,
         output_dir="outputs/dpo_05b_12gb",
         seed=42,
         learning_rate=5e-6,
         beta=0.1,
         num_train_epochs=1,
         per_device_train_batch_size=1,
         per_device_eval_batch_size=1,
         gradient_accumulation_steps=32,
         max_prompt_length=256,
         max_completion_length=16,
         max_length=272,
         precompute_ref_batch_size=1,
         eval_steps=100,
         save_steps=100,
         save_total_limit=3,
         max_steps=-1,
         gradient_checkpointing=False,
         loss_type="sigmoid",
         optim="adamw_torch"):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    datasets = load_dataset("json", data_files={"train": train_jsonl,
                                                "validation": valid_jsonl})
    cols = ["prompt", "chosen", "rejected"]
    train_pairs = datasets["train"].select_columns(cols)
    valid_pairs = datasets["validation"].select_columns(cols)
    print(f"pairs: train={len(train_pairs)} valid={len(valid_pairs)}")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 注意：save_total_limit 的轮转可能删除尚未评估的 checkpoint；若采用
    # "训练后统一选模"，应调大该值或把 eval_dpo_checkpoints 的采样间隔与之对齐。
    args = DPOConfig(
        output_dir=output_dir,
        loss_type=loss_type,
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_length=max_length,
        truncation_mode="keep_end",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=precompute_ref_batch_size,
        reference_free=False,
        sync_ref_model=False,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        optim=optim,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=10,
        report_to="none",
        seed=seed,
        data_seed=seed,
        max_steps=max_steps,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,
        train_dataset=train_pairs,
        eval_dataset=valid_pairs,
    )
    trainer.model.print_trainable_parameters()
    assert trainer.ref_model is None, "PEFT 路径参考策略应通过 disable_adapter 获得"
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert all(("lora" in n) for n in trainable), f"存在非 LoRA 可训练参数：{trainable[:5]}"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    manifest = {
        "git_commit": commit,
        "sft_path": sft_path,
        "sft_model_sha256_16": sha16(os.path.join(sft_path, "model.safetensors")),
        "train_jsonl": train_jsonl, "train_jsonl_sha256_16": sha16(train_jsonl),
        "valid_jsonl": valid_jsonl, "valid_jsonl_sha256_16": sha16(valid_jsonl),
        "config": {k: getattr(args, k) for k in [
            "learning_rate", "beta", "num_train_epochs", "max_steps",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "max_prompt_length", "max_completion_length", "max_length",
            "precompute_ref_log_probs", "loss_type", "optim", "seed"]},
        "lora": {k: (sorted(v) if isinstance(v, set) else v)
                 for k, v in (lora_config.to_dict() if hasattr(lora_config, "to_dict")
                              else {}).items()},
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    runtime = time.time() - t0
    print(f"peak allocated GiB: {peak_gib:.3f} | train runtime: {runtime:.1f}s")

    with open(os.path.join(output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"history": trainer.state.log_history, "global_step": trainer.state.global_step,
                   "peak_allocated_gib": round(peak_gib, 3), "runtime_s": round(runtime, 1)},
                  f, ensure_ascii=False, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"adapter saved to {output_dir}（训练末点，非最佳；选模见 eval_dpo_checkpoints.py）")


if __name__ == "__main__":
    fire.Fire(main)
