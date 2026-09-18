# -*- coding: utf-8 -*-
"""方案第 6 节：CPU 上合并 RL LoRA adapter 到 SFT 底座。
训练进程结束后单独运行，避免与训练争抢显存。
用法: python scripts_win/merge_adapter.py --sft_path ... --adapter_path ... --merged_path ...
"""
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(sft_path: str, adapter_path: str, merged_path: str):
    base = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(merged_path)
    AutoTokenizer.from_pretrained(sft_path).save_pretrained(merged_path)
    print(f"merged model saved to {merged_path}")


if __name__ == "__main__":
    fire.Fire(merge)
