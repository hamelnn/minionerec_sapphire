# -*- coding: utf-8 -*-
"""DPO checkpoint 推荐选模（方案第 9.2 节）。

在每个保存的 adapter 上，对固定验证交互子集做约束 Top-20 解码，按 SID 级
NDCG@10 排序，并把最佳 adapter 复制到 --save_best_dir（默认 <output_dir>/best_adapter）。

- 测试集不参与本流程；只用验证交互。
- 附带碰撞分层（target 是否属于碰撞 SID），SID 消歧前不作精确物品命中解释。
- 20 beams 只支持 @20 以内的指标。

用法：
python scripts_win/eval_dpo_checkpoints.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --checkpoints_dir outputs/dpo_05b_12gb \
  --valid_csv data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --index_path data/Amazon/index/Industrial_and_Scientific.index.json \
  --sample 1000
"""
import json
import math
import os
import shutil
import sys

import fire
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo_data import (  # noqa: E402
    build_prompt, load_sid_catalog, normalize_sid, parse_list_field,
)
from scripts_win.build_dpo_pairs import build_hash_dict, get_hash  # noqa: E402

TOPKS = [5, 10, 20]


def evaluate_adapter(model, tokenizer, hash_dict, samples, num_beams=20,
                     batch_size=4, max_new_tokens=16):
    """对 (prompt, target_sid, in_collision) 样本列表做约束解码并计算 SID 级指标。"""
    from transformers import GenerationConfig, LogitsProcessorList
    from LogitProcessor import ConstrainedLogitsProcessor

    gc = GenerationConfig(num_beams=num_beams, num_return_sequences=num_beams,
                          length_penalty=0.0, max_new_tokens=max_new_tokens,
                          do_sample=False, top_k=None, top_p=None,
                          pad_token_id=tokenizer.eos_token_id,
                          eos_token_id=tokenizer.eos_token_id)
    device = next(model.parameters()).device
    hr = {k: 0 for k in TOPKS}
    ndcg = {k: 0.0 for k in TOPKS}
    hr_c = {k: 0 for k in TOPKS}
    ndcg_c = {k: 0.0 for k in TOPKS}
    n_coll = 0
    total, valid_cnt, cand_total, dup_samples = 0, 0, 0, 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            enc = [tokenizer(p, add_special_tokens=False).input_ids[-256:]
                   for p, _, _ in chunk]
            max_len = max(len(e) for e in enc)
            input_ids = torch.tensor(
                [[tokenizer.eos_token_id] * (max_len - len(e)) + e for e in enc]).to(device)
            attn = torch.tensor(
                [[0] * (max_len - len(e)) + [1] * len(e) for e in enc]).to(device)
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=lambda b, ids: hash_dict.get(get_hash(ids), []),
                num_beams=num_beams, base_model="qwen", eos_token_id=tokenizer.eos_token_id)
            out = model.generate(input_ids, attention_mask=attn, generation_config=gc,
                                 do_sample=False, logits_processor=LogitsProcessorList([clp]),
                                 return_dict_in_generate=True)
            texts = tokenizer.batch_decode(out.sequences[:, max_len:], skip_special_tokens=True)
            for i, (_, target, in_collision) in enumerate(chunk):
                cands = [t.split("Response:\n")[-1].strip()
                         for t in texts[i * num_beams:(i + 1) * num_beams]]
                total += 1
                cand_total += len(cands)
                valid_cnt += sum(1 for c in cands if c)
                if len(set(cands)) < len(cands):
                    dup_samples += 1
                rank = next((j for j, c in enumerate(cands) if c == target), None)
                if in_collision:
                    n_coll += 1
                hit = {"hr": hr_c if in_collision else hr, "ndcg": ndcg_c if in_collision else ndcg}
                if rank is not None:
                    for k in TOPKS:
                        if rank < k:
                            hit["hr"][k] += 1
                            hit["ndcg"][k] += 1.0 / math.log2(rank + 2)
    n = max(total, 1)
    nc = max(n_coll, 1)
    return {
        "n_eval": total, "n_collision_targets": n_coll,
        **{f"HR@{k}": hr[k] / n for k in TOPKS},
        **{f"NDCG@{k}": ndcg[k] / n for k in TOPKS},
        "valid_rate": valid_cnt / max(cand_total, 1),
        "dup_sample_rate": dup_samples / n,
        **{f"HR@{k}_collision": hr_c[k] / nc for k in TOPKS},
        **{f"NDCG@{k}_collision": ndcg_c[k] / nc for k in TOPKS},
    }


def load_valid_samples(valid_csv, sample, seed=42):
    import pandas as pd
    import random as rd
    df = pd.read_csv(valid_csv)
    if sample > 0 and sample < len(df):
        df = df.sample(sample, random_state=seed)
    rows = []
    for _, row in df.iterrows():
        hist = [normalize_sid(s) for s in parse_list_field(row["history_item_sid"])]
        rows.append((build_prompt(hist), normalize_sid(row["item_sid"]), None))
    return rows


def main(sft_path, checkpoints_dir, valid_csv, info_file, index_path,
         sample=1000, num_beams=20, batch_size=4, seed=42,
         save_best_dir=None, result_csv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    ckpts = []
    if os.path.isdir(checkpoints_dir):
        for name in sorted(os.listdir(checkpoints_dir)):
            p = os.path.join(checkpoints_dir, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"no adapter checkpoints under {checkpoints_dir}")
    print(f"found {len(ckpts)} adapters")

    catalog = load_sid_catalog(index_path, info_file)
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    hash_dict = build_hash_dict(tokenizer, info_file)

    samples = load_valid_samples(valid_csv, sample, seed)
    # 碰撞标记
    samples = [(p, t, t in catalog["collision_sids"]) for p, t, _ in samples]
    print(f"valid subset: {len(samples)} interactions "
          f"({sum(1 for _,_,c in samples if c)} collision targets)")

    rows = []
    for ck in ckpts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, ck).to("cuda").eval()
        m = evaluate_adapter(model, tokenizer, hash_dict, samples,
                             num_beams=num_beams, batch_size=batch_size)
        m["checkpoint"] = ck
        m["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        rows.append(m)
        print(json.dumps(m, ensure_ascii=False))
        del model, base
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["NDCG@10"], reverse=True)
    best = rows[0]
    print(f"\nBEST by NDCG@10: {best['checkpoint']} (NDCG@10={best['NDCG@10']:.4f})")

    if save_best_dir is None:
        save_best_dir = os.path.join(checkpoints_dir, "best_adapter")
    if os.path.abspath(best["checkpoint"]) != os.path.abspath(save_best_dir):
        os.makedirs(save_best_dir, exist_ok=True)
        for fn in os.listdir(best["checkpoint"]):
            if fn in ("adapter_config.json", "adapter_model.safetensors",
                      "README.md") or fn.startswith(("tokenizer", "special_tokens",
                                                     "chat_template", "added_tokens",
                                                     "vocab", "merges")):
                shutil.copy2(os.path.join(best["checkpoint"], fn),
                             os.path.join(save_best_dir, fn))
        print(f"best adapter copied to {save_best_dir}")

    result_csv = result_csv or os.path.join(checkpoints_dir, "checkpoint_eval.csv")
    keys = list(rows[0].keys())
    with open(result_csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"table written to {result_csv}")


if __name__ == "__main__":
    fire.Fire(main)
