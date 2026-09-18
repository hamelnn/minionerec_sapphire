# -*- coding: utf-8 -*-
"""构造 DPO 偏好对（方案第 3、4 节）。

流程：固定 SFT checkpoint 在训练交互上做约束 Top-K 候选生成（断点可续），
再按方案 4.4 的过滤顺序构造偏好对，输出标准 3 字段 JSONL + 审计记录 + manifest。

用法（方案第 11 节）：
python scripts_win/build_dpo_pairs.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --split train --sample 6000 --num_beams 20 \
  --negatives_per_example 2 --seed 42 --output_dir outputs/dpo_pairs

注意：
- 每次 generate 新建 ConstrainedLogitsProcessor（有状态 step 计数，不可跨批复用）；
- do_sample=False 必须作为 generate 的显式 kwargs 传入（transformers>=4.50 语义）；
- 候选、偏好对均与 sample_id 绑定，重启跳过已完成样本。
"""
import os
import sys
import json
import hashlib
import random
import subprocess

import fire
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo_data import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION, PairBuildStats, audit_record, build_pairs_for_interaction,
    build_prompt, completion_text, history_target_map_from_train, load_interactions,
    load_sid_catalog, normalize_sid, sha256_file,
)

CATEGORY_DEFAULT = "Industrial_and_Scientific"
DATASET_DEFAULT = "Industrial_and_Scientific_5_2016-10-2018-11"


def get_hash(x):
    return "-".join(str(_) for _ in x)


def build_hash_dict(tokenizer, info_path):
    """合法 SID 前缀树（与 evaluate.py 一致的口径）。"""
    with open(info_path, encoding="utf-8") as f:
        info = f.readlines()
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
    hash_dict = {}
    for entry in info_semantic:
        ID = tokenizer(entry).input_ids
        ID.append(tokenizer.eos_token_id)
        for i in range(3, len(ID)):
            key = get_hash(ID[:3]) if i == 3 else get_hash(ID[3:i])
            hash_dict.setdefault(key, set()).add(ID[i])
    return {k: list(v) for k, v in hash_dict.items()}


def check_token_boundaries(tokenizer, csv_path, template, n=5):
    """方案第 5 节：prompt/completion 拼接的 token 边界校验。

    校验三点：
    1. 模板重建字符串与 data.py 逐字符一致（EvalSidDataset/SidSFTDataset 的
       instruction + BaseDataset.generate_prompt）；
    2. t(instruction)+t(body) 与 t(instruction+body) 的 junction 是否一致；
    3. t(prompt) 末尾必须是 "### Response:\\n" 的末 token，completion 编码
       不受 prompt 影响（无跨边界 merge），拼接解码回原文一致。
    返回 dict 报告；junction 不一致只告警（记录到 manifest），completion 边界
    不一致则抛错拒绝构造。
    """
    import pandas as pd

    from dpo_data import INSTRUCTION, PROMPT_TEMPLATES, _USER_INPUT

    df = pd.read_csv(csv_path).head(n)
    report = {"rows_checked": 0, "template_string_match": True, "junction_match": True,
              "completion_boundary_ok": True, "eos_after_completion": True}
    from dpo_data import parse_list_field
    for _, row in df.iterrows():
        history = [normalize_sid(s) for s in parse_list_field(row["history_item_sid"])]
        target = normalize_sid(row["item_sid"])
        input_text = PROMPT_TEMPLATES[template].format(history=", ".join(history))
        prompt = INSTRUCTION + _USER_INPUT.format(input_text=input_text)
        body = _USER_INPUT.format(input_text=input_text)

        # 1) 与 data.py 的生成路径对比：instruction + generate_prompt(input, '')
        data_py_prompt = INSTRUCTION + (
            "### User Input: \n" + input_text + "\n\n### Response:\n"
        )
        if prompt != data_py_prompt:
            report["template_string_match"] = False

        # 2) junction：分别编码（SFT 的方式）vs 整体编码（TRL 的方式）
        sep_ids = tokenizer(INSTRUCTION, add_special_tokens=False).input_ids + \
                  tokenizer(body, add_special_tokens=False).input_ids
        joint_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if sep_ids != joint_ids:
            report["junction_match"] = False

        # 3) completion 边界：prompt 结尾 token 与 completion 首 token 不融合
        completion = completion_text(target)
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        c_ids = tokenizer(completion, add_special_tokens=False).input_ids
        concat_ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids
        if concat_ids[: len(p_ids)] != p_ids or concat_ids[len(p_ids):] != c_ids:
            report["completion_boundary_ok"] = False
        if tokenizer(prompt + completion + tokenizer.eos_token,
                     add_special_tokens=False).input_ids[-1] != tokenizer.eos_token_id:
            report["eos_after_completion"] = False
        report["rows_checked"] += 1

    if not report["completion_boundary_ok"] or not report["eos_after_completion"]:
        raise RuntimeError(f"token 边界校验失败，拒绝构造数据：{report}")
    return report


def generate_candidates(model, tokenizer, hash_dict, interactions, out_path,
                        num_beams=20, batch_size=4, max_new_tokens=16,
                        length_penalty=0.0):
    """约束 Top-K 候选生成，断点续跑：已完成的 sample_id 跳过。"""
    from transformers import GenerationConfig, LogitsProcessorList
    from LogitProcessor import ConstrainedLogitsProcessor

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["sample_id"])
    todo = [it for it in interactions if it.sample_id not in done]
    print(f"candidates: {len(done)} done, {len(todo)} todo")
    if not todo:
        return

    pad_id = tokenizer.eos_token_id
    gc = GenerationConfig(
        num_beams=num_beams, num_return_sequences=num_beams,
        length_penalty=length_penalty, max_new_tokens=max_new_tokens,
        do_sample=False, top_k=None, top_p=None,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    device = next(model.parameters()).device
    with open(out_path, "a", encoding="utf-8") as fout, torch.no_grad():
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            prompts = [build_prompt(it.history_sids) for it in chunk]
            enc = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
            max_len = max(len(e) for e in enc)
            # prompt 左截断，保留近期历史与 response 前缀（方案第 5 节）
            enc = [e[-256:] for e in enc]
            max_len = max(len(e) for e in enc)
            input_ids = torch.tensor([[pad_id] * (max_len - len(e)) + e for e in enc]).to(device)
            attn = torch.tensor([[0] * (max_len - len(e)) + [1] * len(e) for e in enc]).to(device)
            # 每次 generate 新建 processor：内部 step 计数不可跨批复用
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=lambda b, ids: hash_dict.get(get_hash(ids), []),
                num_beams=num_beams, base_model="qwen", eos_token_id=tokenizer.eos_token_id)
            out = model.generate(input_ids, attention_mask=attn, generation_config=gc,
                                 do_sample=False, logits_processor=LogitsProcessorList([clp]),
                                 return_dict_in_generate=True)
            seqs = out.sequences[:, max_len:]
            texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            for i, it in enumerate(chunk):
                cands = [t.split("Response:\n")[-1].strip()
                         for t in texts[i * num_beams:(i + 1) * num_beams]]
                # 合法 + 去重，保持模型分数序
                seen, ranked = set(), []
                for c in cands:
                    c = normalize_sid(c)
                    if c and c not in seen:
                        seen.add(c)
                        ranked.append(c)
                fout.write(json.dumps({"sample_id": it.sample_id, "sids": ranked},
                                      ensure_ascii=False) + "\n")
            fout.flush()
            if (start // batch_size) % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"  candidates {start + len(chunk)}/{len(todo)} | peak {mem:.2f} GiB",
                      flush=True)


def main(sft_path, split="train", sample=6000, num_beams=20, batch_size=4,
         negatives_per_example=2, seed=42, output_dir="outputs/dpo_pairs",
         category=CATEGORY_DEFAULT, dataset=DATASET_DEFAULT,
         data_root="data/Amazon", prompt_template="eval",
         train_csv=None, valid_csv=None, index_path=None, info_path=None):
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(output_dir, exist_ok=True)
    if train_csv is None:
        train_csv = f"{data_root}/train/{dataset}.csv"
    if valid_csv is None:
        valid_csv = f"{data_root}/valid/{dataset}.csv"
    if index_path is None:
        index_path = f"{data_root}/index/{category}.index.json"
    if info_path is None:
        info_path = f"{data_root}/info/{dataset}.txt"
    csv_path = {"train": train_csv, "valid": valid_csv}[split]

    catalog = load_sid_catalog(index_path, info_path)
    print(f"catalog: {len(catalog['valid_sids'])} unique SIDs, "
          f"{len(catalog['collision_sids'])} collision SIDs")

    # token 边界校验（失败即中止）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    boundary = check_token_boundaries(tokenizer, csv_path, prompt_template)
    print("token boundary check:", boundary)

    # 规则 4 的排除集合：仅由训练集构建（valid 构造同样只用 train 信息）
    train_interactions, train_sha = load_interactions(train_csv, "train", catalog)
    other_pos_map = history_target_map_from_train(train_interactions)

    interactions, data_sha = load_interactions(csv_path, split, catalog)
    if sample > 0 and sample < len(interactions):
        rng = random.Random(f"{seed}-{split}-sample")
        interactions = rng.sample(interactions, sample)
    print(f"{split}: {len(interactions)} interactions sampled")

    # 1) 候选生成（断点续跑）
    model = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    cand_path = os.path.join(output_dir, f"candidates_{split}.jsonl")
    generate_candidates(model, tokenizer, build_hash_dict(tokenizer, info_path),
                        interactions, cand_path, num_beams=num_beams,
                        batch_size=batch_size)
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    cand_hash = sha256_file(cand_path) if os.path.exists(cand_path) else None

    with open(cand_path, encoding="utf-8") as f:
        cand_map = {json.loads(l)["sample_id"]: json.loads(l)["sids"] for l in f if l.strip()}

    # 2) 偏好对构造
    stats = PairBuildStats()
    model_hash = sha256_file(os.path.join(sft_path, "model.safetensors"))[:16]
    pairs_path = os.path.join(output_dir, f"{split}.jsonl")
    audit_path = os.path.join(output_dir, f"{split}_audit.jsonl")
    n_pairs_written = 0
    with open(pairs_path, "w", encoding="utf-8") as fp, \
         open(audit_path, "w", encoding="utf-8") as fa:
        for it in interactions:
            stats.n_interactions += 1
            ranked = cand_map.get(it.sample_id, [])
            other_pos = other_pos_map.get(tuple(it.history_sids), set()) - {it.target_sid}
            rng = random.Random(f"{it.sample_id}|{seed}")
            pairs = build_pairs_for_interaction(it, ranked, catalog, other_pos,
                                                negatives_per_example, rng,
                                                catalog["valid_sids"], stats)
            for neg_index, (pair, neg_source) in enumerate(pairs):
                rejected_sid = normalize_sid(pair["rejected"])
                rank = ranked.index(rejected_sid) if rejected_sid in ranked else None
                rec = audit_record(it, pair, neg_index, neg_source,
                                   candidate_rank=rank, candidate_score=None,
                                   candidate_model_hash=model_hash,
                                   prompt_hash=hashlib.sha256(
                                       pair["prompt"].encode()).hexdigest()[:16])
                rec["rejected_item_ids"] = catalog["sid_to_item_ids"].get(rejected_sid)
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fp.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_pairs_written += 1
                if it.target_in_collision:
                    stats.n_pairs_collision_target += 1
            if 0 < len(pairs) < negatives_per_example:
                stats.n_single_pair_interactions += 1

    # 3) manifest
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    manifest = {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template": prompt_template,
        "sft_path": sft_path,
        "sft_model_sha256_16": model_hash,
        "split": split, "sample": sample, "seed": seed,
        "num_beams": num_beams, "negatives_per_example": negatives_per_example,
        "data_csv": csv_path, "data_csv_sha256": data_sha,
        "train_csv_sha256_16": train_sha[:16],
        "index_sha256_16": sha256_file(index_path)[:16],
        "info_sha256_16": sha256_file(info_path)[:16],
        "candidates_sha256_16": cand_hash[:16] if cand_hash else None,
        "candidate_generation_peak_gib": round(peak_gib, 3),
        "token_boundary_check": boundary,
        "git_commit": commit,
        "stats": vars(stats),
    }
    with open(os.path.join(output_dir, f"manifest_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"pairs": n_pairs_written, **vars(stats)},
                     ensure_ascii=False, indent=2, default=str))
    print(f"written: {pairs_path}\n          {audit_path}\n          {manifest.get('git_commit')}")


if __name__ == "__main__":
    fire.Fire(main)
