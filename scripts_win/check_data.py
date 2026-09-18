# -*- coding: utf-8 -*-
"""数据核对（复现方案第 3 节）：
1. 三份 CSV 的 item_sid / history_item_sid 是否都属于 index.json 定义的同套 SID；
2. info 的 SID 与 index/item 元数据是否一致；
3. tokenizer 扩词后每个 SID 子 token 的编码是否符合预期（单个 token、前缀三元组）；
4. 三种 SFT 任务在 cutoff=256/512 下的截断比例与答案完整性；
5. RL prompt / completion token 长度分布，用于确定 max_prompt_length / max_completion_length。
"""
import os
import sys
import json
import random
import ast

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATEGORY = "Industrial_and_Scientific"
DATASET = "Industrial_and_Scientific_5_2016-10-2018-11"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "data", "Amazon", "index", f"{CATEGORY}.index.json")
ITEM_PATH = os.path.join(ROOT, "data", "Amazon", "index", f"{CATEGORY}.item.json")
INFO_PATH = os.path.join(ROOT, "data", "Amazon", "info", f"{DATASET}.txt")
SPLIT = {s: os.path.join(ROOT, "data", "Amazon", s, f"{DATASET}.csv") for s in ("train", "valid", "test")}
MODEL_ID = "Qwen/Qwen2.5-0.5B"


def load_csv(path):
    df = pd.read_csv(path)
    for col in ("history_item_sid", "history_item_title", "history_item_id"):
        df[col] = df[col].apply(ast.literal_eval)
    return df


def main():
    with open(INDEX_PATH) as f:
        index = json.load(f)
    with open(ITEM_PATH) as f:
        item_feat = json.load(f)
    info = open(INFO_PATH, encoding="utf-8").read().splitlines()
    info_sids = [line.split("\t")[0].strip() for line in info if line.strip()]

    valid_sids = {sids[0] + sids[1] + sids[2] for sids in index.values() if len(sids) >= 3}
    print(f"[1] index 条目数: {len(index)}, 组合 SID 数: {len(valid_sids)}, 碰撞(条目-SID 多对一): "
          f"{len(index) - len({tuple(v) for v in index.values()})}")

    # info 与 index 一致性：info 中每个 SID 都应属于 index 组合集合，且顺序对应 item_id
    missing_in_index = [s for s in info_sids if s not in valid_sids]
    print(f"[2] info 条数: {len(info_sids)}, 不在 index 中的 SID 数: {len(missing_in_index)}")
    id_match = all(
        line.split("\t")[0].strip() in valid_sids and line.split("\t")[2].strip() in index
        for line in info if len(line.split("\t")) >= 3
    )
    print(f"    info 行 SID∈index 且 item_id∈index: {id_match}")
    sid_by_item = {k: "".join(v[:3]) for k, v in index.items()}
    info_pair_ok = all(
        sid_by_item.get(line.split("\t")[2].strip()) == line.split("\t")[0].strip()
        for line in info if len(line.split("\t")) >= 3
    )
    print(f"    info(item_id -> SID) 与 index 映射一致: {info_pair_ok}")

    # CSV 目标 SID 一致性
    for split, path in SPLIT.items():
        df = load_csv(path)
        bad_target = (~df["item_sid"].isin(valid_sids)).sum()
        hist = df["history_item_sid"].explode().dropna()
        bad_hist = (~hist.isin(valid_sids)).sum()
        print(f"[3] {split}: 行数={len(df)}, 非法目标 SID={bad_target}, 非法历史 SID={bad_hist}")

    # tokenizer 扩词与编码
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    orig_vocab = len(tokenizer)
    new_tokens = sorted({t for v in index.values() for t in v})
    tokenizer.add_tokens(new_tokens)
    print(f"[4] 原词表: {orig_vocab}, 新增 SID token: {len(new_tokens)}, 扩词后: {len(tokenizer)}")
    sample_sid = info_sids[0]
    ids = tokenizer(sample_sid).input_ids
    print(f"    示例 SID {sample_sid} -> {len(ids)} tokens (期望 3): {[tokenizer.decode([i]) for i in ids]}")
    resp = "### Response:\n" + sample_sid + "\n"
    rids = tokenizer(resp).input_ids
    print(f"    '### Response:\\n'+SID+'\\n' -> {[tokenizer.decode([i]) for i in rids]}")
    print(f"    前 3 个 token（约束前缀）: {[tokenizer.decode([i]) for i in rids[:3]]}")
    prefix_standalone = tokenizer("### Response:\n").input_ids
    print(f"    独立编码 '### Response:\\n' -> {prefix_standalone}")

    # 每个 SID 的 completion token 数（含换行，不含 EOS）
    comp_lens = [len(tokenizer(s + "\n").input_ids) for s in info_sids]
    print(f"    completion token 数(含\\n, 不含EOS) min/max: {min(comp_lens)}/{max(comp_lens)}")

    # SFT 截断检查
    from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset
    random.seed(42)
    sample_n = 1500
    tasks = {
        "SidSFTDataset": SidSFTDataset(train_file=SPLIT["train"], tokenizer=tokenizer, max_len=100000,
                                       sample=sample_n, seed=42, category=CATEGORY),
        "SidItemFeatDataset": SidItemFeatDataset(item_file=ITEM_PATH, index_file=INDEX_PATH, tokenizer=tokenizer,
                                                 max_len=100000, sample=sample_n, seed=42, category=CATEGORY),
        "FusionSeqRecDataset": FusionSeqRecDataset(train_file=SPLIT["train"], item_file=ITEM_PATH,
                                                   index_file=INDEX_PATH, tokenizer=tokenizer, max_len=100000,
                                                   sample=sample_n, seed=42, category=CATEGORY),
    }
    for name, ds in tasks.items():
        for cutoff in (256, 512):
            n_trunc = 0
            n_ans_cut = 0
            for i in range(len(ds)):
                item = ds[i]
                ids_ = item["input_ids"]
                labels = item["labels"]
                # 原始全长通过 max_len=100000 保证未截断，此处按 cutoff 重切
                full_len = len(ids_)
                if full_len > cutoff:
                    n_trunc += 1
                    ans_len = sum(1 for l in labels if l != -100)
                    if ans_len < full_len - cutoff + ans_len:  # 答案区被左截断
                        pass
                    # 更直接：答案 token 是否全部位于最后 cutoff 窗口内
                    ans_positions = [j for j, l in enumerate(labels) if l != -100]
                    if ans_positions and ans_positions[0] < full_len - cutoff:
                        n_ans_cut += 1
            print(f"[5] {name}: n={len(ds)}, cutoff={cutoff}, 截断比例={n_trunc/len(ds):.2%}, 答案被切样本={n_ans_cut}")

    # RL 长度分布
    from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset
    rl_tasks = {
        "SidDataset(train)": SidDataset(SPLIT["train"], category="industrial and scientific items", sample=1000, seed=42),
        "SidDataset(valid)": SidDataset(SPLIT["valid"], category="industrial and scientific items", sample=500, seed=42),
        "RLTitle2Sid": RLTitle2SidDataset(item_file=ITEM_PATH, index_file=INDEX_PATH, sample=1000, seed=42,
                                          category="industrial and scientific items"),
        "RLSeqTitle2Sid": RLSeqTitle2SidDataset(SPLIT["train"], sample=1000, seed=42,
                                                category="industrial and scientific items"),
    }
    for name, ds in rl_tasks.items():
        p_lens, c_lens = [], []
        for i in range(len(ds)):
            row = ds[i]
            p_lens.append(len(tokenizer(row["prompt"]).input_ids))
            c_lens.append(len(tokenizer(row["completion"]).input_ids))
        pl = pd.Series(p_lens)
        print(f"[6] {name}: n={len(ds)}, prompt tokens p50/p90/p99/max = "
              f"{pl.quantile(.5):.0f}/{pl.quantile(.9):.0f}/{pl.quantile(.99):.0f}/{pl.max()}, "
              f"completion max={max(c_lens)}")


if __name__ == "__main__":
    main()
