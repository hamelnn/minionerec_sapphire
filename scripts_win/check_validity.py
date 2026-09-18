# -*- coding: utf-8 -*-
"""方案第 7.3 节：合法性检查。
calc.py 在命中目标后 break，CC=0 不足以证明全部候选合法。
本脚本扫描每个样本的全部 beam 候选：合法 SID 比例、重复候选统计。
用法: python scripts_win/check_validity.py --result_json ... --info_file ...
"""
import fire
import json


def check(result_json: str, info_file: str):
    with open(info_file, encoding="utf-8") as f:
        valid_sids = {line.split("\t")[0].strip() for line in f if line.strip()}

    with open(result_json, encoding="utf-8") as f:
        test_data = json.load(f)

    total, valid_cnt, dup_groups = 0, 0, 0
    invalid_examples = []
    for sample in test_data:
        preds = [p.strip("\"\n").strip() for p in sample["predict"]]
        total += len(preds)
        valid_cnt += sum(1 for p in preds if p in valid_sids)
        if len(set(preds)) < len(preds):
            dup_groups += 1
        invalid = [p for p in preds if p not in valid_sids]
        if invalid and len(invalid_examples) < 5:
            invalid_examples.append((sample["output"].strip(), invalid))

    print(f"samples: {len(test_data)}, beams/sample: {len(test_data[0]['predict'])}")
    print(f"candidates total: {total}")
    print(f"valid SID ratio: {valid_cnt / total:.4%}")
    print(f"samples with duplicate candidates: {dup_groups}/{len(test_data)}")
    for target, inv in invalid_examples:
        print(f"target={target} invalid={inv[:3]}{'...' if len(inv) > 3 else ''}")


if __name__ == "__main__":
    fire.Fire(check)
