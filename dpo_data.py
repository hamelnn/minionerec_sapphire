# -*- coding: utf-8 -*-
"""DPO 数据构造的共享工具（方案《MiniOneRec_DPO_实施方案.md》第 4、5 节）。

设计要点：
- chosen 直接来自当前交互行（不复用 GRPO 的全局 history2target 查表，规避标签覆盖问题）；
- prompt 模板逐字符取自 data.py 的 EvalSidDataset / SidSFTDataset（BaseDataset.generate_prompt），
  build_dpo_pairs.py 会做逐 token 一致性校验后才允许构造数据；
- 负例过滤顺序遵循方案 4.4 的 7 条规则；
- SID 碰撞信息由 index.json 直接推导，碰撞目标在审计与评估中单独分层。

本模块只做纯数据逻辑，不加载模型；候选生成在 build_dpo_pairs.py 中完成。
"""
import ast
import hashlib
import json
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# prompt 模板：与 data.py 逐字符一致
# ---------------------------------------------------------------------------

# data.py 中 EvalSidDataset.pre / SidSFTDataset.pre 使用的 instruction（注意
# "completes the request. " 行尾有一个空格，结尾两个换行，勿"清理"）。
INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request. \n"
    "\n"
    "### Instruction:\n"
    "Can you predict the next possible item that the user may expect?\n"
    "\n"
)

# BaseDataset.generate_prompt 在 output 为空字符串时的形态（data.py:80-84）。
_USER_INPUT = "### User Input: \n{input_text}\n\n### Response:\n"

# 各任务 input 文本（data.py）：
# - EvalSidDataset.get_history（推荐评估所用模板，DPO 默认对齐评估）
# - SidSFTDataset.get_history（SFT 主任务模板）
PROMPT_TEMPLATES = {
    "eval": "Can you predict the next possible item the user may expect, "
            "given the following chronological interaction history: {history}",
    "sft": "The user has interacted with items {history} in chronological order. "
           "Can you predict the next possible item that the user may expect?",
}

PROMPT_TEMPLATE_VERSION = "dpo-v1-eval"  # 变更模板时必须同步更新 manifest 里的版本号


def build_prompt(history_sids, template="eval"):
    """按 data.py 的 EvalSidDataset（或 SidSFTDataset）逐字符重建完整 prompt。

    末尾恰为 "### Response:\\n"，completion（SID+换行）由此接续。
    """
    history = ", ".join(history_sids)
    input_text = PROMPT_TEMPLATES[template].format(history=history)
    return INSTRUCTION + _USER_INPUT.format(input_text=input_text)


def completion_text(sid):
    """答案文本：SID + 换行。与 SFT 的 golden target 一致；EOS 由 TRL 追加，不在此处拼接。"""
    return sid + "\n"


# ---------------------------------------------------------------------------
# SID 目录与碰撞
# ---------------------------------------------------------------------------

def normalize_sid(text):
    """规范化 SID 文本：去引号/空白/换行（与 calc.py 的 strip 口径一致）。"""
    return str(text).strip().strip("\"").strip("\n").strip()


def load_sid_catalog(index_path, info_path):
    """从 index.json / info 构建合法 SID 目录与映射。

    返回 dict：
      valid_sids           合法组合 SID 集合（index 前 3 层拼接）
      sid_to_item_ids      SID -> [item_id, ...]（碰撞时多于一个）
      item_id_to_sid       item_id -> SID
      collision_sids       出现碰撞（对应多个物品）的 SID 集合
    """
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    sid_to_item_ids = {}
    for item_id, sids in index.items():
        if len(sids) >= 3:
            sid = sids[0] + sids[1] + sids[2]
            sid_to_item_ids.setdefault(sid, []).append(item_id)
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sid_to_item_ids.setdefault(parts[0].strip(), [])
    valid_sids = set(sid_to_item_ids)
    collision_sids = {s for s, ids in sid_to_item_ids.items() if len(ids) > 1}
    item_id_to_sid = {ids[0]: s for s, ids in sid_to_item_ids.items() if len(ids) >= 1}
    return {
        "valid_sids": valid_sids,
        "sid_to_item_ids": sid_to_item_ids,
        "item_id_to_sid": item_id_to_sid,
        "collision_sids": collision_sids,
    }


# ---------------------------------------------------------------------------
# CSV 读取与稳定标识
# ---------------------------------------------------------------------------

def parse_list_field(value):
    """CSV 中以字符串形式存储的列表字段，用 ast.literal_eval 结构化解析。"""
    if isinstance(value, list):
        return list(value)
    return ast.literal_eval(value)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_sample_id(data_file_sha, split, row_idx):
    """稳定 sample_id：数据文件哈希 + split + 原始行号（方案 4.1）。

    同一历史可能对应多个目标，禁止只用 prompt 文本做键。
    """
    return f"{data_file_sha[:8]}-{split}-{row_idx}"


def make_pair_id(sample_id, neg_index):
    return f"{sample_id}-neg{neg_index}"


@dataclass
class Interaction:
    """一条原始交互（方案 4.1），chosen 恒等于该行观测目标。"""
    sample_id: str
    split: str
    source_row: int
    user_id: str
    history_sids: list
    target_item_id: str
    target_sid: str
    target_in_collision: bool = False


def load_interactions(csv_path, split, catalog):
    """逐行读取 split CSV，转为 Interaction 列表。标签取自当前行，不查全局字典。"""
    import pandas as pd

    data_sha = sha256_file(csv_path)
    df = pd.read_csv(csv_path)
    out = []
    for row_idx, row in df.iterrows():
        history_sids = [normalize_sid(s) for s in parse_list_field(row["history_item_sid"])]
        target_sid = normalize_sid(row["item_sid"])
        if target_sid not in catalog["valid_sids"]:
            continue  # 非法目标行跳过并计数（调用方统计）
        out.append(Interaction(
            sample_id=make_sample_id(data_sha, split, row_idx),
            split=split,
            source_row=int(row_idx),
            user_id=str(row.get("user_id", "")),
            history_sids=history_sids,
            target_item_id=str(row.get("item_id", "")),
            target_sid=target_sid,
            target_in_collision=target_sid in catalog["collision_sids"],
        ))
    return out, data_sha


def history_target_map_from_train(interactions):
    """方案 4.4 规则 4：history -> 该历史在训练集中出现过的全部正目标 SID。

    仅使用训练期信息；用于把"同一历史下的其他真实正目标"从负例候选中排除。
    """
    hist2targets = {}
    for it in interactions:
        hist2targets.setdefault(tuple(it.history_sids), set()).add(it.target_sid)
    return hist2targets


# ---------------------------------------------------------------------------
# 偏好对构造（方案 4.4 过滤顺序）
# ---------------------------------------------------------------------------

@dataclass
class PairBuildStats:
    n_interactions: int = 0
    n_pairs: int = 0
    n_hard: int = 0
    n_random: int = 0
    n_fallback_random: int = 0   # 困难负例缺失、由随机负例补齐的对数
    n_skipped_no_negative: int = 0
    n_single_pair_interactions: int = 0
    n_pairs_collision_target: int = 0
    reject_reasons: dict = field(default_factory=dict)

    def reject(self, reason):
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1


def _try_pick(rejected_pool, chosen, used, excluded_sids, rng, source):
    """从候选中选第一个通过过滤的负例。返回 (sid, source) 或 (None, 原因)。"""
    for sid in rejected_pool:
        if not sid or sid not in excluded_sids["valid"]:
            continue  # 规则 1：必须存在于合法目录
        if sid == chosen:
            continue  # 规则 2：与 chosen 相同（即使 item_id 不同）不能成对
        if sid in used:
            continue  # 规则 3：同一交互不重复使用 rejected
        if sid in excluded_sids["other_positives"]:
            continue  # 规则 4：同历史的其他训练正目标
        return sid, source
    return None, "exhausted"


def build_pairs_for_interaction(interaction, ranked_candidates, catalog,
                                other_positives, negatives_per_example, rng,
                                all_valid_sids, stats):
    """为一条交互构造至多 negatives_per_example 个偏好对。

    ranked_candidates: 模型候选 SID 按分数降序（可含非法/重复，函数内部规范化过滤）。
    other_positives:   同历史在训练集出现过的其他正目标集合（不含本行 chosen）。
    all_valid_sids:    随机负例的采样总体（唯一 SID 均匀采样，方案 4.4）。
    """
    chosen = interaction.target_sid
    used = set()
    pairs = []  # [(pair_dict, source), ...]

    # 规范化 + 去重的模型候选（保持分数序）
    seen = set()
    norm_candidates = []
    for sid in ranked_candidates:
        sid = normalize_sid(sid)
        if sid in catalog["valid_sids"] and sid not in seen:
            seen.add(sid)
            norm_candidates.append(sid)

    need = negatives_per_example
    while len(pairs) < need:
        is_first = len(pairs) == 0
        # 首个负例优先取模型困难负例（排名靠前的错误候选）
        if is_first:
            sid, _ = _try_pick(norm_candidates, chosen, used,
                               {"valid": catalog["valid_sids"],
                                "other_positives": other_positives},
                               rng, "model_hard")
            if sid is None:
                source = "random"  # 规则 7：困难负例不足，随机负例补齐
                stats.n_fallback_random += 1
            else:
                source = "model_hard"
                stats.n_hard += 1
        else:
            sid, source = None, "random"
        if source == "random":
            # 随机合法负例：唯一 SID 均匀采样；尝试有限次以满足过滤规则
            for _ in range(64):
                cand = rng.choice(sorted(all_valid_sids)) if all_valid_sids else None
                sid, _ = _try_pick([cand], chosen, used,
                                   {"valid": catalog["valid_sids"],
                                    "other_positives": other_positives},
                                   rng, "random")
                if sid is not None:
                    stats.n_random += 1
                    break
                stats.reject("random_filtered")
        if sid is None:
            stats.reject("no_negative_available")
            break

        used.add(sid)
        pairs.append(({
            "prompt": build_prompt(interaction.history_sids),
            "chosen": completion_text(chosen),
            "rejected": completion_text(sid),
        }, source))

    # 兜底：一条负例都没有则整条交互跳过（规则 7），已通过 stats.reject 记录
    if not pairs:
        stats.n_skipped_no_negative += 1
    stats.n_pairs += len(pairs)
    return pairs


def audit_record(interaction, pair, neg_index, neg_source, candidate_rank=None,
                 candidate_score=None, candidate_model_hash=None, prompt_hash=None):
    """审计记录（方案 4.2）：与标准偏好 JSONL 分离保存。"""
    rejected_sid = normalize_sid(pair["rejected"])
    return {
        "pair_id": make_pair_id(interaction.sample_id, neg_index),
        "sample_id": interaction.sample_id,
        "split": interaction.split,
        "user_id": interaction.user_id,
        "source_row": interaction.source_row,
        "target_item_id": interaction.target_item_id,
        "target_sid": interaction.target_sid,
        "target_in_collision": interaction.target_in_collision,
        "rejected_sid": rejected_sid,
        "rejected_item_ids": None,  # 由调用方依据目录填充（碰撞时不指定单一物品）
        "negative_source": neg_source,
        "candidate_rank": candidate_rank,
        "candidate_score": candidate_score,
        "prompt_hash": prompt_hash,
        "candidate_model_hash": candidate_model_hash,
    }
