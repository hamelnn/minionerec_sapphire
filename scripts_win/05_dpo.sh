#!/bin/bash
# DPO 后训练串联流程（方案第 11 节命令契约）。
# 路径为本地实施配置：SFT checkpoint 位于 MiniOneRec 复现工作区（不入库）。
# 各阶段可独立重跑；候选生成与构造支持断点续跑。
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe

# ---- 本地路径配置（按实施环境修改）----
SFT_PATH=${SFT_PATH:-/d/ctrpred/minionerec/MiniOneRec/outputs/sft_05b_12gb/final_checkpoint}
CATEGORY=Industrial_and_Scientific
DATASET=Industrial_and_Scientific_5_2016-10-2018-11
TRAIN_CSV="data/Amazon/train/${DATASET}.csv"
VALID_CSV="data/Amazon/valid/${DATASET}.csv"
INDEX="data/Amazon/index/${CATEGORY}.index.json"
INFO="data/Amazon/info/${DATASET}.txt"
PAIRS_DIR=${PAIRS_DIR:-outputs/dpo_pairs}
DPO_DIR=${DPO_DIR:-outputs/dpo_05b_12gb}
MERGED_DIR=${MERGED_DIR:-outputs/dpo_05b_12gb_merged}

# ---- 首轮参数（方案 4.5/7.1：6000 交互，最多 12000 对）----
SAMPLE=${SAMPLE:-6000}

# 1) 训练偏好对构造（valid 独立执行同一构造流程）
$PY scripts_win/build_dpo_pairs.py \
  --sft_path "$SFT_PATH" --split train --sample "$SAMPLE" \
  --num_beams 20 --negatives_per_example 2 --seed 42 \
  --output_dir "$PAIRS_DIR" --category "$CATEGORY" --dataset "$DATASET"

$PY scripts_win/build_dpo_pairs.py \
  --sft_path "$SFT_PATH" --split valid --sample 2000 \
  --num_beams 20 --negatives_per_example 2 --seed 42 \
  --output_dir "$PAIRS_DIR" --category "$CATEGORY" --dataset "$DATASET"

# 2) LoRA DPO 训练
$PY dpo.py \
  --sft_path "$SFT_PATH" \
  --train_jsonl "$PAIRS_DIR/train.jsonl" \
  --valid_jsonl "$PAIRS_DIR/valid.jsonl" \
  --output_dir "$DPO_DIR" \
  --learning_rate 5e-6 --beta 0.1 --num_train_epochs 1 --seed 42

# 3) 验证选模（按 NDCG@10，生成 best_adapter）
$PY scripts_win/eval_dpo_checkpoints.py \
  --sft_path "$SFT_PATH" --checkpoints_dir "$DPO_DIR" \
  --valid_csv "$VALID_CSV" --info_file "$INFO" --index_path "$INDEX" \
  --sample 1000

# 4) CPU 合并最佳 adapter 到 SFT 底座
$PY scripts_win/merge_adapter.py \
  --sft_path "$SFT_PATH" \
  --adapter_path "$DPO_DIR/best_adapter" \
  --merged_path "$MERGED_DIR"

# 5) 最终测试评估（与 SFT/GRPO 相同协议：4533×20 beams）
mkdir -p outputs/eval
$PY evaluate.py \
  --base_model "$MERGED_DIR" \
  --info_file "$INFO" --category "$CATEGORY" \
  --test_data_path "data/Amazon/test/${DATASET}.csv" \
  --result_json_data outputs/eval/dpo.json \
  --batch_size 1 --num_beams 20 --max_new_tokens 16 --length_penalty 0.0
$PY calc.py --path outputs/eval/dpo.json --item_path "$INFO"
$PY scripts_win/check_validity.py \
  --result_json outputs/eval/dpo.json --info_file "$INFO"
