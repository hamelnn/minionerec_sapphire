#!/bin/bash
# RL 合并 + 评测（方案 6、7.2）：CPU 合并，单卡 @20 评测
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
category=Industrial_and_Scientific

$PY scripts_win/merge_adapter.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --adapter_path outputs/rl_05b_12gb/final_checkpoint \
  --merged_path outputs/rl_05b_12gb_merged

$PY evaluate.py \
  --base_model outputs/rl_05b_12gb_merged \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/test/${dataset_name}.csv" \
  --result_json_data outputs/eval/rl.json \
  --batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 16 \
  --length_penalty 0.0

$PY calc.py \
  --path outputs/eval/rl.json \
  --item_path "data/Amazon/info/${dataset_name}.txt"

$PY scripts_win/check_validity.py \
  --result_json outputs/eval/rl.json \
  --info_file "data/Amazon/info/${dataset_name}.txt"
