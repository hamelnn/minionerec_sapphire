#!/bin/bash
# 全参 RL 评测：与 04_merge_eval_rl.sh 完全同协议（4533 测试样本 × 20 beams），
# 区别仅在于全参产物无需 merge_adapter，直接评 final_checkpoint。
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
category=Industrial_and_Scientific

$PY evaluate.py \
  --base_model outputs/rl_fullparam_05b_12gb/final_checkpoint \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/test/${dataset_name}.csv" \
  --result_json_data outputs/eval/rl_fullparam.json \
  --batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 16 \
  --length_penalty 0.0

$PY calc.py \
  --path outputs/eval/rl_fullparam.json \
  --item_path "data/Amazon/info/${dataset_name}.txt"

$PY scripts_win/check_validity.py \
  --result_json outputs/eval/rl_fullparam.json \
  --info_file "data/Amazon/info/${dataset_name}.txt"
