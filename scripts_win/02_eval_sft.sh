#!/bin/bash
# SFT 基线评测（方案 7.1）：@20 beams + calc + 合法率检查
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
category=Industrial_and_Scientific

mkdir -p outputs/eval

$PY evaluate.py \
  --base_model outputs/sft_05b_12gb/final_checkpoint \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/test/${dataset_name}.csv" \
  --result_json_data outputs/eval/sft.json \
  --batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 16 \
  --length_penalty 0.0

$PY calc.py \
  --path outputs/eval/sft.json \
  --item_path "data/Amazon/info/${dataset_name}.txt"

$PY scripts_win/check_validity.py \
  --result_json outputs/eval/sft.json \
  --info_file "data/Amazon/info/${dataset_name}.txt"
