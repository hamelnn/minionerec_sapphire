#!/bin/bash
# RL 正式训练（方案 5.4）：LoRA GRPO，ranking reward
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
category=Industrial_and_Scientific

$PY rl.py \
  --model_path outputs/sft_05b_12gb/final_checkpoint \
  --train_file "data/Amazon/train/${dataset_name}.csv" \
  --eval_file "data/Amazon/valid/${dataset_name}.csv" \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --sid_index_path "data/Amazon/index/${category}.index.json" \
  --item_meta_path "data/Amazon/index/${category}.item.json" \
  --category "$category" \
  --output_dir outputs/rl_05b_12gb \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --num_generations 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --reward_type ranking \
  --beta 0.001 \
  --beam_search True \
  --sync_ref_model False \
  --test_during_training False \
  --dynamic_sampling False \
  --mask_all_zero False \
  --sample_train False \
  --add_gt False \
  --dapo False \
  --seed 42 \
  --use_lora True \
  --rl_sample 2000 \
  --rl_seq_sample 2000 \
  --eval_sample 200 \
  --eval_step 0.199
