#!/bin/bash
# 全参 RL 正式训练：与 03_rl.sh（LoRA GRPO）同数据/同种子/同步数，仅三处不同：
#   1. --use_lora False          全参训练（ReReTrainer 内部 create_reference_model 生成参考副本）
#   2. --optim adamw_bnb_8bit    fp32 Adam 状态 4GiB 会挤爆 12GB，换 SFT 同款 8-bit
#   3. --learning_rate 1e-6      LoRA 的 1e-5 作用在 8.8M 参数上；全参 494M 参数按惯例降一个量级
# 注意：SFT checkpoint 位于原始工作仓库（sapphire outputs/ 不含大文件），用绝对路径。
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=/c/Users/coxucu/anaconda3/envs/minionerec/python.exe
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
category=Industrial_and_Scientific
SFT_PATH="D:/ctrpred/minionerec/MiniOneRec/outputs/sft_05b_12gb/final_checkpoint"

$PY rl.py \
  --model_path "$SFT_PATH" \
  --train_file "data/Amazon/train/${dataset_name}.csv" \
  --eval_file "data/Amazon/valid/${dataset_name}.csv" \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --sid_index_path "data/Amazon/index/${category}.index.json" \
  --item_meta_path "data/Amazon/index/${category}.item.json" \
  --category "$category" \
  --output_dir outputs/rl_fullparam_05b_12gb \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --num_generations 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-6 \
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
  --use_lora False \
  --optim adamw_bnb_8bit \
  --rl_sample 2000 \
  --rl_seq_sample 2000 \
  --eval_sample 200 \
  --eval_step 0.199
