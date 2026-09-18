# ============================================================
# 步骤 3: RL / GRPO (单卡 0.5B, 显存峰值约 9-10 GB)
# 运行前: cd 到 MiniOneRec 目录, 已完成 02_sft.ps1
#   powershell -ExecutionPolicy Bypass -File scripts_win\03_rl.ps1
# 省时选项: -RlSample 10000 (约2-3h, README 明确允许 RL 用数万条子集)
# ============================================================
param(
    [string]$ModelPath = "./output_dir/Qwen2.5-0.5B_sft/Industrial_and_Scientific/final_checkpoint",
    [string]$Category = "Industrial_and_Scientific",
    [int]$TrainBatchSize = 32,     # 每卡 batch(会被 num_generations 重复展开); OOM 时减到 16
    [int]$EvalBatchSize = 16,      # 需能被 num_generations 整除
    [int]$NumGenerations = 8,      # 官方为 16; 降到 8 省显存省时间
    [int]$GradAccum = 4,           # 梯度累积: 有效 batch = 32*4=128; 增大可进一步省显存
    [int]$Epochs = 1,              # 官方 rl.sh 为 2; 单卡先用 1
    [int]$RlSample = 20000,        # SidDataset 行数子集; -1 = 全量(36260)
    [string]$OutDir = "./output_dir/Qwen2.5-0.5B_rl"
)
$ErrorActionPreference = "Stop"

$trainFile = "./data/Amazon/train/${Category}_5_2016-10-2018-11.csv"
$evalFile  = "./data/Amazon/valid/${Category}_5_2016-10-2018-11.csv"
$infoFile  = "./data/Amazon/info/${Category}_5_2016-10-2018-11.txt"

# 单卡直接用 python 启动(trainer 内部自建 Accelerator), 不走 accelerate+DeepSpeed
python rl.py `
    --model_path $ModelPath `
    --train_batch_size $TrainBatchSize `
    --eval_batch_size $EvalBatchSize `
    --num_train_epochs $Epochs `
    --gradient_accumulation_steps $GradAccum `
    --train_file $trainFile `
    --eval_file $evalFile `
    --info_file $infoFile `
    --category $Category `
    --sample_train False `
    --eval_step 0.0999 `
    --reward_type ranking `
    --num_generations $NumGenerations `
    --mask_all_zero False `
    --dynamic_sampling False `
    --sync_ref_model True `
    --beam_search True `
    --test_during_training False `
    --temperature 1.0 `
    --learning_rate 1e-5 `
    --add_gt False `
    --beta 1e-3 `
    --dapo False `
    --rl_sample $RlSample `
    --do_eval False `
    --output_dir "$OutDir/$Category" `
    --wandb_run_name "0.5B_rl_$Category" `
    --sid_index_path "./data/Amazon/index/${Category}.index.json" `
    --item_meta_path "./data/Amazon/index/${Category}.item.json"

Write-Host "`nRL 完成。最终权重: $OutDir/$Category/final_checkpoint" -ForegroundColor Green
Write-Host "注意: 训练中产生的 checkpoint-* 目录每个约 5GB, 可手动清理仅保留 final_checkpoint" -ForegroundColor Yellow
