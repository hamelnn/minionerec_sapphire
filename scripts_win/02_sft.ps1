# ============================================================
# 步骤 2: SFT (单卡 0.5B, 显存峰值约 6-8 GB)
# 运行前: cd 到 MiniOneRec 目录, 已激活 conda 环境
#   powershell -ExecutionPolicy Bypass -File scripts_win\02_sft.ps1
# 冒烟测试(约5分钟, 验证管线通不通):
#   powershell -ExecutionPolicy Bypass -File scripts_win\02_sft.ps1 -Sample 200 -Epochs 1
# 快速版(约1小时):            -Epochs 3
# 完整复现(约2-4小时, 默认):   -Epochs 10
# ============================================================
param(
    [string]$Model = "./models/Qwen2.5-0.5B",
    [string]$Category = "Industrial_and_Scientific",   # 或 Office_Products
    [int]$BatchSize = 1024,        # 全局 batch, 与官方 sft.sh 一致(单卡自动转为梯度累积64)
    [int]$MicroBatchSize = 16,     # 每次前向样本数 -> 显存旋钮(OOM就减半)
    [int]$Epochs = 10,             # 官方默认 10
    [int]$Sample = -1,              # -1=全量; 冒烟测试设 200
    [string]$OutDir = "./output_dir/Qwen2.5-0.5B_sft"
)
$ErrorActionPreference = "Stop"

$trainFile = "./data/Amazon/train/${Category}_5_2016-10-2018-11.csv"
$evalFile  = "./data/Amazon/valid/${Category}_5_2016-10-2018-11.csv"

python sft.py `
    --base_model $Model `
    --batch_size $BatchSize `
    --micro_batch_size $MicroBatchSize `
    --num_epochs $Epochs `
    --sample $Sample `
    --train_file $trainFile `
    --eval_file $evalFile `
    --output_dir "$OutDir/$Category" `
    --wandb_project MiniOneRec_SFT `
    --wandb_run_name "0.5B_sft_$Category" `
    --category $Category `
    --train_from_scratch False `
    --seed 42 `
    --sid_index_path "./data/Amazon/index/${Category}.index.json" `
    --item_meta_path "./data/Amazon/index/${Category}.item.json" `
    --freeze_LLM False

Write-Host "`nSFT 完成。最终权重: $OutDir/$Category/final_checkpoint" -ForegroundColor Green
