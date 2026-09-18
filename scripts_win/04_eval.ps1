# ============================================================
# 步骤 4: 离线评测 (单卡, 显存峰值约 4-5 GB, 约 30-60 分钟)
# 运行前: cd 到 MiniOneRec 目录, 已激活 conda 环境
#   powershell -ExecutionPolicy Bypass -File scripts_win\04_eval.ps1 -Checkpoint <模型路径>
# 示例:
#   powershell -ExecutionPolicy Bypass -File scripts_win\04_eval.ps1 `
#       -Checkpoint ./output_dir/Qwen2.5-0.5B_rl/Industrial_and_Scientific/final_checkpoint
#   (对比 SFT 效果可再评一次 SFT checkpoint)
# ============================================================
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,            # 待评测模型路径(SFT 或 RL 的 final_checkpoint)
    [string]$Category = "Industrial_and_Scientific",
    [int]$BatchSize = 8,            # OOM 时减到 4
    [int]$NumBeams = 50,            # 官方评测 beam=50; 减到 20 可提速近一半
    [int]$MaxNewTokens = 256
)
$ErrorActionPreference = "Stop"

$tag = "0.5B_" + (Get-Date -Format "MMdd_HHmm")
$tempDir = "./temp/${Category}-${tag}"
$resultDir = "./results/${Category}-${tag}"
New-Item -ItemType Directory -Force -Path $tempDir, $resultDir | Out-Null

$testFile = "./data/Amazon/test/${Category}_5_2016-10-2018-11.csv"
$infoFile = "./data/Amazon/info/${Category}_5_2016-10-2018-11.txt"

# 1) 单卡无需切分, 但沿用官方 split->evaluate->merge->calc 四步流程
python split.py --input_path $testFile --output_path $tempDir --cuda_list "0"

# 2) 约束 beam search 生成 Top-50
python evaluate.py `
    --base_model $Checkpoint `
    --info_file $infoFile `
    --category $Category `
    --test_data_path "$tempDir/0.csv" `
    --result_json_data "$tempDir/0.json" `
    --batch_size $BatchSize `
    --num_beams $NumBeams `
    --max_new_tokens $MaxNewTokens `
    --length_penalty 0.0

# 3) 合并结果
python merge.py --input_path $tempDir --output_path "$resultDir/final_result_${Category}.json" --cuda_list "0"

# 4) 计算 HR@K / NDCG@K / CC
python calc.py --path "$resultDir/final_result_${Category}.json" --item_path $infoFile

Write-Host "`n评测完成。结果文件: $resultDir/final_result_${Category}.json" -ForegroundColor Green
Write-Host "验收要点: calc.py 输出最后一行 CC 必须为 0 (约束解码成功); 否则参考 README 更换 base 模型" -ForegroundColor Yellow
