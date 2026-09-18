# ============================================================
# 步骤 1: 下载基座模型 Qwen2.5-0.5B (base 版)
# 用 base 而非 Instruct: README 明确指出 Instruct 模型可能触发
# 约束解码失效 (CC>0) 的问题
# 运行前: cd 到 MiniOneRec 目录, 并已激活 conda 环境
# ============================================================
$ErrorActionPreference = "Stop"

# 国内镜像加速; 海外网络可删除此行
$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B

Write-Host "`n模型已保存到 ./models/Qwen2.5-0.5B" -ForegroundColor Green
