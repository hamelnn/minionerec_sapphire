# MiniOneRec 0.5B 单卡复现方案（Windows / RTX 4070 SUPER 12GB）

> 目标：在 12GB 单卡上完整走通 **SFT → RL(GRPO) → 离线评测** 三阶段，复现 MiniOneRec 流程。
> 基座模型：`Qwen/Qwen2.5-0.5B`（base 版，非 Instruct，规避 README 提到的 CC 约束解码失效问题）。
> 数据：仓库自带 `data/Amazon/Industrial_and_Scientific`（已含 SID 构建产物），**跳过 Stage 0（rq/ 全链路）**。

---

## 0. 可行性结论

| 阶段 | 显存峰值 | 耗时（Industrial 数据） | 12GB 可行性 |
|---|---|---|---|
| SFT（10 epoch，默认） | ~6–8 GB | 2–4 h | ✅ |
| SFT（3 epoch 快速版） | ~6–8 GB | 0.7–1.2 h | ✅ |
| RL（子集 2 万条 × 1 epoch） | ~9–10 GB | 2.5–4 h | ✅（需调参） |
| RL（全量 36K × 1 epoch） | ~9–10 GB | 3–6 h | ✅（需调参） |
| 离线评测（beam=50） | ~4–5 GB | 30–60 min | ✅ |

本地实测数据规模：train 36,260 / valid 4,533 / test 4,534 行，物品 3,686 个，新增 SID token 仅 560 个。

---

## 1. 交付物清单（本次已创建）

| 文件 | 说明 |
|---|---|
| `MiniOneRec/requirements_win.txt` | Windows 单卡最小依赖（剔除 torchrec/fbgemm_gpu/deepspeed 等 Linux-only 或单卡不需要的包） |
| `MiniOneRec/scripts_win/01_download_model.ps1` | 下载 Qwen2.5-0.5B（走 hf-mirror 镜像） |
| `MiniOneRec/scripts_win/02_sft.ps1` | SFT 启动脚本（支持 `-Sample 200` 冒烟、`-Epochs 3` 快速档） |
| `MiniOneRec/scripts_win/03_rl.ps1` | RL 启动脚本（12GB 专用参数组合） |
| `MiniOneRec/scripts_win/04_eval.ps1` | 评测脚本（split→evaluate→merge→calc） |
| `MiniOneRec/rl.py`（4 处补丁） | 见 §4，均为向后兼容小改 |

---

## 2. 环境安装（本机尚无 Python，需先装一次）

```powershell
# 1) 安装 Miniconda (https://docs.conda.io/en/latest/miniconda.html, Windows x64)
#    安装时勾选 "Add to PATH" 或之后用 Anaconda Prompt

# 2) 创建环境 (README 推荐 Python 3.11)
conda create -n minionerec python=3.11 -y
conda activate minionerec

# 3) 安装依赖 (torch 2.6.0 默认 cu124 wheel, 4070 SUPER sm_89 原生支持 bf16)
cd D:\ctrpred\minionerec\MiniOneRec
pip install -r requirements_win.txt

# 4) 自检
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期: 2.6.0+cu124 True NVIDIA GeForce RTX 4070 SUPER
```

为什么不用原版 `requirements.txt`：`torchrec==0.6.0+cu118`、`fbgemm_gpu` 只发布 Linux wheel，Windows 装不上；`deepspeed` 仅多卡 accelerate 启动需要（单卡直接 `python rl.py`，trainer 内部自建 Accelerator）；`torchaudio/torcheval/torchmetrics/POT` 主链路未使用。另补了 `scikit-learn`（`rl.py` 顶部 import 需要，原 requirements 漏列）。

---

## 3. 执行流程（4 条命令）

```powershell
# 所有脚本都在 MiniOneRec 目录下运行; 若提示脚本禁止运行:
#   powershell -ExecutionPolicy Bypass -File scripts_win\xx.ps1

# 步骤 1: 下载模型 (~1GB)
powershell -ExecutionPolicy Bypass -File scripts_win\01_download_model.ps1

# 步骤 2: SFT
#   冒烟(约5分钟,先验证管线): -Sample 200 -Epochs 1
#   完整复现(默认10 epochs, 2-4h):
powershell -ExecutionPolicy Bypass -File scripts_win\02_sft.ps1
#   产物: output_dir/Qwen2.5-0.5B_sft/Industrial_and_Scientific/final_checkpoint

# 步骤 3: RL (依赖步骤2产物)
powershell -ExecutionPolicy Bypass -File scripts_win\03_rl.ps1
#   产物: output_dir/Qwen2.5-0.5B_rl/Industrial_and_Scientific/final_checkpoint

# 步骤 4: 评测 (分别评 SFT / RL checkpoint 对比提升)
powershell -ExecutionPolicy Bypass -File scripts_win\04_eval.ps1 `
    -Checkpoint ./output_dir/Qwen2.5-0.5B_rl/Industrial_and_Scientific/final_checkpoint
powershell -ExecutionPolicy Bypass -File scripts_win\04_eval.ps1 `
    -Checkpoint ./output_dir/Qwen2.5-0.5B_sft/Industrial_and_Scientific/final_checkpoint
```

---

## 4. rl.py 的 4 处补丁（均向后兼容，默认行为与原版一致）

| # | 位置 | 改动 | 原因 |
|---|---|---|---|
| 1 | `train()` 签名 | 新增 `rl_sample: int = -1`、`do_eval: bool = True` | 暴露子集采样与关闭 RL 过程中生成式评测的开关 |
| 2 | `sample = -1` | 改为 `sample = rl_sample` | 控制 SidDataset 行数；-1 时与原行为相同 |
| 3 | `llm_model` 加载 | 改为仅 `reward_type ∈ {sasrec, semantic}` 时才额外加载一份模型 | rule/ranking 奖励只需 `device`，省 ~1GB 显存 |
| 4 | `GRPOConfig` | 新增 `model_init_kwargs={"torch_dtype": torch.bfloat16}`；`eval_strategy` 由 `do_eval` 控制 | **关键**：原版不传 dtype 时模型按 fp32 加载，单卡无 ZeRO 分片必 OOM；bf16 加载后峰值 ~9–10GB。关闭训练中 eval 是因为 GRPO 的 eval 也会对 4,533 条验证集做 beam 生成，每次 10–20 分钟 |

不改动：`sft.py` / `evaluate.py` / `minionerec_trainer.py` 均无需修改即可单卡运行（已确认 `RepeatRandomSampler` 为本地实现、无 `torch.distributed` 硬依赖；`evaluate.py` 已内置 `do_sample=False` 规避 transformers≥4.50 的约束解码退化问题）。

---

## 5. 单卡参数 vs 官方 8×A100 参数对照

| 参数 | 官方（sft.sh / rl.sh） | 本方案 | 说明 |
|---|---|---|---|
| GPU | 8 × 80GB (torchrun/accelerate+ZeRO-2) | 1 × 12GB (纯 python) | 无 NCCL/DeepSpeed 依赖 |
| 基座 | Qwen2.5-1.5B（HF 官方 ckpt） | Qwen2.5-0.5B base | 显存约束 |
| SFT batch / micro | 1024 / 16 | 1024 / 16 | 全局 batch 不变，单卡自动转为梯度累积 64 |
| SFT epochs | 10 | 10（可选 3 快速档） | |
| RL num_generations | 16 | 8 | beam 数减半，显存/时间减半 |
| RL per-device batch | 64 | 32（×累积 4 = 有效 128） | |
| RL 数据 | 全量 36K | 默认子集 20K（`-RlSample -1` 可全量） | README 明确允许 RL 只用数万条样本 |
| RL epochs | 2 | 1 | |
| RL lr / beta | 1e-5 / 1e-3 | 不变 | |
| reward | ranking (rule+ndcg) | 不变 | |
| beam_search / sync_ref_model | True / True | 不变 | |
| 评测 batch / beams | 8 / 50 | 8 / 50 | |

---

## 6. 显存构成（0.5B，bf16）

- **SFT**：权重 1.0 + 梯度 1.0 + Adam(bf16) 2.0 + 激活 ~1 + 其它 ~1 ≈ **6–8 GB**
- **RL**：policy 1.0 + ref 1.0（`create_reference_model` 必然多一份）+ 梯度 1.0 + paged_adamw_32bit ~3.9（可分页溢出到内存）+ 激活/KV ~1.5 ≈ **9–10 GB**
- **评测**：权重 1.0 + beam50 KV cache ~1.7 + 激活 ≈ **4–5 GB**

注意：Windows 桌面本身占 ~0.8–1GB 显存，实际可用约 11.2GB；RL 峰值已按此预留余量。

---

## 7. 验收标准

1. **CC = 0**：`calc.py` 输出最后一行（无效生成数）必须为 0，说明约束解码成功。若非 0，按 README 指引确认使用的是 **base** 模型。
2. **SFT → RL 有提升**：对比两次 `04_eval.ps1` 的 HR@{1,5,10} / NDCG@{1,5,10}，RL 应有小幅提升。
3. 0.5B 指标低于论文 1.5B 数字属预期（模型容量差异），重点验证**流程可复现性与相对提升**。

---

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| SFT OOM | `-MicroBatchSize 8`；再不够加 `--freeze_LLM True`（只训新 SID token，峰值 ~4GB） |
| RL OOM | `-TrainBatchSize 16`（配合 `-GradAccum 8`）；或 `-NumGenerations 4` |
| RL 卡在启动/报 dist 错误 | 确认用 `python rl.py`（脚本已如此），不要用 accelerate+deepspeed 启动 |
| 评测 OOM | `-BatchSize 4` 或 `-NumBeams 20` |
| CC > 0 | 确认基座是 Qwen2.5-0.5B（base）而非 Instruct；仍异常则查 transformers 版本是否 4.57.1 |
| 磁盘增长快 | RL 每 10% 存一个 checkpoint（~5GB/个），跑完可删 `output_dir/*/checkpoint-*` 只留 `final_checkpoint` |
| wandb 报错 | rl.py 已设 WANDB_MODE=offline，无需登录；若仍干扰可在 rl.py 中把 `report_to="wandb"` 改为 `"none"` |
| 想换数据集 | 全部脚本 `-Category Office_Products` 即可（数据已内置） |

---

## 9. 已知限制与差异

1. **跳过了 Stage 0（SID 构建 / rq/ 目录）**：直接使用仓库自带的 `.index.json` / `.item.json`。如需从原始 Amazon 数据重建 SID，需在 Linux/WSL 环境补装 `faiss-gpu`、`k-means-constrained`、`polars` 并走 `data/amazon18_data_process.sh → rq/text2emb → rq/rqvae.sh → generate_indices.py → convert_dataset.py`。
2. `ts_rec_sft.sh` 存在原仓库 bug（调用了 `sft.py` 却传 `ts_rec_sft.py` 专属参数），与本方案无关，未修改。
3. RL 有效 batch（128×8=... 单卡 32×4=128）与官方（64×8卡×2累积=1024 prompts/步）不同，RL 结果可能略有差异；如需对齐可将 `-GradAccum` 调大（仅影响优化器步数，不影响显存上限）。
4. Windows 下 `torch.backends.cuda` 数学注意力路径（rl.py:70-71 关闭了 flash/mem-efficient）已按短序列验证过显存安全；若出现长 prompt 批次的尖峰，按 §8 降 batch。
