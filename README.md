# minionerec_v0918

MiniOneRec 的 12GB 单卡复现实验项目（2026-09-17 ~ 09-18 执行）。独立仓库：`github.com/hamelnn/minionerec_sapphire`。

## 项目内容

基于 [AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)（commit `5f4f733`），在单张 RTX 4070 SUPER 12GB 上完成 **SID → Qwen2.5-0.5B 全参数 SFT → LoRA GRPO → 合并 adapter → 单卡评测** 的完整闭环。本目录包含：

- 修改后的训练/评测代码：`sft.py`、`rl.py`、`minionerec_trainer.py`、`data.py`、`evaluate.py`、`calc.py`、`LogitProcessor.py`
- Windows 单卡运行脚本：`scripts_win/`（check_data / merge_adapter / check_validity / 各阶段 shell 脚本）
- 复现方案与实验记录：`report/`
- 评测结果 JSON：`outputs/eval/`（SFT 与 RL 各 4533×20 beams）
- 数据：`data/Amazon/Industrial_and_Scientific` 品类（与仓库一致，未改动划分）

## 主要改动（相对上游）

1. `sft.py`：`optim="adamw_bnb_8bit"`、梯度检查点、`report_to="none"`、`max_steps`/`eval_step` 参数、峰值显存打印。
2. `rl.py`：GRPOConfig 按 12GB 配置（`max_prompt_length=256`、`max_completion_length=16`、`use_vllm=False`、`optim="adamw_torch"`、`report_to="none"`）；注入 LoRA（r=16, alpha=32, 七个投影层）；`ref_model=None`（PEFT disable_adapter 计算参考策略，不加载第二份模型）。
3. `minionerec_trainer.py`：新增 `group_hit_ratio` / `nonzero_adv_ratio` 指标（奖励稀疏性监控）。
4. `data.py`：修复 `CSVBaseDataset.sample` 超过数据集大小时的崩溃。
5. RL 关闭梯度检查点：PEFT 冻结底座 + reentrant checkpoint 在 transformers 4.57 下导致 loss 无梯度（详见 `report/复现实验记录.md` 第 3.1 节）。

## 结果（Industrial_and_Scientific，4533 测试样本 × 20 beams，约束解码）

| 实验 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | 合法率 | 峰值显存 | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT（0.5B 全参） | 8.41% | 6.79% | 9.93% | 7.29% | 13.28% | 8.13% | 100% | 4.33 GiB | 2.57 h |
| SFT + LoRA RL | 8.80% | 6.97% | 11.07% | 7.69% | 14.12% | 8.46% | 100% | 3.10 GiB | 58 min |

说明：SFT 正式训练使用 sample=6000/任务（全量 3 epochs 需约 14.5h，超出可执行窗口）；RL 为单 seed、4 候选、1 epoch。结论限定为"闭环完成且 RL 有有效更新"，未做多 seed 稳定提升宣称。

## 环境

conda `minionerec`：Python 3.11.16、torch 2.6.0+cu124、transformers 4.57.1、trl 0.24.0、accelerate 1.10.1、bitsandbytes 0.48.1、peft 0.21.0。模型 `Qwen/Qwen2.5-0.5B`（经 hf-mirror.com 下载）。单进程启动，无 torchrun / vLLM / DeepSpeed。

## 复现路径

```bash
# 1. 数据核对（SID 映射、tokenizer 扩词、截断比例）
python scripts_win/check_data.py

# 2. SFT 短训练 → 正式训练（先改 sft.py 中 TrainingArguments 的 max_steps 做验证）
python sft.py --base_model Qwen/Qwen2.5-0.5B \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_index_path data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path data/Amazon/index/Industrial_and_Scientific.item.json \
  --category Industrial_and_Scientific --output_dir outputs/sft_05b_12gb \
  --micro_batch_size 1 --batch_size 32 --cutoff_len 256 --learning_rate 5e-5 \
  --num_epochs 3 --sample 6000 --seed 42 --train_from_scratch False --freeze_LLM False

# 3. SFT 基线评测
bash scripts_win/02_eval_sft.sh

# 4. RL 短训练 → 正式训练
bash scripts_win/03_rl_short.sh
bash scripts_win/03_rl.sh

# 5. 合并 adapter + RL 评测
bash scripts_win/04_merge_eval_rl.sh
```

详见 `report/MiniOneRec_12GB_复现方案.md` 与 `report/复现实验记录.md`。
