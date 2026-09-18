# MiniOneRec：12GB 单卡、0.5B 模型复现方案

> 方案日期：2026-09-17。根据本会话中已读取的 MiniOneRec 源码整理。本文提供实施配置和代码修改示例，尚未在 12GB 显卡上完成训练实测；显存峰值、运行时间和指标提升均需验证。本文中的代码是待实施方案，不表示仓库已经完成修改。

## 1. 目标与技术路线

推荐路线：**复用仓库 SID → Qwen2.5-0.5B 全参数 SFT → LoRA GRPO → 合并 adapter → 单卡评测**。

这条路线保留语义 ID、联合监督训练、推荐奖励和约束解码等主要环节。相对原始训练配置，调整了批量、候选数量、RL 参数更新方式和参考策略同步方式，应称为“12GB 资源约束下的方法复现”，不能直接宣称复现论文指标。

第一轮复用已有 SID，验证 SFT 与 RL；第二轮如需完整重建数据流程，再执行文本编码、RQ-VAE、SID 导出和数据转换。两轮应分别记录结果。

| 项目 | 第一轮配置 |
|---|---|
| GPU | 单张 NVIDIA 12GB 显卡，以支持 BF16 为前提 |
| 模型 | `Qwen/Qwen2.5-0.5B` Base |
| 数据 | 仓库自带 Industrial_and_Scientific |
| SID | 复用该品类已有 index、item、info 和 CSV |
| SFT | 全参数更新，保留原有三种训练任务 |
| RL | 在 SFT checkpoint 上新建 LoRA，使用 ranking reward |
| 推理 | 单卡、小 batch、约束 beam search |
| 指标 | HR/NDCG@5、10、20；合法 SID 比例；实际显存峰值 |

BF16 权重本身约 1GB，但训练还有梯度、优化器状态、激活、词表 logits 和生成缓存。RL 的多候选生成及参考策略计算会进一步占用显存，不能只根据权重大小判断能否训练。

若显卡不支持 BF16，需要同时适配模型加载精度、Trainer 的 `bf16/fp16` 设置及评测精度，并验证数值稳定性；不能只替换其中一个开关。

## 2. 环境准备和版本管理

使用独立 Python 环境，以下版本来自本次核对的仓库依赖，可作为兼容性验证起点：

| 组件 | 起始版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.6.0 |
| Transformers | 4.57.1 |
| TRL | 0.24.0 |
| Accelerate | 1.10.1 |
| bitsandbytes | 0.48.1 |
| PEFT | 选择与上述组件兼容的版本，短训练通过后记录并锁定 |

这些版本不是已验证的兼容组合。当前 requirements 混有 CUDA 11/12 组件和其他框架依赖，不宜未经检查直接安装到已有环境。按显卡驱动选择 PyTorch 安装包，再安装训练路径实际需要的依赖，包括数据处理及 reward 导入使用的依赖。

自定义 Trainer 导入了 TRL 内部接口。实施时先确认导入成功，再执行短训练，最后保存准确依赖版本和仓库 commit。遇到接口不兼容，应先对照当前代码处理，避免无依据地整体升级库。

本方案直接使用单进程 `python` 启动，不使用默认八卡训练脚本，也不依赖 vLLM 或 DeepSpeed。

## 3. 数据准备

在仓库根目录执行后续 shell 命令；每次新开 shell 先设置：

```bash
export CUDA_VISIBLE_DEVICES=0
category=Industrial_and_Scientific
dataset_name=Industrial_and_Scientific_5_2016-10-2018-11
```

输入使用仓库已有的训练、验证、测试 CSV，以及同品类的 info、index、item 元数据。运行前核对：

1. 所有 CSV 中的目标 SID 都属于同一套物品索引。
2. info 中 SID 与 item/index 的映射一致。
3. tokenizer 扩词后每个 SID 子 token 的编码符合预期。
4. 不改变已有训练、验证、测试划分，不使用测试集调参。
5. 三种 SFT 任务截断后仍保留完整监督答案，且没有把答案放进可见输入。

减少训练样本数主要缩短运行时间；控制峰值显存主要依靠 micro batch、序列长度、候选数量和优化器。

## 4. SFT：训练新增 SID 与推荐能力

### 4.1 代码修改

在 [sft.py](https://github.com/AkaliKong/MiniOneRec/blob/main/sft.py) 现有 `TrainingArguments` 中新增或替换以下参数。已有同名参数直接替换，不能重复传入。

```python
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
optim="adamw_bnb_8bit",
report_to="none",
```

保持 BF16 加载、`freeze_LLM=False` 和 `train_from_scratch=False`。仓库原有训练前的 `model.config.use_cache=False` 应保留。

8-bit 优化器减少优化器状态占用，不会把模型权重量化成 8-bit。全参数 SFT 能直接训练新增 SID embedding，避免一开始使用普通 LoRA 时把新增随机 token 冻结的问题。

### 4.2 起步配置

| 参数 | 值 | 说明 |
|---|---:|---|
| micro_batch_size | 1 | 每次前向和反向的样本数 |
| batch_size | 32 | 当前单卡代码对应累积 32 次 |
| cutoff_len | 256 | 检查截断比例，有余量可升到 512 |
| learning_rate | 5e-5 | 起始值，按验证集选择 |
| num_epochs | 3 | 首轮训练预算，代码可能提前停止 |
| freeze_LLM | False | 训练全模型及新增 SID |
| seed | 42 | 固定第一轮随机种子 |

256 token 不是越短越好。三种任务包含商品文本和历史混合输入，应记录各任务的截断比例；若截断严重，优先调整到 512 或按任务设计保留策略。

### 4.3 运行命令

完成上述代码修改后执行：

```bash
python sft.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --train_file "data/Amazon/train/${dataset_name}.csv" \
  --eval_file "data/Amazon/valid/${dataset_name}.csv" \
  --sid_index_path "data/Amazon/index/${category}.index.json" \
  --item_meta_path "data/Amazon/index/${category}.item.json" \
  --category "$category" \
  --output_dir outputs/sft_05b_12gb \
  --micro_batch_size 1 \
  --batch_size 32 \
  --cutoff_len 256 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --seed 42 \
  --train_from_scratch False \
  --freeze_LLM False
```

短训练阶段可在 `TrainingArguments` 临时设置 `max_steps=50`，并将评估和保存间隔设为适当的整数步数，例如 25；保证至少经历一次验证和 checkpoint 保存。正式训练时移除 `max_steps` 限制，恢复预定评估策略。

完成 SFT 后，先按第 7 节单独评测 SFT checkpoint，记录基线，再启动 RL。

## 5. RL：LoRA GRPO 的必要改造

### 5.1 删除冗余模型加载

[rl.py](https://github.com/AkaliKong/MiniOneRec/blob/main/rl.py) 在创建 Trainer 前额外加载了一个 `llm_model`，随后 Trainer 又加载训练模型。本方案只使用 `ranking` reward，不需要前一份完整模型。

删除这两行：

```python
llm_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
device = llm_model.device
```

替换为：

```python
device = torch.device("cuda:0")
```

若保留 semantic reward 分支，也将其中 `.to(llm_model.device)` 改成 `.to(device)`，确保没有残留引用。ranking 路径无须加载 SASRec 或语义奖励模型。

### 5.2 显式指定加载精度与长度

在现有 `GRPOConfig` 中新增或替换：

```python
model_init_kwargs={
    "torch_dtype": torch.bfloat16,
    "use_cache": False,
},
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
max_prompt_length=256,
max_completion_length=16,
use_vllm=False,
optim="adamw_torch",
report_to="none",
save_total_limit=2,
```

保留 `bf16=True`。该开关负责训练混合精度，不能代替 `model_init_kwargs` 的权重加载精度设置。

`max_completion_length=16` 是待核对的起步值。用保存后的 SFT tokenizer 检查全部目标 SID，并计入实际模板中需要生成的换行和 EOS；若最大值超过 16，需同步调高 RL 和评测长度。

当前 Trainer 会对 prompt token 左侧截断。抽查截断后任务说明、近期历史与 response 前缀，确认任务仍可辨认。奖励映射继续使用原始 prompt，不应随意重写 prompt 字符串而不更新映射。

### 5.3 注入 LoRA

新增导入并在构建 Trainer 前创建：

```python
from peft import LoraConfig

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

在现有 `ReReTrainer(...)` 参数列表增加 `peft_config=peft_config`，保留其他参数。

RL 必须加载本轮训练后的完整 SFT checkpoint。不要重新扩词，也不要回到原始 Qwen checkpoint；SID embedding 已在 SFT 中完成学习，本阶段冻结它们，仅更新 LoRA。

[minionerec_trainer.py](https://github.com/AkaliKong/MiniOneRec/blob/main/minionerec_trainer.py) 已包含 PEFT 分支：关闭 adapter 后使用冻结的 SFT 底座计算参考策略，因而不用再保存一份独立参考模型。

必须设置 `sync_ref_model=False`，否则原有参考模型同步逻辑不适用于此处的无独立参考模型路径。构造完成后可检查：

```python
trainer.model.print_trainable_parameters()
assert trainer.ref_model is None
```

短训练中还要确认 LoRA 梯度非零、参数确实更新。关闭 adapter 的输出应对应冻结 SFT 模型；checkpoint 恢复后也应验证这一点。

### 5.4 运行命令

```bash
python rl.py \
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
  --seed 42
```

原脚本会将 WandB 设为 offline；本方案将 Trainer 的 `report_to` 改为 `"none"`，无需在线实验追踪服务。

### 5.5 batch 与奖励的关键约束

- 当前 Trainer 要求 `per_device_batch_size × GPU 数量` 能被 `num_generations` 整除，训练和验证均应满足；梯度累积不参与这项检查。
- 单卡 `batch=4, generations=4` 表示一个 prompt 的四条候选轨迹，不是四个独立用户。累积八次约为八个 prompt、32 条轨迹。
- 显存不足时可以同时降到 `batch=2, generations=2`，但只适合作为流程验证起点，奖励稀疏问题会更明显。
- 当前 ranking reward 在整组都未命中目标时没有有效的组内差异。应记录组命中率、组内奖励标准差或非零 advantage 比例，不能仅凭 loss 有输出判定 RL 有效。
- 若绝大多数组没有学习信号，优先改善 SFT；显存允许时将训练 batch、验证 batch、候选数量一起提高到 8。
- 保持原有约束生成实现，抽查候选合法性和去重结果；不要将关闭约束解码作为省显存手段。
- `beta=0` 并不自动消除原实现的参考模型及参考 log-prob 计算；本方案通过 PEFT 分支减少模型副本。

## 6. 合并 RL adapter

PEFT 模型调用 `save_pretrained()` 保存的是 adapter。原评测脚本按完整模型加载，因此需要合并后再评测。

先结束训练进程，再在独立进程运行以下 Python 代码；CPU 合并避免与训练模型争抢显存。需要足够主机内存容纳模型加载与合并。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sft_path = "outputs/sft_05b_12gb/final_checkpoint"
adapter_path = "outputs/rl_05b_12gb/final_checkpoint"
merged_path = "outputs/rl_05b_12gb_merged"

base = AutoModelForCausalLM.from_pretrained(
    sft_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()
model.config.use_cache = True
model.save_pretrained(merged_path)
AutoTokenizer.from_pretrained(sft_path).save_pretrained(merged_path)
```

保留 SFT 底座与 adapter 的对应关系。合并前后可以对少量样本比较输出或 logits，允许正常数值误差，确认 adapter 没有加载到错误底座。

## 7. 单卡评测

不使用默认八卡评测脚本，直接调用 [evaluate.py](https://github.com/AkaliKong/MiniOneRec/blob/main/evaluate.py) 和 [calc.py](https://github.com/AkaliKong/MiniOneRec/blob/main/calc.py)。

### 7.1 SFT 基线

```bash
mkdir -p outputs/eval

python evaluate.py \
  --base_model outputs/sft_05b_12gb/final_checkpoint \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/test/${dataset_name}.csv" \
  --result_json_data outputs/eval/sft.json \
  --batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 16 \
  --length_penalty 0.0

python calc.py \
  --path outputs/eval/sft.json \
  --item_path "data/Amazon/info/${dataset_name}.txt"
```

### 7.2 RL 结果

```bash
python evaluate.py \
  --base_model outputs/rl_05b_12gb_merged \
  --info_file "data/Amazon/info/${dataset_name}.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/test/${dataset_name}.csv" \
  --result_json_data outputs/eval/rl.json \
  --batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 16 \
  --length_penalty 0.0

python calc.py \
  --path outputs/eval/rl.json \
  --item_path "data/Amazon/info/${dataset_name}.txt"
```

若第 5 节的 token 长度检查要求超过 16，应修改以上两组评测命令。SFT 与 RL 使用相同数据、beam 数、生成长度和指标计算逻辑。

20 beams 最多报告 @20。需要 @50 时，两种模型均使用 50 beams 重新评测；不能用 20 个候选冒充 @50。

### 7.3 合法性检查

除查看 `CC`，还要检查全部候选是否属于 info 定义的合法 SID 集合，并统计重复候选。当前 calc 实现在遇到正确答案后停止扫描该样本，因而 `CC=0` 不足以证明所有候选均合法。

如果合法率异常，先核对 tokenizer、response 前缀、EOS、info/index 一致性和约束解码行为，再解释 HR/NDCG。不要在约束解码失效时讨论 RL 效果。

## 8. 显存测量与 OOM 处理

每个阶段单独运行，结束后释放进程。记录训练、验证、生成和保存阶段，而不只测模型加载完成时的占用。

可以在训练开始前、结束后分别加入：

```python
# 训练开始前
torch.cuda.reset_peak_memory_stats()

# 训练结束后
torch.cuda.synchronize()
print("peak allocated GiB:", torch.cuda.max_memory_allocated() / 2**30)
print("peak reserved GiB:", torch.cuda.max_memory_reserved() / 2**30)
```

这些是 PyTorch 分配器统计；还应记录设备整体显存占用，以覆盖 CUDA 上下文和其他分配。不要把 allocated 数值等同于设备总占用。

| 问题 | 优先处理 |
|---|---|
| SFT OOM | 确认 micro batch=1、8-bit 优化器和梯度检查点生效；检查实际长度与 logits 开销 |
| RL 加载即 OOM | 检查冗余 llm_model 已删除、加载精度显式指定、ref_model 为 None |
| RL 生成 OOM | 4/4 同时降为 2/2；核对生成长度；确认无额外训练中评测 |
| RL 反向 OOM | 检查仅 LoRA 参数可训练、梯度检查点生效，必要时缩短 prompt |
| 评测 OOM | batch 保持 1，先完成 @20；@50 单独安排 |
| loss/梯度 NaN | 检查精度支持、学习率、奖励统计和异常样本 |
| RL 没有改善 | 检查组命中率和 advantage、SFT 质量，再调整候选数和学习率 |

梯度累积降低实现相同有效 batch 所需的 micro batch，但不会减少一次生成必须同时保留的候选组。也不能仅靠增加梯度累积绕过 Trainer 的整除检查。

如果上述 SFT 配置仍无法满足显存，可进一步设计 SFT LoRA 或仅训练 SID 的过渡方案，但必须保证新增 token 可训练并正确保存；这属于额外实现分支，不应直接把所有 embedding 冻结。

## 9. 分阶段验收与实验记录

| 阶段 | 验收标准 |
|---|---|
| 环境 | 关键导入通过，模型可加载，准确记录依赖版本和 GPU 型号 |
| 数据 | SID 映射一致，token 扩充正确，监督答案完整 |
| SFT 短训练 | 连续几十个 optimizer step 无 OOM/NaN，完成至少一次验证和保存 |
| SFT 正式训练 | 有效 SID 输出，非零 HR/NDCG，选定基线 checkpoint |
| RL 短训练 | 只有 LoRA 参数可训练，ref_model 为 None，有非零 advantage 且参数更新 |
| 合并 | adapter 使用正确 SFT 底座和 tokenizer，合并模型可加载 |
| 最终评测 | 相同测试集与解码配置，报告指标、全部候选合法率和资源使用 |

建议保存以下结果表：

| 实验 | seed | prompt 长度 | RL 候选数 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | 合法率 | 峰值显存 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 42 | 256 | — | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| SFT+LoRA RL | 42 | 256 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

第一轮目标是完成闭环并确认 RL 有有效更新。若要对“RL 稳定提升”作结论，应增加多个随机种子，报告均值与波动；不应仅凭一次实验下结论。

## 10. 可选：完整重建 SID 阶段

第一轮成功后，如需从原始商品文本开始复现：

1. 沿用选定数据集的预处理和划分规则。
2. 选择与原方案一致的文本编码器；如果更换为小编码器，明确记录为另一项方法变化。
3. 单卡分批编码 title+description，逐批保存 embedding。编码器是否能放入 12GB 需独立评估，不能由 0.5B 推荐模型大小推断。
4. 使用原有三层 RQ-VAE 结构，从较小 batch 起步，按实际 embedding 维度和模型结构测量显存；不要直接沿用文档中的大 batch。
5. 导出 SID 并检查碰撞处理、唯一性和物品覆盖情况。
6. 用新索引重新转换 train/valid/test、info 和元数据，保证整套映射一致。
7. 从原始 Qwen 模型重新执行 SFT 和 RL，不复用旧 SID 训练得到的 checkpoint。

若选择 RQ-Kmeans 等替代方案，单独标注量化方法变化。完整 SID 重建仍需要根据所选编码器和具体数据规模补充配置，不能视为已经通过显存验证。

## 11. 与原方案的差异及已知限制

- 默认多卡训练改为 12GB 单卡，吞吐会下降。
- 第一轮复用 SID，不覆盖文本编码和量化训练的复现。
- SFT 使用更小 batch、较短序列和 8-bit 优化器。
- RL 从全参数更新改为 LoRA，参考策略固定为 SFT 底座，关闭参考模型同步。
- RL 候选数量从默认 16 降到 4，可能增加奖励稀疏性。
- 依赖兼容性、显存峰值和推荐指标尚未实测。
- 本方案未给出训练耗时承诺；时间受具体 GPU、数据规模、长度、评测频率及磁盘速度影响。

## 12. 参考资料

- [MiniOneRec 项目与 README](https://github.com/AkaliKong/MiniOneRec)
- [SFT 实现](https://github.com/AkaliKong/MiniOneRec/blob/main/sft.py)
- [RL 入口](https://github.com/AkaliKong/MiniOneRec/blob/main/rl.py)
- [自定义 GRPO Trainer](https://github.com/AkaliKong/MiniOneRec/blob/main/minionerec_trainer.py)
- [Qwen2.5-0.5B 官方模型页](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [PEFT LoRA 官方文档](https://huggingface.co/docs/peft/main/en/package_reference/lora)

以上源码链接指向 main 分支，内容可能更新；实际实施时应记录并固定使用的仓库 commit。
