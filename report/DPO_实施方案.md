# MiniOneRec Sapphire：DPO 后训练实施方案

> 版本：v1.0  
> 日期：2026-09-18  
> 目标仓库：[hamelnn/minionerec_sapphire](https://github.com/hamelnn/minionerec_sapphire)  
> 状态：开发实施设计。本文中的新增模块、命令和训练代码为待实现方案，未在本轮创建到目标仓库，未执行训练、测试或效果验证。超参数是实验初值，不代表最优值。

## 1. 目标与方案决策

在现有 Qwen2.5-0.5B SFT 模型上，用 LoRA DPO 替换 GRPO 后训练阶段，保留原有合法 SID 约束解码和基线评估能力。

DPO 是离线偏好优化，不是在线策略梯度 RL。训练阶段使用已构造的正负答案对，不需要在线 rollout、GRPO advantage、Critic 或奖励模型。

第一版采用以下固定决策：

- 仅训练“用户历史 → 下一物品 SID”，其他辅助任务保留在 SFT 阶段。
- 使用现有全参数 SFT checkpoint 作为冻结底座和参考策略，新增一个 DPO LoRA adapter。
- 每条交互最多构造两个偏好对：一个模型困难负例，一个随机合法负例。
- 首先使用标准 sigmoid DPO，不混入辅助 loss，明确测量 DPO 本身的作用。
- 负例生成和最终推理使用 SID 约束；训练使用完整词表下的标准序列 log-prob。
- 首轮保留现有 SID 编码，结果明确按 SID 级报告，碰撞目标单独分层；消歧编码作为独立升级实验。
- 用验证集 NDCG@10 选择 checkpoint，不能只按 DPO loss 或偏好准确率选择。

## 2. 当前基础与必须处理的问题

### 2.1 可复用基础

| 项目 | 当前基础 |
| --- | --- |
| 模型 | Qwen2.5-0.5B，已完成 SID 扩词和全参数 SFT |
| 后训练 | LoRA r=16、alpha=32，七个投影层 |
| 环境 | 仓库记录为 torch 2.6.0、transformers 4.57.1、trl 0.24.0、peft 0.21.0 |
| 设备 | RTX 4070 SUPER 12GB 单卡 |
| 数据规模 | train 36,259；valid 4,532；test 4,533 条交互 |
| 输出 | 当前 SID 为三个 token，另有换行与 EOS |
| 推理 | 合法 SID 前缀约束、20 beams |

依据：[README](https://github.com/hamelnn/minionerec_sapphire/blob/main/README.md)、[实验记录](https://github.com/hamelnn/minionerec_sapphire/blob/main/report/复现实验记录.md)。实施前应固定实际仓库 commit、模型文件和数据哈希；本文不把上游基础 commit 当作当前仓库版本。

### 2.2 必须绕开的旧实现问题

1. **目标标签覆盖。** 原奖励通过全局历史→目标字典查找，同一历史对应不同目标时会覆盖。DPO chosen 必须直接来自当前交互行，不复用该查表链路。
2. **SID 碰撞。** 前序数据核对发现 3,686 个物品对应 3,670 个唯一 SID，15 个碰撞组涉及 31 个物品；317 条测试交互的目标属于碰撞组。不能把这些 SID 的命中直接解释为精确 item_id 命中。
3. **多任务提示词差异。** 从 SFT/推荐评估提取完整模板，包含 instruction、标点、换行与 response 前缀，不能用简化版 RL prompt 代替。
4. **冻结底座与梯度检查点。** 仓库记录过 PEFT 冻结底座时的 reentrant checkpoint 断梯度问题。首版关闭梯度检查点，后续独立验证后再启用。

上述统计来自前序对发布数据的分析，实际开发需随数据版本重新核对；不等同于当前抽样训练集的受影响数量。

## 3. 端到端流程

```text
固定 SFT checkpoint / tokenizer / SID 索引 / 数据划分
                         │
             共享推荐 prompt formatter
                         │
        SFT 对训练历史生成 Top-20 合法 SID
                         │
       真实下一物品 + 候选池 + 随机合法负例
                         │
         过滤歧义、去重、构造偏好 JSONL
                         │
         固定 SFT 参考概率预计算与缓存
                         │
               LoRA DPO 训练
                         │
        验证集约束解码与 NDCG@10 选模
                         │
          adapter 合并、最终测试评估
```

## 4. 数据设计

### 4.1 原始样本与稳定标识

从现有 train/valid CSV 逐行读取原始物品 ID、历史 SID 和目标 SID。列表字段应使用结构化解析或 `ast.literal_eval`，不要新增 `eval` 调用。

每条交互生成稳定 sample_id，例如由“数据文件哈希、split、原始行号”组成。每个偏好对生成 pair_id，由 sample_id 和负例标识组成。不能只用 prompt 文本作为唯一键，因为相同历史可能对应多个目标。

### 4.2 训练数据格式

标准 DPO 输入只需三个字符串字段：

```json
{
  "prompt": "完整共享模板生成的提示词，结尾为 ### Response:\n",
  "chosen": "<a_12><b_34><c_56>\n",
  "rejected": "<a_12><b_34><c_78>\n"
}
```

以上数字仅为格式示例，真实 token 必须存在于固定 SID 索引和 SFT tokenizer 中。不得在 chosen/rejected 中重复拼接 prompt 或 response 标题。

审计信息单独保存，不依赖 Trainer 保留自定义字段：

| 字段 | 含义 |
| --- | --- |
| pair_id / sample_id / split | 样本定位与划分 |
| user_id / source_row | 用户与原始行号 |
| target_item_id / target_sid | 该行观测目标 |
| rejected_sid / rejected_item_ids | 负例及对应物品集合；碰撞时不可随意指定一个物品 |
| negative_source | model_hard / random / prefix_hard |
| candidate_rank / candidate_score | 模型候选排名及可选序列分数 |
| prompt_hash / candidate_model_hash | 复现提示词和生成模型 |

### 4.3 候选生成

使用固定 SFT checkpoint，在训练交互上生成候选池：

- `num_beams=20`、`num_return_sequences=20`。
- `do_sample=False` 必须作为 generate 的显式参数传入，延续现有评估实现。
- `max_new_tokens=16`、`length_penalty=0.0`。
- 使用仓库 SID 约束，保存按模型分数排序的合法、去重候选。
- 每个候选输出与 sample_id 绑定，支持断点续跑。
- 每次 generate 创建新的有状态约束 processor，避免步数状态跨批次复用。

完整数据量候选生成也需要计算预算。首轮只生成固定 6,000 条训练交互；后续扩大全量。候选池可复用，成本纳入总 GPU 小时。

### 4.4 正负例选择与过滤

chosen 直接取当前行真实目标 SID。先选排序靠前的有效模型错误候选作为困难负例，再选一个随机合法 SID。随机采样的单位应固定并记录；首版按唯一 SID 均匀采样，避免碰撞物品重复加权。

过滤顺序：

1. 规范化 SID 文本并核验其存在于合法目录。
2. 排除与 chosen 相同的 SID，即使物品 ID 不同也不能形成偏好对。
3. 排除同一交互已经使用的 rejected SID。
4. 可保守排除相同历史在训练集中已出现的其他正目标 SID；此集合仅使用训练期信息。
5. 不读取验证或测试未来行为来筛训练负例。
6. 不一律排除历史已交互物品，因为下一物品可能是重复购买。
7. 困难负例不足时用另一个随机有效负例补齐，并记录实际来源；没有任何负例则跳过并统计。

优先保证每条交互两个不同负例，控制交互权重。确实只能生成一对的样本必须统计，不能静默产生不均衡；首版可统一只保留能生成两对的交互，并报告覆盖率。后续若允许变长负例数，需要按原交互归一化权重。

未交互物品属于弱负例，不代表真实负反馈。困难负例只取少量，避免训练集中出现大量互相矛盾的偏好关系。

### 4.5 数据划分与预算

- 训练偏好对仅由原 train 生成。
- 验证偏好对由原 valid 独立生成，固定候选池，仅用于诊断。
- 测试集不参与负例构造、阈值选择或训练。
- 不将同一历史生成的多个偏好对随机拆分到 train/valid。
- 不因相同 prompt 存在于不同原始划分就合并目标字典；保留各行真实目标，报告重复情况。
- 首轮：6,000 条交互，过滤前最多 12,000 对。
- 全量：36,259 条交互，过滤前最多 72,518 对。

## 5. DPO 目标与分词约定

定义答案序列概率：

\[
\ell_\theta(y\mid x)=\sum_{t\in\mathrm{completion}}\log\pi_\theta(y_t\mid x,y_{<t})
\]

定义参考调整后的偏好 margin：

\[
m=[\ell_\theta(y^+\mid x)-\ell_\theta(y^-\mid x)]-[\ell_{\rm ref}(y^+\mid x)-\ell_{\rm ref}(y^-\mid x)]
\]

\[
L_{\rm DPO}=-\log\sigma(\beta m)
\]

实现要求：

- 仅累计 completion 的 log-prob，不累计 prompt 和 padding。
- 使用序列 log-prob 之和，首版不引入自定义长度归一化。
- chosen/rejected 保留与 SFT 一致的换行。
- TRL 0.24.0 的 DPO 预处理追加 EOS，原始答案不要再手动附加 EOS。
- 真实 EOS 必须保留在答案 mask 中；pad 与 EOS 共用 token ID 时不能按 token 值统一屏蔽。
- 检查单独编码 prompt、completion 后的拼接是否符合 SFT 使用的 token 边界，尤其是末尾换行和 SID 起始 token。
- SID tokenizer 必须来自 SFT checkpoint，不在 DPO 阶段扩词或重排 token。
- prompt 左截断保留近期历史和 response 前缀；答案不得截断。

训练采用完整词表概率，候选生成和推理采用 SID 约束。第一版不自定义约束归一化 DPO loss；如后续研究该变体，策略与参考概率都要采用相同约束重新计算，不能复用旧概率缓存。

## 6. 模型与参考策略

| 角色 | 配置 |
| --- | --- |
| 底座 | 现有全参数 SFT checkpoint，冻结 |
| 策略 | SFT 底座 + 新建 DPO LoRA adapter |
| 参考策略 | 禁用 DPO adapter 后的原 SFT |
| 可训练参数 | 七个投影层的 LoRA 参数 |
| 冻结参数 | 原模型、embedding、lm_head |

使用 PEFT 时，标准 DPOTrainer 支持 `ref_model=None` 并禁用 adapter 获取参考概率。首版不设置 `modules_to_save`，避免改变 embedding 或输出头导致参考策略不再固定。

启用 `precompute_ref_log_probs=True`。缓存绑定以下元信息：SFT checkpoint 哈希、tokenizer 哈希、偏好数据哈希、prompt 模板版本、最大长度、EOS 规则及概率口径。候选或模板变更后重新预计算。

注意：Trainer 自动预计算不等于已生成可跨作业复用的独立缓存文件。若要跨训练进程复用，应显式导出带参考概率列的数据及 manifest，并校验 pair_id/行序和版本一致性。

## 7. 初始配置与训练骨架

### 7.1 建议初值

| 参数 | 初值 | 后续实验 |
| --- | --- | --- |
| LoRA r / alpha | 16 / 32 | 首轮不变 |
| LoRA dropout | 0 | 与固定参考策略行为对齐 |
| learning_rate | 5e-6 | 1e-6、1e-5 |
| beta | 0.1 | 0.05、0.2 |
| epochs | 1 | 验证后决定是否延长 |
| micro-batch | 1 个偏好对 | 测量显存后逐级提高 |
| gradient accumulation | 32 | 提高 micro-batch 时维持有效 batch |
| prompt / completion / total | 256 / 16 / 272 | 统计截断后调整 |
| precision | bf16 | 沿用现有环境 |
| optimizer | adamw_torch | 首轮不增加变量 |
| gradient checkpointing | 关闭 | 单独验证后再开启 |
| eval / save interval | 100 优化步 | 小规模训练可改为 25 步 |

DPO beta 与 GRPO 中 KL loss 的权重不是可直接等值迁移的超参数。每个偏好对含正负两条序列，batch=1 不等于只处理一条序列。显存和耗时需实测，不能沿用 GRPO 的记录值。

### 7.2 训练入口参考骨架

以下代码用于说明训练主体，拟作为新增入口的核心逻辑。它假定偏好 JSONL 已通过校验，参数解析、哈希记录、断点恢复和推荐评估流程需按本文补齐。代码未经本轮运行验证。

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def train_dpo(sft_path, train_jsonl, valid_jsonl, output_dir, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    datasets = load_dataset(
        "json",
        data_files={"train": train_jsonl, "validation": valid_jsonl},
    )
    train_pairs = datasets["train"].select_columns(
        ["prompt", "chosen", "rejected"]
    )
    valid_pairs = datasets["validation"].select_columns(
        ["prompt", "chosen", "rejected"]
    )

    lora_config = LoraConfig(
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

    args = DPOConfig(
        output_dir=output_dir,
        loss_type="sigmoid",
        beta=0.1,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        max_prompt_length=256,
        max_completion_length=16,
        max_length=272,
        truncation_mode="keep_end",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        reference_free=False,
        sync_ref_model=False,
        gradient_checkpointing=False,
        bf16=True,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,
        train_dataset=train_pairs,
        eval_dataset=valid_pairs,
    )
    trainer.model.print_trainable_parameters()
    assert trainer.ref_model is None
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
```

该骨架保存的是训练结束时的 adapter，不自动等于最佳推荐 checkpoint。正式版本必须实现下一节的推荐选模；不能将最终保存结果直接称为“最佳模型”。

## 8. 仓库改造与模块接口

本节文件名均为拟新增接口，不表示已经存在。现有文件引用使用仓库链接。

| 拟新增模块 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `dpo_data.py` | 共享 prompt、逐行标签、规范化、偏好对构造 | CSV、SID 索引、候选池 | 标准偏好记录和审计记录 |
| `scripts_win/build_dpo_pairs.py` | 生成候选、筛负例、固定样本与缓存 | SFT、split CSV、配置 | train/valid JSONL、manifest、过滤统计 |
| `dpo.py` | 参数解析、LoRA DPO、保存、记录资源 | 偏好数据、SFT、配置 | adapter、tokenizer、训练日志 |
| `scripts_win/eval_dpo_checkpoints.py` | 在验证交互上评估保存点并保护最佳模型 | adapters、SFT、valid CSV | checkpoint 指标表、最佳路径 |
| `scripts_win/05_dpo.sh` | 串联经确认的本地训练流程 | 本地路径参数 | 可复现运行记录 |

现有组件处理方式：

- [data.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/data.py)：提取完整共享 prompt；DPO 不使用全局 history2target。
- [evaluate.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/evaluate.py)：复用模型加载、合法候选生成及输出格式；新增 sample_id 与可选候选分数输出。
- [LogitProcessor.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/LogitProcessor.py)：只用于候选构造和推荐评估，不参与 DPO loss。
- [minionerec_trainer.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/minionerec_trainer.py)：保留作 GRPO 对照，不作为 DPO 的父类。
- [merge_adapter.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/scripts_win/merge_adapter.py)：复用 CPU 合并，将 DPO adapter 合并到对应 SFT 底座。
- [calc.py](https://github.com/hamelnn/minionerec_sapphire/blob/main/calc.py)：复用 SID 指标，同时补充去重、碰撞分层和用户分组统计。

## 9. 选模、监控与最终评估

### 9.1 诊断指标

| 类别 | 指标 |
| --- | --- |
| 偏好优化 | DPO loss、参考调整后的 margin、偏好准确率 |
| 原始模型行为 | chosen/rejected log-prob、原始概率差、chosen NLL |
| 推荐效果 | HR@5/10/20、NDCG@5/10/20 |
| 目录与分布 | 合法率、重复候选率、覆盖率、热门/长尾分层 |
| 歧义分析 | 碰撞/无碰撞目标子集的独立指标 |
| 训练资源 | 梯度范数、峰值 allocated/reserved 显存、样本吞吐、GPU 小时 |

参考调整后的 margin 增大，不意味着 chosen 的绝对概率提高；偏好准确率提高也不保证 Top-K 推荐变好。

### 9.2 推荐选模策略

1. 固定验证偏好集，定期计算 loss 与 margin。
2. 每个保存间隔或较稀疏的推荐评估间隔，在固定验证交互子集上运行 Top-20 约束解码。
3. 按验证 NDCG@10 保存独立的最佳 adapter；用全量验证集复核少数候选。
4. 锁定配置后再进行最终测试评估，测试集不用于筛超参数。

存储注意：如果采用训练后统一评估，必须保留所有计划参与比较的 checkpoint；不能用 `save_total_limit=3` 删除尚未评估的模型。若采用训练中评估，则在轮转清理前将当前最佳 adapter 保存到独立位置。参考骨架的轮转配置仅用于闭环，不替代正式选模实现。

### 9.3 对照与指标口径

- 使用同一 SFT 起点、同一 SID 目录、同一测试交互、20 beams 和相同长度配置。
- 报告 SFT、修复标签问题后的 GRPO、DPO；历史原 GRPO 结果可保留但标明版本差异。
- 训练数据量不同时，增加固定交互集合和固定计算预算对照。
- 报告候选构造、参考概率预计算、训练和评估的分项成本。
- 最佳配置至少三个随机种子，报告均值、标准差及按用户分组的配对 bootstrap 区间。
- SID 消歧前不宣称精确物品命中提升；消歧后的正式主指标采用 item_id 级 NDCG@10。

## 10. 实施阶段与验收清单

### 阶段 A：固定基线和数据

- [ ] 固定仓库 commit、SFT checkpoint、tokenizer、SID 索引和数据哈希。
- [ ] prompt 与 SFT/评估模板一致，样本目标来自当前行。
- [ ] chosen/rejected 均为合法 SID，且两者不同。
- [ ] 报告负例来源、过滤数、实际偏好对数、交互覆盖率。
- [ ] 没有利用验证/测试未来行为筛训练负例。
- [ ] 记录 SID 碰撞及其对训练、评估的影响。

### 阶段 B：小规模训练闭环

建议在执行阶段先用 200 条交互构造数据，运行 20～50 个优化步。以下是未来验收要求，本轮未执行。

- [ ] 仅 LoRA 参数可训练，参考底座、embedding 和 lm_head 不变。
- [ ] 固定偏好对的参考 log-prob 在训练前后保持一致。
- [ ] 答案 SID、换行与 EOS 未被截断，prompt/padding 不计入答案概率。
- [ ] LoRA 梯度非零且有限，无 NaN/Inf 或 OOM。
- [ ] adapter 保存、重新加载、合并流程可完成。
- [ ] adapter 模式与合并模型在相同输入上的 log-prob 接近，允许 bf16 误差；近似并列候选可能发生次序变化。
- [ ] 实测记录显存与训练吞吐。

### 阶段 C：首轮效果实验

- [ ] 使用固定 6,000 条交互、最多 12,000 对，训练一个 epoch。
- [ ] 在验证交互上选择 checkpoint，最终保存与最佳保存明确区分。
- [ ] 与相同 SFT 起点比较推荐质量、合法率、覆盖率及成本。
- [ ] 若指标无增益，先分析困难负例、chosen NLL、模板和标签，不直接增加 epoch。

### 阶段 D：全量与消融

| 实验 | 改动 | 目的 |
| --- | --- | --- |
| D0 | SFT 基线 | 量化后训练净收益 |
| D1 | 纯 DPO，困难+随机各一个 | 最小可行方案 |
| D2 | 纯 DPO，仅随机负例，负例总数不变 | 判断困难负例贡献 |
| D3 | D1 + SFT 辅助损失 | 判断是否需要维护 chosen 概率 |
| D4 | D1 扩至全量交互 | 判断数据覆盖收益 |
| D5 | 最佳模型刷新困难负例，再做一轮 | 判断静态负例是否过时 |

一次只改变一个主要因素。若 DPO margin 上升但 chosen log-prob 和推荐效果下降，可试 `L = L_DPO + λ L_SFT`，从 λ=0.1 起步。可使用 TRL 的 `rpo_alpha`，或显式组合 loss，二者择一，避免重复加入辅助损失。该实验应命名为 DPO+SFT，不与纯 DPO 混淆。

迭代困难负例时：策略可延续上一轮 adapter，参考策略继续固定原 SFT，明确记录此决策；重新生成偏好对和参考概率。若改为每轮重置参考策略，应作为另一种实验单独报告。

## 11. 拟定命令接口

以下是开发后的命令契约示例，新增脚本当前尚未实现，不能当作已经验证可执行的命令。路径由实施者在目标仓库内配置。

```bash
# 1. 在 train 上生成偏好对；valid 需要独立执行同样的数据构造流程
python scripts_win/build_dpo_pairs.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --split train \
  --sample 6000 \
  --num_beams 20 \
  --negatives_per_example 2 \
  --seed 42 \
  --output_dir outputs/dpo_pairs

# 2. 训练；实际入口还应接收并记录数据与模型 manifest
python dpo.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --train_jsonl outputs/dpo_pairs/train.jsonl \
  --valid_jsonl outputs/dpo_pairs/valid.jsonl \
  --output_dir outputs/dpo_05b_12gb \
  --learning_rate 5e-6 \
  --beta 0.1 \
  --num_train_epochs 1 \
  --seed 42

# 3. 验证选模后，复用已有合并入口
python scripts_win/merge_adapter.py \
  --sft_path outputs/sft_05b_12gb/final_checkpoint \
  --adapter_path outputs/dpo_05b_12gb/best_adapter \
  --merged_path outputs/dpo_05b_12gb_merged
```

`best_adapter` 必须由推荐选模流程实际生成；不能凭目录名假定存在。构造入口还需通过配置或参数取得 train/valid CSV、SID 索引和 info 文件，不能隐式使用错误品类。

## 12. 风险、停止条件与交付标准

| 现象 | 优先排查与处理 |
| --- | --- |
| loss 降低、推荐指标退化 | 负例质量、chosen NLL、参考 margin 与原始排序是否背离 |
| 偏好对几乎没有差异 | 相同 SID、分词错误、答案被截断、负例过易 |
| 参考概率随训练变化 | 底座意外解冻、adapter 切换错误、缓存版本不匹配 |
| 长尾表现明显下降 | 负例采样分布、热门困难负例比例、训练覆盖 |
| 显存不足 | 降低 micro-batch/预计算 batch；测量 logits 开销；再单独验证梯度检查点 |
| 合法率下降 | tokenizer/索引不一致、合并路径错误、约束 processor 状态污染 |

首轮验收必须先满足数据与训练正确性。效果结论以验证选模后的统一测试协议为准，不预承诺提升百分比。若多随机种子无稳定收益或固定预算下劣于 SFT，应保留 SFT，并回到负例构造和数据覆盖分析。

最终开发交付物应包含：偏好数据与 manifest、构造入口、DPO 训练入口、验证选模入口、最佳 adapter 及 tokenizer、可复现实验配置、分项成本和指标报告。本文本身是一份独立实施文档，不包含已训练模型或已完成的仓库改动。

## 13. 参考资料

- [目标仓库](https://github.com/hamelnn/minionerec_sapphire)
- [仓库 SFT 入口](https://github.com/hamelnn/minionerec_sapphire/blob/main/sft.py)
- [仓库 RL 入口](https://github.com/hamelnn/minionerec_sapphire/blob/main/rl.py)
- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [TRL 0.24.0 DPOTrainer 文档](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)
- [TRL 0.24.0 DPOTrainer 源码](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_trainer.py)

设计依据为本次会话已读取的仓库代码、公开实验记录、前序数据统计及上述版本接口。实施阶段应固定版本并完成验收，不将示例配置当作已经验证的兼容性或效果承诺。
