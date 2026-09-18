"""
MiniOneRec 环境自检脚本 (Windows 单卡)
用法: python scripts_win\00_check_env.py   (在 MiniOneRec 目录下运行)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[OK]   {name}: {msg}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


# 1) PyTorch + GPU
def _torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return (f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
            f"sm_{cap[0]}{cap[1]} | {vram:.1f} GB | bf16={torch.cuda.is_bf16_supported()}")


def _bf16_matmul():
    import torch
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    return f"bf16 矩阵乘 OK {tuple(y.shape)} {y.dtype}"


# 2) 训练栈
def _stack():
    import accelerate, bitsandbytes, datasets, sklearn, transformers, trl
    return (f"transformers {transformers.__version__} | trl {trl.__version__} | "
            f"datasets {datasets.__version__} | accelerate {accelerate.__version__} | bnb {bitsandbytes.__version__}")


def _paged_optim():
    import bitsandbytes as bnb
    import torch
    p = torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
    opt = bnb.optim.PagedAdamW32bit([p], lr=1e-3)  # rl.py 中 optim="paged_adamw_32bit"
    p.grad = torch.randn_like(p)
    opt.step()
    return "PagedAdamW32bit 单步 OK (RL 阶段依赖)"


# 3) 项目模块
def _proj():
    import LogitProcessor, data, minionerec_trainer, sasrec  # noqa: F401
    return "data / LogitProcessor / minionerec_trainer / sasrec 均可导入"


# 4) 数据文件
def _data():
    cat = "Industrial_and_Scientific"
    files = {
        "index": f"data/Amazon/index/{cat}.index.json",
        "item": f"data/Amazon/index/{cat}.item.json",
        "info": f"data/Amazon/info/{cat}_5_2016-10-2018-11.txt",
        "train": f"data/Amazon/train/{cat}_5_2016-10-2018-11.csv",
        "valid": f"data/Amazon/valid/{cat}_5_2016-10-2018-11.csv",
        "test": f"data/Amazon/test/{cat}_5_2016-10-2018-11.csv",
    }
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"缺少 {missing}")
    return f"{cat} 的 6 个数据文件齐全"


def _model():
    p = "models/Qwen2.5-0.5B"
    if not os.path.isdir(p):
        raise FileNotFoundError("未下载 (先运行 scripts_win/01_download_model.ps1)")
    return "本地基座模型已就绪"


print(f"MiniOneRec 环境自检 | Python {sys.version.split()[0]} | {sys.executable}\n")
check("PyTorch / GPU", _torch)
check("GPU 计算", _bf16_matmul)
check("训练栈版本", _stack)
check("bitsandbytes 优化器", _paged_optim)
check("项目模块导入", _proj)
check("数据集文件", _data)
check("基座模型", _model)
print("\n全部通过，可以开始复现流程。" if ok else "\n存在未通过项，请按上面提示处理。")
sys.exit(0 if ok else 1)
