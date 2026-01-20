"""
使用 Weights & Biases 进行实验追踪的训练脚本

安装 wandb:
    pip install wandb

使用方法:
    python train_with_wandb.py
"""

import os
import sys
import json
import torch
import pathlib
import numpy as np
import argparse
from tqdm import tqdm
from utils import _to_device_and_compile
from model import BasicsTransformerLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import *
import time

# 尝试导入 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. Install with: pip install wandb")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "valid.dat")
CONFIG_PATH = "scripts/config.json"

def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")
    return arr

# 注释掉原来的实现，直接使用 adapters.py 中的 run_get_batch
# def get_batch(memmap_arr, batch_size, context_length):
#     N = len(memmap_arr)
#     ix = np.random.randint(0, N-context_length-1, size=(batch_size,))
#     x = np.stack([memmap_arr[i:i+context_length] for i in ix])
#     y = np.stack([memmap_arr[i+1:i+context_length+1] for i in ix])
#     return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    """
    验证集迭代器：顺序遍历数据集（不使用随机采样）
    注意：这里不能用 run_get_batch，因为验证需要顺序遍历而不是随机采样
    """
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
def main():
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="Train Transformer LM with W&B tracking")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to config file (default: scripts/config.json)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for W&B (default: auto-generated from timestamp)"
    )
    cmd_args = parser.parse_args()
    
    # 1. 加载配置
    config_path = cmd_args.config
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config["model"])  

    # 合并所有参数
    params = {}
    for group in config.values():
        params.update(group)

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = DotDict(params)

    # 2. 初始化 Weights & Biases
    if WANDB_AVAILABLE:
        # 从配置文件名推断实验名称
        config_name = pathlib.Path(config_path).stem  # e.g., "baseline", "large_model"
        experiment_name = cmd_args.experiment_name or f"{config_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # 添加额外的元数据方便对比
        notes = f"Config: {config_name} | Model: d_model={params['d_model']}, layers={params['num_layers']} | Optim: lr={params['lr']}, bs={params['batch_size']}"
        
        wandb.init(
            project="cs336-transformer-lm",  # 项目名称
            name=experiment_name,  # 运行名称
            config=params,  # 记录所有超参数
            tags=[config_name, "tinystories"],  # 标签（自动包含配置名称）
            notes=notes,  # 简短描述，方便快速查看
        )
        print(f"🚀 W&B run: {experiment_name}")
        print(f"📝 Notes: {notes}")
        # 注意: wandb.watch() 与 torch.compile() 不兼容，已禁用
        # 如果需要梯度统计，请在 utils.py 中禁用 torch.compile
        # wandb.watch(model, log='all', log_freq=100)

    model, device = _to_device_and_compile(model)
    os.makedirs(args.save_path, exist_ok=True)

    # 3. 加载数据集
    train_data = get_memmap_dataset(TRAIN_DATA_PATH)
    val_data = get_memmap_dataset(VAL_DATA_PATH)

    # 4. 构建优化器
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. 恢复断点
    start_iter = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint {args.resume_checkpoint}")
        resume_ckpt_path = pathlib.Path(__file__).resolve().parent.parent / f"checkpoints/ckpt_iter{args.resume_checkpoint}.pt"
        start_iter = run_load_checkpoint(resume_ckpt_path, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # 6. 训练循环
    start_time = time.time()
    
    for iteration in tqdm(range(start_iter, args.train_steps), desc="Training"):
        model.train()
        
        # 获取批次并训练
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = run_cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        # 更新学习率
        lr = run_get_lr_cosine_schedule(
            iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # 记录训练指标
        if (iteration + 1) % args.log_interval == 0:
            current_time = time.time() - start_time
            
            metrics = {
                'train/loss': loss.item(),
                'train/learning_rate': lr,
                'train/iteration': iteration + 1,
                'train/wallclock_time': current_time,
            }
            
            if WANDB_AVAILABLE:
                wandb.log(metrics, step=iteration + 1)
            
            print(f"iter {iteration+1:05d}: TRAIN loss = {loss.item():.4f}, "
                  f"lr = {lr:.6f}, time = {current_time:.2f}s")

        # 验证
        if (iteration+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in memmap_val_iterator(val_data, args.batch_size, args.context_length):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_loss = run_cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1)
                    )
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_losses)
                current_time = time.time() - start_time
                
                # 记录验证指标
                val_metrics = {
                    'val/loss': val_loss_mean,
                    'val/iteration': iteration + 1,
                    'val/wallclock_time': current_time,
                }
                
                if WANDB_AVAILABLE:
                    wandb.log(val_metrics, step=iteration + 1)
                
                print(f"iter {iteration+1:05d}: VALID loss = {val_loss_mean:.4f}, "
                      f"time = {current_time:.2f}s")

        # 保存检查点
        if (iteration+1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{iteration+1}.pt")
            run_save_checkpoint(model, optimizer, iteration+1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")
            
            # 保存检查点到 wandb
            if WANDB_AVAILABLE and iteration+1 == args.train_steps:
                wandb.save(ckpt_name)
    
    # 训练完成 - 记录最终指标
    total_time_minutes = (time.time() - start_time) / 60
    
    if WANDB_AVAILABLE:
        # 记录最终的 summary metrics（会显示在表格中）
        wandb.run.summary["final_train_loss"] = loss.item() if 'loss' in locals() else None
        wandb.run.summary["final_val_loss"] = val_loss_mean if 'val_loss_mean' in locals() else None
        wandb.run.summary["total_time_minutes"] = total_time_minutes
        wandb.run.summary["final_iteration"] = iteration + 1 if 'iteration' in locals() else args.train_steps
        
        wandb.finish()
    
    print(f"\n✅ Training completed! Total time: {total_time_minutes:.2f} minutes")

if __name__ == "__main__":
    main()
