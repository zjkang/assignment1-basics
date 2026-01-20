# 超参数配置管理 📋

本目录包含不同的超参数配置文件，用于运行多个实验。

---

## 📁 **可用配置**

### **1. baseline.json** (基准配置)
- **模型**: d_model=512, num_layers=4, num_heads=16
- **优化器**: lr=0.0002, batch_size=32
- **用途**: 基准实验，用于对比

### **2. large_model.json** (大模型)
- **模型**: d_model=768, num_layers=6, num_heads=16
- **优化器**: lr=0.0001, batch_size=24 (减小以适应内存)
- **用途**: 测试更大模型的性能

### **3. high_lr.json** (高学习率)
- **模型**: 与 baseline 相同
- **优化器**: lr=0.0005 (2.5倍于 baseline)
- **用途**: 测试学习率对训练的影响

### **4. small_fast.json** (小模型快速训练)
- **模型**: d_model=256, num_layers=2, num_heads=8
- **优化器**: lr=0.0003, batch_size=64
- **用途**: 快速原型验证，调试代码

---

## 🚀 **使用方法**

### **方法 1: 指定配置文件**

```bash
cd /Users/zhengjiankang/Downloads/cs336/assignment1-basics

# 设置 API key
export WANDB_API_KEY="your_wandb_api_key_here"

# 运行 baseline 配置
uv run python scripts/train_with_wandb.py --config configs/baseline.json

# 运行大模型配置
uv run python scripts/train_with_wandb.py --config configs/large_model.json

# 运行高学习率配置
uv run python scripts/train_with_wandb.py --config configs/high_lr.json

# 运行小模型配置
uv run python scripts/train_with_wandb.py --config configs/small_fast.json
```

### **方法 2: 自定义实验名称**

```bash
# 使用自定义名称
uv run python scripts/train_with_wandb.py \
    --config configs/baseline.json \
    --experiment-name "my-first-experiment"

# 不指定名称时，会自动使用: {config_name}-{timestamp}
# 例如: "baseline-20260119-153045"
```

---

## 📊 **在 wandb 中查看结果**

所有实验会自动记录到 wandb，并带有标签：

- **Tag 1**: 配置文件名（如 `baseline`, `large_model`）
- **Tag 2**: `tinystories`

你可以在 wandb dashboard 中按标签筛选和对比实验：
- https://wandb.ai/zk299-new-york-university/cs336-transformer-lm

---

## 🎯 **推荐实验流程**

### **1. 快速验证（5-10分钟）**
```bash
# 使用 small_fast 配置快速验证代码
uv run python scripts/train_with_wandb.py --config configs/small_fast.json
```

### **2. 基准实验（~30分钟）**
```bash
# 运行 baseline 作为对比基准
uv run python scripts/train_with_wandb.py --config configs/baseline.json
```

### **3. 超参数探索**
```bash
# 并行运行多个配置（在不同终端）
# 终端 1
uv run python scripts/train_with_wandb.py --config configs/high_lr.json

# 终端 2
uv run python scripts/train_with_wandb.py --config configs/large_model.json
```

### **4. 对比结果**
在 wandb 中：
1. 选择多个实验
2. 点击 "Compare" 按钮
3. 查看 loss 曲线、学习率变化等指标

---

## ✏️ **创建自定义配置**

### **步骤 1: 复制现有配置**
```bash
cp configs/baseline.json configs/my_config.json
```

### **步骤 2: 修改参数**
编辑 `my_config.json`，例如：
```json
{
  "model": {
    "d_model": 1024,
    "num_layers": 8
  },
  "optimizer": {
    "lr": 0.0001
  },
  "train": {
    "batch_size": 16,
    "save_path": "./checkpoints/my_config/"
  }
}
```

### **步骤 3: 运行实验**
```bash
uv run python scripts/train_with_wandb.py --config configs/my_config.json
```

---

## 💡 **配置参数说明**

### **模型参数 (model)**
- `vocab_size`: 词表大小（默认 10000）
- `context_length`: 上下文长度（默认 256）
- `d_model`: 模型维度（越大越强，但越慢）
- `num_layers`: Transformer 层数
- `num_heads`: 注意力头数（必须能整除 d_model）
- `d_ff`: 前馈网络维度（通常是 d_model 的 2.5-4 倍）

### **优化器参数 (optimizer)**
- `lr`: 学习率（通常 0.0001-0.0005）
- `min_lr`: 最小学习率
- `warmup_iters`: 学习率预热步数
- `weight_decay`: 权重衰减（L2 正则化）
- `clip_grad_norm`: 梯度裁剪阈值

### **训练参数 (train)**
- `batch_size`: 批次大小（越大越快，但需要更多内存）
- `train_steps`: 训练步数
- `val_interval`: 验证间隔
- `save_interval`: 保存检查点间隔
- `save_path`: 检查点保存路径

---

## 🔍 **常见问题**

### Q: 如何恢复训练？
在配置文件中设置：
```json
"train": {
  "resume_checkpoint": "./checkpoints/baseline/ckpt_iter3000.pt"
}
```

### Q: 如何减少显存使用？
调整：
- 减小 `batch_size`
- 减小 `d_model` 或 `num_layers`
- 减小 `context_length`

### Q: 如何加快训练速度？
调整：
- 增大 `batch_size`（如果显存足够）
- 减小模型大小
- 减少 `log_interval`

---

## 📈 **实验记录模板**

建议在笔记中记录：

```markdown
## 实验 1: Baseline
- **配置**: configs/baseline.json
- **目标**: 建立基准
- **结果**: Final train loss: 2.34, val loss: 2.56
- **观察**: 训练稳定，收敛良好

## 实验 2: High LR
- **配置**: configs/high_lr.json
- **目标**: 测试更高学习率的影响
- **结果**: Final train loss: 2.12, val loss: 2.48
- **观察**: 比 baseline 收敛更快，loss 更低
```

---

祝实验顺利！🎉
