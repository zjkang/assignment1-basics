# Weights & Biases 实验追踪指南 🚀

使用 W&B 进行专业的实验追踪和可视化。

---

## 🎯 为什么使用 Weights & Biases？

✅ **实时可视化** - 训练过程中实时查看损失曲线  
✅ **自动记录** - 所有指标自动保存，无需手动管理  
✅ **实验对比** - 轻松比较多个实验的结果  
✅ **云端存储** - 所有数据自动备份到云端  
✅ **团队协作** - 轻松分享实验结果  
✅ **专业报告** - 自动生成漂亮的图表和表格  

---

## 📦 安装和设置

### ✅ 当前登录状态

你已成功登录 wandb！

- **用户名**: `zk299`
- **组织**: `zk299-new-york-university`
- **Dashboard**: https://wandb.ai/zk299-new-york-university

### 🔑 API Key 设置方法

#### 方法1：每次手动 export（简单）

```bash
cd /Users/zhengjiankang/Downloads/cs336/assignment1-basics

# 设置 API key
export WANDB_API_KEY="your_wandb_api_key_here"

# 运行训练
uv run python scripts/train_with_wandb.py
```

#### 方法2：添加到 shell 配置（推荐，一劳永逸）

编辑 `~/.zshrc`（如果使用 zsh）或 `~/.bashrc`（如果使用 bash）：

```bash
# 在文件末尾添加
export WANDB_API_KEY="your_wandb_api_key_here"
```

然后重新加载配置：

```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

之后每次打开终端都会自动设置，无需再次 export。

---

## 🚀 快速开始

### 运行训练（带 W&B 追踪）

```bash
cd /Users/zhengjiankang/Downloads/cs336/assignment1-basics

# 设置 API key（如果还没添加到 shell 配置）
export WANDB_API_KEY="your_wandb_api_key_here"

# 运行训练
uv run python scripts/train_with_wandb.py
```

### 查看结果

训练开始后，终端会显示类似这样的信息：

```
wandb: Syncing run experiment-20260119-143052
wandb: ⭐️ View run at https://wandb.ai/your-username/cs336-transformer-lm/runs/abc123
```

**点击链接**，在浏览器中实时查看：
- 📈 损失曲线（训练和验证）
- 📊 学习率变化
- ⚡ 训练速度
- 💾 系统资源使用
- 📝 所有超参数配置

---

## 📊 W&B 记录的指标

### 训练指标（每个 log_interval）
- `train/loss` - 训练损失
- `train/learning_rate` - 当前学习率
- `train/iteration` - 当前迭代次数
- `train/wallclock_time` - 累计训练时间（秒）

### 验证指标（每个 val_interval）
- `val/loss` - 验证损失
- `val/iteration` - 验证时的迭代次数
- `val/wallclock_time` - 验证时的累计时间

### 模型指标（自动记录）
- 梯度统计（均值、方差、最大值）
- 参数统计
- 梯度直方图

---

## 🎨 W&B Dashboard 功能

### 1. Charts（图表）

**默认图表**：
- **train/loss vs iteration** - 训练损失曲线
- **val/loss vs iteration** - 验证损失曲线
- **train/learning_rate vs iteration** - 学习率调度
- **gradients/** - 各层梯度统计

**自定义图表**：
- 点击 "Add Chart" 创建新图表
- 可以绘制任意指标的组合
- 支持多种图表类型（线图、散点图、直方图等）

### 2. Overview（概览）

- 运行状态（Running / Finished / Failed）
- 运行时长
- 最终指标值
- 系统信息（GPU、CPU、内存）

### 3. Config（配置）

自动记录的所有超参数：
```python
{
  "vocab_size": 10000,
  "context_length": 256,
  "d_model": 512,
  "num_layers": 4,
  "num_heads": 16,
  "lr": 0.0002,
  "batch_size": 32,
  ...
}
```

### 4. Files（文件）

自动保存的文件：
- 代码快照
- 检查点（如果启用）
- 日志文件
- 输出

---

## 🔄 运行多个实验

### 实验1：Baseline

```bash
# 使用默认配置
export WANDB_API_KEY="your_wandb_api_key_here"
uv run python scripts/train_with_wandb.py
```

### 实验2：2倍学习率

```bash
# 修改 config.json
# "lr": 0.0004  (从 0.0002 改为 0.0004)

export WANDB_API_KEY="your_wandb_api_key_here"
uv run python scripts/train_with_wandb.py
```

### 实验3：更大模型

```bash
# 修改 config.json
# "d_model": 768  (从 512 改为 768)
# "num_layers": 6  (从 4 改为 6)

export WANDB_API_KEY="your_wandb_api_key_here"
uv run python scripts/train_with_wandb.py
```

### 在 W&B 中比较实验

1. 打开 W&B 项目页面
2. 选择多个运行（勾选复选框）
3. 点击 "Compare" 按钮
4. 查看并排的图表对比！

**提示**：可以添加标签（tags）来组织实验：
```python
wandb.init(
    project="cs336-transformer-lm",
    name="exp-2xlr",
    tags=["lr-ablation", "experiment"],
)
```

---

## 📝 创建实验日志

### 在 W&B 中添加笔记

1. 打开运行详情页
2. 点击 "Overview" 标签
3. 在 "Notes" 区域添加观察和结论

示例笔记：
```markdown
## Experiment: 2x Learning Rate

### Observations
- 训练损失下降更快
- 验证损失在 3500 步达到最佳
- 比 baseline 快 1000 步收敛

### Conclusion
- 2倍学习率有效提升收敛速度
- 最终性能提升 5%
- 建议后续实验尝试更激进的学习率
```

### 导出实验报告

W&B 支持导出为：
- PDF 报告
- Markdown
- HTML

**步骤**：
1. 选择要包含的运行
2. 点击 "Create Report"
3. 拖拽图表和添加文字
4. 导出为 PDF 或分享链接

---

## 💡 高级功能

### 1. Sweeps（超参数扫描）

自动运行多个实验，搜索最佳超参数。

**创建 sweep 配置**：
```yaml
# sweep.yaml
program: scripts/train_with_wandb.py
method: bayes  # 或 grid, random
metric:
  name: val/loss
  goal: minimize
parameters:
  lr:
    values: [0.0001, 0.0002, 0.0004]
  batch_size:
    values: [16, 32, 64]
```

**运行 sweep**：
```bash
# 初始化 sweep
wandb sweep sweep.yaml

# 运行 agent（可以并行多个）
wandb agent your-username/cs336-transformer-lm/sweep-id
```

### 2. Artifacts（模型管理）

保存和版本控制模型检查点。

```python
# 保存检查点为 artifact
artifact = wandb.Artifact('model', type='model')
artifact.add_file('checkpoints/ckpt_iter6000.pt')
wandb.log_artifact(artifact)
```

### 3. Tables（数据表格）

记录结构化数据，如样本预测。

```python
# 创建表格
table = wandb.Table(columns=["iteration", "input", "prediction", "target"])
table.add_data(1000, "The cat", "sat", "sat")
wandb.log({"predictions": table})
```

---

## 🎓 满足作业要求

使用 W&B，你已经自动完成了：

✅ **实验追踪基础设施**
- 自动记录所有指标
- 实时可视化
- 云端存储

✅ **损失曲线**
- 训练损失 vs 步数
- 训练损失 vs 时间
- 验证损失 vs 步数
- 验证损失 vs 时间

✅ **实验日志**
- 自动保存配置
- 添加笔记和观察
- 生成专业报告

✅ **实验对比**
- 并排比较多个实验
- 表格汇总
- 图表叠加

---

## 📸 截图示例

### Dashboard 示例

访问 W&B 后你会看到：

```
┌─────────────────────────────────────────────────┐
│  cs336-transformer-lm                           │
├─────────────────────────────────────────────────┤
│  Charts                                         │
│  ┌───────────────────┬───────────────────────┐ │
│  │ train/loss        │ val/loss              │ │
│  │       ↓           │       ↓               │ │
│  │      ╱            │      ╱ ╲              │ │
│  │    ╱              │    ╱     ╲            │ │
│  │  ╱                │  ╱         ╲__        │ │
│  └───────────────────┴───────────────────────┘ │
│                                                 │
│  ┌───────────────────┬───────────────────────┐ │
│  │ train/lr          │ gradients/...         │ │
│  │  ╲                │                       │ │
│  │   ╲___            │  📊 Histogram         │ │
│  │       ╲___        │                       │ │
│  └───────────────────┴───────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 🔧 自定义 train_with_wandb.py

如果需要记录额外指标：

```python
# 在训练循环中添加
if (iteration + 1) % args.log_interval == 0:
    metrics = {
        'train/loss': loss.item(),
        'train/learning_rate': lr,
        
        # 添加自定义指标
        'train/perplexity': torch.exp(loss).item(),
        'train/tokens_per_sec': (batch_size * context_length) / step_time,
    }
    
    wandb.log(metrics, step=iteration + 1)
```

---

## 📚 更多资源

- **W&B 官方文档**: https://docs.wandb.ai/
- **快速开始指南**: https://docs.wandb.ai/quickstart
- **示例项目**: https://wandb.ai/wandb/examples
- **视频教程**: https://www.youtube.com/c/WeightsBiases

---

## ❓ 常见问题

### Q1: torch.compile 和 wandb.watch 冲突怎么办？

**问题**：运行训练时出现 `AssertionError: SpeculationLog diverged` 错误。

**原因**：`torch.compile` (PyTorch 2.0+) 与 `wandb.watch()` 不兼容。

**解决方案**（已在代码中实现）：
- ✅ 禁用 `wandb.watch()`（推荐）- 训练 loss、学习率等核心指标仍会正常记录
- ⚠️ 只是不会记录每层的梯度统计

如果确实需要梯度统计，可以在 `utils.py` 中注释掉 `torch.compile(model)`。

---

### Q2: 如何关闭 W&B？

如果暂时不想使用 W&B，设置环境变量：
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_MODE=disabled
uv run python scripts/train_with_wandb.py
```

或在代码中：
```python
wandb.init(mode="disabled")
```

### Q3: 如何在本地运行（不上传云端）？

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_MODE=offline
uv run python scripts/train_with_wandb.py
```

### Q4: 数据会占用多少空间？

- 每个运行通常 < 10 MB（仅指标数据）
- 如果保存检查点，取决于模型大小
- 免费账户有 100 GB 存储空间

### Q5: 如何删除实验？

1. 打开运行详情页
2. 点击右上角的 "..." 菜单
3. 选择 "Delete run"

---

## 🎉 现在开始使用 W&B！

```bash
# 1. 设置 API key（添加到 ~/.zshrc）
export WANDB_API_KEY="your_wandb_api_key_here"

# 2. 重新加载配置
source ~/.zshrc

# 3. 运行第一个实验
uv run python scripts/train_with_wandb.py

# 4. 在浏览器中查看结果 🚀
# 访问：https://wandb.ai/zk299-new-york-university
```

享受专业的实验追踪体验！✨
