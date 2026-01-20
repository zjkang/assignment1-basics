# 📊 Wandb 查看和对比实验指南

详细说明如何在 Weights & Biases 中查看超参数和对比实验结果。

---

## 🎯 **快速访问**

**你的 Dashboard**: https://wandb.ai/zk299-new-york-university/cs336-transformer-lm

---

## 📋 **方法 1: 查看单个实验的所有参数**

### **步骤 1: 打开实验**
1. 访问你的 dashboard
2. 在左侧 runs 列表中，点击任意一个实验名称
   - 例如: `baseline-20260119-153045`

### **步骤 2: 查看 Overview（概览）**
在 **Overview** 标签页，你会看到：
- **Notes**: 实验的简短描述（包含关键参数）
  ```
  Config: baseline | Model: d_model=512, layers=4 | Optim: lr=0.0002, bs=32
  ```
- **Tags**: `baseline`, `tinystories`
- **Summary Metrics**: 最终的 loss 和训练时间

### **步骤 3: 查看 Config（完整配置）**
点击顶部的 **"Config"** 标签，查看所有超参数：

```yaml
# 模型参数
vocab_size: 10000
context_length: 256
d_model: 512
num_layers: 4
num_heads: 16
d_ff: 1344
rope_theta: 10000.0

# 优化器参数
lr: 0.0002
min_lr: 0.00001
warmup_iters: 500
cosine_iters: 10000
weight_decay: 0.01
clip_grad_norm: 1.0

# 训练参数
batch_size: 32
train_steps: 5000
val_interval: 100
...
```

**提示**: 你可以点击参数名称旁边的 📋 图标复制参数值。

---

## 📊 **方法 2: 表格视图对比（最常用）**

### **步骤 1: 切换到表格视图**
在左侧导航栏，点击 **"Table"** 或 **"Runs"** 图标

### **步骤 2: 添加参数列**

默认表格只显示部分列。要添加更多参数列：

1. 点击表格右上角的 **"Columns"** 按钮（或 **"+"** 图标）
2. 在搜索框输入参数名称，例如：
   - `config.lr` - 学习率
   - `config.d_model` - 模型维度
   - `config.num_layers` - 层数
   - `config.batch_size` - 批次大小
3. 勾选你想显示的参数
4. 点击 "Apply"

### **步骤 3: 查看对比**

现在表格会显示类似这样的内容：

| Name | Tags | config.lr | config.d_model | config.batch_size | final_train_loss | final_val_loss |
|------|------|-----------|----------------|-------------------|-----------------|----------------|
| baseline-xxx | baseline | 0.0002 | 512 | 32 | 2.34 | 2.56 |
| high_lr-xxx | high_lr | **0.0005** | 512 | 32 | **2.12** | **2.48** |
| large_model-xxx | large_model | 0.0001 | **768** | 24 | 2.28 | 2.51 |
| small_fast-xxx | small_fast | 0.0003 | **256** | 64 | 2.89 | 3.02 |

**观察**：
- ✅ `high_lr` 的学习率更高（0.0005），loss 最低（2.12）
- ✅ `large_model` 的模型更大（768），loss 适中（2.28）
- ✅ `small_fast` 模型最小（256），loss 最高（2.89）

### **步骤 4: 排序和筛选**

**排序**：
- 点击列标题可以排序
- 例如，点击 `final_val_loss` 列，找到验证 loss 最低的配置

**筛选**：
- 点击顶部的 **"Filter"** 按钮
- 添加条件，例如：
  - `config.lr > 0.0002` - 只显示学习率高于 0.0002 的实验
  - `Tags contains "baseline"` - 只显示 baseline 相关的实验

---

## 🔄 **方法 3: Compare（并排对比）**

### **步骤 1: 选择实验**
在表格视图中，勾选 2-3 个你想对比的实验（左边的复选框）

### **步骤 2: 点击 Compare**
点击顶部的 **"Compare"** 按钮

### **步骤 3: 查看对比结果**

#### **3.1 Config Diff（配置差异）**

wandb 会高亮显示不同的参数：

```diff
Parameter        | baseline       | high_lr        | large_model
-----------------|----------------|----------------|----------------
lr               | 0.0002         | 0.0005 ⚠️      | 0.0001
d_model          | 512            | 512            | 768 ⚠️
num_layers       | 4              | 4              | 6 ⚠️
batch_size       | 32             | 32             | 24 ⚠️
```

**⚠️ 标记 = 参数不同**

#### **3.2 Charts（图表对比）**

所有图表会在同一个画布上显示多条曲线：

**Loss 曲线**:
```
train/loss
  ┌─────────────────────────────────────┐
4 │                                     │
  │ ╲                                   │
3 │  ╲╲                                 │
  │    ╲╲─── baseline (blue)            │
2 │      ╲─── high_lr (green) ⭐ 最低   │
  │       ╲── large_model (red)         │
1 │                                     │
  └─────────────────────────────────────┘
    0    1000   2000   3000   4000   5000
              Iteration
```

**学习率曲线**:
```
train/learning_rate
  ┌─────────────────────────────────────┐
  │      ╱─── high_lr (0.0005)          │
  │     ╱─── baseline (0.0002)          │
  │    ╱─── large_model (0.0001)        │
  └─────────────────────────────────────┘
```

#### **3.3 Summary（最终指标）**

| Run | final_train_loss | final_val_loss | total_time_minutes |
|-----|-----------------|----------------|-------------------|
| baseline | 2.34 | 2.56 | 42.3 |
| high_lr | **2.12 ⭐** | **2.48 ⭐** | 43.1 |
| large_model | 2.28 | 2.51 | **68.7** ⚠️ 慢 |

---

## 🎨 **方法 4: 自定义图表**

### **创建自定义对比图表**

1. 在 workspace 页面，点击 **"Add Panel"**
2. 选择 **"Line Plot"**
3. 配置：
   - **X 轴**: `Step` 或 `train/wallclock_time`
   - **Y 轴**: `train/loss` 或 `val/loss`
   - **Group by**: 选择多个 runs
4. 点击 "Apply"

### **创建参数-性能散点图**

1. 点击 **"Add Panel"** → **"Scatter Plot"**
2. 配置：
   - **X 轴**: `config.lr` (学习率)
   - **Y 轴**: `final_val_loss` (最终验证 loss)
   - **Color**: `config.d_model` (模型大小)
3. 这样可以看到学习率与性能的关系

---

## 🔍 **方法 5: 使用 Tags 快速筛选**

### **按配置类型筛选**

在 runs 列表上方，点击 tag 进行筛选：

- 点击 **`baseline`** → 只显示 baseline 实验
- 点击 **`high_lr`** → 只显示高学习率实验
- 点击 **`large_model`** → 只显示大模型实验

### **组合 tags**

选择多个 tags 可以进行组合筛选：
- `baseline` + `tinystories` → 显示所有 baseline 的 tinystories 实验

---

## 📈 **方法 6: 导出数据**

### **导出为 CSV**

1. 在表格视图，点击右上角的 **"Export"** 按钮
2. 选择 **"CSV"**
3. 所有参数和指标会导出到 CSV 文件

### **使用 API 下载**

```python
import wandb

# 下载所有 runs 的数据
api = wandb.Api()
runs = api.runs("zk299-new-york-university/cs336-transformer-lm")

for run in runs:
    print(f"Run: {run.name}")
    print(f"Config: {run.config}")
    print(f"Summary: {run.summary}")
```

---

## 💡 **快速对比清单**

### **要快速回答这些问题，查看哪里？**

| 问题 | 在哪里查看 |
|------|-----------|
| **这个实验用了什么学习率？** | Run → Config → `lr` |
| **哪个学习率最好？** | Table → 添加 `config.lr` 和 `final_val_loss` 列，排序 |
| **大模型比小模型好多少？** | Compare → 选择两个 runs → 查看 loss 曲线 |
| **训练用了多长时间？** | Run → Overview → `total_time_minutes` |
| **哪个配置收敛最快？** | Compare → X 轴选 `train/wallclock_time` |

---

## 🎯 **实用技巧**

### **技巧 1: 保存自定义视图**

创建自定义表格列配置后：
1. 点击 **"Save View"**
2. 命名（如 "Parameter Comparison"）
3. 下次可以快速加载这个视图

### **技巧 2: 使用 Notes 快速识别**

在表格中添加 **"Notes"** 列：
- 一眼看出每个实验的关键配置
- 例如: `Config: baseline | Model: d_model=512, layers=4 | Optim: lr=0.0002, bs=32`

### **技巧 3: 按时间排序**

点击 **"Created"** 列：
- 查看最新的实验
- 追踪实验进度

### **技巧 4: 使用 Workspace**

创建自定义 workspace：
1. 点击顶部 **"Create workspace"**
2. 添加你最常用的图表和表格
3. 保存，作为快速查看面板

---

## 🚀 **推荐工作流程**

### **每次实验后：**

1. ✅ 在 **Table** 中添加新列（如果需要）
2. ✅ 查看 **final_val_loss**，确认是否改进
3. ✅ 点击实验名称，查看 **Config**，记录关键参数
4. ✅ 如果效果好，复制配置创建新实验

### **对比多个实验时：**

1. ✅ 勾选 2-3 个实验
2. ✅ 点击 **Compare**
3. ✅ 查看 **Config Diff**（参数差异）
4. ✅ 查看 **Charts**（loss 曲线）
5. ✅ 记录最佳配置

---

## 📚 **更多资源**

- **wandb 文档**: https://docs.wandb.ai/
- **视频教程**: https://www.youtube.com/watch?v=gnD8BFuyVUA
- **示例项目**: https://wandb.ai/wandb/examples

---

## ✅ **总结**

查看参数的 5 种方法：

1. **Overview / Config** - 查看单个实验的完整配置
2. **Table** - 并排对比多个实验的参数和结果
3. **Compare** - 深入对比 2-3 个实验
4. **Tags** - 按配置类型快速筛选
5. **Custom Charts** - 创建参数-性能关系图

**最推荐**: 使用 **Table** 添加参数列，快速对比！📊✨
