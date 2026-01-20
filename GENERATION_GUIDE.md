# 文本生成（Decoding）指南 🎲

本指南介绍如何使用训练好的 Transformer 语言模型生成文本。

---

## ✨ 功能特性

✅ **自动补全** - 根据提示词生成后续文本  
✅ **温度控制** - 调节生成的随机性  
✅ **Top-k 采样** - 从概率最高的 k 个 token 中采样  
✅ **Top-p 采样** - 核采样（Nucleus Sampling）  
✅ **EOS 处理** - 遇到 `<|endoftext|>` 自动停止  
✅ **长度控制** - 限制最大生成 token 数量  

---

## 🚀 快速开始

### 基本用法

```bash
cd /Users/zhengjiankang/Downloads/cs336/assignment1-basics

# 使用默认设置生成
python scripts/generate.py --prompt "Once upon a time"
```

### 自定义参数

```bash
# 生成更长的文本
python scripts/generate.py --prompt "Once upon a time" --max_new_tokens 512

# 调整温度（更随机）
python scripts/generate.py --prompt "Once upon a time" --temperature 1.5

# 调整温度（更确定性）
python scripts/generate.py --prompt "Once upon a time" --temperature 0.7

# 使用 top-k 采样
python scripts/generate.py --prompt "Once upon a time" --top_k 50

# 使用 top-p（核采样）
python scripts/generate.py --prompt "Once upon a time" --top_p 0.9

# 指定检查点
python scripts/generate.py --prompt "Once upon a time" --ckpt_path checkpoints/ckpt_iter6000.pt
```

---

## 📚 参数说明

### `--prompt` (必需)

输入提示词，模型会基于此生成后续文本。

```bash
--prompt "The cat sat on the"
--prompt "In a galaxy far, far away"
--prompt "def fibonacci(n):"
```

### `--max_new_tokens` (默认: 256)

生成的最大 token 数量。

```bash
--max_new_tokens 100   # 简短回复
--max_new_tokens 512   # 较长文本
--max_new_tokens 1024  # 长文本
```

**注意**：实际生成可能更短（如果遇到 EOS token）。

### `--temperature` (默认: 1.0)

控制生成的随机性。

| Temperature | 效果 | 适用场景 |
|------------|------|---------|
| 0.1 - 0.5 | 非常确定性，重复性高 | 代码生成、事实回答 |
| 0.7 - 0.9 | 平衡，较为合理 | 一般文本生成 |
| 1.0 | 标准采样 | 默认设置 |
| 1.2 - 1.5 | 更随机，更有创意 | 创意写作、头脑风暴 |
| > 2.0 | 非常随机，可能不连贯 | 实验性 |

**原理**：
```python
# 温度缩放
scaled_logits = logits / temperature

# temperature < 1: 分布更尖锐（更确定）
# temperature = 1: 不变
# temperature > 1: 分布更平坦（更随机）
```

### `--top_k` (可选)

仅从概率最高的 k 个 token 中采样。

```bash
--top_k 10   # 非常保守
--top_k 50   # 平衡
--top_k 100  # 较开放
```

**⚠️ 注意**：不能与 `--top_p` 同时使用。

**原理**：
1. 计算所有 token 的概率
2. 选择概率最高的 k 个
3. 将其他 token 的概率设为 0
4. 重新归一化并采样

### `--top_p` (可选，推荐)

核采样（Nucleus Sampling）- 从累积概率超过 p 的最小 token 集合中采样。

```bash
--top_p 0.9   # 常用设置（覆盖 90% 概率质量）
--top_p 0.95  # 更保守
--top_p 0.8   # 更开放
```

**⚠️ 注意**：不能与 `--top_k` 同时使用。

**原理**（论文：Holtzman et al., 2020）：
1. 按概率降序排列所有 token
2. 计算累积概率
3. 选择累积概率刚好超过 p 的最小集合
4. 从这个集合中采样

**为什么比 top-k 更好？**
- 动态调整候选集大小
- 对于高置信度预测，候选集更小
- 对于低置信度预测，候选集更大

### `--ckpt_path` (可选)

指定要加载的检查点文件。

```bash
--ckpt_path checkpoints/ckpt_iter6000.pt
```

默认使用 `checkpoints/ckpt_iter5000.pt`。

---

## 🎯 使用示例

### 示例1：创意故事生成

```bash
python scripts/generate.py \
    --prompt "Once upon a time, there was a brave knight" \
    --max_new_tokens 300 \
    --temperature 1.2 \
    --top_p 0.9
```

**输出示例**：
```
Once upon a time, there was a brave knight named Sir Galahad. 
He lived in a castle on top of a tall mountain. One day, he 
heard about a dragon that was terrorizing the nearby village...
```

### 示例2：事实性文本生成

```bash
python scripts/generate.py \
    --prompt "The capital of France is" \
    --max_new_tokens 50 \
    --temperature 0.3 \
    --top_p 0.9
```

**输出示例**：
```
The capital of France is Paris. It is one of the most 
beautiful cities in the world...
```

### 示例3：对话生成

```bash
python scripts/generate.py \
    --prompt "Human: What is the meaning of life?
AI:" \
    --max_new_tokens 150 \
    --temperature 0.8 \
    --top_p 0.9
```

### 示例4：代码生成

```bash
python scripts/generate.py \
    --prompt "def factorial(n):
    # Calculate the factorial of n
    " \
    --max_new_tokens 100 \
    --temperature 0.2 \
    --top_k 10
```

---

## 🔬 温度和采样策略对比

### 实验：生成相同提示的不同版本

```bash
# 确定性（温度=0.3，top-k=10）
python scripts/generate.py --prompt "The cat" --temperature 0.3 --top_k 10

# 平衡（温度=1.0，top-p=0.9）
python scripts/generate.py --prompt "The cat" --temperature 1.0 --top_p 0.9

# 随机性（温度=1.5，top-p=0.95）
python scripts/generate.py --prompt "The cat" --temperature 1.5 --top_p 0.95
```

**预期结果**：
- 低温度：输出更一致、重复性更高
- 高温度：输出更多样、更有创意（但可能不连贯）

---

## ⚙️ 实现细节

### 生成算法（自回归采样）

```python
for _ in range(max_new_tokens):
    # 1. 获取最后一个位置的 logits
    logits = model(input_ids)[:, -1, :]
    
    # 2. 应用温度缩放
    scaled_logits = logits / temperature
    
    # 3. 应用 top-k 或 top-p 过滤
    if top_k:
        # 保留 top-k 个 token
        mask out non-top-k tokens
    elif top_p:
        # 保留累积概率 > top_p 的 token
        mask out low-probability tokens
    
    # 4. 计算概率分布
    probs = softmax(scaled_logits)
    
    # 5. 采样下一个 token
    next_token = sample(probs)
    
    # 6. 检查 EOS
    if next_token == eos_token_id:
        break
    
    # 7. 追加到序列
    input_ids = cat(input_ids, next_token)
```

### Top-p（核采样）算法

```python
# 1. 排序
sorted_probs, sorted_indices = sort(probs, descending=True)

# 2. 计算累积概率
cumulative_probs = cumsum(sorted_probs)

# 3. 找到截断点
# 保留累积概率刚好超过 p 的最小集合
mask = cumulative_probs > top_p

# 4. 至少保留1个 token
mask[0] = False

# 5. 应用 mask
probs[mask] = 0
probs = normalize(probs)

# 6. 采样
next_token = sample(probs)
```

---

## 🎓 满足作业要求

实现的功能：

✅ **生成补全** - `generate()` 方法支持自回归生成  
✅ **最大长度控制** - `--max_new_tokens` 参数  
✅ **温度缩放** - `--temperature` 参数  
✅ **Top-p 采样** - `--top_p` 参数（Holtzman et al., 2020）  
✅ **EOS 处理** - 自动检测 `<|endoftext|>` 并停止  

---

## 🐛 常见问题

### Q1: 生成的文本重复

**原因**：温度太低或 top-k/top-p 过于保守。

**解决**：
```bash
# 增加温度
--temperature 1.2

# 或使用更大的 top-p
--top_p 0.95
```

### Q2: 生成的文本不连贯

**原因**：温度太高。

**解决**：
```bash
# 降低温度
--temperature 0.7

# 或使用更小的 top-p
--top_p 0.85
```

### Q3: 生成速度慢

**原因**：模型未编译或运行在 CPU 上。

**解决**：
- 确保有 GPU 可用
- 模型会自动使用 `torch.compile` 加速
- 首次生成会较慢（编译时间）

### Q4: 内存不足

**原因**：生成序列太长。

**解决**：
```bash
# 减少最大长度
--max_new_tokens 256  # 而不是 1024
```

---

## 📖 参考文献

- **Top-p (Nucleus) Sampling**: 
  Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"
  https://arxiv.org/abs/1904.09751

- **Temperature Scaling**:
  Standard technique in language model sampling

---

## 🎉 现在开始生成！

```bash
# 简单开始
python scripts/generate.py --prompt "Once upon a time"

# 高质量生成（推荐设置）
python scripts/generate.py \
    --prompt "Your prompt here" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.9
```

享受创造性的文本生成！✨
