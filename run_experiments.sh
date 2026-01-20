#!/bin/bash
# 批量运行多个配置的实验

# 设置 wandb API key
# IMPORTANT: Set your WANDB_API_KEY environment variable before running this script
# export WANDB_API_KEY="your_api_key_here"
# Or add it to your ~/.zshrc: export WANDB_API_KEY="your_api_key_here"

if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: WANDB_API_KEY environment variable is not set"
    echo "Please set it first:"
    echo "  export WANDB_API_KEY=\"your_api_key_here\""
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

echo "🚀 开始批量运行实验..."
echo ""

# 配置列表
configs=(
    "configs/baseline.json"
    "configs/high_lr.json"
    "configs/large_model.json"
    "configs/small_fast.json"
)

# 遍历运行每个配置
for config in "${configs[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Running: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 运行训练
    uv run python scripts/train_with_wandb.py --config "$config"
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "✅ $config completed successfully"
    else
        echo "❌ $config failed"
    fi
    
    echo ""
    echo "等待 5 秒后继续..."
    sleep 5
done

echo "🎉 所有实验完成！"
echo "📊 查看结果: https://wandb.ai/zk299-new-york-university/cs336-transformer-lm"
