import os
import sys
import pickle
import pathlib

# Add the parent directory to Python path before importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.adapters import run_train_bpe

# 数据集路径
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

# Tokenizer保存路径
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl") # 词表（token → ID 的映射）
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl") # 合并规则（BPE 算法的核心）

# 训练参数
vocab_size = 10_000 # GPT-2: 50,257 tokens
special_tokens = ["<|endoftext|>"]

# 训练
print("开始训练BPE tokenizer...")
vocab, merges = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)
print("BPE tokenizer训练完成")

# 序列化到磁盘
os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merges, f)

# 统计最长 token
longest_token = max(vocab.values(), key=len)
print("最长token:", longest_token, "长度:", len(longest_token))

# -----------------------------------------------------------------
# 1. 读取训练数据
#    TinyStoriesV2-GPT4-train.txt
#    ↓
# 2. 训练 BPE
#    run_train_bpe()
#    - 统计字符对频率
#    - 不断合并高频对
#    - 直到词表达到 10,000
#    ↓
# 3. 得到两个东西
#    vocab: {0: b"a", 1: b"b", ...}
#    merges: [(b"t", b"h"), ...]
#    ↓
# 4. 保存到磁盘
#    vocab.pkl, merges.pkl
#    ↓
# 5. 完成！下次直接加载使用

