import os
import sys
import json
import torch
import pickle
import pathlib
import argparse
from utils import _to_device_and_compile
from model import BasicsTransformerLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import Tokenizer


# 1. 设定路径
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")
special_tokens = ["<|endoftext|>"]

# 2. 加载词表和 merges
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
with open(MERGES_PATH, 'rb') as f:
    merges = pickle.load(f)

# 3. 构造 tokenizer
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

CKPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints/ckpt_iter5000.pt"
CONFIG_PATH = "scripts/config.json"
def main():
    parser = argparse.ArgumentParser(description="Text Generation with Transformer LM")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a pretty girl", 
                       help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, 
                       help="Sampling temperature (higher=more random, lower=more deterministic)")
    parser.add_argument("--top_k", type=int, default=None, 
                       help="Top-k sampling (if specified, only sample from top k tokens)")
    parser.add_argument("--top_p", type=float, default=None, 
                       help="Top-p (nucleus) sampling (if specified, sample from smallest set with cumulative prob > top_p)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                       help="Path to checkpoint file (default: checkpoints/ckpt_iter5000.pt)")
    args = parser.parse_args()

    # ==== 加载模型结构 ====
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config["model"])
    model, device = _to_device_and_compile(model)
    model.eval()

    # ==== 导入模型权重 ====
    ckpt_path = args.ckpt_path if args.ckpt_path else CKPT_PATH
    with open(ckpt_path, 'rb') as f:
        checkpoint = torch.load(f, weights_only=False, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {ckpt_path}")
    print(f"   Trained for {checkpoint['iteration']} iterations")

    # ==== Encode prompt ====
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"   Encoded as {len(input_ids)} tokens")
    
    # ==== 生成 ====
    print(f"\n🎲 Generating with:")
    print(f"   max_new_tokens: {args.max_new_tokens}")
    print(f"   temperature: {args.temperature}")
    if args.top_k is not None:
        print(f"   top_k: {args.top_k}")
    if args.top_p is not None:
        print(f"   top_p: {args.top_p}")
    
    # Get EOS token ID
    eos_token_bytes = special_tokens[0].encode("utf-8")  # <|endoftext|>
    eos_token_id = tokenizer.byte_to_token_id.get(eos_token_bytes, None)
    
    output_tokens = model.generate(
        input_tensor, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id
    )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    
    print(f"\n✨ Generated {len(output_ids)} new tokens")
    
    # ==== Decode ====
    full_ids = input_ids + output_ids
    # generated_text = tokenizer.decode(output_ids)
    full_text = tokenizer.decode(full_ids)
    
    print(f"\n{'='*60}")
    print(f"📖 Generated Text:")
    print(f"{'='*60}")
    print(full_text)
    print(f"{'='*60}")


# uv run python scripts/generate.py --prompt "Once upon a time" --temperature 0.8 --top_p 0.9 --max_new_tokens 256
if __name__ == "__main__":
    main()