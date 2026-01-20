from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int


logger = logging.getLogger(__name__)


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            in_features: int
                The number of input features.
            out_features: int
                The number of output features.
            device: torch.device | None, optional
                Device to store the parameters on.
            dtype: torch.dtype | None, optional
                Data type of the parameters.
        """
        
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        # Store weight as (out_features, in_features) - not transposed
        weight_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight: Float[Tensor, " out_features in_features"] = nn.Parameter(
            nn.init.trunc_normal_(weight_tensor, mean=0.0, std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
    
    def extra_repr(self):
        return f"Linear(out_features={self.weight.shape[0]}, in_features={self.weight.shape[1]})"


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """An embedding layer initialized with truncated normal.

        Args:
            num_embeddings: int
                Size of the vocabulary.
            embedding_dim: int
                Dimension of the embedding vectors, i.e., d_model.
            device: torch.device | None, optional
                Device to store the parameters on.
            dtype: torch.dtype | None, optional
                Data type of the parameters.
        """
        super().__init__()
        std = 1.0
        # Store embedding matrix with embedding_dim as the final dimension: (num_embeddings, embedding_dim)
        embedding_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(embedding_tensor, mean=0.0, std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids, :]
    
    def extra_repr(self):
        return f"Embedding(num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]})"


class RMSNorm(nn.Module):
    """
    This module implements root mean square (RMS) layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        d_model: int
            Hidden dimension of the model.
        eps: float, default is 1e-5
            Epsilon value for numerical stability.
        device: torch.device | None, optional
            Device to store the parameters on.
        dtype: torch.dtype | None, optional
            Data type of the parameters.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Tensor of same shape as input.
        """
        # Upcast to float32 to prevent overflow when squaring the input
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Perform RMSNorm
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        
        # Apply weight and return in original dtype
        result = self.weight * x
        return result.to(in_dtype)
    
    def extra_repr(self):
        return f"RMSNorm(d_model={self.weight.shape[0]}, eps={self.eps})"


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).
    
    Applies rotary positional embeddings to input tensors as described in
    "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    
    Args:
        theta: float
            Θ value for the RoPE.
        d_k: int
            Dimension of query and key vectors.
        max_seq_len: int
            Maximum sequence length that will be inputted.
        device: torch.device | None, optional
            Device to store the buffer on.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies: 1 / (theta^(2i/d_k)) for i in [0, d_k//2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # Precompute positions: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device).float()
        
        # Compute frequency-position matrix: (max_seq_len, d_k // 2)
        freqs = freqs.unsqueeze(0) * positions.unsqueeze(1)  # (max_seq_len, d_k // 2)
        
        # Precompute cos and sin
        cos = torch.cos(freqs)  # (max_seq_len, d_k // 2)
        sin = torch.sin(freqs)  # (max_seq_len, d_k // 2)
        
        # Register as buffers
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Tensor of shape (..., seq_len) specifying token positions.
        
        Returns:
            Tensor of same shape as x with RoPE applied.
        """
        # Get cos and sin for the specified token positions
        # token_positions shape: (..., seq_len)
        # We need to index cos/sin which are (max_seq_len, d_k // 2)
        # Result should be (..., seq_len, d_k // 2)
        
        # Flatten token_positions for indexing, then reshape back
        original_shape = token_positions.shape
        flat_positions = token_positions.flatten()  # (N,)
        
        # Index cos and sin: (N, d_k // 2)
        cos_selected = self.cos[flat_positions.long()]
        sin_selected = self.sin[flat_positions.long()]
        
        # Reshape back to (..., seq_len, d_k // 2)
        cos_selected = cos_selected.view(*original_shape, -1)
        sin_selected = sin_selected.view(*original_shape, -1)
        
        # Split x into even and odd dimensions
        # x shape: (..., seq_len, d_k)
        x_even = x[..., ::2]   # (..., seq_len, d_k // 2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k // 2)
        
        # Apply rotation: [x_even', x_odd'] = [cos -sin; sin cos] [x_even; x_odd]
        x_even_rot = x_even * cos_selected - x_odd * sin_selected
        x_odd_rot = x_even * sin_selected + x_odd * cos_selected
        
        # Interleave back: [x_even_rot[0], x_odd_rot[0], x_even_rot[1], x_odd_rot[1], ...]
        result = torch.zeros_like(x)
        result[..., ::2] = x_even_rot
        result[..., 1::2] = x_odd_rot
        
        return result
    
    def extra_repr(self):
        return f"RotaryPositionalEmbedding(theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len})"


class BasicsTransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta: float
            The theta value for the RoPE positional encoding.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_head,
            max_seq_len=context_length
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the lm_head parameters get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()

        return n_params

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        """
        Args:
            x: Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        # (batch size, sequence_length, d_model)
        x = self.token_embeddings(x)

        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)

        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)

        # (batch size, sequence_length, vocab_size)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Generate text from the model using various sampling strategies.
        
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float, default=1.0
                Temperature for softmax scaling. Higher values (>1) make the distribution
                more uniform (more random), lower values (<1) make it more peaked (more deterministic).
            top_k: int | None
                If provided, only sample from the `top_k` vocab items (by probability).
                Cannot be used together with top_p.
            top_p: float | None
                If provided, use nucleus sampling: only sample from the smallest set of tokens
                whose cumulative probability exceeds top_p. Cannot be used together with top_k.
            eos_token_id: int | None
                If provided, stop generation when we generate this ID (e.g., <|endoftext|>).

        Returns: A LongTensor of shape (1, generated_length) with the generated tokens
                 (excluding the input prompt).
        """
        # Ensure x is 2D (batch_size, seq_len)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Cannot use both top_k and top_p
        if top_k is not None and top_p is not None:
            raise ValueError("Cannot use both top_k and top_p sampling simultaneously")
            
        original_sequence_length = x.size(-1)
        
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x_cond = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            
            # Get the logits from the model
            logits = self.forward(x_cond)
            
            # Take the logits for the next token (last position)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature scaling
            scaled_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                topk_values, topk_indices = torch.topk(
                    scaled_logits,
                    min(top_k, scaled_logits.size(-1)),
                    dim=-1
                )
                # Mask out tokens not in top-k
                threshold = topk_values[:, -1, None]  # (batch_size, 1)
                mask = scaled_logits < threshold
                scaled_logits = scaled_logits.masked_fill(mask, float("-inf"))
            
            # Apply top-p (nucleus) sampling
            elif top_p is not None:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                
                # Compute cumulative probabilities
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Keep tokens where cumulative probability (before adding current token) < top_p
                # This ensures we keep the smallest set whose cumulative probability exceeds top_p
                keep_mask = (cumulative_probs - sorted_probs) < top_p
                
                # Apply mask to sorted logits (in-place)
                sorted_logits[~keep_mask] = float("-inf")
                
                # Scatter back to original indices (directly modify scaled_logits)
                scaled_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            
            # Compute probabilities and sample
            next_token_probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)
            
            # Check for EOS token
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            
            # Append the new token
            x = torch.cat((x, next_token_id), dim=-1)
        
        # Return only the newly generated tokens
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        positional_encoder: RotaryPositionalEmbedding | None, optional
            The RoPE module to use. If None, no positional encoding is applied.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) activation function.
    
    Uses torch.sigmoid for numerical stability.
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.
    
    SwiGLU combines SiLU activation with GLU (Gated Linear Unit).
    The formula is: SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
    
    Args:
        d_model: int
            Dimensionality of the model embeddings (input and output).
        d_ff: int
            Dimensionality of the feed-forward inner layer.
            Should be approximately 8/3 × d_model and a multiple of 64.
        device: torch.device | None, optional
            Device to store the parameters on.
        dtype: torch.dtype | None, optional
            Data type of the parameters.
    """
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model).
            
        Returns:
            Output tensor of shape (..., d_model).
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
    def extra_repr(self):
        return f"SwiGLU(d_model={self.w1.weight.shape[1]}, d_ff={self.w1.weight.shape[0]})"


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        # mask=True: allow attention (keep attention_scores)
        # mask=False: mask out (set to -inf, which becomes 0 after softmax)
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        positional_encoder: RotaryPositionalEmbedding | None, optional
            The RoPE module to use. If None, no positional encoding is applied.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE (optional)

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            token_positions: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size() # b 只包含批次维度，不包含 heads
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        # Apply RoPE if positional_encoder is provided
        if self.positional_encoder is not None:
            if token_positions is None:
                token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

            # Duplicate token positions for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b)) 
        causal_mask = qi >= kj  # (query, key)

        # Shape: (..., num_heads, sequence_length, d_v)
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(attn_output, "... heads seq d_v -> ... seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output
