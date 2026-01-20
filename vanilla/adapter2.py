import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# PyTorch Learning Notes
# Difference between .reshape() and .view()
# .view() - does not copy data
# x = torch.randn(2, 3, 4)
# y = x.view(2, -1)  # does not copy data, just changes view
# print(x.data_ptr() == y.data_ptr())  # True - shares same memory

# .reshape() - may copy data:
# x = torch.randn(2, 3, 4)
# x_t = x.transpose(1, 2)  # not contiguous
# y = x_t.reshape(2, -1)   # if original tensor is not contiguous, will copy data
# print(x_t.data_ptr() == y.data_ptr())  # False - different memory addresses

# Add & Norm layer
# This can be merged into EncoderLayer
# class SublayerConnection(nn.Module):
#     """
#         Alternative implementation of sublayer connection with residual connection implemented directly in this module.

#         Parameters:
#             feature_size: The dimension size of input features, i.e., the dimension to normalize.
#             dropout: Dropout probability in residual connection.
#             epsilon: Small constant to prevent division by zero.
#         """
#     def __init__(self, feature_size, dropout=0.1, epsilon=1e-9):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(feature_size, epsilon)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, sublayer):
#         # Apply dropout to sublayer output, then residual connection, then normalization
#         # Add: Residual (Skip) Connection - x + self.dropout(sublayer(x))
#         return self.norm(x + self.dropout(sublayer(x)))

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, h, d_ff, dropout):
#         """
#         Encoder layer.

#         Parameters:
#             d_model: Embedding dimension
#             h: Number of attention heads
#             d_ff: Hidden layer dimension of feed-forward network
#             dropout: Dropout probability
#         """
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, h)  # Multi-Head Self-Attention
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # Feed-forward network

#         # Define two sublayer connections for multi-head attention and feed-forward network
#         self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
#         self.d_model = d_model

#     def forward(self, x, src_mask):
#         """
#         Forward propagation function.

#         Parameters:
#             x: Input tensor with shape (batch_size, seq_len, d_model).
#             src_mask: Source sequence mask for self-attention.

#         Returns:
#             Output of encoder layer with shape (batch_size, seq_len, d_model).
#         """
#         x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, src_mask))  # Self-attention sublayer
#         x = self.sublayers[1](x, self.feed_forward)  # Feed-forward sublayer
#         return x

# EncoderLayer - Modern Pre-Norm Architecture
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, h)
#         self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Pre-Norm architecture: normalize first, then sublayer, then residual connection
#         """
#         # First sublayer: Self-Attention + Add & Norm
#         norm_x = self.norm1(x)
#         attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)
#         x = x + self.dropout(attn_output)  # residual connection
        
#         # Second sublayer: FFN + Add & Norm
#         norm_x = self.norm2(x)
#         ffn_output = self.ffn(norm_x)
#         return x + self.dropout(ffn_output)  # residual connection



# ------------------------------------------------------------------------------------------------
# Vanilla Transformer Paper Implementation


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention computation.

    Parameters:
        Q: Query matrix (batch_size, seq_len_q, embed_size)
        K: Key matrix (batch_size, seq_len_k, embed_size)
        V: Value matrix (batch_size, seq_len_v, embed_size)
        mask: Mask matrix, supports boolean mask or float mask (optional)

    Returns:
        output: Attention-weighted output matrix
        attention_weights: Attention weight matrix
    """
    # Dimension validation
    assert Q.size(-1) == K.size(-1), "Q and K must have the same embedding dimension"
    assert K.size(-2) == V.size(-2), "K and V must have the same sequence length"
    
    embed_size = Q.size(-1)  # embed_size

    # Compute dot product and scale (using PyTorch approach)
    scale = embed_size ** -0.5  # equivalent to 1/sqrt(embed_size)
    attention_scores = (Q @ K.transpose(-2, -1)) * scale

    # Handle mask (supports both boolean mask and float mask)
    if mask is not None:
        if mask.dtype == torch.bool:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        else:  # float mask
            attention_scores = attention_scores + mask

    # Apply Softmax function to scaled scores to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Weighted sum to compute output
    output = attention_weights @ V

    return output, attention_weights


# Test - Example parameters
# batch_size = 2
# num_heads = 2
# seq_len_q = 3  # query sequence length
# seq_len_k = 3  # key sequence length
# head_dim = 4

# # Simulate query matrix Q and key-value matrices K, V
# Q = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
# K = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
# V = torch.randn(batch_size, num_heads, seq_len_k, head_dim)

# # causal mask generates lower triangular mask matrix, broadcast to all heads
# mask = torch.tril(torch.ones(seq_len_q, seq_len_k)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k)
# mask = mask.masked_fill(mask == 0, float('-inf'))  # replace 0.0 with -inf for float mask

# # Execute scaled dot-product attention with lower triangular mask
# output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

# # Print results
# print("Mask matrix (lower triangular):")
# print(mask[0, 0])

# print("\nAttention weight matrix:")
# print(attn_weights)

# ------------------------------------------------------------------------------------------------

# Multi-Head Attention
# 
# Multi-head attention in Transformer plays a similar role to convolutional kernels in CNNs.
# CNNs use multiple different kernels to capture various local features in the spatial domain,
# while Transformer's multi-head attention uses multiple heads in parallel to attend to 
# dependencies across different representation subspaces of the input data.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        """
        Multi-head attention mechanism: each head has its own linear layers.

        Parameters:
            d_model: Embedding dimension of input sequence.
            h: Number of attention heads.
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h."

        self.d_model = d_model
        self.h = h

        # "Shared" Q, K, V linear layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer, maps multi-head concatenated output back to d_model
        # Because num_heads * d_k = d_model
        # So concat_out: (batch_size, seq_len, d_model)
        # With linear layer: learnable linear transformation where W^O is learnable d_model × d_model matrix
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            q: Query matrix (batch_size, seq_len_q, d_model)
            k: Key matrix (batch_size, seq_len_k, d_model)
            v: Value matrix (batch_size, seq_len_v, d_model)
            mask: Mask matrix, supports boolean mask or float mask

        Returns:
            out: Attention-weighted output (batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # Get sequence lengths for query and key-value
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # Split the linearly transformed "shared" matrices into multi-heads, adjust dimensions to (batch_size, h, seq_len, d_k)
        # d_k is the dimension of each attention head
        # query Q and key K might not have the same sequence length, 
        # but the value V always has the same sequence length as the key K.
        Q = self.w_q(q).view(batch_size, seq_len_q, self.h, -1).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len_k, self.h, -1).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len_k, self.h, -1).transpose(1, 2)

        # Execute scaled dot-product attention
        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate multi-heads and restore to (batch_size, seq_len_q, d_model)
        concat_out = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Through output linear layer where W^O is learnable d_model × d_model matrix
        # This is necessary because multi-head attention output is concatenated multi-head result, needs linear layer to map back to d_model
        # Multi-head attention output:
        # Head 1: (batch_size, seq_len, d_k)
        # Head 2: (batch_size, seq_len, d_k)
        # ...
        # Head h: (batch_size, seq_len, d_k)

        # After concatenation:
        # (batch_size, seq_len, num_heads * d_k) = (batch_size, seq_len, d_model)
        # Concatenation is just simple connection, but model needs to learn how to combine these features
        # Output projection provides learnable linear transformation:
        # output = W^O * concat_output
        # where W^O is learnable d_model × d_model matrix
        out = self.fc_out(concat_out)  # (batch_size, seq_len_q, d_model)

        return out

# ------------------------------------------------------------------------------------------------

# FFN (Position-wise Feed-Forward Network) - similar to MLP, trains hidden layer
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Position-wise Feed-Forward Network.

        Parameters:
            d_model: Dimension of input and output vectors
            d_ff: Hidden layer dimension of FFN (usually 4x d_model)
            dropout: Dropout rate for preventing overfitting
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.w_2 = nn.Linear(d_ff, d_model)  # Second linear layer d_ff = 4x d_model
        self.dropout = nn.Dropout(dropout)   # Dropout layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation: d_model → d_ff → d_model
        """
        # Calculate activation function first, then randomly drop to avoid unnecessary computation
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ------------------------------------------------------------------------------------------------

# LayerNorm: nn.LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, feature_size, epsilon=1e-9):
        """
        Layer normalization for normalizing the last dimension.

        Parameters:
            feature_size: Dimension size of input features, i.e., the feature dimension to normalize.
            epsilon: Small constant to prevent division by zero.
        Model can learn whether to change the numerical range, gamma and beta are learnable parameters,
        if model finds no need for scaling, gamma will approach 1
        if model finds no need for offset, beta will approach 0
        but model can selectively use these parameters
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_size))  # learnable scaling parameter, initial value 1
        self.beta = nn.Parameter(torch.zeros(feature_size))  # learnable offset parameter, initial value 0
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta  # Element-wise multiplication

# ------------------------------------------------------------------------------------------------

# Post-Norm architecture for Encoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation - Original Transformer's Post-Norm architecture.
        """
        # First sublayer: self-attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Second sublayer: feed-forward network + Add & Norm
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))


class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Encoder, stacked from N EncoderLayers.

        Parameters:
            d_model: Embedding dimension
            N: Number of encoder layers
            h: Number of attention heads
            d_ff: Hidden layer dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        # Note: Original Transformer paper has no final layer normalization
        # Each EncoderLayer already has its own layer normalization

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Input mask

        Returns:
            Encoder output (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x  # Directly return output of last EncoderLayer

# ------------------------------------------------------------------------------------------------

# Post-Norm architecture for Decoder
class DecoderLayer(nn.Module):
    """
    Decoder layer - Original Transformer paper implementation (without SublayerConnection).
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)  # masked multi-head self-attention
        self.cross_attn = MultiHeadAttention(d_model, h)  # multi-head cross-attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Three LayerNorms for three sublayers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(
        self, 
        x: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation - Original Transformer's Post-Norm architecture.
        
        Parameters:
            x: Decoder input (batch_size, seq_len_tgt, d_model)
            memory: Encoder output (batch_size, seq_len_src, d_model)
            src_mask: Source sequence mask for cross-attention
            tgt_mask: Target sequence mask for self-attention
        """
        # First sublayer: masked multi-head self-attention + Add & Norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Second sublayer: cross multi-head attention + Add & Norm
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Third sublayer: feed-forward network + Add & Norm
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, N: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Decoder, stacked from N DecoderLayers.

        Parameters:
            d_model: Embedding dimension
            N: Number of decoder layers
            h: Number of attention heads
            d_ff: Hidden layer dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        # Note: Original Transformer paper (Post-Norm) has no final layer normalization
        # Each DecoderLayer already has its own layer normalization

    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            x: Decoder input (batch_size, seq_len_tgt, d_model)
            memory: Encoder output (batch_size, seq_len_src, d_model)
            src_mask: Source sequence mask for cross-attention (optional)
            tgt_mask: Target sequence mask for self-attention (optional)

        Returns:
            Decoder output (batch_size, seq_len_tgt, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # Directly return output of last DecoderLayer

# ------------------------------------------------------------------------------------------------
# Embedding and Positional Encoding

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional encoding that adds unique positional representations to each position in the input sequence.

        Parameters:
            d_model: Embedding dimension, i.e., the dimension of the encoding vector for each position.
            dropout: Dropout probability applied after positional encoding.
            max_len: Maximum length of positional encoding to accommodate sequences of different lengths.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # As mentioned in paper section 5.4, dropout should be applied to the sum of embeddings and positional encoding

        # Create positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # Position indices (max_len, 1)

        # Calculate frequency for each dimension
        # Use explicit float dtype to ensure numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        # Combine position and frequency, compute sin and cos
        # Fix: Ensure it works correctly when d_model is odd
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        else:
            # If d_model is odd, there is one less odd column than even columns
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # Add batch dimension for easy addition with input, shape becomes (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register positional encoding as a buffer (not a parameter, no gradient updates)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            x: Input embedding vectors with shape (batch_size, seq_len, d_model).

        Returns:
            Embedding vectors with positional encoding and dropout applied, shape (batch_size, seq_len, d_model).
        """
        # Extract positional encodings matching the input sequence length and add to input
        # No requires_grad needed since positional encoding is fixed
        x = x + self.pe[:, :x.size(1), :]

        # Apply dropout
        return self.dropout(x)


class Embeddings(nn.Module):
    """
    Token Embeddings with scaling.

    This module converts token IDs to embedding vectors and applies scaling
    by sqrt(d_model) as described in the Transformer paper.
    Original Transformer paper: "We multiply those embeddings by sqrt(d_model)"
    GPT series (usually not used), Modern Transformer variants (many don't use)
    Modern initialization techniques: 
    1) Better weight initialization can replace scaling;
    2) If LayerNorm is present, scaling may not be necessary;
    3) Training stability: modern optimizers and training techniques make scaling less critical;

    Parameters:
        vocab_size: Vocabulary size
        d_model: Embedding vector dimension
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            x: Input tensor with shape (batch_size, seq_len), where each element is a token ID

        Returns:
            Scaled embedding vectors with shape (batch_size, seq_len, d_model)
        """
        return self.embed(x) * math.sqrt(self.d_model)


class SourceEmbedding(nn.Module):
    def __init__(self, src_vocab_size, d_model, dropout=0.1):
        """
        Source sequence embedding that converts input token sequences to embedding vectors and adds positional encoding.

        Parameters:
            src_vocab_size: Size of the source language vocabulary
            d_model: Dimension of the embedding vectors
            dropout: Dropout probability applied after positional encoding
        """
        super().__init__()
        self.embed = Embeddings(src_vocab_size, d_model)  # Token embedding layer
        self.positional_encoding = PositionalEncoding(d_model, dropout)  # Positional encoding layer

    def forward(self, x):
        """
        Forward propagation function.

        Parameters:
            x: Input tensor of source language sequence with shape (batch_size, seq_len_src), where each element is a token ID.

        Returns:
            Embedding vectors with positional encoding added, shape (batch_size, seq_len_src, d_model).
        """
        x = self.embed(x)  # Generate token embeddings (batch_size, seq_len_src, d_model)
        return self.positional_encoding(x)  # Add positional encoding


class TargetEmbedding(nn.Module):
    def __init__(self, tgt_vocab_size: int, d_model: int, dropout: float = 0.1):
        """
        Target sequence embedding that converts target token IDs to embedding vectors and adds positional encoding.

        Parameters:
            tgt_vocab_size: Size of the target language vocabulary
            d_model: Dimension of the embedding vectors
            dropout: Dropout probability applied after positional encoding
        """
        super().__init__()
        self.embed = Embeddings(tgt_vocab_size, d_model)  # Token embedding layer
        self.positional_encoding = PositionalEncoding(d_model, dropout)  # Positional encoding layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function.

        Parameters:
            x: Input tensor of target sequence with shape (batch_size, seq_len_tgt), where each element is a token ID.

        Returns:
            Embedding vectors with positional encoding added, shape (batch_size, seq_len_tgt, d_model).
        """
        x = self.embed(x)  # Generate token embeddings (batch_size, seq_len_tgt, d_model)
        return self.positional_encoding(x)  # Add positional encoding


# ------------------------------------------------------------------------------------------------
# Masking Mechanisms 
# Two types of masks in Transformer:
# 1. Padding mask - prevents attention to padding tokens
# 2. Look-ahead mask (causal mask) - prevents attention to future tokens

def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create padding mask to prevent attention to padding tokens.
    
    Parameters:
        seq: Input sequence with shape (batch_size, seq_len)
        pad_token: Padding token ID (default: 0)
    
    Returns:
        Boolean mask with shape (batch_size, 1, 1, seq_len)
        - True: valid token (can attend)
        - False: padding token (masked out)
    
    Example:
        >>> seq = torch.tensor([[5, 7, 9, 0, 0], [8, 6, 0, 0, 0]])  # 0 = <PAD>
        >>> create_padding_mask(seq)
        tensor([[[[ True,  True,  True, False, False]]],
                [[[ True,  True, False, False, False]]]])
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask


def create_look_ahead_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Create look-ahead mask (causal mask) to prevent attention to future positions.
    Used in decoder to prevent "peeking" at future tokens during training.
    
    Parameters:
        size: Sequence length
        device: Device to create tensor on (default: None, uses CPU)
    
    Returns:
        Boolean lower triangular mask with shape (size, size)
        - True: can attend
        - False: masked (future position)
    
    Example:
        >>> create_look_ahead_mask(5)
        tensor([[ True, False, False, False, False],  # Pos 0 can only see itself
                [ True,  True, False, False, False],  # Pos 1 can see 0, 1
                [ True,  True,  True, False, False],  # Pos 2 can see 0, 1, 2
                [ True,  True,  True,  True, False],  # Pos 3 can see 0, 1, 2, 3
                [ True,  True,  True,  True,  True]]) # Pos 4 can see all
    """
    mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return mask


def create_decoder_mask(tgt_seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create combined decoder mask (padding + look-ahead) for self-attention in decoder.
    
    Combines two constraints:
    1. Cannot attend to padding tokens
    2. Cannot attend to future positions (causal constraint)
    
    Parameters:
        tgt_seq: Target sequence with shape (batch_size, seq_len_tgt)
        pad_token: Padding token ID (default: 0)
    
    Returns:
        Combined boolean mask with shape (batch_size, 1, seq_len_tgt, seq_len_tgt)
        - Dimension 0: batch_size (different samples may have different padding)
        - Dimension 1: 1 (broadcasts to all attention heads)
        - Dimension 2-3: (seq_len, seq_len) attention matrix
    
    Note:
        Although padding positions can attend to previous valid tokens, their outputs
        are ignored during loss calculation and not used in inference.
    """
    # Get masks
    padding_mask = create_padding_mask(tgt_seq, pad_token)  # (batch_size, 1, 1, seq_len_tgt)
    look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1), device=tgt_seq.device)  # (seq_len_tgt, seq_len_tgt)
    
    # Combine: position can attend only if BOTH conditions are True
    # PyTorch broadcasting automatically handles the dimension alignment
    combined_mask = look_ahead_mask & padding_mask  # (batch_size, 1, seq_len_tgt, seq_len_tgt)
    return combined_mask

# ------------------------------------------------------------------------------------------------
# Full Transformer Model (Encoder-Decoder Architecture)

class Transformer(nn.Module):
    """
    Complete Transformer model as described in "Attention is All You Need" (Vaswani et al., 2017).
    
    Architecture:
        Input → Embedding + Positional Encoding → Encoder Stack → Decoder Stack → Output Projection
        
    The model consists of:
        - Source embedding layer (for encoder input)
        - Target embedding layer (for decoder input)
        - Encoder stack (N layers of self-attention + FFN)
        - Decoder stack (N layers of masked self-attention + cross-attention + FFN)
        - Output projection layer (to vocabulary)
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        share_embeddings: bool = False,
        pad_token: int = 0,
    ):
        """
        Initialize Transformer model.

        Parameters:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension (embedding dimension)
            num_layers: Number of encoder and decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward network hidden dimension (typically 4 * d_model)
            dropout: Dropout probability (default: 0.1)
            share_embeddings: Whether to share source and target embeddings (default: False)
            pad_token: Padding token ID (default: 0)
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token = pad_token

        # Embedding layers with positional encoding
        # src corresponds to encoder input, tgt corresponds to decoder input
        self.src_embedding = SourceEmbedding(src_vocab_size, d_model, dropout)
        
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            # Share embeddings between source and target (common in some tasks)
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TargetEmbedding(tgt_vocab_size, d_model, dropout)

        # Encoder and Decoder stacks
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)

        # Output projection layer (logits over target vocabulary)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Parameters:
            src: Source sequence (batch_size, seq_len_src) with token IDs
            tgt: Target sequence (batch_size, seq_len_tgt) with token IDs
            src_mask: Custom source mask (optional). If None, automatically creates padding mask.
                     Only provide this if you need custom masking logic beyond standard padding.
            tgt_mask: Custom target mask (optional). If None, automatically creates combined
                     padding + causal mask. Only provide this for custom masking logic.

        Returns:
            Output logits with shape (batch_size, seq_len_tgt, tgt_vocab_size)
            Note: Output is NOT softmaxed (raw logits for loss calculation)
            
        Examples:
            >>> # Standard usage (recommended): masks are auto-generated
            >>> output = model(src, tgt)
            
            >>> # Custom mask (rare): for special masking requirements
            >>> custom_mask = create_custom_mask(...)
            >>> output = model(src, tgt, src_mask=custom_mask)
        """
        # Auto-generate masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_token)
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt, self.pad_token)

        # Encoder: embed and encode source sequence
        src_embedded = self.src_embedding(src)  # (batch_size, seq_len_src, d_model)
        memory = self.encoder(src_embedded, src_mask)  # (batch_size, seq_len_src, d_model)

        # Decoder: embed target, decode with encoder output, and project to vocabulary
        tgt_embedded = self.tgt_embedding(tgt)  # (batch_size, seq_len_tgt, d_model)
        dec_output = self.decoder(tgt_embedded, memory, src_mask, tgt_mask)  # (batch_size, seq_len_tgt, d_model)
        output = self.output_projection(dec_output)  # (batch_size, seq_len_tgt, tgt_vocab_size)

        return output