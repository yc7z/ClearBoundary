import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    "Scaled Dot-Product Attention'"
    d_k = query.size(-1)
    # Transpose the last two dimensions since dim0 might be the batch size.
    scores = torch.matmul(query, torch.transpose(key, -1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e12)
    attn_probs = scores.softmax(dim=-1)
    if dropout is not None:
        attn_probs = dropout(attn_probs)
    outputs = torch.matmul(attn_probs, value)
    return outputs, attn_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout_p: float = 0.1) -> None:
        assert (d_model % n_heads == 0) and (0 <= dropout_p < 1)
        super(MultiHeadAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.linear_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.attn_probs: Tensor
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = q.size(0)
        query = self.linear_q.forward(q)
        key = self.linear_k.forward(k)
        value = self.linear_v.forward(v)

        # Reshape the queries, keys, and values into (batch_size, n_heads, seq_length, d_k)
        query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # attn_outputs shape: (batch_size, n_heads, seq_length, d_k)
        # attn_probs shape: (batch_size, n_heads, seq_length, seq_length)
        attn_outputs, attn_probs = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)
        self.attn_probs = attn_probs

        # "Concatenate" the outputs of all heads and apply W_O
        # (batch_size, n_heads, seq_length, d_k) -> (batch_size, seq_length, n_heads, d_k)
        # -> (batch_size, seq_length, d_model)
        x = attn_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)

        del query
        del key
        del value
        return self.linear_o.forward(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout_p: float = 0.1) -> None:
        assert 0 <= dropout_p < 1
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout_p)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_hidden: int, dropout_p: float):
        super(TransformerBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(n_heads, d_model, dropout_p)
        self.ffn = FeedForwardNetwork(d_model, d_hidden, dropout_p)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x1 = self.layer_norm1(x)
        x = x + self.multi_head_attn.forward(q=x1, k=x1, v=x1, mask=mask)
        x2 = self.layer_norm2(x)
        return x + self.ffn.forward(x2)


class Transformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        d_model: int,
        d_hidden: int,
        d_input: int,
        d_output: int,
        dropout_p: float,
    ) -> None:
        super(Transformer, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_heads, d_model, d_hidden, dropout_p) for _ in range(n_blocks)]
        )
        self.last_layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.first_embedding_layer = nn.Linear(in_features=d_input, out_features=d_model)
        self.final_projection_layer = nn.Linear(in_features=d_model, out_features=d_output)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_embedding_layer.forward(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        x = self.last_layer_norm(x)
        return self.final_projection_layer.forward(x[:, -1, :])


class RMSTransformerBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_hidden: int, dropout_p: float):
        super(RMSTransformerBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(n_heads, d_model, dropout_p)
        self.ffn = FeedForwardNetwork(d_model, d_hidden, dropout_p)
        self.rms_norm1 = nn.RMSNorm(normalized_shape=d_model)
        self.rms_norm2 = nn.RMSNorm(normalized_shape=d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x1 = self.rms_norm1(x)
        x = x + self.multi_head_attn.forward(q=x1, k=x1, v=x1, mask=mask)
        x2 = self.rms_norm2(x)
        return x + self.ffn.forward(x2)


class RMSTransformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        d_model: int,
        d_hidden: int,
        d_input: int,
        d_output: int,
        dropout_p: float,
    ) -> None:
        super(RMSTransformer, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [RMSTransformerBlock(n_heads, d_model, d_hidden, dropout_p) for _ in range(n_blocks)]
        )
        self.last_rms_norm = nn.RMSNorm(normalized_shape=d_model)
        self.first_embedding_layer = nn.Linear(in_features=d_input, out_features=d_model)
        self.final_projection_layer = nn.Linear(in_features=d_model, out_features=d_output)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_embedding_layer.forward(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        x = self.last_rms_norm(x)
        return self.final_projection_layer.forward(x[:, -1, :])
