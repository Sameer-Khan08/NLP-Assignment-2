import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_idx: int = 0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        tok_emb = self.token_embedding(input_ids)                    # [B, T, D]
        pos_emb = self.position_embedding(positions)                 # [B, T, D]

        return tok_emb + pos_emb


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k, v: [B, H, T, Dh]
        mask:    [B, 1, 1, T] or [B, 1, T, T]
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # [B, H, T, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)                     # [B, H, T, T]
        output = torch.matmul(attn_weights, v)                           # [B, H, T, Dh]

        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] -> [B, H, T, Dh]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, T, Dh] -> [B, T, D]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, num_heads * head_dim)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        attn_output, attn_weights = self.attention(q, k, v, mask=mask)
        attn_output = self.combine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attn_weights


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()

        self.embedding = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            pad_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.cls_head = nn.Linear(d_model, num_classes)

    def make_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        attention_mask: [B, T]
        returns: [B, 1, 1, T]
        """
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.embedding(input_ids)           # [B, T, D]
        x = self.dropout(x)

        attn_mask = self.make_attention_mask(attention_mask)  # [B,1,1,T]

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=attn_mask)
            all_attn_weights.append(attn_weights)

        cls_rep = x[:, 0, :]                    # [B, D]
        logits = self.cls_head(cls_rep)         # [B, C]

        return logits, all_attn_weights