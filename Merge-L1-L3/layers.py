"""Transformer building blocks, copied verbatim from the official ECERC release
(IEMOCAP/model.py) so that parameter names match the released checkpoints.
Only change: PositionalEncoding takes a `scale` (MELD release multiplies it by 0.)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200, scale=1.0):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.scale = scale

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.scale * self.pos_table[:, :x.size(1), :x.size(2)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        return self.layer_norm(q)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(F.relu(self.w_1(x))))
        x += residual
        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model * 2, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        return self.pos_ffn(self.slf_attn(q, k, v, mask=mask))


class Encoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(n_head, d_model, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layer)])

    def forward(self, q, k, v, mode, mask=None):
        for enc_layer in self.layer_stack:
            q = enc_layer(q, k, v, mask=mask)
            if mode == "self":
                k = v = q
        return q


class ModalFilterV2(nn.Module):
    """Evidence Gating (stage 1). Also returns the per-modality gate activations,
    which CACG uses as part of its evidence-quality descriptor."""

    def __init__(self, d_t, d_a, d_v, hidden):
        super().__init__()
        self.proj_t = nn.Linear(d_t, hidden)
        self.proj_a = nn.Linear(d_a, hidden)
        self.proj_v = nn.Linear(d_v, hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)

    def forward(self, t, a, v):
        t, a, v = self.proj_t(t), self.proj_a(a), self.proj_v(v)
        w_t = torch.sigmoid(self.q(t) + self.k(a) + self.k(v))
        w_a = torch.sigmoid(self.q(a) + self.k(t) + self.k(v))
        w_v = torch.sigmoid(self.q(v) + self.k(t) + self.k(a))
        gates = torch.stack([w_t.mean(-1), w_a.mean(-1), w_v.mean(-1)], dim=-1)  # (B, L, 3)
        return torch.cat([w_t * t, w_a * a, w_v * v], dim=-1), gates
