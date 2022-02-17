# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# Adapted from https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import math
import torch
from torch import nn


def attention(
    query, key, value, mask=None, dropout=None, bias=None, attn_bias_type=None
):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if bias is not None:
        if attn_bias_type == "dot":
            assert scores.size(0) == bias.size(0) and scores.size(-1) == bias.size(-1)
            scores = scores + bias[:, None, None, :]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -10000.0)

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, config, attn_bias_type=None):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads
        self.linears = self.clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.attn = None
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_bias_type = attn_bias_type

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None, bias=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        def gate(x, p):
            assert x.size(0) == p.size(0) and x.size(1) == p.size(-1)
            return x + self.dropout(p.unsqueeze(-1) * x)

        if bias is not None:
            if self.attn_bias_type == "key_only":
                key = gate(key, bias)
            elif self.attn_bias_type == "value_only":
                value = gate(value, bias)
            elif self.attn_bias_type == "both":
                key = gate(key, bias)
                value = gate(value, bias)

        n_b = query.size(0)
        query, key, value = [
            lin(x).view(n_b, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            bias=bias,
            attn_bias_type=self.attn_bias_type,
        )

        x = x.transpose(1, 2).contiguous().view(n_b, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadedAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, mask=None, bias=None):
        return x + self.dropout(self.self_attn(x, x, x, mask, bias))
