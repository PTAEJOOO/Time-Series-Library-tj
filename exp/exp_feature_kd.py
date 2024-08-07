import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from math import sqrt
import random
from utils.masking import TriangularCausalMask, ProbMask

def student_attention(s_query, s_key, s_value):
    #, t_attn, t_query, t_key, t_value
    B, L, H, E = s_query.shape
    _, S, _, D = s_value.shape
    scale = 1. / sqrt(E)

    # r = random.uniform(0,1)

    scores = torch.einsum("blhe,bshe->bhls", s_query, s_key)
    attn_mask = TriangularCausalMask(B, L, device=s_query.device)
    scores.masked_fill_(attn_mask.mask, -np.inf)

    dropout = nn.Dropout(0.1)
    s_attn = dropout(torch.softmax(scale * scores, dim=-1))
    V = torch.einsum("bhls,bshd->blhd", s_attn, s_value)
    # return V.contiguous(), A


    return V.contiguous(), s_attn

def calculate_value(s_value, t_value):
    _, _, _, E = s_value.shape
    scale = 1./ sqrt(E)
    s_out = torch.einsum("blhe,bshe->bhls", s_value, s_value) * scale
    t_out = torch.einsum("blhe,bshe->bhls", t_value, t_value) * scale
    return s_out, t_out