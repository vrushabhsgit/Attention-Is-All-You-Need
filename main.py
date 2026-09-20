"""
Attention Is All You Need
=========================

A from-scratch implementation of the original encoder-decoder
Transformer architecture using PyTorch.

Components:
- Token and positional embeddings
- Scaled dot-product attention
- Multi-head attention
- Encoder and decoder blocks
- Padding and causal masks
- Cross-attention
- Training and greedy inference

Reference:
https://arxiv.org/abs/1706.03762
"""

import math

import torch
import torch.nn as nn