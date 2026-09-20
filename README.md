# Attention Is All You Need

A simple implementation of the original encoder-decoder Transformer architecture from scratch using PyTorch.

## Implemented Components

* Token embeddings
* Positional encoding
* Scaled dot-product attention
* Multi-head attention
* Padding and causal masks
* Encoder and decoder blocks
* Cross-attention
* Training and greedy inference

## Purpose

This project was built to understand how Transformers work internally without using PyTorch’s built-in `nn.Transformer`.

## Setup

```bash
git clone <your-repository-url>
cd attention-is-all-you-need

uv sync
```

## Run

```bash
uv run python main.py
```

## Reference

Based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
