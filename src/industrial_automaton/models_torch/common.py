"""Common PyTorch model components: embeddings, base class, and pipeline.

Mirrors models_jax/common.py but uses nn.Module and torch ops.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from industrial_automaton.vocab import SIZE as VOCAB_SIZE, PAD, SEP, YIELD


class BaseAutomata(nn.Module, ABC):
    """Base class for all PyTorch neural automata models."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension fed into OutputHead."""
        pass

    @abstractmethod
    def init_state(self, batch_size=1, device=None):
        """Initialize model-specific state. Returns state or None."""
        pass

    @abstractmethod
    def forward(self, inputs, state, pad_mask, input_length=None):
        """Process embedded token sequence.

        Args:
            inputs:       (B, T, embedding_dim) float tensor
            state:        model-specific state
            pad_mask:     (B, T) bool tensor — True for real tokens
            input_length: (B,) int tensor, tokens before SEP/YIELD (for TapeRNN jumps)

        Returns:
            hidden:    (B, T, output_dim)
            new_state: updated state
        """
        pass


# ── Embedding layers ─────────────────────────────────────────────────────────

class LearnableEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)  # (T, embedding_dim)


class BinaryEmbedding(nn.Module):
    """Token index → binary bits → orthonormal projection.

    Mirrors the JAX BinaryEmbedding exactly:
      - QR decompose a random (embedding_dim, bits) matrix
      - Use Q.T as projection (total_bits, embedding_dim)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, fixed: bool = False, generator: torch.Generator = None):
        super().__init__()
        total_bits = math.ceil(math.log2(vocab_size))
        self.register_buffer('bits', torch.arange(total_bits))

        matrix = torch.randn(embedding_dim, total_bits, generator=generator)
        q, _ = torch.linalg.qr(matrix)  # (embedding_dim, total_bits)
        proj = q.T  # (total_bits, embedding_dim)

        if fixed:
            self.register_buffer('projection_matrix', proj)
        else:
            self.projection_matrix = nn.Parameter(proj)

    def forward(self, x):
        # x: (T,) int token indices
        binary = ((x.unsqueeze(-1) >> self.bits.unsqueeze(0)) & 1).float()  # (T, bits)
        return binary @ self.projection_matrix  # (T, embedding_dim)


class CosineEmbedding(nn.Module):
    """Sinusoidal positional embedding (non-learnable)."""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: (B, T)
        shape = x.shape
        x_flat = x.view(-1)
        position = x_flat.float().unsqueeze(-1)  # (B*T, 1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=x.device, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = torch.zeros(x_flat.shape[0], self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.view(*shape, self.embedding_dim)


class OneHotEmbedding(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        return F.one_hot(x, self.vocab_size).float()  # (T, vocab_size)


# ── Output head ───────────────────────────────────────────────────────────────

class OutputHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, vocab_size)

    def forward(self, hidden):
        return self.projection(hidden)  # (T, vocab_size)


# ── Full pipeline ─────────────────────────────────────────────────────────────

class ModelPipeline(nn.Module):
    """Token IDs → embedding → core model → logits.

    Mirrors the JAX ModelPipeline:
      - Computes PAD mask and input_length from raw token IDs
      - Passes embedded tokens + mask to model
      - Projects hidden states to vocab logits
    """

    def __init__(self, config, model_cls, embedding_type: str, generator: torch.Generator = None):
        super().__init__()

        if embedding_type == 'learnable':
            self.embedding = LearnableEmbedding(VOCAB_SIZE, config.embedding_dim)
        elif embedding_type == 'cosine':
            self.embedding = CosineEmbedding(VOCAB_SIZE, config.embedding_dim)
        elif embedding_type == 'binary':
            self.embedding = BinaryEmbedding(VOCAB_SIZE, config.embedding_dim, fixed=False, generator=generator)
        elif embedding_type == 'binary_fixed':
            self.embedding = BinaryEmbedding(VOCAB_SIZE, config.embedding_dim, fixed=True, generator=generator)
        elif embedding_type == 'one_hot':
            self.embedding = OneHotEmbedding(VOCAB_SIZE)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        self.model = model_cls(config, generator=generator)
        self.head = OutputHead(self.model.output_dim, VOCAB_SIZE)

    def init_state(self, batch_size=1, device=None):
        return self.model.init_state(batch_size=batch_size, device=device)

    def forward(self, x, state):
        """
        Args:
            x:     (B, T) int token indices
            state: model-specific state (from init_state)

        Returns:
            logits:    (B, T, vocab_size)
            new_state: updated model state
        """
        pad_mask = (x != PAD)  # (B, T) bool

        # input_length = index of first SEP or YIELD (= number of input tokens)
        yield_or_sep = (x == SEP) | (x == YIELD)
        B, T = x.shape
        positions = torch.where(
            yield_or_sep,
            torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1),
            torch.full_like(x, T),
        )
        input_length = positions.min(dim=1).values # (B,)

        embeds = self.embedding(x)                                   # (B, T, embedding_dim)
        hidden, new_state = self.model(embeds, state, pad_mask, input_length=input_length)
        logits = self.head(hidden)                                   # (B, T, vocab_size)
        return logits, new_state
