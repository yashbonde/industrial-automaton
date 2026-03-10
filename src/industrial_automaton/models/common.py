# This file contains the common embedding architecture and model pipeline

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from industrial_automaton.vocab import SIZE as VOCAB_SIZE, PAD, SEP
import math


class BaseAutomata(eqx.Module, ABC):
    """Base class for all neural automata models.

    All models process token sequences with explicit PAD masking.
    Subclasses implement PAD handling appropriate to their architecture.
    """

    autoregressive_input: bool = eqx.field(default=True, static=True)

    @property
    def output_dim(self) -> int:
        """Output dimension of the model (fed into OutputHead).
        Defaults to embedding_dim. Override if hidden_size != embedding_dim.
        """
        return self.embedding_dim

    @abstractmethod
    def __call__(self, inputs, state, pad_mask, input_length=None):
        """Process embedded token sequence with PAD masking.

        Args:
            inputs: (T, embedding_dim) - embedded token sequence
            state: Model-specific state (None for stateless models)
            pad_mask: (T,) boolean - True for real tokens, False for PAD
            input_length: int - number of input tokens before SEP (optional, used by TapeRNN)

        Returns:
            hidden: (T, output_dim) - hidden states for each position
            new_state: Updated model state
        """
        pass

    @abstractmethod
    def init_state(self):
        """Initialize model-specific state."""
        pass


class LearnableEmbedding(eqx.Module):
    """All embedding models operate in a specific ways. Each embedding method is
    different, but the output must be in the same embedding space while maximising
    the usage of the embedding dimension.
    
    Each models follows it's own patterns."""
    embedding: eqx.nn.Embedding

    def __init__(self, vocab_size: int, embedding_dim: int, *, key):
        self.embedding = eqx.nn.Embedding(vocab_size, embedding_dim, key=key)

    def __call__(self, x):
        # x: (T,) indices
        return jax.vmap(self.embedding)(x)


class CosineEmbedding(eqx.Module):
    """The basic positional embedding from GPT-2 paper.
    This produces [Vocab_Size x emb_dim] matrix."""
    max_len: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, embedding_dim: int, *, key):
        self.max_len = vocab_size
        self.embedding_dim = embedding_dim

    def __call__(self, x):
        # x: (T,) indices (positions)
        # Create sinusoidal positional embeddings
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        position = x[:, None].astype(jnp.float32)  # (T, 1)
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim))  # (D/2,)

        # Initialize embedding array
        pe = jnp.zeros((x.shape[0], self.embedding_dim))

        # Apply sin to even indices
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        # Apply cos to odd indices
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe


class BinaryEmbedding(eqx.Module):
    """Encodes token indices as binary representations and projects them.
    
    If fixed=True, the projection matrix is initialized randomly and fixed (non-trainable).
    """
    vocab_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    total_bits_required: int = eqx.field(static=True)
    fixed: bool = eqx.field(static=True)
    projection_matrix: jnp.ndarray

    def __init__(self, vocab_size: int, embedding_dim: int, fixed: bool = False, *, key):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.total_bits_required = math.ceil(math.log2(vocab_size))
        self.fixed = fixed
        
        # Initialize projection matrix: (total_bits_required, embedding_dim)
        # We use QR on the transpose to get an orthonormal basis of size total_bits_required in embedding_dim
        matrix = jax.random.normal(key, (embedding_dim, self.total_bits_required))
        q, _ = jnp.linalg.qr(matrix) # (embedding_dim, total_bits_required)
        self.projection_matrix = q.T if not fixed else jax.lax.stop_gradient(q.T)

    def __call__(self, x):
        # x: (T,) indices
        bits = jnp.arange(self.total_bits_required)
        binary = (x[:, None] >> bits[None, :]) & 1
        binary = binary.astype(jnp.float32)
        return binary @ self.projection_matrix

class OneHotEmbedding(eqx.Module):
    """One-hot encoding followed by a projection.
    
    If fixed=True, the projection matrix is initialized randomly and fixed (non-trainable).
    """
    vocab_size: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, embedding_dim: int = 0, fixed: bool = False, *, key):
        self.vocab_size = vocab_size

    def __call__(self, x):
        # x: (T,) indices
        return jax.nn.one_hot(x, self.vocab_size)
        

class OutputHead(eqx.Module):
    projection: eqx.nn.Linear

    def __init__(self, embedding_dim: int, vocab_size: int, *, key):
        self.projection = eqx.nn.Linear(embedding_dim, vocab_size, key=key)

    def __call__(self, hidden):
        # hidden: (T, D) hidden states
        return jax.vmap(self.projection)(hidden)


class ModelPipeline(eqx.Module):
    embedding: eqx.Module
    model: BaseAutomata  # Type hint enforces inheritance
    head: OutputHead

    def __init__(self, config, model_cls, embedding_type: str, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        if embedding_type == 'learnable':
            self.embedding = LearnableEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, key=k1)
        elif embedding_type == 'cosine':
            self.embedding = CosineEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, key=k1)
        elif embedding_type == 'binary':
            self.embedding = BinaryEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, fixed=False, key=k1)
        elif embedding_type == 'binary_fixed':
            self.embedding = BinaryEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, fixed=True, key=k1)
        elif embedding_type == 'one_hot':
            self.embedding = OneHotEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, fixed=False, key=k1)
        elif embedding_type == 'one_hot_fixed':
            self.embedding = OneHotEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=config.embedding_dim, fixed=True, key=k1)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        self.model = model_cls(config, key=k2)
        self.head = OutputHead(embedding_dim=self.model.output_dim, vocab_size=VOCAB_SIZE, key=k3)

    def init_state(self):
        return self.model.init_state()

    def __call__(self, x, state):
        # NOTE: Users only provide token sequences. PAD masking is handled
        # entirely by this pipeline. Individual models should never be
        # called directly — always go through ModelPipeline.

        # x: (T,) token indices [3, 5, 1, PAD, PAD, 7, PAD]

        # 1. Create PAD mask (model-agnostic detection)
        pad_mask = (x != PAD)  # (T,) [True, True, True, False, False, True, False]

        # 2. Compute input_length = number of tokens before SEP or YIELD
        #    Used by TapeRNN for correct jump distance
        from industrial_automaton.vocab import YIELD
        sep_positions = jnp.where((x == SEP) | (x == YIELD), jnp.arange(x.shape[0]), x.shape[0])
        input_length = jnp.min(sep_positions)

        # 3. Embed tokens
        embeds = self.embedding(x)  # (T, embedding_dim)

        # 4. Pass to model with mask and input_length
        hidden, new_state = self.model(embeds, state, pad_mask, input_length=input_length)  # (T, output_dim), state

        # 4. Project to vocabulary
        logits = self.head(hidden)  # (T, vocab_size)

        return logits, new_state
