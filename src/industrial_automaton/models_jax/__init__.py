"""Neural automata model architectures, organized by computation type.

Architecture files:
- common.py: Shared components (ModelPipeline, BaseAutomata, Embedding, OutputHead)
- implicit.py: Sequential models with hidden state
- transformers.py: Attention-based models
- stack_rnns.py: Stack-based automata
- tape.py: Memory-augmented models
"""

from .common import ModelPipeline, LearnableEmbedding, OutputHead, BaseAutomata, BinaryEmbedding, CosineEmbedding
from .implicit import LSTM, LSTMConfig, LSTMState
from .transformers import Transformer, TransformerConfig, TransformerBlock
from .stack_rnns import SuzgunStackRNN, SuzgunStackRNNConfig, StackRNNState, BabyNTM, BabyNTMModelConfig, BabyNTMState
# TapeRNN moved to models_torch — use TorchTrainer when model="tape_rnn"
# from .tape import TapeRNN, TapeRNNConfig, TapeRNNState

__all__ = [
    # Common
    "ModelPipeline", "Embedding", "OutputHead", "BaseAutomata",
    "BinaryEmbedding", "CosineEmbedding",
    # LSTM
    "LSTM", "LSTMConfig", "LSTMState", "LSTMCell",
    # Transformer
    "Transformer", "TransformerConfig", "TransformerBlock",
    # Stack RNN
    "SuzgunStackRNN", "SuzgunStackRNNConfig", "StackRNNState",
    # Tape models (BabyNTM only — TapeRNN lives in models_torch)
    "BabyNTM", "BabyNTMModelConfig", "BabyNTMState",
    # "TapeRNN", "TapeRNNConfig", "TapeRNNState",  # → models_torch
]
