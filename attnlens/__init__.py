"""AttnLens: multi-head self-attention from scratch, with pattern analysis."""

from .model import Params, forward, loss_and_grads, predict
from .train import train

__version__ = "0.1.0"

__all__ = ["Params", "forward", "loss_and_grads", "predict", "train"]
