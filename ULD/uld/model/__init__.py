"""Model registry for RECAP assistant training."""

from .utils import create_peft_model

TRAIN_INIT_FUNCS = {"uld": create_peft_model}

__all__ = ["TRAIN_INIT_FUNCS"]
