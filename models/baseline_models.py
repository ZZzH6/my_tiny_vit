"""Backward-compatible re-exports for the old single-file model layout."""

from .deit_factory import COMMON_MODEL_KEYS, DEFAULT_LOCAL_FFN_BLOCKS, build_model, build_model_from_cfg
from .local_ffn import LocalFFN
from .precnn_adapter import PreCNNLocalAdapter
from .prepatch_adapter import PrePatchLocalAdapter

__all__ = [
    "COMMON_MODEL_KEYS",
    "DEFAULT_LOCAL_FFN_BLOCKS",
    "LocalFFN",
    "PreCNNLocalAdapter",
    "PrePatchLocalAdapter",
    "build_model",
    "build_model_from_cfg",
]
