"""モデル層。GUI はここで定義した型のみに依存する."""

from .base import (
    EMBEDDING_CLS,
    EMBEDDING_CONCAT,
    EMBEDDING_OUTPUT,
    EMBEDDING_PATCH_MEAN,
    FeatureExtractor,
    ModelEntry,
    ModelLoader,
    PreprocessConfig,
)
from .loaders._torch_common import resolve_device
from .registry import (
    ModelRegistry,
    available_loaders,
    get_loader,
    load_builtin_loaders,
    load_user_loaders,
    register_loader,
)

__all__ = [
    "EMBEDDING_CLS",
    "EMBEDDING_CONCAT",
    "EMBEDDING_OUTPUT",
    "EMBEDDING_PATCH_MEAN",
    "FeatureExtractor",
    "ModelEntry",
    "ModelLoader",
    "ModelRegistry",
    "PreprocessConfig",
    "available_loaders",
    "get_loader",
    "load_builtin_loaders",
    "load_user_loaders",
    "register_loader",
    "resolve_device",
]
