"""モデル層の共通インタフェース.

GUI とスコア算出はここで定義した型のみに依存し、モデルの実体を知らない
（docs/spec.md 3.8.1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# embedding 種別の標準名。モデルが対応するものだけを supported_embeddings に並べる。
EMBEDDING_CLS = "cls"
EMBEDDING_PATCH_MEAN = "patch_mean"
EMBEDDING_CONCAT = "concat"
EMBEDDING_OUTPUT = "output"


@dataclass(frozen=True)
class PreprocessConfig:
    """前処理に必要なパラメータ。モデルごとに異なるのでモデル側が持つ."""

    input_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@runtime_checkable
class FeatureExtractor(Protocol):
    """任意のモデルをこの形に揃える."""

    name: str
    preprocess: PreprocessConfig
    supported_embeddings: list[str]

    def embed(self, batch: np.ndarray, embedding_type: str) -> np.ndarray:
        """前処理済みバッチ (N, 3, S, S) float32 から (N, D) float32 を返す."""
        ...

    @property
    def device(self) -> str: ...


@dataclass
class ModelEntry:
    """レジストリ（YAML）の 1 エントリ（docs/spec.md 3.8.3）."""

    name: str
    loader: str
    args: dict = field(default_factory=dict)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embeddings: list[str] = field(default_factory=lambda: [EMBEDDING_OUTPUT])

    @classmethod
    def from_dict(cls, d: dict) -> ModelEntry:
        pre = d.get("preprocess") or {}
        return cls(
            name=d["name"],
            loader=d["loader"],
            args=d.get("args") or {},
            preprocess=PreprocessConfig(
                input_size=int(pre.get("input_size", 224)),
                mean=tuple(pre.get("mean", (0.485, 0.456, 0.406))),
                std=tuple(pre.get("std", (0.229, 0.224, 0.225))),
            ),
            embeddings=list(d.get("embeddings") or [EMBEDDING_OUTPUT]),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "loader": self.loader,
            "args": self.args,
            "preprocess": {
                "input_size": self.preprocess.input_size,
                "mean": list(self.preprocess.mean),
                "std": list(self.preprocess.std),
            },
            "embeddings": self.embeddings,
        }


class ModelLoader(Protocol):
    """モデル形式ごとの読み込み手続き。追加はこれを実装して register するだけ."""

    def load(self, entry: ModelEntry, device: str) -> FeatureExtractor: ...
