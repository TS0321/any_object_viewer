"""コサイン類似度とスコア集約."""

from __future__ import annotations

from enum import Enum

import numpy as np


class Aggregation(str, Enum):
    """マルチビュー登録時のスコア集約方式（docs/spec.md 3.6.2）."""

    MAX = "max"
    MEAN = "mean"


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """最終次元について L2 正規化する。(D,) と (N, D) の両方を受け付ける."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def cosine_similarity(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """照合 embedding (D,) と登録 embedding 群 (N, D) のコサイン類似度 (N,)."""
    q = l2_normalize(np.atleast_2d(query))
    r = l2_normalize(np.atleast_2d(references))
    return (r @ q.T).ravel()


def aggregate(scores: np.ndarray, method: Aggregation) -> float:
    """1 エントリが複数の登録画像を持つ場合のスコアをまとめる."""
    if scores.size == 0:
        return float("nan")
    if method is Aggregation.MAX:
        return float(np.max(scores))
    return float(np.mean(scores))
