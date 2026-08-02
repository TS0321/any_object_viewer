"""PyTorch 系ローダの共通部分."""

from __future__ import annotations

import numpy as np
import torch

from ..base import (
    EMBEDDING_CLS,
    EMBEDDING_CONCAT,
    EMBEDDING_OUTPUT,
    EMBEDDING_PATCH_MEAN,
    PreprocessConfig,
)


def resolve_device(preferred: str | None = None) -> str:
    """cuda > mps > cpu の順に利用可能なデバイスを選ぶ."""
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchExtractor:
    """torch.nn.Module を FeatureExtractor に適合させる.

    トークンの取り出し方はモデルによって違うので、サブクラスが `_tokens` を
    実装する。既定では forward の出力をそのまま 1 本のベクトルとして扱う。
    """

    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        preprocess: PreprocessConfig,
        supported_embeddings: list[str],
        device: str,
    ) -> None:
        self.name = name
        self.preprocess = preprocess
        self.supported_embeddings = supported_embeddings
        self._device = device
        self._module = module.eval().to(device)

    @property
    def device(self) -> str:
        return self._device

    def _tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(cls 相当 (B, D), patch トークン (B, N, D) または None) を返す."""
        out = self._module(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.flatten(1), None

    @torch.inference_mode()
    def embed(self, batch: np.ndarray, embedding_type: str) -> np.ndarray:
        if embedding_type not in self.supported_embeddings:
            raise ValueError(
                f"{self.name} は embedding '{embedding_type}' に非対応。"
                f"対応: {', '.join(self.supported_embeddings)}"
            )

        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self._device)
        cls, patches = self._tokens(x)

        if embedding_type in (EMBEDDING_CLS, EMBEDDING_OUTPUT):
            feat = cls
        elif embedding_type == EMBEDDING_PATCH_MEAN:
            feat = self._require_patches(patches).mean(dim=1)
        elif embedding_type == EMBEDDING_CONCAT:
            feat = torch.cat([cls, self._require_patches(patches).mean(dim=1)], dim=-1)
        else:
            raise ValueError(f"未知の embedding 種別: {embedding_type}")

        return feat.float().cpu().numpy()

    def _require_patches(self, patches: torch.Tensor | None) -> torch.Tensor:
        if patches is None:
            raise ValueError(f"{self.name} はパッチトークンを返さない")
        return patches
