"""HuggingFace transformers のモデルを読み込むローダ.

Hub ID（例 "facebook/dinov2-base"）でもローカルディレクトリでも読める。
一度ダウンロード済みなら ~/.cache/huggingface に残っているので、
オフラインでも `local_files_only: true` で読み込める。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..base import ModelEntry
from ..registry import register_loader
from ._torch_common import TorchExtractor

logger = logging.getLogger(__name__)


class _HFExtractor(TorchExtractor):
    """transformers の出力から CLS とパッチトークンを取り出す.

    Dinov2Model など ViT 系は last_hidden_state が (B, 1+N, D) で、
    先頭が CLS、残りがパッチトークン。
    """

    def _tokens(self, x: torch.Tensor):
        out = self._module(pixel_values=x)

        hidden = getattr(out, "last_hidden_state", None)
        if hidden is not None and hidden.dim() == 3:
            return hidden[:, 0], hidden[:, 1:]

        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled.flatten(1), None

        raise ValueError(
            f"{self.name}: last_hidden_state も pooler_output も返さないモデルです"
        )


@register_loader("huggingface")
class HuggingFaceLoader:
    """args:
    path: Hub ID（"facebook/dinov2-base"）またはローカルディレクトリ
    local_files_only: True ならダウンロードせずキャッシュのみ使う（既定 False）
    revision: 任意のリビジョン
    """

    def load(self, entry: ModelEntry, device: str) -> _HFExtractor:
        from transformers import AutoModel

        path = str(entry.args["path"])
        local_only = bool(entry.args.get("local_files_only", False))
        revision = entry.args.get("revision")

        expanded = Path(path).expanduser()
        if expanded.is_dir():
            path = str(expanded)
            logger.info("ローカルディレクトリから読み込み: %s", path)
        else:
            logger.info(
                "HuggingFace から読み込み: %s（local_files_only=%s）", path, local_only
            )

        kwargs: dict = {"local_files_only": local_only}
        if revision:
            kwargs["revision"] = revision

        module = AutoModel.from_pretrained(path, **kwargs)

        params = sum(p.numel() for p in module.parameters())
        logger.info(
            "取得完了: %s パラメータ数 %.1fM hidden=%s",
            path, params / 1e6, getattr(module.config, "hidden_size", "?"),
        )

        return _HFExtractor(
            name=entry.name,
            module=module,
            preprocess=entry.preprocess,
            supported_embeddings=entry.embeddings,
            device=device,
        )
