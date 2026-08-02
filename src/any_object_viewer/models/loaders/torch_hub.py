"""torch.hub 経由でモデルを読み込むローダ（DINOv2 の既定経路）."""

from __future__ import annotations

import logging

import torch

from ..base import ModelEntry
from ..registry import register_loader
from ._torch_common import TorchExtractor

logger = logging.getLogger(__name__)


class _HubExtractor(TorchExtractor):
    """DINOv2 のように forward_features がトークンを dict で返すモデル向け."""

    def _tokens(self, x: torch.Tensor):
        module = self._module
        if hasattr(module, "forward_features"):
            out = module.forward_features(x)
            if isinstance(out, dict):
                cls = out.get("x_norm_clstoken")
                patches = out.get("x_norm_patchtokens")
                if cls is not None:
                    return cls, patches
            elif torch.is_tensor(out) and out.dim() == 3:
                # (B, N+1, D) 形式。先頭を CLS とみなす。
                return out[:, 0], out[:, 1:]
        return super()._tokens(x)


@register_loader("torch_hub")
class TorchHubLoader:
    """`repo` と `model` を指定して torch.hub.load を呼ぶ.

    args:
      repo:  例 "facebookresearch/dinov2"
      model: 例 "dinov2_vitb14"
      source: "github"（既定）または "local"
    """

    def load(self, entry: ModelEntry, device: str) -> _HubExtractor:
        repo = entry.args["repo"]
        model_name = entry.args["model"]
        source = entry.args.get("source", "github")

        logger.info("torch.hub から取得: repo=%s model=%s source=%s", repo, model_name, source)
        logger.info("キャッシュ先: %s", torch.hub.get_dir())
        logger.info("初回はリポジトリと重みのダウンロードが走ります（数分かかることがあります）")

        # verbose=True にしておくこと。torch のダウンロード進捗がここからしか出ない。
        module = torch.hub.load(repo, model_name, source=source, verbose=True)

        params = sum(p.numel() for p in module.parameters())
        logger.info("取得完了: %s パラメータ数 %.1fM", model_name, params / 1e6)
        return _HubExtractor(
            name=entry.name,
            module=module,
            preprocess=entry.preprocess,
            supported_embeddings=entry.embeddings,
            device=device,
        )
