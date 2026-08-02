"""TorchScript の .pt / .pth をパス指定で読み込むローダ.

「任意のモデルパスを指定して読む」経路の最小実装（docs/spec.md 3.8.2）。
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..base import ModelEntry
from ..registry import register_loader
from ._torch_common import TorchExtractor


@register_loader("torchscript")
class TorchScriptLoader:
    """args:
    path: TorchScript ファイルへのパス
    """

    def load(self, entry: ModelEntry, device: str) -> TorchExtractor:
        path = Path(entry.args["path"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")

        module = torch.jit.load(str(path), map_location=device)
        return TorchExtractor(
            name=entry.name,
            module=module,
            preprocess=entry.preprocess,
            supported_embeddings=entry.embeddings,
            device=device,
        )
