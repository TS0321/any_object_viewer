"""フレーム供給源の抽象化.

いまは画像フォルダのみ。動画リーダーは FrameSource を実装して差し込む
（docs/spec.md 3.1, 7.2）。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@runtime_checkable
class FrameSource(Protocol):
    """フレーム列を index でランダムアクセスできるもの."""

    def __len__(self) -> int: ...

    def get(self, index: int) -> np.ndarray:
        """RGB uint8 の (H, W, 3) を返す."""
        ...

    def name(self, index: int) -> str:
        """表示用の名前（ファイル名など）."""
        ...


def _natural_key(path: Path) -> tuple:
    """frame_2 が frame_10 より前に来るように分割してソートする."""
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


class ImageFolderSource:
    """フォルダ内の画像をフレーム列として扱う.

    フレームごとにサイズが違ってよい。読み込んだ画像は LRU でキャッシュする。
    """

    def __init__(self, folder: str | Path, cache_size: int = 16) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"フォルダが見つかりません: {self.folder}")

        self.paths: list[Path] = sorted(
            (p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
            key=_natural_key,
        )
        if not self.paths:
            raise ValueError(f"画像が 1 枚もありません: {self.folder}")

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        with Image.open(self.paths[index]) as im:
            array = np.asarray(im.convert("RGB"), dtype=np.uint8)

        self._cache[index] = array
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return array

    def name(self, index: int) -> str:
        return self.paths[index].name

    def path(self, index: int) -> Path:
        return self.paths[index]
