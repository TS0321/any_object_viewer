"""BBox 切り出しからモデル入力までの前処理.

アスペクト比を維持して長辺を input_size に合わせ、短辺を黒でパディングする
（docs/spec.md 3.7）。入力サイズと正規化パラメータはモデル側が決めるため、
引数で受け取る。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .bbox import BBox


def crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    """フレームから BBox 領域を切り出す。BBox は画像内にクランプされる."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = bbox.clamped(width, height).to_pixels()
    x1 = min(x1, width)
    y1 = min(y1, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"BBox が画像の外にあります: {bbox}")
    return frame[y0:y1, x0:x1]


def letterbox(image: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """アスペクト比を保って長辺を size に合わせ、余白を pad_value で埋める.

    返り値は (size, size, 3) uint8。中央寄せで配置する。
    """
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, min(size, int(round(width * scale))))
    new_h = max(1, min(size, int(round(height * scale))))

    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def normalize(
    image: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 に正規化する."""
    array = image.astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def prepare(
    frame: np.ndarray,
    bbox: BBox,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """切り出し -> レターボックス -> 正規化 を通して (3, size, size) を返す."""
    return normalize(letterbox(crop(frame, bbox), size), mean, std)


def thumbnail(frame: np.ndarray, bbox: BBox, size: int = 128) -> np.ndarray:
    """GUI 表示用のサムネイル。前処理と同じ見え方になるようレターボックスする."""
    return letterbox(crop(frame, bbox), size)
