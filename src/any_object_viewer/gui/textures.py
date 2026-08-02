"""サムネイル用の GL テクスチャキャッシュ."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, immvision


def tex_ref(texture: immvision.GlTexture) -> imgui.ImTextureRef:
    """GlTexture の ID を imgui の描画 API が要求する参照型に包む."""
    return imgui.ImTextureRef(texture.texture_id)


def to_uploadable(image: np.ndarray) -> np.ndarray:
    """GlTexture に渡せる配列にする.

    immvision のバインディングは書き込み可能な連続配列しか受け付けない。
    PIL 由来の配列は read-only なので、その場合だけコピーする。
    """
    if image.flags.writeable and image.flags.c_contiguous:
        return image
    return np.array(image, dtype=image.dtype, copy=True, order="C")


class TextureCache:
    """キーごとに GlTexture を保持する。GUI から都度アップロードしないため."""

    def __init__(self) -> None:
        self._textures: dict[object, immvision.GlTexture] = {}

    def get(self, key: object, image: np.ndarray) -> immvision.GlTexture:
        texture = self._textures.get(key)
        if texture is None:
            texture = immvision.GlTexture()
            texture.update_from_image(to_uploadable(image))
            self._textures[key] = texture
        return texture

    def drop(self, key: object) -> None:
        self._textures.pop(key, None)

    def clear(self) -> None:
        self._textures.clear()
