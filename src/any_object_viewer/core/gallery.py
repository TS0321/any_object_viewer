"""ギャラリー（1:N 照合の N 側）のデータ構造（docs/spec.md 3.6）."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .bbox import BBox
from .scoring import Aggregation, aggregate, cosine_similarity

_ids = itertools.count(1)


@dataclass
class RegisteredImage:
    """エントリを構成する登録画像 1 枚."""

    bbox: BBox
    frame_index: int | None = None
    """画像群から取った場合のフレーム番号."""

    source_path: str | None = None
    """事前設定ファイルから取った場合のパス."""

    thumbnail: np.ndarray | None = None
    embedding: np.ndarray | None = None
    embedding_key: tuple | None = None
    """embedding を計算したときの (モデル名, embedding 種別)."""

    uid: int = field(default_factory=lambda: next(_ids))

    def origin_label(self) -> str:
        if self.source_path is not None:
            return self.source_path.rsplit("/", 1)[-1]
        return f"frame {self.frame_index}"

    def is_stale(self, key: tuple) -> bool:
        return self.embedding is None or self.embedding_key != key

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "frame_index": self.frame_index,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegisteredImage:
        return cls(
            bbox=BBox.from_dict(d["bbox"]),
            frame_index=d.get("frame_index"),
            source_path=d.get("source_path"),
        )


@dataclass
class GalleryEntry:
    """1 つの物体。マルチビューのため登録画像を複数持てる."""

    name: str
    images: list[RegisteredImage] = field(default_factory=list)
    uid: int = field(default_factory=lambda: next(_ids))

    def to_dict(self) -> dict:
        return {"name": self.name, "images": [i.to_dict() for i in self.images]}

    @classmethod
    def from_dict(cls, d: dict) -> GalleryEntry:
        return cls(
            name=d["name"],
            images=[RegisteredImage.from_dict(i) for i in d.get("images", [])],
        )


@dataclass
class MatchResult:
    """1 エントリに対する照合結果."""

    entry: GalleryEntry
    score: float
    per_image: np.ndarray

    def is_ok(self, threshold: float) -> bool:
        return self.score >= threshold


class Gallery:
    def __init__(self) -> None:
        self.entries: list[GalleryEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: GalleryEntry) -> None:
        self.entries.append(entry)

    def remove(self, entry: GalleryEntry) -> None:
        if entry in self.entries:
            self.entries.remove(entry)

    def unique_name(self, base: str = "object") -> str:
        existing = {e.name for e in self.entries}
        for i in itertools.count(1):
            name = f"{base}_{i:02d}"
            if name not in existing:
                return name
        raise AssertionError("unreachable")

    def stale_images(self, key: tuple) -> list[RegisteredImage]:
        return [img for e in self.entries for img in e.images if img.is_stale(key)]

    def match(
        self, query: np.ndarray, method: Aggregation
    ) -> list[MatchResult]:
        """照合 embedding を全エントリと比較し、スコア降順で返す."""
        results: list[MatchResult] = []
        for entry in self.entries:
            embeddings = [i.embedding for i in entry.images if i.embedding is not None]
            if not embeddings:
                continue
            per_image = cosine_similarity(query, np.stack(embeddings))
            results.append(
                MatchResult(entry=entry, score=aggregate(per_image, method), per_image=per_image)
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results
