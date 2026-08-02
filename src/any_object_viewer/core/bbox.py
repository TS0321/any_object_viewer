"""BBox とその由来."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BBoxSource(str, Enum):
    """BBox がどこから来たか。

    検証時に「人が引いた枠」と「モデルが出した枠」を区別する必要があるため、
    BBox 自体に持たせる。DETECTOR は将来の自動検出用（docs/spec.md 7.1）。
    """

    MANUAL = "manual"
    DETECTOR = "detector"


@dataclass
class BBox:
    """画像座標系の矩形。x, y は左上、w, h は正の値に正規化して保持する."""

    x: float
    y: float
    w: float
    h: float
    source: BBoxSource = BBoxSource.MANUAL

    def __post_init__(self) -> None:
        if self.w < 0:
            self.x += self.w
            self.w = -self.w
        if self.h < 0:
            self.y += self.h
            self.h = -self.h

    @classmethod
    def from_corners(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        source: BBoxSource = BBoxSource.MANUAL,
    ) -> BBox:
        return cls(x0, y0, x1 - x0, y1 - y0, source)

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def clamped(self, width: int, height: int) -> BBox:
        """画像の内側に収める."""
        x0 = max(0.0, min(self.x, width))
        y0 = max(0.0, min(self.y, height))
        x1 = max(0.0, min(self.x1, width))
        y1 = max(0.0, min(self.y1, height))
        return BBox.from_corners(x0, y0, x1, y1, self.source)

    def to_pixels(self) -> tuple[int, int, int, int]:
        """切り出し用の整数座標 (x0, y0, x1, y1)。最低 1px は確保する."""
        x0 = int(round(self.x))
        y0 = int(round(self.y))
        x1 = max(x0 + 1, int(round(self.x1)))
        y1 = max(y0 + 1, int(round(self.y1)))
        return x0, y0, x1, y1

    def is_valid(self, min_size: float = 2.0) -> bool:
        return self.w >= min_size and self.h >= min_size

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BBox:
        return cls(
            x=d["x"],
            y=d["y"],
            w=d["w"],
            h=d["h"],
            source=BBoxSource(d.get("source", "manual")),
        )
