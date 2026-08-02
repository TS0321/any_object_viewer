"""GUI に依存しないロジック層."""

from .bbox import BBox, BBoxSource
from .frames import FrameSource, ImageFolderSource
from .preprocess import crop, letterbox, normalize, prepare, thumbnail
from .scoring import Aggregation, aggregate, cosine_similarity, l2_normalize

__all__ = [
    "BBox",
    "BBoxSource",
    "FrameSource",
    "ImageFolderSource",
    "crop",
    "letterbox",
    "normalize",
    "prepare",
    "thumbnail",
    "Aggregation",
    "aggregate",
    "cosine_similarity",
    "l2_normalize",
]
