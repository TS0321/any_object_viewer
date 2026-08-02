"""動作確認用のサンプルフレーム列を生成する.

同じ物体が動くフレーム列を作るので、「別フレームの同一物体」と
「別物体」でスコアがどう変わるかを試せる。

    uv run python scripts/make_sample_frames.py sample_data/frames
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 960, 540
N_FRAMES = 60

# (名前, 色, 形, 半径, 軌道の位相)
OBJECTS = [
    ("red_circle", (220, 60, 60), "circle", 46, 0.0),
    ("blue_square", (60, 110, 230), "square", 44, 2.1),
    ("green_tri", (70, 190, 90), "triangle", 50, 4.2),
    ("yellow_circle", (235, 200, 60), "circle", 40, 1.0),
]


def draw_object(draw: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, r: int, color) -> None:
    if kind == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)], fill=color
        )


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    # 背景にノイズテクスチャを敷いて、切り出し位置の違いが効くようにする
    background = rng.integers(28, 46, (HEIGHT, WIDTH, 3), dtype=np.uint8)

    for i in range(N_FRAMES):
        t = i / N_FRAMES
        image = Image.fromarray(background.copy())
        draw = ImageDraw.Draw(image)
        for _, color, kind, radius, phase in OBJECTS:
            cx = WIDTH * (0.15 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * t + phase)))
            cy = HEIGHT * (0.25 + 0.5 * (0.5 + 0.5 * math.cos(2 * math.pi * t * 1.3 + phase)))
            draw_object(draw, kind, cx, cy, radius, color)
        image.save(out / f"frame_{i:04d}.png")

    print(f"{N_FRAMES} 枚を生成しました: {out.resolve()}")
    print("含まれる物体:", ", ".join(o[0] for o in OBJECTS))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample_data/frames")
