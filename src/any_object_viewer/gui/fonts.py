"""日本語が表示できるフォントの読み込み.

Dear ImGui 1.92 はグリフを動的にロードするため、範囲指定は不要。
TTF/TTC を渡すだけでよい。
"""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import imgui

# 上から順に試す。macOS 標準に含まれるもの。
FONT_CANDIDATES: list[tuple[str, int]] = [
    ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0),
]


def load_japanese_font(size: float = 16.0) -> bool:
    """日本語フォントを読み込む。見つからなければ False を返す."""
    atlas = imgui.get_io().fonts
    for path, index in FONT_CANDIDATES:
        if not Path(path).is_file():
            continue
        config = imgui.ImFontConfig()
        config.font_no = index
        atlas.add_font_from_file_ttf(path, size, config)
        return True
    return False
