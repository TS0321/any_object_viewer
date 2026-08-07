"""日本語が表示できるフォントの読み込み.

Dear ImGui 1.92 はグリフを動的にロードするため、範囲指定は不要。
TTF/TTC を渡すだけでよい。

UI ラベルが日本語なので、ここで失敗すると全部豆腐になる。
OS ごとに候補を並べ、見つからなければ警告を出す。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from imgui_bundle import imgui

logger = logging.getLogger(__name__)


def _windows_fonts() -> list[tuple[str, int]]:
    root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    names = [
        "YuGothR.ttc",   # 游ゴシック Regular（Win10 以降の標準）
        "YuGothM.ttc",   # 游ゴシック Medium
        "meiryo.ttc",    # メイリオ
        "msgothic.ttc",  # MS ゴシック
    ]
    return [(str(root / n), 0) for n in names]


def _macos_fonts() -> list[tuple[str, int]]:
    return [
        ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0),
    ]


def _linux_fonts() -> list[tuple[str, int]]:
    return [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf", 0),
        ("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ]


def font_candidates() -> list[tuple[str, int]]:
    """このプラットフォームで試すフォントの一覧（パス, TTC インデックス）."""
    if sys.platform == "win32":
        return _windows_fonts()
    if sys.platform == "darwin":
        return _macos_fonts()
    return _linux_fonts()


def load_japanese_font(size: float = 16.0) -> bool:
    """日本語フォントを読み込む。見つからなければ False を返す.

    AOV_FONT 環境変数でフォントファイルを明示指定できる
    （候補に無いフォントを使いたい場合や、候補が全滅した場合の逃げ道）。
    """
    atlas = imgui.get_io().fonts

    candidates = list(font_candidates())
    override = os.environ.get("AOV_FONT")
    if override:
        candidates.insert(0, (override, int(os.environ.get("AOV_FONT_INDEX", "0"))))

    for path, index in candidates:
        if not Path(path).is_file():
            continue
        config = imgui.ImFontConfig()
        config.font_no = index
        atlas.add_font_from_file_ttf(path, size, config)
        logger.info("フォント: %s", path)
        return True

    logger.warning(
        "日本語フォントが見つかりませんでした（%s）。UI の日本語が表示されません。\n"
        "  試したパス: %s\n"
        "  AOV_FONT=<ttf/ttc のパス> で明示指定できます。",
        sys.platform,
        ", ".join(p for p, _ in candidates),
    )
    return False
