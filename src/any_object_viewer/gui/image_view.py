"""フレーム表示 + ズーム/パン + BBox 描画のウィジェット.

immvision.image() ではなく自前描画にしている。BBox の描画・移動・リサイズ・
正方形拘束を思い通りに制御する必要があるため（docs/spec.md 3.3）。
テクスチャのアップロードだけ immvision.GlTexture を借りる。

操作:
  左ドラッグ       新規 BBox を描く / ハンドルを掴んでリサイズ / 内側を掴んで移動
  中 or 右ドラッグ  パン
  ホイール         カーソル位置を中心にズーム
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imgui_bundle import ImVec2, imgui, immvision

from ..core.bbox import BBox, BBoxSource
from .textures import tex_ref, to_uploadable

HANDLE_RADIUS = 5.0
"""ハンドルの当たり判定半径（スクリーン px）."""

MIN_ZOOM = 0.02
MAX_ZOOM = 64.0

# ハンドルの並び: 四隅
_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


@dataclass
class _Drag:
    """ドラッグ中の状態."""

    mode: str  # "new" | "move" | "resize" | "pan"
    anchor_img: ImVec2 | None = None  # 固定される側の画像座標
    corner: int = -1
    grab_offset: ImVec2 | None = None
    start_pan: ImVec2 | None = None
    start_mouse: ImVec2 | None = None


class ImageView:
    """1 枚のフレームを表示し、BBox を 1 つ編集させるウィジェット."""

    def __init__(self, label: str) -> None:
        self.label = label
        # GlTexture の生成は OpenGL コンテキストを要求するため、描画ループ内まで遅らせる
        self._texture: immvision.GlTexture | None = None
        self._tex_key: object = None
        self._image_size = (0, 0)  # (w, h)

        self.zoom = 1.0
        self.pan = ImVec2(0.0, 0.0)  # キャンバス左上に対応する画像座標
        self._fit_pending = True

        self.bbox: BBox | None = None
        self.square_mode = True
        self._drag: _Drag | None = None
        self._changed = False

        # 照合枠の真上に出すラベル（1 位の名前とスコアなど）
        self.bbox_label = ""
        self.bbox_label_color = 0

        # 参考表示用（登録済み BBox などを薄く重ねる）
        self.overlays: list[tuple[BBox, int]] = []

    # ------------------------------------------------------------------
    # 座標変換
    # ------------------------------------------------------------------
    def _to_screen(self, x: float, y: float) -> ImVec2:
        return ImVec2(
            self._origin.x + (x - self.pan.x) * self.zoom,
            self._origin.y + (y - self.pan.y) * self.zoom,
        )

    def _to_image(self, pos: ImVec2) -> ImVec2:
        return ImVec2(
            (pos.x - self._origin.x) / self.zoom + self.pan.x,
            (pos.y - self._origin.y) / self.zoom + self.pan.y,
        )

    # ------------------------------------------------------------------
    # 画像の差し替え
    # ------------------------------------------------------------------
    def set_image(self, image: np.ndarray, key: object) -> None:
        """フレームを差し替える。key が同じならアップロードを省く.

        描画ループ内から呼ぶこと（GlTexture が OpenGL コンテキストを要求する）。
        """
        if key == self._tex_key and self._texture is not None:
            return
        if self._texture is None:
            self._texture = immvision.GlTexture()
        self._texture.update_from_image(to_uploadable(image))
        self._tex_key = key
        new_size = (image.shape[1], image.shape[0])
        if new_size != self._image_size:
            self._image_size = new_size
            self._fit_pending = True

    def request_fit(self) -> None:
        self._fit_pending = True

    def _fit(self, canvas: ImVec2) -> None:
        width, height = self._image_size
        if width == 0 or height == 0 or canvas.x <= 0 or canvas.y <= 0:
            return
        self.zoom = min(canvas.x / width, canvas.y / height)
        self.pan = ImVec2(
            (width - canvas.x / self.zoom) / 2,
            (height - canvas.y / self.zoom) / 2,
        )
        self._fit_pending = False

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------
    def draw(self, size: ImVec2 | None = None) -> bool:
        """1 フレーム分描画する。BBox が変化したら True を返す."""
        self._changed = False
        if self._texture is None or self._image_size == (0, 0):
            imgui.text_disabled("画像が読み込まれていません")
            return False

        canvas = size or imgui.get_content_region_avail()
        canvas = ImVec2(max(canvas.x, 32.0), max(canvas.y, 32.0))
        self._origin = imgui.get_cursor_screen_pos()

        if self._fit_pending:
            self._fit(canvas)

        self._canvas_max = ImVec2(self._origin.x + canvas.x, self._origin.y + canvas.y)
        imgui.invisible_button(f"##canvas_{self.label}", canvas)
        hovered = imgui.is_item_hovered()
        active = imgui.is_item_active()

        self._handle_input(hovered, active)

        draw_list = imgui.get_window_draw_list()
        canvas_max = ImVec2(self._origin.x + canvas.x, self._origin.y + canvas.y)
        draw_list.push_clip_rect(self._origin, canvas_max, True)

        draw_list.add_rect_filled(self._origin, canvas_max, imgui.IM_COL32(30, 30, 34, 255))

        width, height = self._image_size
        draw_list.add_image(
            tex_ref(self._texture),
            self._to_screen(0, 0),
            self._to_screen(width, height),
        )

        for box, color in self.overlays:
            self._draw_box(draw_list, box, color, thickness=1.5, handles=False)

        if self.bbox is not None:
            color = imgui.IM_COL32(70, 220, 120, 255)
            self._draw_box(draw_list, self.bbox, color, thickness=2.0, handles=True)
            if self.bbox_label:
                self._draw_label(draw_list, self.bbox, self.bbox_label,
                                 self.bbox_label_color or color)

        draw_list.pop_clip_rect()
        return self._changed

    def _draw_label(
        self, draw_list: imgui.ImDrawList, box: BBox, text: str, color: int
    ) -> None:
        """枠の上辺に貼り付くラベル。枠が画面上端に近いときは内側に回り込む."""
        size = imgui.calc_text_size(text)
        top_left = self._to_screen(box.x, box.y)
        pad = 4.0
        width = size.x + pad * 2
        height = size.y + pad

        y = top_left.y - height - pad
        if y < self._origin.y:
            y = top_left.y + pad

        # キャンバスからはみ出さないよう左右に寄せる
        x = min(top_left.x, self._canvas_max.x - width - 2)
        x = max(x, self._origin.x + 2)

        p0 = ImVec2(x, y)
        p1 = ImVec2(p0.x + width, p0.y + height)
        draw_list.add_rect_filled(p0, p1, imgui.IM_COL32(20, 20, 24, 220), 3.0)
        draw_list.add_rect(p0, p1, color, 3.0, 1.0)
        draw_list.add_text(ImVec2(p0.x + pad, p0.y + pad * 0.5), color, text)

    def _draw_box(
        self,
        draw_list: imgui.ImDrawList,
        box: BBox,
        color: int,
        thickness: float,
        handles: bool,
    ) -> None:
        p0 = self._to_screen(box.x, box.y)
        p1 = self._to_screen(box.x1, box.y1)
        draw_list.add_rect(p0, p1, color, 0.0, thickness)

        if not handles:
            return
        for cx, cy in _CORNERS:
            pos = self._to_screen(
                box.x + cx * box.w,
                box.y + cy * box.h,
            )
            draw_list.add_rect_filled(
                ImVec2(pos.x - HANDLE_RADIUS, pos.y - HANDLE_RADIUS),
                ImVec2(pos.x + HANDLE_RADIUS, pos.y + HANDLE_RADIUS),
                color,
            )

    # ------------------------------------------------------------------
    # 入力
    # ------------------------------------------------------------------
    def _handle_input(self, hovered: bool, active: bool) -> None:
        io = imgui.get_io()
        mouse = io.mouse_pos

        if hovered and io.mouse_wheel != 0.0:
            self._zoom_at(mouse, io.mouse_wheel)

        # パン（中ボタン or 右ボタン）
        if hovered and (
            imgui.is_mouse_clicked(imgui.MouseButton_.middle)
            or imgui.is_mouse_clicked(imgui.MouseButton_.right)
        ):
            self._drag = _Drag(
                mode="pan", start_pan=ImVec2(self.pan.x, self.pan.y), start_mouse=mouse
            )

        if self._drag is not None and self._drag.mode == "pan":
            if imgui.is_mouse_down(imgui.MouseButton_.middle) or imgui.is_mouse_down(
                imgui.MouseButton_.right
            ):
                dx = (mouse.x - self._drag.start_mouse.x) / self.zoom
                dy = (mouse.y - self._drag.start_mouse.y) / self.zoom
                self.pan = ImVec2(
                    self._drag.start_pan.x - dx, self._drag.start_pan.y - dy
                )
            else:
                self._drag = None
            return

        # BBox 操作（左ボタン）
        if active and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self._begin_left_drag(mouse)

        if self._drag is not None and self._drag.mode in ("new", "move", "resize"):
            if imgui.is_mouse_down(imgui.MouseButton_.left):
                self._update_left_drag(mouse)
            else:
                self._end_left_drag()

    def _zoom_at(self, mouse: ImVec2, wheel: float) -> None:
        before = self._to_image(mouse)
        self.zoom = float(np.clip(self.zoom * (1.15**wheel), MIN_ZOOM, MAX_ZOOM))
        after = self._to_image(mouse)
        self.pan = ImVec2(
            self.pan.x + (before.x - after.x), self.pan.y + (before.y - after.y)
        )

    def _begin_left_drag(self, mouse: ImVec2) -> None:
        img = self._to_image(mouse)

        if self.bbox is not None:
            corner = self._hit_handle(mouse)
            if corner >= 0:
                # 掴んだ角の対角を固定点にする
                ox, oy = _CORNERS[(corner + 2) % 4]
                self._drag = _Drag(
                    mode="resize",
                    anchor_img=ImVec2(
                        self.bbox.x + ox * self.bbox.w,
                        self.bbox.y + oy * self.bbox.h,
                    ),
                    corner=corner,
                )
                return

            if (
                self.bbox.x <= img.x <= self.bbox.x1
                and self.bbox.y <= img.y <= self.bbox.y1
            ):
                self._drag = _Drag(
                    mode="move",
                    grab_offset=ImVec2(img.x - self.bbox.x, img.y - self.bbox.y),
                )
                return

        self._drag = _Drag(mode="new", anchor_img=img)

    def _update_left_drag(self, mouse: ImVec2) -> None:
        img = self._to_image(mouse)
        drag = self._drag
        assert drag is not None

        if drag.mode == "move":
            assert self.bbox is not None and drag.grab_offset is not None
            self.bbox = BBox(
                img.x - drag.grab_offset.x,
                img.y - drag.grab_offset.y,
                self.bbox.w,
                self.bbox.h,
                BBoxSource.MANUAL,
            )
        else:  # "new" / "resize"
            assert drag.anchor_img is not None
            self.bbox = self._box_from_drag(drag.anchor_img, img)

        self._changed = True

    def _end_left_drag(self) -> None:
        if self.bbox is not None and not self.bbox.is_valid():
            self.bbox = None
            self._changed = True
        self._drag = None

    def _box_from_drag(self, anchor: ImVec2, current: ImVec2) -> BBox:
        dx = current.x - anchor.x
        dy = current.y - anchor.y

        if self.square_mode:
            side = max(abs(dx), abs(dy))
            dx = side if dx >= 0 else -side
            dy = side if dy >= 0 else -side

        return BBox(anchor.x, anchor.y, dx, dy, BBoxSource.MANUAL)

    def _hit_handle(self, mouse: ImVec2) -> int:
        if self.bbox is None:
            return -1
        for i, (cx, cy) in enumerate(_CORNERS):
            pos = self._to_screen(
                self.bbox.x + cx * self.bbox.w,
                self.bbox.y + cy * self.bbox.h,
            )
            if (
                abs(mouse.x - pos.x) <= HANDLE_RADIUS + 2
                and abs(mouse.y - pos.y) <= HANDLE_RADIUS + 2
            ):
                return i
        return -1

    # ------------------------------------------------------------------
    def clear_bbox(self) -> None:
        self.bbox = None
        self._changed = True
