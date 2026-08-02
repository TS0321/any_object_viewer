"""任意物体認証 検証ツールのエントリポイント（最小構成）.

実装済み: 画像フォルダ読込 / フレーム送り・自動再生 / BBox 描画 /
          ギャラリー登録 / 1:N コサイン類似度 / 閾値による OK/NG
未実装:   セッション保存、モデルの GUI 追加、検出モデル（docs/spec.md 7）
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from imgui_bundle import ImVec2, ImVec4, hello_imgui, imgui, immapp, portable_file_dialogs

from ..core.bbox import BBox
from ..core.frames import ImageFolderSource
from ..core.gallery import Gallery, GalleryEntry, MatchResult, RegisteredImage
from ..core.preprocess import prepare, thumbnail
from ..core.scoring import Aggregation
from ..models import ModelRegistry, load_builtin_loaders, load_user_loaders, resolve_device
from .fonts import load_japanese_font
from .image_view import ImageView
from .textures import TextureCache, tex_ref

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"
USER_LOADERS_DIR = PROJECT_ROOT / "loaders"

THUMB_STORE = 224
"""サムネイルを保持する解像度。拡大表示してもぼやけないよう表示サイズより大きく持つ."""

THUMB_MIN, THUMB_MAX = 48.0, 200.0
GALLERY_W = 320.0
RESULTS_W = 400.0
COLOR_OK = ImVec4(0.35, 0.85, 0.45, 1.0)
COLOR_NG = ImVec4(0.9, 0.45, 0.40, 1.0)
COLOR_DIM = ImVec4(0.6, 0.6, 0.6, 1.0)
COLOR_WARN = ImVec4(0.95, 0.75, 0.30, 1.0)
COLOR_INFO = ImVec4(0.55, 0.75, 1.0, 1.0)


@dataclass
class MatchInfo:
    """いつ・どの設定で照合したかの記録。UI に出して状態を可視化する."""

    seq: int
    elapsed_ms: float
    at_time: float
    frame_index: int
    model_name: str
    embedding_type: str
    aggregation: str
    n_entries: int
    n_images: int


@dataclass
class ModelState:
    """モデルの読み込み状態。読み込みはワーカースレッドで行う."""

    extractor: object | None = None
    name: str = ""
    embedding_type: str = "cls"
    loading: bool = False
    loading_name: str = ""
    started_at: float = 0.0
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.extractor is not None

    @property
    def key(self) -> tuple:
        return (self.name, self.embedding_type)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.loading else 0.0

    def start_load(self, registry: ModelRegistry, name: str, device: str) -> None:
        if self.loading:
            logger.warning("すでに読み込み中です: %s", self.loading_name)
            return
        self.loading = True
        self.loading_name = name
        self.started_at = time.monotonic()
        self.error = ""

        entry = registry.get(name)
        logger.info("=" * 60)
        logger.info("モデル読込を開始: %s", name)
        logger.info("  loader=%s device=%s", entry.loader, device)
        logger.info("  args=%s", entry.args)
        logger.info("  input_size=%s embeddings=%s", entry.preprocess.input_size, entry.embeddings)

        def worker() -> None:
            try:
                extractor = registry.build(name, device)
                with self._lock:
                    self.extractor = extractor
                    self.name = name
                    if self.embedding_type not in entry.embeddings:
                        logger.info(
                            "embedding 種別を %s から %s に変更（このモデルが未対応のため）",
                            self.embedding_type, entry.embeddings[0],
                        )
                        self.embedding_type = entry.embeddings[0]
                logger.info(
                    "モデル読込 完了: %s（%.1f 秒, device=%s）",
                    name, time.monotonic() - self.started_at, device,
                )
            except Exception:
                with self._lock:
                    self.extractor = None
                    self.error = traceback.format_exc(limit=5)
                logger.error(
                    "モデル読込 失敗: %s（%.1f 秒）", name, time.monotonic() - self.started_at
                )
                logger.exception("原因:")
            finally:
                self.loading = False
                logger.info("=" * 60)

        threading.Thread(target=worker, daemon=True, name=f"load-{name}").start()


class App:
    def __init__(self) -> None:
        load_builtin_loaders()
        self.user_loaders = load_user_loaders(USER_LOADERS_DIR)

        self.registry = ModelRegistry(MODELS_YAML)
        self.device = resolve_device()
        self.model = ModelState()
        self.selected_model = (
            "dinov2_base"
            if "dinov2_base" in self.registry.names()
            else (self.registry.names()[0] if self.registry.names() else "")
        )

        self.source: ImageFolderSource | None = None
        self.frame_index = 0
        self.playing = False
        self.fps = 10.0
        self._last_advance = 0.0

        self.view = ImageView("main")
        self.textures = TextureCache()

        self.gallery = Gallery()
        self.selected_entry: GalleryEntry | None = None

        self.query_thumb: np.ndarray | None = None
        self.query_uid = 0
        self.results: list[MatchResult] = []
        self.match_info: MatchInfo | None = None
        self._match_seq = 0
        self._matched_signature: tuple | None = None

        self.thumb_size = 110.0
        """ギャラリーのサムネイル表示サイズ（GUI から変更可）."""

        self.threshold = 0.70
        self.aggregation = Aggregation.MAX
        self.live_match = True
        self.status = "画像フォルダを開いてください"
        self.status_color = COLOR_DIM

    # ------------------------------------------------------------------
    # データ
    # ------------------------------------------------------------------
    def open_folder(self, folder: str) -> None:
        try:
            self.source = ImageFolderSource(folder)
        except Exception as exc:
            logger.error("画像フォルダの読み込みに失敗: %s (%s)", folder, exc)
            self.set_status(f"読み込み失敗: {exc}", COLOR_NG)
            return
        self.frame_index = 0
        self.view.request_fit()
        self.view.clear_bbox()
        self.results = []
        self.query_thumb = None
        logger.info("画像フォルダを読み込み: %s（%d 枚）", self.source.folder, len(self.source))
        self.set_status(f"{len(self.source)} 枚を読み込みました", COLOR_INFO)

    def current_frame(self) -> np.ndarray | None:
        if self.source is None:
            return None
        return self.source.get(self.frame_index)

    # ------------------------------------------------------------------
    # 特徴抽出
    # ------------------------------------------------------------------
    def _embed(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        extractor = self.model.extractor
        assert extractor is not None
        config = extractor.preprocess
        batch = prepare(
            frame, bbox, config.input_size, config.mean, config.std
        )[np.newaxis]
        return extractor.embed(batch, self.model.embedding_type)[0]

    def refresh_gallery_embeddings(self) -> None:
        """モデルや embedding 種別が変わった登録画像を計算し直す."""
        if not self.model.ready or self.source is None:
            return
        stale = self.gallery.stale_images(self.model.key)
        for image in stale:
            frame = self._frame_for(image)
            if frame is None:
                continue
            image.embedding = self._embed(frame, image.bbox)
            image.embedding_key = self.model.key
        if stale:
            logger.info(
                "登録画像 %d 件の embedding を再計算（%s / %s）",
                len(stale), self.model.name, self.model.embedding_type,
            )
            self.set_status(f"登録画像 {len(stale)} 件を再計算しました", COLOR_INFO)

    def _frame_for(self, image: RegisteredImage) -> np.ndarray | None:
        if image.frame_index is not None and self.source is not None:
            return self.source.get(image.frame_index)
        return None

    # ------------------------------------------------------------------
    # 照合の状態
    # ------------------------------------------------------------------
    def set_status(self, message: str, color: ImVec4 = COLOR_DIM) -> None:
        self.status = message
        self.status_color = color

    def _signature(self) -> tuple:
        """照合結果を左右する入力すべての指紋。変わっていれば結果は古い."""
        bbox = self.view.bbox
        return (
            self.frame_index,
            None
            if bbox is None
            else (round(bbox.x, 1), round(bbox.y, 1), round(bbox.w, 1), round(bbox.h, 1)),
            self.model.name,
            self.model.embedding_type,
            self.aggregation,
            tuple((e.uid, len(e.images)) for e in self.gallery.entries),
        )

    def blocker(self) -> tuple[str, str, ImVec4] | None:
        """照合できない理由と、次にすべきこと。できるなら None."""
        # 直前に起きたことを優先して見せる（読込中・失敗）。
        # その後で、手順として足りないものを順に案内する。
        if self.model.loading:
            return (
                f"モデル読込中… {self.model.elapsed():.0f} 秒",
                "進捗はコンソールに出ます（初回はダウンロードあり）",
                COLOR_INFO,
            )
        if self.model.error:
            return ("モデル読込に失敗", "コンソールのログを確認してください", COLOR_NG)
        if self.source is None:
            return ("画像フォルダが未選択", "上部の「画像フォルダを開く」から選択", COLOR_WARN)
        if not self.model.ready:
            return ("モデル未読込", "上部の「モデル読込」を押す", COLOR_NG)
        if self.view.bbox is None or not self.view.bbox.is_valid():
            return ("照合枠が未指定", "画像上を左ドラッグして枠を描く", COLOR_WARN)
        if len(self.gallery) == 0:
            return ("ギャラリーが空", "左の「現在の枠を新規登録」で登録", COLOR_WARN)
        return None

    def needs_match(self) -> bool:
        """照合可能で、かつ前回の計算から入力が変わっているか."""
        if self.blocker() is not None:
            return False
        return self._matched_signature != self._signature()

    def is_stale(self) -> bool:
        """表示中の結果が今の入力と食い違っているか（表示用）."""
        return bool(self.results) and self._matched_signature != self._signature()

    def run_match(self) -> None:
        frame = self.current_frame()
        bbox = self.view.bbox
        if frame is None or bbox is None or not bbox.is_valid():
            self.results = []
            self.query_thumb = None
            self.match_info = None
            self._matched_signature = None
            return
        if not self.model.ready:
            self.set_status("モデル未読込のため照合できません", COLOR_NG)
            return

        started = time.perf_counter()
        self.refresh_gallery_embeddings()
        self.query_thumb = thumbnail(frame, bbox, THUMB_STORE)
        self.query_uid += 1
        query = self._embed(frame, bbox)
        self.results = self.gallery.match(query, self.aggregation)
        elapsed = (time.perf_counter() - started) * 1000

        self._match_seq += 1
        self.match_info = MatchInfo(
            seq=self._match_seq,
            elapsed_ms=elapsed,
            # imgui.get_time() は描画ループ外だと落ちるので使わない
            at_time=time.monotonic(),
            frame_index=self.frame_index,
            model_name=self.model.name,
            embedding_type=self.model.embedding_type,
            aggregation=self.aggregation.value,
            n_entries=len(self.results),
            n_images=sum(len(e.images) for e in self.gallery.entries),
        )
        self._matched_signature = self._signature()
        logger.debug(
            "照合 #%d: frame=%d %s/%s/%s %d件 %.1fms -> %s",
            self._match_seq, self.frame_index, self.model.name,
            self.model.embedding_type, self.aggregation.value,
            len(self.results), elapsed,
            f"{self.results[0].entry.name} {self.results[0].score:.4f}" if self.results else "該当なし",
        )

    def register_current(self, into: GalleryEntry | None = None) -> None:
        frame = self.current_frame()
        bbox = self.view.bbox
        if frame is None or bbox is None or not bbox.is_valid():
            self.set_status("登録する BBox がありません", COLOR_WARN)
            return

        image = RegisteredImage(
            bbox=BBox(bbox.x, bbox.y, bbox.w, bbox.h, bbox.source),
            frame_index=self.frame_index,
            thumbnail=thumbnail(frame, bbox, THUMB_STORE),
        )
        if self.model.ready:
            image.embedding = self._embed(frame, bbox)
            image.embedding_key = self.model.key

        if into is None:
            entry = GalleryEntry(name=self.gallery.unique_name(), images=[image])
            self.gallery.add(entry)
            self.selected_entry = entry
            logger.info(
                "ギャラリーに登録: %s（frame %d, bbox=%.0f,%.0f,%.0f,%.0f）",
                entry.name, self.frame_index, bbox.x, bbox.y, bbox.w, bbox.h,
            )
            self.set_status(f"{entry.name} を登録しました", COLOR_OK)
        else:
            into.images.append(image)
            logger.info(
                "登録画像を追加: %s（frame %d, 計 %d 枚）",
                into.name, self.frame_index, len(into.images),
            )
            self.set_status(f"{into.name} に画像を追加（{len(into.images)} 枚）", COLOR_OK)

        if self.live_match:
            self.run_match()

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------
    def gui(self) -> None:
        self._advance_playback()
        self._toolbar()
        imgui.separator()

        avail = imgui.get_content_region_avail()
        bottom_h = 92.0
        main_h = max(120.0, avail.y - bottom_h)

        if imgui.begin_child("##gallery", ImVec2(GALLERY_W, main_h), imgui.ChildFlags_.borders):
            self._gallery_panel()
        imgui.end_child()

        imgui.same_line()
        center_w = max(200.0, imgui.get_content_region_avail().x - RESULTS_W)
        if imgui.begin_child("##viewer", ImVec2(center_w, main_h), imgui.ChildFlags_.borders):
            self._viewer_panel()
        imgui.end_child()

        imgui.same_line()
        if imgui.begin_child("##results", ImVec2(0, main_h), imgui.ChildFlags_.borders):
            self._results_panel()
        imgui.end_child()

        if imgui.begin_child("##bottom", ImVec2(0, 0), imgui.ChildFlags_.borders):
            self._playback_panel()
            self._settings_panel()
        imgui.end_child()

        # 入力（フレーム / 枠 / モデル / embedding / 集約 / ギャラリー）が
        # 変わっていれば、ここで一度だけ再照合する。
        if self.live_match and self.needs_match():
            self.run_match()

    # -- toolbar --------------------------------------------------------
    def _toolbar(self) -> None:
        if imgui.button("画像フォルダを開く"):
            picked = portable_file_dialogs.select_folder("画像フォルダを選択").result()
            if picked:
                self.open_folder(picked)

        imgui.same_line()
        imgui.text_disabled(
            str(self.source.folder) if self.source else "（未選択）"
        )

        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 24)
        imgui.set_next_item_width(180)
        if imgui.begin_combo("##model", self.selected_model or "モデルなし"):
            for name in self.registry.names():
                clicked, _ = imgui.selectable(name, name == self.selected_model)
                if clicked:
                    self.selected_model = name
            imgui.end_combo()

        imgui.same_line()
        disabled = self.model.loading or not self.selected_model
        imgui.begin_disabled(disabled)
        if imgui.button("モデル読込"):
            self.model.start_load(self.registry, self.selected_model, self.device)
            self.set_status(f"{self.selected_model} を読み込み中…（初回はダウンロードあり）", COLOR_INFO)
        imgui.end_disabled()

        imgui.same_line()
        if self.model.loading:
            imgui.text_colored(
                COLOR_INFO,
                f"{self.model.loading_name} を読み込み中… {self.model.elapsed():.0f}s"
                "（進捗はコンソール）",
            )
        elif self.model.error:
            imgui.text_colored(COLOR_NG, "読み込み失敗（詳細はコンソール / ここにカーソル）")
            if imgui.is_item_hovered():
                imgui.set_tooltip(self.model.error)
        elif self.model.ready:
            imgui.text_colored(COLOR_OK, f"{self.model.name} @ {self.device}")
        else:
            imgui.text_colored(COLOR_DIM, f"未読込（device: {self.device}）")

    # -- gallery --------------------------------------------------------
    def _gallery_panel(self) -> None:
        imgui.text("ギャラリー")
        imgui.separator()

        if imgui.button("現在の枠を新規登録", ImVec2(-1, 0)):
            self.register_current()

        add_disabled = self.selected_entry is None
        imgui.begin_disabled(add_disabled)
        if imgui.button("選択エントリに追加", ImVec2(-1, 0)):
            self.register_current(into=self.selected_entry)
        imgui.end_disabled()

        imgui.text_disabled("表示サイズ")
        imgui.same_line()
        imgui.set_next_item_width(-1)
        _, self.thumb_size = imgui.slider_float(
            "##thumbsize", self.thumb_size, THUMB_MIN, THUMB_MAX, "%.0f px"
        )
        imgui.separator()

        size = ImVec2(self.thumb_size, self.thumb_size)
        # 1 行に何枚並ぶか。パネル幅から折り返し位置を決める。
        spacing = imgui.get_style().item_spacing.x
        per_row = max(1, int((GALLERY_W - 24) // (self.thumb_size + spacing)))

        for entry in list(self.gallery.entries):
            imgui.push_id(str(entry.uid))
            selected = entry is self.selected_entry
            if imgui.selectable(
                f"{entry.name}  ({len(entry.images)} 枚)##sel", selected
            )[0]:
                self.selected_entry = entry

            for i, image in enumerate(entry.images):
                if image.thumbnail is None:
                    continue
                if i % per_row != 0:
                    imgui.same_line()
                texture = self.textures.get(image.uid, image.thumbnail)
                imgui.image(tex_ref(texture), size)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"{entry.name}\n{image.origin_label()}\n"
                        f"bbox {image.bbox.w:.0f}x{image.bbox.h:.0f}"
                    )

            if entry.images:
                imgui.text_disabled(
                    "  ".join(i.origin_label() for i in entry.images[:per_row])
                    + ("  …" if len(entry.images) > per_row else "")
                )

            if imgui.small_button("削除"):
                for image in entry.images:
                    self.textures.drop(image.uid)
                self.gallery.remove(entry)
                if self.selected_entry is entry:
                    self.selected_entry = None
                self.results = [r for r in self.results if r.entry is not entry]
                logger.info("ギャラリーから削除: %s", entry.name)
            imgui.pop_id()
            imgui.separator()

        if not self.gallery.entries:
            imgui.text_disabled("未登録")

    # -- viewer ---------------------------------------------------------
    def _viewer_panel(self) -> None:
        frame = self.current_frame()
        if frame is None or self.source is None:
            imgui.text_disabled("画像フォルダを開いてください")
            return

        imgui.text(
            f"{self.source.name(self.frame_index)}   "
            f"{frame.shape[1]}x{frame.shape[0]}   "
            f"[{self.frame_index + 1}/{len(self.source)}]"
        )
        imgui.same_line()
        if imgui.small_button("全体表示"):
            self.view.request_fit()
        imgui.same_line()
        if imgui.small_button("枠をクリア"):
            self.view.clear_bbox()
            self.results = []
            self.query_thumb = None

        self.view.set_image(frame, key=self.frame_index)
        self.view.overlays = [
            (img.bbox, imgui.IM_COL32(90, 150, 255, 160))
            for entry in self.gallery.entries
            for img in entry.images
            if img.frame_index == self.frame_index
        ]

        # 枠の真上に 1 位の結果を出す。スコアを探しに行かなくて済むように。
        if self.results and not self.is_stale():
            top = self.results[0]
            ok = top.is_ok(self.threshold)
            self.view.bbox_label = f"{top.entry.name}  {top.score:.3f}  {'OK' if ok else 'NG'}"
            self.view.bbox_label_color = (
                imgui.IM_COL32(90, 220, 120, 255) if ok else imgui.IM_COL32(230, 115, 100, 255)
            )
        elif self.results:
            self.view.bbox_label = "要再照合"
            self.view.bbox_label_color = imgui.IM_COL32(240, 190, 75, 255)
        else:
            self.view.bbox_label = ""

        # 再照合の要否は gui() の末尾で一括判定する（フレーム送りや設定変更も拾うため）
        self.view.draw()

    # -- results --------------------------------------------------------
    def _banner(self, title: str, details: list[str], color: ImVec4) -> None:
        """パネル上部に出す状態表示。今なぜこうなっているかを常に示す."""
        draw_list = imgui.get_window_draw_list()
        p0 = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        line = imgui.get_text_line_height_with_spacing()
        height = line * (1 + len(details)) + 12
        p1 = ImVec2(p0.x + width, p0.y + height)

        fill = imgui.IM_COL32(int(color.x * 60), int(color.y * 60), int(color.z * 60), 255)
        edge = imgui.color_convert_float4_to_u32(color)
        draw_list.add_rect_filled(p0, p1, fill, 4.0)
        draw_list.add_rect(p0, p1, edge, 4.0, 1.5)
        draw_list.add_rect_filled(p0, ImVec2(p0.x + 4, p1.y), edge, 4.0)

        imgui.dummy(ImVec2(width, 2))
        imgui.indent(12)
        imgui.text_colored(color, title)
        for detail in details:
            imgui.text_disabled(detail)
        imgui.unindent(12)
        imgui.dummy(ImVec2(width, 2))

    def _score_bar(self, score: float, ok: bool) -> None:
        """スコアを 0..1 の横棒で描き、閾値の位置に目盛りを立てる."""
        width = max(40.0, imgui.get_content_region_avail().x - 14)
        height = 10.0
        p0 = imgui.get_cursor_screen_pos()
        p1 = ImVec2(p0.x + width, p0.y + height)
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(p0, p1, imgui.IM_COL32(55, 55, 62, 255), 2.0)

        fraction = float(np.clip(score, 0.0, 1.0))
        if fraction > 0:
            draw_list.add_rect_filled(
                p0,
                ImVec2(p0.x + width * fraction, p1.y),
                imgui.color_convert_float4_to_u32(COLOR_OK if ok else COLOR_NG),
                2.0,
            )
        tick = p0.x + width * float(np.clip(self.threshold, 0.0, 1.0))
        draw_list.add_line(
            ImVec2(tick, p0.y - 2), ImVec2(tick, p1.y + 2),
            imgui.IM_COL32(240, 240, 240, 200), 1.5,
        )
        imgui.dummy(ImVec2(width, height))

    def _results_panel(self) -> None:
        imgui.text("照合結果")
        imgui.separator()

        blocker = self.blocker()
        if blocker is not None:
            title, detail, color = blocker
            self._banner(title, [detail], color)
        elif self.is_stale():
            self._banner(
                "結果が古い（入力が変わりました）",
                ["「照合実行」を押すか「即時照合」を ON に"],
                COLOR_WARN,
            )
        elif self.match_info is not None:
            info = self.match_info
            self._banner(
                f"照合済み  #{info.seq}   {info.elapsed_ms:.0f} ms",
                [
                    f"frame {info.frame_index} · {info.embedding_type} · {info.aggregation}",
                    f"{info.model_name} · {info.n_entries}件 {info.n_images}枚",
                ],
                COLOR_OK,
            )
        else:
            self._banner("待機中", ["枠を描くと照合されます"], COLOR_DIM)

        imgui.separator()

        # 照合画像（クエリ）
        if self.query_thumb is not None:
            texture = self.textures.get(("query", self.query_uid), self.query_thumb)
            imgui.image(tex_ref(texture), ImVec2(self.thumb_size, self.thumb_size))
            imgui.same_line()
            imgui.begin_group()
            imgui.text("照合画像 (Query)")
            if self.match_info is not None:
                imgui.text_disabled(f"frame {self.match_info.frame_index}")
            imgui.text_disabled(f"vs ギャラリー {len(self.gallery)} 件")
            imgui.end_group()
        else:
            imgui.text_disabled("照合画像なし")

        imgui.separator()

        if not self.results:
            imgui.text_disabled("結果なし")
            return

        # 1 位を大きく
        top = self.results[0]
        top_ok = top.is_ok(self.threshold)
        imgui.text_disabled("最も近い登録")
        imgui.text_colored(COLOR_OK if top_ok else COLOR_NG, f"{top.entry.name}")
        imgui.same_line()
        imgui.text_colored(
            COLOR_OK if top_ok else COLOR_NG,
            f"  {top.score:.4f}  {'OK' if top_ok else 'NG'}",
        )
        imgui.separator()

        crossed = False
        for rank, result in enumerate(self.results, start=1):
            if not crossed and result.score < self.threshold:
                imgui.text_disabled(f"──────── 閾値 {self.threshold:.2f} ────────")
                crossed = True

            ok = result.is_ok(self.threshold)
            color = COLOR_OK if ok else COLOR_NG
            imgui.push_id(f"res{result.entry.uid}")

            first = next((i for i in result.entry.images if i.thumbnail is not None), None)
            if first is not None:
                row = max(40.0, self.thumb_size * 0.55)
                imgui.image(tex_ref(self.textures.get(first.uid, first.thumbnail)),
                            ImVec2(row, row))
                imgui.same_line()

            imgui.begin_group()
            imgui.text_colored(color, f"{rank}. {result.entry.name}")
            imgui.same_line()
            imgui.text_colored(color, f"  {result.score:.4f}  {'OK' if ok else 'NG'}")
            self._score_bar(result.score, ok)
            imgui.end_group()

            if len(result.per_image) > 1:
                detail = ", ".join(f"{s:.3f}" for s in result.per_image)
                imgui.text_disabled(f"    {self.aggregation.value} ← {detail}")
            imgui.pop_id()

    # -- playback -------------------------------------------------------
    def _advance_playback(self) -> None:
        if not self.playing or self.source is None:
            return
        now = imgui.get_time()
        if now - self._last_advance < 1.0 / max(self.fps, 0.1):
            return
        self._last_advance = now
        self.frame_index = (self.frame_index + 1) % len(self.source)
        # 再照合は gui() 末尾の needs_match() が拾う

    def _playback_panel(self) -> None:
        n = len(self.source) if self.source else 0
        imgui.begin_disabled(n == 0)

        if imgui.button("|<"):
            self.frame_index = 0
        imgui.same_line()
        if imgui.button("<"):
            self.frame_index = max(0, self.frame_index - 1)
        imgui.same_line()
        if imgui.button("一時停止" if self.playing else "再生"):
            self.playing = not self.playing
            self._last_advance = imgui.get_time()
        imgui.same_line()
        if imgui.button(">"):
            self.frame_index = min(n - 1, self.frame_index + 1) if n else 0
        imgui.same_line()
        if imgui.button(">|"):
            self.frame_index = max(0, n - 1)

        imgui.same_line()
        imgui.set_next_item_width(-220)
        changed, value = imgui.slider_int(
            "##seek", self.frame_index, 0, max(0, n - 1)
        )
        if changed:
            self.frame_index = value

        imgui.same_line()
        imgui.set_next_item_width(120)
        _, self.fps = imgui.slider_float("fps", self.fps, 1.0, 60.0, "%.0f")

        imgui.end_disabled()

    # -- settings -------------------------------------------------------
    def _settings_panel(self) -> None:
        if imgui.radio_button("正方形", self.view.square_mode):
            self.view.square_mode = True
        imgui.same_line()
        if imgui.radio_button("長方形", not self.view.square_mode):
            self.view.square_mode = False

        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 16)
        imgui.set_next_item_width(140)
        entry = (
            self.registry.get(self.model.name)
            if self.model.name in self.registry.names()
            else None
        )
        options = entry.embeddings if entry else ["cls"]
        if imgui.begin_combo("embedding", self.model.embedding_type):
            for option in options:
                clicked, _ = imgui.selectable(option, option == self.model.embedding_type)
                if clicked:
                    self.model.embedding_type = option
            imgui.end_combo()

        imgui.same_line()
        imgui.set_next_item_width(120)
        if imgui.begin_combo("集約", self.aggregation.value):
            for method in Aggregation:
                clicked, _ = imgui.selectable(method.value, method is self.aggregation)
                if clicked:
                    self.aggregation = method
            imgui.end_combo()

        imgui.same_line()
        imgui.set_next_item_width(180)
        # 閾値は OK/NG の線引きを変えるだけで、再計算は不要（signature に含めない）
        _, self.threshold = imgui.slider_float("閾値", self.threshold, -1.0, 1.0, "%.2f")

        imgui.same_line()
        _, self.live_match = imgui.checkbox("即時照合", self.live_match)
        imgui.same_line()
        stale = self.is_stale() or (self.blocker() is None and self._matched_signature is None)
        imgui.begin_disabled(self.live_match or not stale)
        if imgui.button("照合実行" + ("  ●" if stale and not self.live_match else "")):
            self.run_match()
        imgui.end_disabled()

        imgui.text_colored(self.status_color, self.status)


def setup_logging() -> None:
    """コンソールにログを出す。AOV_LOG=DEBUG で照合ごとのログも出る."""
    level = os.environ.get("AOV_LOG", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # torch.hub のダウンロード進捗は tqdm が stderr に直接書くので、
    # torch のロガーは触らない（INFO にすると内部ログまで流れてくる）
    logging.getLogger("torch").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    logger.info("任意物体認証 検証ツール")

    app = App()
    logger.info("device=%s", app.device)
    logger.info("モデルレジストリ: %s", MODELS_YAML)
    logger.info("  登録モデル: %s", ", ".join(app.registry.names()) or "(なし)")
    if app.user_loaders:
        logger.info("  ユーザー定義ローダ: %s", ", ".join(app.user_loaders))
    logger.info("ログレベルを上げるには AOV_LOG=DEBUG uv run aov")

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "任意物体認証 検証ツール"

    # ウィンドウ位置やパネル幅の保存先。既定はカレントディレクトリに
    # ウィンドウタイトル由来の .ini を撒くので、ユーザー設定フォルダに逃がす。
    runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    runner_params.ini_filename_use_app_window_title = False
    runner_params.ini_filename = "any_object_viewer/ui.ini"
    runner_params.app_window_params.window_geometry.size = (1600, 950)
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    )
    runner_params.imgui_window_params.enable_viewports = False
    runner_params.callbacks.load_additional_fonts = lambda: load_japanese_font(17.0)
    runner_params.callbacks.show_gui = app.gui
    runner_params.fps_idling.fps_idle = 12.0

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
