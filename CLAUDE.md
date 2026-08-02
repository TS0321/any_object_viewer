# any_object_viewer

任意物体認証（Arbitrary Object Verification）の検証ツール。
画像群を眺めながら物体を BBox で囲み、特徴量のコサイン類似度で 1:N 照合する GUI。

**仕様は [docs/spec.md](docs/spec.md) が正。** 機能を足す前にそこを読むこと。

## コマンド

```bash
uv run aov                                          # GUI 起動
AOV_LOG=DEBUG uv run aov                            # 照合 1 回ごとのログも出す
uv run python scripts/make_sample_frames.py         # 動作確認用フレーム生成
```

## 構成

```
src/any_object_viewer/
├── core/           GUI 非依存のロジック（CLI / テストからも呼べる）
│   ├── bbox.py         BBox と由来（manual / detector）
│   ├── frames.py       FrameSource プロトコル + ImageFolderSource
│   ├── preprocess.py   切り出し -> レターボックス -> 正規化
│   ├── scoring.py      コサイン類似度、max / mean 集約
│   └── gallery.py      ギャラリー、エントリ、照合結果
├── models/         モデル層（差し替え前提）
│   ├── base.py         FeatureExtractor / ModelEntry / PreprocessConfig
│   ├── registry.py     ローダのレジストリ + YAML モデル一覧
│   └── loaders/        形式ごとのローダ
└── gui/            imgui-bundle
    ├── app.py          アプリ本体
    ├── image_view.py   ズーム/パン + BBox 描画
    ├── textures.py     サムネイル用テクスチャキャッシュ
    └── fonts.py        日本語フォント

configs/models.yaml     利用可能なモデル一覧
loaders/                ユーザー定義ローダの置き場（起動時に自動 import）
```

## 設計上の約束

- **GUI はモデルの実体を知らない。** `FeatureExtractor` インタフェース越しにのみ触る。
  入力サイズ・正規化パラメータもモデル側が申告する（`PreprocessConfig`）。
  DINOv2 前提のコードを GUI 層に書かないこと。
- **モデル形式の追加はローダを 1 つ足すだけで済ませる。** `@register_loader("名前")` を
  付けたクラスを `models/loaders/` か `loaders/` に置く。GUI 側は変更しない。
- **core は imgui に依存しない。** セッション保存や CLI 実行で再利用するため。
- **BBox は必ず `source` を持つ。** 手動描画とモデル検出を区別するのが検証の前提。
- **再照合の判定は `App._signature()` の一箇所に集約する。** 結果を左右する入力
  （フレーム / 枠 / モデル / embedding 種別 / 集約方式 / ギャラリー構成）を指紋にし、
  `gui()` の末尾で `needs_match()` が変化を検出して 1 回だけ再計算する。
  個別のウィジェットから `run_match()` を呼ぶと、拾い漏れと二重実行が生まれる。
  閾値は OK/NG の線引きを変えるだけなので指紋に含めない（再計算不要）。
- **「いま何が起きているか」を常に画面に出す。** 照合できない理由（`blocker()`）、
  結果が古いこと（`is_stale()`）、いつ何 ms でどの設定で計算したか（`MatchInfo`）。
  検証ツールなので、数字が出ていること自体より、その数字の出どころが分かることが重要。

## 実装上の注意

- **`immvision.GlTexture()` は OpenGL コンテキスト内でしか生成できない**（外だと SIGSEGV）。
  必ず描画ループ内で遅延生成する。
- モデル読み込みは重いのでワーカースレッドで行う（`ModelState.start_load`）。
  DINOv2 は torch.hub 経由で初回にダウンロードが走る。
- embedding は `(モデル名, embedding 種別)` をキーにキャッシュする。
  どちらかが変わったら再計算が必要（`Gallery.stale_images`）。
- Dear ImGui 1.92 はグリフを動的ロードするため、日本語フォントに範囲指定は不要。
- **既定モデルは `huggingface` ローダ経由の `dinov2_base`。** `~/.cache/huggingface`
  を先に見るのでオフラインでも動く。`torch_hub` ローダは github.com への接続が要る
  （この環境では通らないことがある）ため既定にしない。
- **`torch.hub.load` は必ず `verbose=True` で呼ぶ。** ダウンロード進捗（tqdm）が
  ここからしか出ず、`False` にすると 350MB の取得が完全に無言になる。
- **時間のかかる処理・失敗しうる処理には必ず `logger` を入れる。** GUI の表示だけでは
  「押したのに何も起きない」に見える。逆に torch のロガーを INFO にすると内部ログが
  流れ込むので `WARNING` に抑えること。

## 未実装（docs/spec.md 参照）

- セッション保存（3.12）
- GUI からのモデル追加（3.8.3）
- timm / huggingface / onnx ローダ（3.8.2）
- 検出モデルによる BBox 自動生成（7.1）
- 動画ファイル入力（7.2）、結果のエクスポート（7.3）
