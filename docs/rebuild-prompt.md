# 再構築用プロンプト（Copilot / 他のコーディングエージェント向け）

このリポジトリをゼロから、なるべく少ないやりとりで再現するためのプロンプト。

**使い方**: VS Code の Copilot Chat を **Agent モード**にして、空のディレクトリで
以下の「プロンプト本体」をそのまま貼る。

**なぜ長いのか**: 元の開発では 15 往復ほどかけて仕様を詰め、さらに実行して
初めて分かる不具合を 6 件踏んだ。その結論と落とし穴を全部前倒しで書いてあるので、
このまま渡せば同じ回り道をしない。特に「6. 既知の落とし穴」は削らないこと。

---

## プロンプト本体（ここから下をコピー）

````text
任意物体認証（Arbitrary Object Verification）の検証ツールを Python で実装してください。
仕様は下記で確定済みです。質問せずに最後まで実装しきってください。

# 1. 何を作るか

画像フォルダをフレーム列として読み込み、ユーザーが GUI 上で物体を BBox で囲むと、
特徴抽出モデルの embedding のコサイン類似度で 1:N 照合してスコアを出す GUI ツール。
「このモデルはこの 2 つを同一と判断するか」を対話的に確かめるのが目的。

# 2. 技術スタック（決定済み・変更しない）

- Python 3.12 / パッケージ管理は uv
- GUI: imgui-bundle（Dear ImGui + immvision + portable-file-dialogs）
- 数値計算: PyTorch。デバイスは cuda > mps > cpu の順に自動判定
- モデル: transformers 経由の DINOv2（facebook/dinov2-base）を既定とする
- 依存: imgui-bundle numpy pillow pyyaml torch torchvision transformers

# 3. 機能仕様

## 3.1 入力
- 画像フォルダを読み込む。フレームごとにサイズが違ってよい
- ファイル名は自然順ソート（frame_2 が frame_10 より前）
- 将来 mp4 に対応できるよう FrameSource プロトコルで抽象化する

## 3.2 ビューア
- ズーム（ホイール、カーソル位置中心）・パン（右/中ドラッグ）
- コマ送り / 自動再生（fps 指定）/ シークバー
- フレーム名・解像度・番号を表示

## 3.3 BBox
- 左ドラッグで描画。角のハンドルでリサイズ、内側ドラッグで移動
- デフォルトは正方形拘束。長方形（自由アスペクト）にトグルで切替
- BBox は source フィールド（manual / detector）を必ず持つ
  ※ detector は将来の自動検出用。今は manual のみ生成するが構造は用意する

## 3.4 前処理
1. BBox で切り出す
2. アスペクト比を維持して長辺を input_size に合わせる
3. 短辺を黒でパディングして正方形にする（レターボックス、中央寄せ）
4. mean / std で正規化
入力サイズと正規化パラメータはモデル側が申告する（ハードコードしない）。

## 3.5 ギャラリーと照合
- 登録画像を複数保持し、1:N 照合してスコア降順でランキング表示
- 1 エントリに複数枚（マルチビュー）登録可能。集約は max / mean を GUI で選択
- 1:1 は「エントリ 1 件の 1:N」として同じ機構で扱う
- embedding を L2 正規化してコサイン類似度
- 閾値を GUI で設定し、OK/NG を判定・色分け表示

## 3.6 embedding 種別
GUI で切替可能にする。モデルが対応するものだけ選択肢に出す。
- cls: CLS トークン
- patch_mean: 全パッチトークンの平均
- concat: 上記 2 つを連結
- output: モデル出力をそのまま（ViT 以外向け）

# 4. モデルは差し替え前提（この設計が本質）

DINOv2 固定にしないこと。GUI は FeatureExtractor インタフェースのみに依存し、
モデルの実体を知らない構造にする。

```python
class FeatureExtractor(Protocol):
    name: str
    preprocess: PreprocessConfig      # input_size, mean, std
    supported_embeddings: list[str]
    def embed(self, batch: np.ndarray, embedding_type: str) -> np.ndarray: ...
    @property
    def device(self) -> str: ...
```

形式ごとの「ローダ」をレジストリに登録する方式にする。
ローダを 1 つ追加するだけで新形式に対応でき、GUI 側は無変更で済むこと。

```python
@register_loader("torchscript")
class TorchScriptLoader:
    def load(self, entry: ModelEntry, device: str) -> FeatureExtractor: ...
```

同梱するローダ: huggingface / torch_hub / torchscript
利用可能なモデルは configs/models.yaml で管理し、GUI のドロップダウンに出す。

```yaml
models:
  - name: dinov2_base
    loader: huggingface
    args: { path: facebook/dinov2-base }
    preprocess:
      input_size: 224
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    embeddings: [cls, patch_mean, concat]
```

`loaders/` ディレクトリに置いた .py を起動時に import して、
ユーザー定義ローダを追加できるようにする。

# 5. 状態の可視化（ここを手を抜くと使い物にならない）

検証ツールなので、数字が出ていること自体より
「その数字がいつ・どの設定で計算されたか」が分かることが重要。

1. **再照合の判定を一箇所に集約する。**
   結果を左右する入力（フレーム番号 / BBox 座標 / モデル名 / embedding 種別 /
   集約方式 / ギャラリー構成）から指紋（signature）を作り、
   毎フレーム末尾で変化を検出したときだけ 1 回再計算する。
   個別のウィジェットのコールバックから照合を呼ぶと、必ず拾い漏れる。
   閾値は OK/NG の線引きを変えるだけなので指紋に含めない（再計算不要）。

2. **照合結果パネルの先頭に常時ステータスバナーを出す。**
   - 照合できない理由 + 次にすべきこと（モデル未読込 / 枠が未指定 / ギャラリーが空）
   - 「結果が古い（入力が変わりました）」
   - 「照合済み #5 23ms / frame 12 · cls · max / dinov2_base · 3件 4枚」
     通し番号・所要時間・使った設定を必ず出す

3. **BBox の真上に 1 位の名前とスコアを重ねて描く。**
   スコアを探しに視線を動かさなくて済むように。

4. **時間のかかる処理・失敗しうる処理には必ずログを入れる。**
   モデル読込は別スレッドで行い、開始・完了・失敗・所要秒数をコンソールに出す。
   GUI 側にも経過秒数を出す。AOV_LOG=DEBUG で照合 1 回ごとのログも出す。

# 6. 既知の落とし穴（実際に踏んだもの。必ず守ること）

## imgui-bundle / Dear ImGui 1.92
- `immvision.GlTexture()` は **OpenGL コンテキスト内でしか生成できない**。
  描画ループの外で作ると SIGSEGV。必ず遅延生成すること。
- `GlTexture.update_from_image()` は **書き込み可能な連続配列しか受け付けない**。
  PIL 由来の `np.asarray(...)` は read-only で、`np.ascontiguousarray()` では
  コピーされず read-only のまま。`np.array(x, copy=True, order="C")` を使う。
- `GlTexture.texture_id` は int を返すが、`imgui.image()` と
  `draw_list.add_image()` は `ImTextureRef` を要求する。
  `imgui.ImTextureRef(texture.texture_id)` で包むこと。
- `draw_list.add_rect(p_min, p_max, col, rounding, thickness, flags)` の順。
  thickness は 5 番目。flags と取り違えると線が消える。
- `imgui.push_id()` は str のみ。int を渡すと落ちる。
- ImGui 1.92 はグリフを動的ロードするので、**日本語フォントに範囲指定は不要**。
  `add_font_from_file_ttf(path, size, config)` だけでよい。
  macOS: `/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc`
  （TTC は `ImFontConfig.font_no` でインデックス指定）
- hello_imgui は既定でウィンドウタイトル由来の .ini を
  **カレントディレクトリに撒く**（日本語タイトルだと `______.ini` になる）。
  以下でユーザー設定フォルダに逃がすこと:
  ```python
  runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
  runner_params.ini_filename_use_app_window_title = False
  runner_params.ini_filename = "any_object_viewer/ui.ini"
  ```
- `imgui.get_time()` は描画ループ外で呼ぶと落ちる。
  ロジック層では `time.monotonic()` を使うこと（テストから呼べなくなる）。
- BBox の描画は immvision.image() ではなく自前で行う。
  `imgui.invisible_button()` で入力を取り、`ImDrawList` で描く。
  正方形拘束やハンドル操作を思い通りに制御するため。

## モデル読み込み
- `torch.hub` は github.com への接続が必要で、環境によっては通らない。
  **既定は huggingface ローダにする。** `~/.cache/huggingface` を先に見るので
  一度取得済みならオフラインでも動く（`local_files_only` を立てなくてよい）。
- `torch.hub.load()` は必ず `verbose=True`。False にすると
  350MB のダウンロードが完全に無言になる。
- `logging.getLogger("torch")` を INFO にしないこと。内部ログが流れ込む。WARNING に。
- HuggingFace の Dinov2Model: `last_hidden_state[:, 0]` が CLS、
  `[:, 1:]` がパッチトークン。
- torch.hub の DINOv2: `forward_features(x)` が dict を返し、
  `x_norm_clstoken` と `x_norm_patchtokens` が入っている。

# 7. ディレクトリ構成

```
pyproject.toml            [project.scripts] aov = "any_object_viewer.gui.app:main"
configs/models.yaml
loaders/                  ユーザー定義ローダ置き場
scripts/make_sample_frames.py
src/any_object_viewer/
├── core/                 GUI 非依存。imgui を import しないこと
│   ├── bbox.py           BBox（source フィールド付き）
│   ├── frames.py         FrameSource プロトコル + ImageFolderSource（LRU キャッシュ）
│   ├── preprocess.py     crop / letterbox / normalize
│   ├── scoring.py        cosine_similarity / max・mean 集約
│   └── gallery.py        Gallery / GalleryEntry / RegisteredImage / MatchResult
├── models/
│   ├── base.py           FeatureExtractor / ModelEntry / PreprocessConfig
│   ├── registry.py       ローダのレジストリ + YAML 管理
│   └── loaders/          huggingface.py / torch_hub.py / torchscript.py
└── gui/
    ├── app.py            アプリ本体
    ├── image_view.py     ズーム/パン + BBox 編集
    ├── textures.py       テクスチャキャッシュ
    └── fonts.py          日本語フォント
```

画面レイアウト: 左=ギャラリー(320px) / 中央=ビューア / 右=照合結果(400px) /
下=再生コントロールと設定。ギャラリーのサムネイル表示サイズはスライダーで
48〜200px に変更でき、保持解像度は 224px（拡大してもぼやけないように）。

# 8. 検証（実装後に必ず実行すること）

1. サンプルデータ生成スクリプトを作る。
   960x540 の 60 フレームに、色と形の違う 4 物体（赤丸・青四角・緑三角・黄丸）を
   別々の軌道で動かす。背景は弱いノイズテクスチャ。

2. ロジック層を GUI なしでテストする。
   BBox の負の幅の正規化 / 自然順ソート / レターボックスの黒帯 /
   コサイン類似度 / max・mean 集約。

3. **GUI を実際に起動して描画パスを通す。**
   コードレビューだけでは上記 6 の不具合は 1 つも見つからない。
   ヘッドレス検証スクリプトを作ること:
   - App を組み立て、フォルダ読込・ギャラリー登録・照合まで済ませる
   - `runner_params.callbacks.show_gui` で 40 フレーム描画したら
     `app_shall_exit = True`
   - gui() を try/except で囲み、例外を捕捉して非ゼロ終了
   - `hello_imgui.final_app_window_screenshot()` を PNG 保存
   - 「モデル未読込」「結果が古い」「照合済み」の各状態を撮り分ける

4. 実物の DINOv2 で 1:N 照合し、同一物体が別フレームでも 1 位になるか確認する。
   （注: DINOv2 は形状バイアスが強く、赤丸と黄丸のように「色だけ違う同形状」は
   混同することがある。これはモデルの性質でありバグではない）

# 9. 成果物

上記すべてを実装し、README.md と CLAUDE.md（または .github/copilot-instructions.md）に
落とし穴と設計上の約束を記録してください。
````

---

## 補足: 一度に通らなかった場合の分割案

エージェントが途中で力尽きたら、次の順で 3 回に分ける。
各回の冒頭に「6. 既知の落とし穴」を毎回貼り直すこと。

1. **土台** — uv セットアップ + `core/` + `models/` + サンプル生成 + ロジックのテスト
   （GUI なしで完結するので確実に通る）
2. **GUI** — `gui/` 一式 + ヘッドレス検証スクリプト + 起動確認
3. **仕上げ** — 状態バナー・ログ・サムネイルサイズ・README

## 補足: このプロンプトで**省いた**もの

意図的に落とした。必要なら追加で指示する。

- セッション保存（名前付き JSON + 自動保存）
- GUI からのモデル追加ダイアログ
- timm / onnx ローダ
- 検出モデルによる BBox 自動生成、動画ファイル入力、結果の CSV 出力
