"""ローダのレジストリと、YAML によるモデル一覧の管理.

ローダを追加したいときは `@register_loader("名前")` を付けたクラスを
`loaders/` 以下に置くだけでよい。GUI 側の変更は要らない。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import yaml

from .base import ModelEntry, ModelLoader

_LOADERS: dict[str, ModelLoader] = {}


def register_loader(name: str) -> Callable[[type], type]:
    """ローダクラスをレジストリに登録するデコレータ."""

    def decorator(cls: type) -> type:
        _LOADERS[name] = cls()
        return cls

    return decorator


def available_loaders() -> list[str]:
    return sorted(_LOADERS)


def get_loader(name: str) -> ModelLoader:
    if name not in _LOADERS:
        raise KeyError(
            f"未知のローダ '{name}'。利用可能: {', '.join(available_loaders())}"
        )
    return _LOADERS[name]


def load_builtin_loaders() -> None:
    """同梱ローダを import して登録させる."""
    package = f"{__package__}.loaders"
    loaders_dir = Path(__file__).parent / "loaders"
    for path in sorted(loaders_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        importlib.import_module(f"{package}.{path.stem}")


def load_user_loaders(directory: str | Path) -> list[str]:
    """ユーザー定義ローダを読み込む.

    指定ディレクトリの .py を import する。import 時に register_loader が
    走ることで登録される。読み込んだファイル名のリストを返す。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    loaded: list[str] = []
    for path in sorted(directory.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module_name = f"aov_user_loader_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded.append(path.name)
    return loaded


class ModelRegistry:
    """YAML で管理されるモデル一覧."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, ModelEntry] = {}
        if self.path.exists():
            self.reload()

    def reload(self) -> None:
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.entries = {
            e["name"]: ModelEntry.from_dict(e) for e in data.get("models", [])
        }

    def names(self) -> list[str]:
        return list(self.entries)

    def get(self, name: str) -> ModelEntry:
        return self.entries[name]

    def add(self, entry: ModelEntry, save: bool = True) -> None:
        self.entries[entry.name] = entry
        if save:
            self.save()

    def remove(self, name: str, save: bool = True) -> None:
        self.entries.pop(name, None)
        if save:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"models": [e.to_dict() for e in self.entries.values()]}
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def build(self, name: str, device: str):
        """エントリ名からモデルを実体化する."""
        entry = self.get(name)
        return get_loader(entry.loader).load(entry, device)
