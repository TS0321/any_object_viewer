# ユーザー定義ローダ

このディレクトリに `.py` を置くと、起動時に自動で import されます。
`@register_loader("名前")` を付けたクラスがレジストリに登録され、
`configs/models.yaml` の `loader:` から参照できるようになります。

```python
from any_object_viewer.models import register_loader, ModelEntry

@register_loader("my_format")
class MyLoader:
    def load(self, entry: ModelEntry, device: str):
        ...  # FeatureExtractor を返す
```
