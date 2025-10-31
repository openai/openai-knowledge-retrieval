# Vector Store Plugins

Implement your own vector store by providing a class that implements the `VectorStore` protocol:

```python
from stores.base import VectorStore, Chunk, ScoredChunk

class MyVectorStore(VectorStore):
  def __init__(self, **kwargs):
    ...
  def upsert(self, chunks: list[Chunk]) -> None:
    ...
  def delete(self, doc_ids: list[str]) -> None:
    ...
  def search(self, query: str, k: int, filters: dict | None = None) -> list[ScoredChunk]:
    ...
  def stats(self) -> dict:
    ...
```

Point your YAML to it:

```yaml
vector_store:
  backend: custom
  custom:
    kind: plugin
    plugin:
      module_path: "./extensions/my_store.py"
      class_name: "MyVectorStore"
      init_args: {}
```