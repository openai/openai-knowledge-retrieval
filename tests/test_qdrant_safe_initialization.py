from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from cli.config import RootConfig
from stores.custom_qdrant import QdrantStore


def config() -> RootConfig:
    return RootConfig(
        project="test",
        data={"paths": []},
        embeddings={"model": "text-embedding-3-small"},
        vector_store={
            "backend": "custom",
            "custom": {"kind": "qdrant", "qdrant": {"collection": "knowledge"}},
        },
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=VectorParams(size=1536, distance=Distance.COSINE))
        ),
        points_count=7,
    )
    monkeypatch.setattr("stores.custom_qdrant.QdrantClient", lambda **kwargs: client)
    return client


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        PermissionError("denied"),
        UnexpectedResponse(503, "unavailable", b"unavailable", {}),
    ],
)
def test_existence_errors_propagate_without_mutating_collections(
    client: Mock, error: Exception
) -> None:
    client.collection_exists.side_effect = error
    with pytest.raises(type(error)) as caught:
        QdrantStore(config())
    assert caught.value is error
    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()
    client.delete_collection.assert_not_called()


def test_existing_collection_is_reused_without_creation(client: Mock) -> None:
    QdrantStore(config())
    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()


def test_missing_collection_is_created_non_destructively(client: Mock) -> None:
    client.collection_exists.return_value = False
    QdrantStore(config())
    client.create_collection.assert_called_once_with(
        collection_name="knowledge",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    client.recreate_collection.assert_not_called()


def test_http_creation_conflict_validates_the_winning_collection(client: Mock) -> None:
    client.collection_exists.return_value = False
    client.create_collection.side_effect = UnexpectedResponse(409, "conflict", b"exists", {})
    QdrantStore(config())
    client.get_collection.assert_called_once_with("knowledge")


class CreationError(grpc.RpcError):
    def __init__(self, status: grpc.StatusCode) -> None:
        self.status = status

    def code(self) -> grpc.StatusCode:
        return self.status


def test_grpc_creation_conflict_validates_the_winning_collection(client: Mock) -> None:
    client.collection_exists.return_value = False
    client.create_collection.side_effect = CreationError(grpc.StatusCode.ALREADY_EXISTS)
    QdrantStore(config())
    client.get_collection.assert_called_once_with("knowledge")


@pytest.mark.parametrize(
    "error",
    [
        UnexpectedResponse(403, "forbidden", b"denied", {}),
        CreationError(grpc.StatusCode.UNAVAILABLE),
        TimeoutError("timeout"),
    ],
)
def test_non_conflict_creation_errors_are_propagated(client: Mock, error: Exception) -> None:
    client.collection_exists.return_value = False
    client.create_collection.side_effect = error
    with pytest.raises(type(error)) as caught:
        QdrantStore(config())
    assert caught.value is error
    client.get_collection.assert_not_called()
    client.recreate_collection.assert_not_called()


@pytest.mark.parametrize(
    "vectors",
    [
        VectorParams(size=3072, distance=Distance.COSINE),
        VectorParams(size=1536, distance=Distance.DOT),
        {"named": VectorParams(size=1536, distance=Distance.COSINE)},
        None,
    ],
)
def test_incompatible_collection_is_rejected_without_mutation(
    client: Mock, vectors: VectorParams | dict[str, VectorParams] | None
) -> None:
    client.get_collection.return_value.config.params.vectors = vectors
    with pytest.raises((ValueError, TypeError)):
        QdrantStore(config())
    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()


def test_existing_points_survive_reinitialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with closing(QdrantClient(path=str(tmp_path / "qdrant"))) as local:
        local.create_collection(
            "knowledge", vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        point = PointStruct(id=1, vector=[1.0] + [0.0] * 1535, payload={"text": "keep me"})
        local.upsert("knowledge", [point], wait=True)
        monkeypatch.setattr("stores.custom_qdrant.QdrantClient", lambda **kwargs: local)
        QdrantStore(config())
        QdrantStore(config())
        records = local.retrieve("knowledge", [1], with_payload=True, with_vectors=True)
        assert len(records) == 1
        assert records[0].id == point.id
        assert records[0].payload == point.payload
        assert records[0].vector == point.vector


@pytest.mark.parametrize("lookup", ["collection_exists", "get_collection"])
def test_lookup_failure_preserves_existing_local_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookup: str
) -> None:
    with closing(QdrantClient(path=str(tmp_path / "qdrant"))) as local:
        local.create_collection(
            "knowledge", vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        point = PointStruct(id=1, vector=[1.0] + [0.0] * 1535, payload={"text": "preserved"})
        local.upsert("knowledge", [point], wait=True)
        monkeypatch.setattr("stores.custom_qdrant.QdrantClient", lambda **kwargs: local)
        with monkeypatch.context() as failure:
            failure.setattr(local, lookup, Mock(side_effect=TimeoutError("metadata timeout")))
            with pytest.raises(TimeoutError):
                QdrantStore(config())
        assert local.count("knowledge", exact=True).count == 1
        records = local.retrieve("knowledge", [1], with_vectors=True, with_payload=True)
        assert records[0].payload == point.payload
        assert records[0].vector == point.vector


def test_concurrently_created_incompatible_collection_is_rejected(client: Mock) -> None:
    client.collection_exists.return_value = False
    client.create_collection.side_effect = UnexpectedResponse(409, "conflict", b"exists", {})
    client.get_collection.return_value.config.params.vectors = VectorParams(
        size=3072, distance=Distance.COSINE
    )
    with pytest.raises(ValueError, match="incompatible vector configuration"):
        QdrantStore(config())
    client.recreate_collection.assert_not_called()
    client.delete_collection.assert_not_called()
