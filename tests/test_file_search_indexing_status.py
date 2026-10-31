from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from cli.config import RootConfig
from stores.openai_file_search import OpenAIFileSearchStore


@pytest.fixture
def ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[OpenAIFileSearchStore, Mock, Path]:
    path = tmp_path / "document.txt"
    path.write_text("Example document", encoding="utf-8")
    client = Mock()
    client.files.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="file-test", filename=path.name, bytes=path.stat().st_size)],
        has_more=False,
    )
    client.vector_stores.files.list.return_value = SimpleNamespace(data=[], has_more=False)
    client.vector_stores.files.create.return_value = SimpleNamespace(id="file-test")
    monkeypatch.setattr("stores.openai_file_search.build_openai_client", lambda cfg: client)
    monkeypatch.setattr("stores.openai_file_search.time.sleep", lambda seconds: None)
    monkeypatch.setenv("RAG_SKIP_PREFLIGHT", "false")
    monkeypatch.setenv("RAG_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("RAG_INGEST_WORKERS", "1")
    cfg = RootConfig(
        project="test",
        data={"paths": [str(tmp_path)]},
        vector_store={
            "backend": "openai_file_search",
            "openai_file_search": {"vector_store_name": "test"},
        },
    )
    store = OpenAIFileSearchStore(cfg)
    store.set_vector_store_id("vs-test")
    return store, client, path


def existing(client: Mock, status: str | None) -> None:
    record = SimpleNamespace(id="file-test")
    if status is not None:
        record.status = status
    client.vector_stores.files.list.return_value = SimpleNamespace(data=[record], has_more=False)


@pytest.mark.parametrize("status", ["completed", "processed", "succeeded"])
def test_completed_attachment_is_reused_without_upload_or_poll(ingestion, status: str) -> None:
    store, client, path = ingestion
    existing(client, status)
    store.upsert_paths([str(path)])
    client.files.create.assert_not_called()
    client.vector_stores.files.create.assert_not_called()
    client.vector_stores.files.retrieve.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "cancelled", "errored"])
def test_terminal_attachment_makes_ingestion_fail(
    ingestion, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    store, client, path = ingestion
    existing(client, status)
    with pytest.raises(RuntimeError, match="vector_store=1"):
        store.upsert_paths([str(path)])
    assert f"status={status}" in capsys.readouterr().out
    client.files.create.assert_not_called()
    client.vector_stores.files.create.assert_not_called()
    client.vector_stores.files.retrieve.assert_not_called()


@pytest.mark.parametrize("status", ["in_progress", None, "unknown"])
def test_pending_or_unrecognized_attachment_is_verified_before_success(
    ingestion, status: str | None
) -> None:
    store, client, path = ingestion
    existing(client, status)
    client.vector_stores.files.retrieve.side_effect = [
        SimpleNamespace(status="in_progress"),
        SimpleNamespace(status="completed"),
    ]
    store.upsert_paths([str(path)])
    assert (
        client.vector_stores.files.retrieve.call_args_list
        == [call(vector_store_id="vs-test", file_id="file-test")] * 2
    )
    client.vector_stores.files.create.assert_not_called()
    client.files.create.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_pending_attachment_that_fails_does_not_report_success(ingestion, status: str) -> None:
    store, client, path = ingestion
    existing(client, "in_progress")
    client.vector_stores.files.retrieve.return_value = SimpleNamespace(status=status)
    with pytest.raises(RuntimeError, match="vector_store=1"):
        store.upsert_paths([str(path)])
    client.vector_stores.files.retrieve.assert_called_once_with(
        vector_store_id="vs-test", file_id="file-test"
    )
    client.vector_stores.files.create.assert_not_called()


def test_pending_attachment_polling_is_bounded(ingestion) -> None:
    store, client, path = ingestion
    existing(client, "in_progress")
    client.vector_stores.files.retrieve.return_value = SimpleNamespace(status="in_progress")
    with pytest.raises(RuntimeError, match="vector_store=1"):
        store.upsert_paths([str(path)])
    assert client.vector_stores.files.retrieve.call_count == 10
    client.vector_stores.files.create.assert_not_called()


def test_new_attachment_still_waits_for_indexing(ingestion) -> None:
    store, client, path = ingestion
    client.vector_stores.files.retrieve.return_value = SimpleNamespace(status="completed")
    store.upsert_paths([str(path)])
    client.vector_stores.files.create.assert_called_once_with(
        vector_store_id="vs-test", file_id="file-test"
    )
    client.vector_stores.files.retrieve.assert_called_once_with(
        vector_store_id="vs-test", file_id="file-test"
    )


def test_attachment_statuses_are_collected_across_pages(ingestion) -> None:
    store, client, path = ingestion
    client.vector_stores.files.list.side_effect = [
        SimpleNamespace(data=[SimpleNamespace(id="other", status="completed")], has_more=True),
        SimpleNamespace(data=[SimpleNamespace(id="file-test", status="failed")], has_more=False),
    ]
    with pytest.raises(RuntimeError, match="vector_store=1"):
        store.upsert_paths([str(path)])
    assert client.vector_stores.files.list.call_args_list == [
        call(limit=100, vector_store_id="vs-test"),
        call(limit=100, vector_store_id="vs-test", after="other"),
    ]
    client.vector_stores.files.create.assert_not_called()
