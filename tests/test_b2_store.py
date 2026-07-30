"""
B2FileStore tests. Fast-suite tests substitute a fake S3 client (matching
boto3's put_object/get_object/list_objects_v2 shape) so nothing here ever
makes a real network call or needs B2 credentials - exactly the same fake-the-
boundary pattern as the fake LLMClient in test_extractor.py/test_generation.py.

One `live_api`-marked test at the bottom does a real round trip against the
actual B2 bucket named in .env, run deliberately (see pyproject.toml's
`live_api` marker, broadened to cover any live external provider, not just
LLMs).
"""

from __future__ import annotations

import json

import pytest

from app.storage.store import B2FileStore, SourceDocument


class _NoSuchKey(Exception):
    pass


class _FakeExceptions:
    NoSuchKey = _NoSuchKey


class _FakeS3Client:
    """In-memory stand-in for boto3's S3 client, exposing only the methods
    B2FileStore actually calls."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self.exceptions = _FakeExceptions()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self._objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self._objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": _FakeBody(self._objects[Key])}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int = 1000) -> dict:
        matches = [key for key in self._objects if key.startswith(Prefix)]
        if not matches:
            return {}
        return {"Contents": [{"Key": key} for key in matches[:MaxKeys]]}


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


@pytest.fixture
def store(monkeypatch) -> B2FileStore:
    fake_client = _FakeS3Client()

    def _fake_boto3_client(*args, **kwargs):
        return fake_client

    import boto3

    monkeypatch.setattr(boto3, "client", _fake_boto3_client)

    return B2FileStore(
        key_id="fake-key-id",
        application_key="fake-app-key",
        endpoint="s3.us-east-005.backblazeb2.com",
        bucket_name="fake-bucket",
    )


def test_put_source_returns_two_distinct_linked_ids(store: B2FileStore):
    pair = store.put_source(original=b"%PDF-fake-bytes", text="fake bytes as text")
    assert isinstance(pair, SourceDocument)
    assert pair.text_id != pair.original_id


def test_text_id_retrieves_the_extracted_text(store: B2FileStore):
    pair = store.put_source(original=b"original raw content", text="extracted text")
    assert store.get_text(pair.text_id) == "extracted text"


def test_original_id_retrieves_the_untouched_original_bytes(store: B2FileStore):
    original_bytes = b"\x25\x50\x44\x46-raw original bytes, not text"
    pair = store.put_source(original=original_bytes, text="whatever was extracted")
    assert store.get_bytes(pair.original_id) == original_bytes


def test_get_original_id_from_text_id_resolves_the_link(store: B2FileStore):
    pair = store.put_source(original=b"original", text="text")
    assert store.get_original_id(pair.text_id) == pair.original_id


def test_get_original_id_raises_for_a_document_never_stored_via_put_source(
    store: B2FileStore,
):
    lone_id = store.put(b"some bytes")
    with pytest.raises(FileNotFoundError):
        store.get_original_id(lone_id)


def test_get_bytes_raises_for_an_unknown_document_id(store: B2FileStore):
    with pytest.raises(FileNotFoundError):
        store.get_bytes("never-stored")


def test_put_source_preserves_original_suffix(store: B2FileStore):
    pair = store.put_source(original=b"%PDF-fake", text="cv text", original_suffix=".pdf")
    key = store._resolve_key(pair.original_id)
    assert key.endswith(".pdf")


def test_endpoint_gets_https_scheme_prepended(monkeypatch):
    """B2_ENDPOINT is stored bare (no scheme) - boto3 needs the full URL."""
    captured = {}

    def _capture_client(service_name, **kwargs):
        captured.update(kwargs)
        return _FakeS3Client()

    import boto3

    monkeypatch.setattr(boto3, "client", _capture_client)

    B2FileStore(
        key_id="k",
        application_key="a",
        endpoint="s3.us-east-005.backblazeb2.com",
        bucket_name="b",
    )

    assert captured["endpoint_url"] == "https://s3.us-east-005.backblazeb2.com"


# --- live_api-marked: real Backblaze B2, run deliberately --------------------


@pytest.mark.live_api
def test_real_b2_round_trip():
    """Proves the actual B2 bucket in .env is reachable and read/write works -
    a real network call, run deliberately, never during routine iteration."""
    import os

    store = B2FileStore(
        key_id=os.environ["B2_KEY_ID"],
        application_key=os.environ["B2_APPLICATION_KEY"],
        endpoint=os.environ["B2_ENDPOINT"],
        bucket_name=os.environ["B2_BUCKET_NAME"],
    )
    pair = store.put_source(original=b"live b2 smoke test", text="live b2 smoke test text")
    assert store.get_text(pair.text_id) == "live b2 smoke test text"
    assert store.get_original_id(pair.text_id) == pair.original_id
