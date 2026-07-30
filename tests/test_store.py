"""
FileStore tests: every parsed source must produce two independently
retrievable, linked documents - the untouched original and the extracted
text - not just the text. A grounded claim's document_id points at the text;
get_original_id must always resolve back to the original from there.
"""

from __future__ import annotations

import pytest

from app.storage.store import LocalFileStore, SourceDocument


@pytest.fixture
def store(tmp_path) -> LocalFileStore:
    return LocalFileStore(root=tmp_path / "uploads")


def test_put_source_returns_two_distinct_linked_ids(store: LocalFileStore):
    pair = store.put_source(original=b"%PDF-fake-bytes", text="fake bytes as text")
    assert isinstance(pair, SourceDocument)
    assert pair.text_id != pair.original_id


def test_text_id_retrieves_the_extracted_text(store: LocalFileStore):
    pair = store.put_source(original=b"original raw content", text="extracted text")
    assert store.get_text(pair.text_id) == "extracted text"


def test_original_id_retrieves_the_untouched_original_bytes(store: LocalFileStore):
    original_bytes = b"\x25\x50\x44\x46-raw original bytes, not text"
    pair = store.put_source(original=original_bytes, text="whatever was extracted")
    assert store.get_bytes(pair.original_id) == original_bytes


def test_get_original_id_from_text_id_resolves_the_link(store: LocalFileStore):
    pair = store.put_source(original=b"original", text="text")
    assert store.get_original_id(pair.text_id) == pair.original_id


def test_get_original_id_raises_for_a_document_never_stored_via_put_source(
    store: LocalFileStore,
):
    lone_id = store.put(b"some bytes")
    with pytest.raises(FileNotFoundError):
        store.get_original_id(lone_id)


def test_put_source_preserves_original_suffix(store: LocalFileStore):
    pair = store.put_source(original=b"%PDF-fake", text="cv text", original_suffix=".pdf")
    resolved = store._resolve(pair.original_id)
    assert resolved.suffix == ".pdf"
