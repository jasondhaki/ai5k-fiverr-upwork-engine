"""
Storage abstraction for original files.

Retaining original files is non-negotiable if grounding is real: the validator
re-reads the source span at generation time, which means re-reading the stored
file. This interface hides WHERE that file lives.

Locally, files are on disk under data/uploads and text is read straight back.
In the hosted version, LocalFileStore is swapped for an S3/R2-backed store -
a one-file change - because everything upstream depends only on the Protocol,
never on the filesystem.

Every parsed source produces TWO linked documents, not one: the untouched
original (the uploaded PDF, the raw API response, the pasted text as
submitted) and the extracted text a claim's indices are actually valid
against. They're linked rather than merged because they serve different
readers - `SourceSpan.document_id` must point at the text document (indices
are only meaningful against that exact string), while a human wants to see
the original page. A parser upgrade (a new Docling version, say) can then
reprocess the retained original without ever touching the spans that already
point at the old extracted text.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import NamedTuple, Protocol


class SourceDocument(NamedTuple):
    """The two linked ids produced by storing one parsed source.

    `text_id` is what SourceSpan.document_id must reference. `original_id`
    is retrievable via get_original_id(text_id) so the untouched source can
    always be re-derived - re-read at generation time, or shown to the user
    as "page 2 of your CV" - independent of how extraction happened to format it.
    """

    text_id: str
    original_id: str


class FileStore(Protocol):
    """The only storage contract the rest of the system knows about."""

    def put(self, content: bytes, suffix: str = "") -> str:
        """Store raw bytes, return a document_id."""
        ...

    def get_bytes(self, document_id: str) -> bytes:
        """Retrieve raw bytes by document_id."""
        ...

    def get_text(self, document_id: str) -> str:
        """
        Retrieve decoded text. This is what the span validator calls to
        re-read a source and confirm a substring still matches its indices.
        """
        ...

    def put_source(
        self, *, original: bytes, text: str, original_suffix: str = ""
    ) -> SourceDocument:
        """
        Store a parsed source's original bytes and its extracted text as two
        linked documents. Use `text_id` for SourceSpan.document_id; the
        original stays retrievable via get_original_id/get_bytes.
        """
        ...

    def get_original_id(self, text_document_id: str) -> str:
        """Given a text document's id, return the id of the original document
        it was extracted from. Raises if `text_document_id` was never stored
        via put_source (e.g. it's an original itself, not a text document)."""
        ...


class LocalFileStore:
    """Disk-backed implementation for local development and the sprint."""

    def __init__(self, root: str | Path = "data/uploads") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Kept in a subdirectory (rather than named like `{text_id}.json` next
        # to the content) so it never collides with _resolve()'s document_id*
        # glob over self.root.
        self._meta_dir = self.root / "_meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes, suffix: str = "") -> str:
        document_id = uuid.uuid4().hex
        path = self.root / f"{document_id}{suffix}"
        path.write_bytes(content)
        return document_id

    def _resolve(self, document_id: str) -> Path:
        matches = [p for p in self.root.glob(f"{document_id}*") if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"No stored file for document_id {document_id}")
        return matches[0]

    def get_bytes(self, document_id: str) -> bytes:
        return self._resolve(document_id).read_bytes()

    def get_text(self, document_id: str) -> str:
        return self._resolve(document_id).read_text(encoding="utf-8", errors="replace")

    def put_source(
        self, *, original: bytes, text: str, original_suffix: str = ""
    ) -> SourceDocument:
        original_id = self.put(original, suffix=original_suffix)
        text_id = self.put(text.encode("utf-8"), suffix=".txt")
        meta_path = self._meta_dir / f"{text_id}.json"
        meta_path.write_text(json.dumps({"original_id": original_id}), encoding="utf-8")
        return SourceDocument(text_id=text_id, original_id=original_id)

    def get_original_id(self, text_document_id: str) -> str:
        meta_path = self._meta_dir / f"{text_document_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No original linked to text document {text_document_id} - "
                "it wasn't stored via put_source()"
            )
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data["original_id"]


# The single instance the app imports. Swap this line (and only this line) for
# an S3FileStore when the system goes hosted.
file_store: FileStore = LocalFileStore()
