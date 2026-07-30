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
import os
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


class B2FileStore:
    """
    Backblaze B2-backed implementation for the hosted demo. B2 is
    S3-compatible, so this is a thin wrapper over boto3's S3 client pointed
    at B2's S3-compatible endpoint - no B2-specific SDK needed, same as the
    extractor reusing Groq's OpenAI-compatible REST endpoint instead of a
    second LLM SDK.

    Conforms to the exact same FileStore protocol as LocalFileStore, so
    everything upstream (span validator re-reads, CV parser, etc.) works
    unchanged regardless of which backend STORAGE_BACKEND selected.

    Mirrors LocalFileStore's on-disk layout one-for-one: `put` writes one
    object per document_id(+suffix); `put_source`'s original/text link is a
    small JSON object under a `_meta/` key prefix, exactly like
    LocalFileStore's `_meta` subdirectory - not S3 object metadata headers,
    so both stores' persistence model (and this class's tests) stay
    identical in shape.
    """

    def __init__(
        self, *, key_id: str, application_key: str, endpoint: str, bucket_name: str
    ) -> None:
        import boto3  # local import: only the b2 backend needs boto3 installed

        self._bucket = bucket_name
        # endpoint is stored bare (e.g. "s3.us-east-005.backblazeb2.com") -
        # boto3's endpoint_url needs the scheme, prepended here rather than
        # in the env var so B2_ENDPOINT reads the same as B2's own console.
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{endpoint}",
            aws_access_key_id=key_id,
            aws_secret_access_key=application_key,
        )

    def put(self, content: bytes, suffix: str = "") -> str:
        document_id = uuid.uuid4().hex
        self._client.put_object(Bucket=self._bucket, Key=f"{document_id}{suffix}", Body=content)
        return document_id

    def _resolve_key(self, document_id: str) -> str:
        """S3 equivalent of LocalFileStore's glob-by-prefix `_resolve` - the
        suffix isn't known at retrieval time, so list by prefix instead of
        get_object-ing a guessed key."""
        response = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=document_id, MaxKeys=1
        )
        contents = response.get("Contents") or []
        if not contents:
            raise FileNotFoundError(f"No stored object for document_id {document_id}")
        return contents[0]["Key"]

    def get_bytes(self, document_id: str) -> bytes:
        key = self._resolve_key(document_id)
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def get_text(self, document_id: str) -> str:
        return self.get_bytes(document_id).decode("utf-8", errors="replace")

    def put_source(
        self, *, original: bytes, text: str, original_suffix: str = ""
    ) -> SourceDocument:
        original_id = self.put(original, suffix=original_suffix)
        text_id = self.put(text.encode("utf-8"), suffix=".txt")
        self._client.put_object(
            Bucket=self._bucket,
            Key=f"_meta/{text_id}.json",
            Body=json.dumps({"original_id": original_id}).encode("utf-8"),
        )
        return SourceDocument(text_id=text_id, original_id=original_id)

    def get_original_id(self, text_document_id: str) -> str:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=f"_meta/{text_document_id}.json")
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(
                f"No original linked to text document {text_document_id} - "
                "it wasn't stored via put_source()"
            ) from None
        data = json.loads(obj["Body"].read())
        return data["original_id"]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set when STORAGE_BACKEND=b2")
    return value


def _build_file_store() -> FileStore:
    """
    The one place a storage backend is chosen, mirroring build_default_client
    in app/ingestion/extractor.py. Defaults to 'local' so the fast test suite
    (and a fresh `uvicorn --reload`) never needs real B2 credentials - only
    STORAGE_BACKEND=b2, set explicitly (e.g. in render.yaml), switches it.
    """
    backend = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalFileStore()
    if backend == "b2":
        return B2FileStore(
            key_id=_require_env("B2_KEY_ID"),
            application_key=_require_env("B2_APPLICATION_KEY"),
            endpoint=_require_env("B2_ENDPOINT"),
            bucket_name=_require_env("B2_BUCKET_NAME"),
        )
    raise ValueError(f"Unknown STORAGE_BACKEND '{backend}' - expected 'local' or 'b2'.")


# The single instance the app imports. Backend picked by STORAGE_BACKEND
# (local|b2) - swap the env var, not this line, to go from disk to B2.
file_store: FileStore = _build_file_store()
