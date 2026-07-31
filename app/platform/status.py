"""
In-process run-status tracking for the async /analyze flow - a StatusStore
protocol, same spirit as FileStore (app/storage/store.py) and Repository
(app/storage/repository.py): callers depend on the interface, never a
specific backend, so this could move to Postgres later without touching
api.py or pipeline.py.

KNOWN LIMITATION: InProcessStatusStore lives only in this process's memory,
and the background task that updates it lives only in this process too.
If the process restarts mid-run, the status and the job that would have
finished it die together - which is at least coherent: a lost job doesn't
leave a status claiming it's still running forever, and there's no orphaned
row anywhere durable that anyone could be misled by. A Postgres-backed
store would need to actively decide what to do about a status row whose
owning process is gone (mark it errored? try to resume the run?) - a
problem this version doesn't have, by construction, because nothing
survives a restart on either side.
"""

from __future__ import annotations

import threading
from typing import Literal, Optional, Protocol

from pydantic import BaseModel

Stage = Literal[
    "queued",
    "parsing_cv",
    "fetching_repos",
    "extracting_claims",
    "assigning_tiers",
    "scoring",
    "generating",
    "done",
    "error",
]


class RunStatus(BaseModel):
    """Everything the progress page needs to render - JSON-serializable as-is."""

    run_id: str
    stage: Stage = "queued"
    detail: str = ""
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    claims_found: int = 0
    claims_provable: int = 0
    error: Optional[str] = None

    # Set by GroqClient's 429 retry loop (via status_reporting - see
    # extractor.py) to an absolute unix timestamp when a rate-limit wait is
    # expected to clear, and reset back to None once that retry succeeds.
    # Deliberately an absolute server-clock timestamp rather than "seconds
    # remaining" - api.py's /analyze/{run_id}/status handler recomputes the
    # remaining seconds fresh against its own clock on every single poll, so
    # a slow poll or a paused laptop tab never leaves a stale countdown; the
    # browser's clock is never consulted for this at all.
    rate_limit_resume_at: Optional[float] = None


class StatusStore(Protocol):
    """The only status-tracking contract the rest of the system knows about."""

    def create(self, run_id: str) -> None:
        """Register a new run at stage='queued', before its background task starts."""
        ...

    def update(self, run_id: str, **fields: object) -> None:
        """Merge `fields` into the run's current status. A run_id nobody
        ever create()'d (or one that's since been evicted) is a silent
        no-op, not an error - a stray late update from a run nobody's
        watching anymore isn't worth raising over."""
        ...

    def get(self, run_id: str) -> Optional[RunStatus]:
        ...


class InProcessStatusStore:
    """Plain dict behind a lock. See this module's docstring for the one
    thing that wouldn't survive moving to Postgres unchanged: a process
    restart, which this version doesn't need to reason about at all."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._statuses: dict[str, RunStatus] = {}

    def create(self, run_id: str) -> None:
        with self._lock:
            self._statuses[run_id] = RunStatus(run_id=run_id)

    def update(self, run_id: str, **fields: object) -> None:
        with self._lock:
            current = self._statuses.get(run_id)
            if current is None:
                return
            self._statuses[run_id] = current.model_copy(update=fields)

    def get(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            return self._statuses.get(run_id)


# The single instance api.py uses - mirrors app/storage/store.py's
# `file_store` singleton and app/storage/repository.py's `repository`.
status_store: StatusStore = InProcessStatusStore()
