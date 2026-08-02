"""
Database engine/session setup, read from DATABASE_URL (app/config/env.py
loads .env before this module is ever imported).

`create_engine` is lazy - constructing it here at import time opens no
connection and makes no network call, even when DATABASE_URL points at a
real Neon instance, which is what makes it safe for every test in the fast
suite to import app.storage.repository (and transitively this module)
without ever touching the network. A connection is only opened the moment
something actually executes a query - the /analyze endpoint in production,
or an explicit fixture in tests/test_repository.py's `db`-marked cases.

Falls back to a local SQLite file (data/app.db) when DATABASE_URL is unset,
matching LocalFileStore's own "local disk now" default - so `uvicorn
app.platform.api:app --reload` still works with zero cloud setup.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

DEFAULT_LOCAL_DATABASE_URL = "sqlite:///./data/app.db"


def _use_psycopg3_driver(url: str) -> str:
    """Neon (and most guides) hand out a bare `postgresql://` / `postgres://`
    connection string, which SQLAlchemy maps to psycopg2 by default. This
    project installs psycopg3 (`psycopg[binary]`) instead, so rewrite the
    scheme to `postgresql+psycopg://` rather than asking the user to hand-edit
    DATABASE_URL in .env."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _database_url() -> str:
    raw = os.environ.get("DATABASE_URL", "").strip() or DEFAULT_LOCAL_DATABASE_URL
    return _use_psycopg3_driver(raw)


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or _database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(
        url,
        connect_args=connect_args,
        # Neon (and poolers generally) close idle server-side connections
        # out from under a long-lived client pool - without pre_ping, the
        # next checkout hands out that dead connection and the query fails
        # with "SSL connection has been closed unexpectedly" instead of
        # transparently reconnecting. pre_ping issues a cheap liveness check
        # (SELECT 1) on checkout and replaces the connection if it's dead.
        # pool_recycle proactively retires connections before Neon's own
        # idle-close window, rather than waiting to discover they're dead.
        pool_pre_ping=True,
        pool_recycle=280,
    )


engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
