"""Single Redshift-access wrapper for the FDE Email Agent (Phase 4 debugging).

This is the ONLY module in the codebase that opens a connection to the Plivo
data warehouse. Everything else (app/redshift_tools.py) calls the helpers here.
Mirrors the app/llm.py single-access-point pattern: one place to hold the
driver, the credentials, and the safety posture, so the rest of the code can't
bypass them.

Hard rules honored here (CLAUDE.md — internal data tools):
- Credentials are read from the environment (gitignored `.env`); never hardcoded.
- READ-ONLY. The connection is opened read-only AND every statement is funnelled
  through `query()`, which refuses anything that is not a single SELECT/WITH.
  There is deliberately NO method on this module that runs INSERT/UPDATE/DELETE
  or any DDL — there is no write/DDL code path to call.
- PARAMETERIZED. `query(sql, params)` binds params via the driver (psycopg's
  server-side parameter binding); SQL is never built by string-interpolating
  caller values. Account scoping is enforced one layer up, in redshift_tools.py,
  by always binding account_id as a parameter inside each function.

Driver: psycopg 3 (already a project dependency for Postgres). Redshift speaks
the Postgres wire protocol, so psycopg connects to it directly — no extra
dependency. (Amazon's `redshift-connector` is the alternative; psycopg keeps a
single driver story and is already installed/tested here.)
"""

from __future__ import annotations

import os
import re
import threading

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

# --- Connection config (warehouse creds live only in .env) -------------------
_HOST = os.getenv("REDSHIFT_HOST")
_PORT = int(os.getenv("REDSHIFT_PORT", "5439"))
_DB = os.getenv("REDSHIFT_DB")
_USER = os.getenv("REDSHIFT_USER")
_PASSWORD = os.getenv("REDSHIFT_PASSWORD")

# A short statement timeout so a runaway analytical query can't hang the agent.
_STATEMENT_TIMEOUT_MS = int(os.getenv("REDSHIFT_STATEMENT_TIMEOUT_MS", "30000"))

_conn: psycopg.Connection | None = None
_lock = threading.Lock()  # serialize connect/reconnect (one warehouse conn)


# --- SELECT-only guard -------------------------------------------------------
# Backstop against anything but a single read. Our own tool queries are the only
# callers, so this is defense-in-depth, not untrusted-input sanitization.
_FORBIDDEN = re.compile(
    r"(?is)\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|unload|merge|call|vacuum|analyze|comment|lock|prepare)\b"
)


def _assert_select(sql: str) -> None:
    """Raise unless `sql` is a single SELECT/WITH statement with no write/DDL."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("empty SQL")
    # No statement chaining — one read at a time.
    if ";" in stripped:
        raise ValueError("multiple statements are not allowed (read-only)")
    head = stripped.split(None, 1)[0].lower()
    if head not in ("select", "with"):
        raise ValueError(f"only SELECT/WITH queries are allowed, got: {head!r}")
    if _FORBIDDEN.search(stripped):
        raise ValueError("statement contains a non-SELECT keyword (read-only)")


# --- Connection management ---------------------------------------------------
def _connect() -> psycopg.Connection:
    missing = [n for n, v in (("REDSHIFT_HOST", _HOST), ("REDSHIFT_DB", _DB),
                              ("REDSHIFT_USER", _USER), ("REDSHIFT_PASSWORD", _PASSWORD))
               if not v]
    if missing:
        raise RuntimeError(f"Redshift config missing from .env: {', '.join(missing)}")
    # Credentials all come from .env (see module top); assembled as a dict so no
    # secret is ever inlined here.
    conn_params = {
        "host": _HOST, "port": _PORT, "dbname": _DB,
        "user": _USER, "password": _PASSWORD,
        "autocommit": True, "row_factory": dict_row,
        "connect_timeout": int(os.getenv("REDSHIFT_CONNECT_TIMEOUT", "15")),
        "application_name": "fde-email-agent",
        # Redshift reports client_encoding as "UNICODE", which psycopg 3 cannot
        # map to a Python codec ("codec not available: 'UNICODE'"). Pin utf8 in
        # the startup packet so the negotiated encoding is one psycopg knows.
        "client_encoding": "utf8",
    }
    conn = psycopg.connect(**conn_params)
    # Defense-in-depth: ask the server for a read-only session. The hard
    # guarantee remains `_assert_select` + the absence of any write code path;
    # this is best-effort because not every warehouse honors it.
    try:
        conn.read_only = True
    except Exception:
        pass
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO %s", (_STATEMENT_TIMEOUT_MS,))
    except Exception:
        pass
    return conn


def get_connection() -> psycopg.Connection:
    """Lazily open (and cache) the single read-only warehouse connection."""
    global _conn
    with _lock:
        if _conn is None or _conn.closed:
            _conn = _connect()
        return _conn


def query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a single SELECT/WITH and return rows as dicts.

    `params` are bound by the driver (parameterized) — never string-formatted
    into `sql`. Reconnects once if the cached connection has gone stale.
    """
    _assert_select(sql)
    for attempt in (1, 2):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if cur.description is None:
                    return []
                return cur.fetchall()
        except (psycopg.OperationalError, psycopg.InterfaceError):
            # Stale/broken connection — drop it and retry once with a fresh one.
            with _lock:
                global _conn
                try:
                    if _conn is not None:
                        _conn.close()
                finally:
                    _conn = None
            if attempt == 2:
                raise
    return []


def query_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    """Run a SELECT expected to return at most one row; None if no rows."""
    rows = query(sql, params)
    return rows[0] if rows else None
