"""Zero-Retention SQLite audit store for Pokrov."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from pokrov.core.config import AuditConfig
from pokrov.core.contracts import (
    Action,
    AuditEvent,
    AuditStats,
    ComplianceRef,
    ControlPoint,
    EntityCategory,
    Severity,
)

logger = logging.getLogger(__name__)

_MIN_HMAC_SALT_LEN: Final[int] = 16

_CREATE_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id_hmac TEXT NOT NULL DEFAULT '',
    principal TEXT NOT NULL DEFAULT 'anonymous',
    control_point TEXT NOT NULL,
    action TEXT NOT NULL,
    severity TEXT NOT NULL,
    categories TEXT NOT NULL,
    value_hmacs TEXT NOT NULL,
    detector_names TEXT NOT NULL,
    evasion_detected INTEGER NOT NULL DEFAULT 0,
    upstream TEXT NOT NULL DEFAULT '',
    policy_profile TEXT NOT NULL DEFAULT '',
    latency_ms REAL NOT NULL DEFAULT 0.0,
    compliance_refs TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT ''
);
"""

_CREATE_INDEX_TS_SQL: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON audit_events (timestamp_utc);"
)
_CREATE_INDEX_ACTION_SQL: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_audit_events_action ON audit_events (action);"
)
_CREATE_INDEX_REQ_SQL: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_audit_events_req ON audit_events (request_id);"
)

_INSERT_EVENT_SQL: Final[str] = """
INSERT INTO audit_events (
    event_id,
    timestamp,
    timestamp_utc,
    request_id,
    session_id_hmac,
    principal,
    control_point,
    action,
    severity,
    categories,
    value_hmacs,
    detector_names,
    evasion_detected,
    upstream,
    policy_profile,
    latency_ms,
    compliance_refs,
    reason,
    tool_name
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(event_id) DO UPDATE SET
    timestamp = excluded.timestamp,
    timestamp_utc = excluded.timestamp_utc,
    request_id = excluded.request_id,
    session_id_hmac = excluded.session_id_hmac,
    principal = excluded.principal,
    control_point = excluded.control_point,
    action = excluded.action,
    severity = excluded.severity,
    categories = excluded.categories,
    value_hmacs = excluded.value_hmacs,
    detector_names = excluded.detector_names,
    evasion_detected = excluded.evasion_detected,
    upstream = excluded.upstream,
    policy_profile = excluded.policy_profile,
    latency_ms = excluded.latency_ms,
    compliance_refs = excluded.compliance_refs,
    reason = excluded.reason,
    tool_name = excluded.tool_name;
"""

_QUERY_EVENTS_SQL: Final[str] = """
SELECT
    event_id,
    timestamp,
    request_id,
    session_id_hmac,
    principal,
    control_point,
    action,
    severity,
    categories,
    value_hmacs,
    detector_names,
    evasion_detected,
    upstream,
    policy_profile,
    latency_ms,
    compliance_refs,
    reason,
    tool_name
FROM audit_events
WHERE (? IS NULL OR action = ?)
  AND (? IS NULL OR EXISTS (SELECT 1 FROM json_each(categories) WHERE value = ?))
  AND (? IS NULL OR timestamp_utc >= ?)
ORDER BY timestamp_utc DESC, rowid DESC
LIMIT ? OFFSET ?;
"""

_STATS_SUMMARY_ALL_SQL: Final[str] = """
SELECT
    COUNT(*) AS total_events,
    COUNT(DISTINCT request_id) AS total_requests,
    MIN(timestamp_utc) AS min_ts,
    MAX(timestamp_utc) AS max_ts
FROM audit_events;
"""

_STATS_SUMMARY_SINCE_SQL: Final[str] = """
SELECT
    COUNT(*) AS total_events,
    COUNT(DISTINCT request_id) AS total_requests,
    MIN(timestamp_utc) AS min_ts,
    MAX(timestamp_utc) AS max_ts
FROM audit_events
WHERE timestamp_utc >= ?;
"""

_STATS_ACTIONS_ALL_SQL: Final[str] = """
SELECT action, COUNT(*) AS cnt
FROM audit_events
GROUP BY action;
"""

_STATS_ACTIONS_SINCE_SQL: Final[str] = """
SELECT action, COUNT(*) AS cnt
FROM audit_events
WHERE timestamp_utc >= ?
GROUP BY action;
"""

_STATS_CP_ALL_SQL: Final[str] = """
SELECT control_point, COUNT(*) AS cnt
FROM audit_events
GROUP BY control_point;
"""

_STATS_CP_SINCE_SQL: Final[str] = """
SELECT control_point, COUNT(*) AS cnt
FROM audit_events
WHERE timestamp_utc >= ?
GROUP BY control_point;
"""

_STATS_CAT_ALL_SQL: Final[str] = """
SELECT j.value AS category, COUNT(*) AS cnt
FROM audit_events AS e, json_each(e.categories) AS j
GROUP BY j.value;
"""

_STATS_CAT_SINCE_SQL: Final[str] = """
SELECT j.value AS category, COUNT(*) AS cnt
FROM audit_events AS e, json_each(e.categories) AS j
WHERE e.timestamp_utc >= ?
GROUP BY j.value;
"""

_STATS_LATENCY_ALL_SQL: Final[str] = """
SELECT latency_ms
FROM audit_events
ORDER BY latency_ms ASC;
"""

_STATS_LATENCY_SINCE_SQL: Final[str] = """
SELECT latency_ms
FROM audit_events
WHERE timestamp_utc >= ?
ORDER BY latency_ms ASC;
"""

_DELETE_EXPIRED_SQL: Final[str] = """
DELETE FROM audit_events
WHERE timestamp_utc < ?;
"""

_VALID_CHECKPOINT_MODES: Final[frozenset[str]] = frozenset(
    {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
)


def _to_utc_iso(dt: datetime) -> str:
    """Convert a naive or aware datetime to fixed-width UTC ISO-8601 string."""
    dt_utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact_json(value: object) -> str:
    """Serialize JSON compactly without unnecessary whitespace or ASCII escaping."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_posix() -> bool:
    """Return True when running on a POSIX operating system."""
    return os.name == "posix"


def _apply_posix_0600(db_path: Path) -> None:
    """Restrict SQLite database and WAL/SHM sidecar file permissions to 0o600 on POSIX."""
    if not _is_posix():
        return
    for candidate in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if candidate.exists():
            with contextlib.suppress(OSError):
                os.chmod(candidate, 0o600)


def _stream_percentiles(cursor: sqlite3.Cursor, count: int) -> tuple[float, float]:
    """Compute linear-interpolated p50 and p95 in O(1) RAM from an ordered SQLite cursor."""
    if count <= 0:
        return (0.0, 0.0)
    pos_50 = 0.50 * (count - 1)
    lo_50, hi_50 = math.floor(pos_50), math.ceil(pos_50)
    pos_95 = 0.95 * (count - 1)
    lo_95, hi_95 = math.floor(pos_95), math.ceil(pos_95)
    needed = {lo_50, hi_50, lo_95, hi_95}
    picked: dict[int, float] = {}
    for idx, row in enumerate(cursor):
        if idx in needed:
            picked[idx] = float(row[0])
            if idx >= hi_95 and len(picked) == len(needed):
                break
    if len(picked) < len(needed):
        return (0.0, 0.0)
    p50 = picked[lo_50] + (picked[hi_50] - picked[lo_50]) * (pos_50 - lo_50)
    p95 = picked[lo_95] + (picked[hi_95] - picked[lo_95]) * (pos_95 - lo_95)
    return (float(p50), float(p95))


class SqliteAuditStore:
    """Thread-safe Zero-Retention SQLite backend implementing ``AuditStore``.

    Only HMAC digests, entity categories, actions, compliance citations, and
    operational metadata are persisted; raw PII, secrets, and prompts are never
    stored in the database schema.
    """

    def __init__(self, config: AuditConfig | None = None) -> None:
        self._config = config or AuditConfig()
        self._lock = threading.RLock()
        self._closed = False
        self._salt_bytes = (
            self._config.hmac_salt.encode("utf-8") if self._config.hmac_salt else os.urandom(32)
        )

        self._is_memory = self._config.db_path == ":memory:"
        self._db_path: Path | None = None
        if self._is_memory:
            db_target = ":memory:"
        else:
            if not self._config.hmac_salt:
                logger.warning(
                    "AuditConfig.hmac_salt is empty for file-backed audit DB (%s); "
                    "using ephemeral per-process random salt (cross-restart HMAC "
                    "correlation disabled).",
                    self._config.db_path,
                )
            elif len(self._config.hmac_salt) < _MIN_HMAC_SALT_LEN:
                logger.warning(
                    "AuditConfig.hmac_salt is shorter than %d characters (%d chars) for "
                    "file-backed audit DB (%s); configure a high-entropy salt in production.",
                    _MIN_HMAC_SALT_LEN,
                    len(self._config.hmac_salt),
                    self._config.db_path,
                )

            db_path = Path(self._config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                with contextlib.suppress(OSError):
                    fd = os.open(db_path, os.O_CREAT | os.O_RDWR, 0o600)
                    os.close(fd)
            self._db_path = db_path
            db_target = str(db_path)

        self._conn = sqlite3.connect(db_target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA busy_timeout=5000")
            with self._conn:
                self._conn.execute(_CREATE_TABLE_SQL)
                self._conn.execute(_CREATE_INDEX_TS_SQL)
                self._conn.execute(_CREATE_INDEX_ACTION_SQL)
                self._conn.execute(_CREATE_INDEX_REQ_SQL)
            if self._db_path is not None:
                _apply_posix_0600(self._db_path)
                self.prune_expired()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteAuditStore is closed")

    def compute_hmac(self, raw_value: str) -> str:
        """Return hex-encoded HMAC-SHA256 digest of ``raw_value``."""
        return hmac.new(
            self._salt_bytes,
            raw_value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def prune_expired(self, *, now: datetime | None = None) -> int:
        """Delete audit rows older than ``config.retention_days`` and return deleted count."""
        ref_now = now if now is not None else datetime.now(UTC)
        cutoff = _to_utc_iso(ref_now - timedelta(days=self._config.retention_days))
        with self._lock:
            self._ensure_open()
            with self._conn:
                cursor = self._conn.execute(_DELETE_EXPIRED_SQL, (cutoff,))
                deleted = int(cursor.rowcount) if cursor.rowcount is not None else 0
        return max(0, deleted)

    def record(self, event: AuditEvent) -> None:
        """Persist a single Zero-Retention ``AuditEvent`` to SQLite."""
        params = (
            event.event_id,
            event.timestamp.isoformat(),
            _to_utc_iso(event.timestamp),
            event.request_id,
            event.session_id_hmac,
            event.principal,
            event.control_point.value,
            event.action.value,
            event.severity.value,
            _compact_json([cat.value for cat in event.categories]),
            _compact_json(list(event.value_hmacs)),
            _compact_json(list(event.detector_names)),
            1 if event.evasion_detected else 0,
            event.upstream,
            event.policy_profile,
            float(event.latency_ms),
            _compact_json([ref.model_dump(mode="json") for ref in event.compliance_refs]),
            event.reason,
            event.tool_name,
        )
        with self._lock:
            self._ensure_open()
            with self._conn:
                self._conn.execute(_INSERT_EVENT_SQL, params)
            if self._db_path is not None:
                _apply_posix_0600(self._db_path)

    def query(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        action: Action | None = None,
        category: EntityCategory | None = None,
        since: datetime | None = None,
    ) -> list[AuditEvent]:
        """Query persisted audit events newest-first with optional filters."""
        if limit <= 0:
            return []
        safe_offset = max(0, offset)
        action_val = action.value if action is not None else None
        category_val = category.value if category is not None else None
        since_val = _to_utc_iso(since) if since is not None else None

        with self._lock:
            self._ensure_open()
            cursor = self._conn.execute(
                _QUERY_EVENTS_SQL,
                (
                    action_val,
                    action_val,
                    category_val,
                    category_val,
                    since_val,
                    since_val,
                    limit,
                    safe_offset,
                ),
            )
            rows = cursor.fetchall()

        return [self._row_to_event(row) for row in rows]

    def stats(self, *, since: datetime | None = None) -> AuditStats:
        """Aggregate action, category, control-point, and latency statistics in SQL."""
        since_params: tuple[str, ...] = (_to_utc_iso(since),) if since is not None else ()
        summary_sql = _STATS_SUMMARY_SINCE_SQL if since is not None else _STATS_SUMMARY_ALL_SQL
        actions_sql = _STATS_ACTIONS_SINCE_SQL if since is not None else _STATS_ACTIONS_ALL_SQL
        cp_sql = _STATS_CP_SINCE_SQL if since is not None else _STATS_CP_ALL_SQL
        cat_sql = _STATS_CAT_SINCE_SQL if since is not None else _STATS_CAT_ALL_SQL
        latency_sql = _STATS_LATENCY_SINCE_SQL if since is not None else _STATS_LATENCY_ALL_SQL

        with self._lock:
            self._ensure_open()
            summary_row = self._conn.execute(summary_sql, since_params).fetchone()
            total_events = int(summary_row["total_events"]) if summary_row is not None else 0
            if total_events == 0 or summary_row is None:
                return AuditStats()

            total_requests = int(summary_row["total_requests"])
            period_start = datetime.fromisoformat(str(summary_row["min_ts"]))
            period_end = datetime.fromisoformat(str(summary_row["max_ts"]))

            action_rows = self._conn.execute(actions_sql, since_params).fetchall()
            cp_rows = self._conn.execute(cp_sql, since_params).fetchall()
            cat_rows = self._conn.execute(cat_sql, since_params).fetchall()
            with contextlib.closing(self._conn.execute(latency_sql, since_params)) as lat_cursor:
                p50, p95 = _stream_percentiles(lat_cursor, total_events)

        blocked = 0
        masked = 0
        routed_local = 0
        flagged = 0
        allowed = 0
        approval_required = 0

        for row in action_rows:
            act = Action(str(row["action"]))
            cnt = int(row["cnt"])
            if act == Action.BLOCK:
                blocked += cnt
            elif act in (Action.MASK, Action.MASK_REVERSIBLE):
                masked += cnt
            elif act == Action.ROUTE_LOCAL:
                routed_local += cnt
            elif act == Action.FLAG:
                flagged += cnt
            elif act == Action.ALLOW:
                allowed += cnt
            elif act == Action.REQUIRE_APPROVAL:
                approval_required += cnt

        by_control_point = {str(row["control_point"]): int(row["cnt"]) for row in cp_rows}
        by_category = {str(row["category"]): int(row["cnt"]) for row in cat_rows}

        return AuditStats(
            total_requests=total_requests,
            blocked_count=blocked,
            masked_count=masked,
            routed_local_count=routed_local,
            flagged_count=flagged,
            allowed_count=allowed,
            approval_required_count=approval_required,
            by_category=by_category,
            by_control_point=by_control_point,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            period_start=period_start,
            period_end=period_end,
        )

    def checkpoint(self, mode: str = "PASSIVE") -> None:
        """Run a SQLite WAL checkpoint on file-backed databases."""
        normalized = mode.upper()
        if normalized not in _VALID_CHECKPOINT_MODES:
            raise ValueError(f"Unsupported WAL checkpoint mode: {mode!r}")
        if self._is_memory:
            return
        with self._lock:
            self._ensure_open()
            self._conn.execute(f"PRAGMA wal_checkpoint({normalized})")

    def close(self) -> None:
        """Close the underlying SQLite connection idempotently."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def __enter__(self) -> SqliteAuditStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AuditEvent:
        categories = tuple(EntityCategory(str(item)) for item in json.loads(str(row["categories"])))
        value_hmacs = tuple(str(item) for item in json.loads(str(row["value_hmacs"])))
        detector_names = tuple(str(item) for item in json.loads(str(row["detector_names"])))
        compliance_refs = tuple(
            ComplianceRef.model_validate(item) for item in json.loads(str(row["compliance_refs"]))
        )
        return AuditEvent(
            event_id=str(row["event_id"]),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            request_id=str(row["request_id"]),
            session_id_hmac=str(row["session_id_hmac"]),
            principal=str(row["principal"]),
            control_point=ControlPoint(str(row["control_point"])),
            action=Action(str(row["action"])),
            severity=Severity(str(row["severity"])),
            categories=categories,
            value_hmacs=value_hmacs,
            detector_names=detector_names,
            evasion_detected=bool(row["evasion_detected"]),
            upstream=str(row["upstream"]),
            policy_profile=str(row["policy_profile"]),
            latency_ms=float(row["latency_ms"]),
            compliance_refs=compliance_refs,
            reason=str(row["reason"]),
            tool_name=str(row["tool_name"]),
        )


__all__ = ["SqliteAuditStore"]
