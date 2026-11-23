"""Zero-Retention SQLite audit store for Pokrov."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import threading
from datetime import UTC, datetime
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

_STATS_ALL_SQL: Final[str] = """
SELECT timestamp_utc, control_point, action, categories, latency_ms
FROM audit_events
ORDER BY timestamp_utc ASC, rowid ASC;
"""

_STATS_SINCE_SQL: Final[str] = """
SELECT timestamp_utc, control_point, action, categories, latency_ms
FROM audit_events
WHERE timestamp_utc >= ?
ORDER BY timestamp_utc ASC, rowid ASC;
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


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile in ``[0, 100]`` over sorted values."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if count == 1:
        return float(sorted_values[0])
    pos = (pct / 100.0) * (count - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    fraction = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * fraction)


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
        if self._is_memory:
            db_target = ":memory:"
        else:
            db_path = Path(self._config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Aggregate action, category, control-point, and latency statistics."""
        with self._lock:
            self._ensure_open()
            if since is None:
                rows = self._conn.execute(_STATS_ALL_SQL).fetchall()
            else:
                rows = self._conn.execute(_STATS_SINCE_SQL, (_to_utc_iso(since),)).fetchall()

        if not rows:
            return AuditStats()

        blocked = 0
        masked = 0
        routed_local = 0
        flagged = 0
        allowed = 0
        approval_required = 0
        by_category: dict[str, int] = {}
        by_control_point: dict[str, int] = {}
        latencies: list[float] = []

        for row in rows:
            act = Action(str(row["action"]))
            if act == Action.BLOCK:
                blocked += 1
            elif act in (Action.MASK, Action.MASK_REVERSIBLE):
                masked += 1
            elif act == Action.ROUTE_LOCAL:
                routed_local += 1
            elif act == Action.FLAG:
                flagged += 1
            elif act == Action.ALLOW:
                allowed += 1
            elif act == Action.REQUIRE_APPROVAL:
                approval_required += 1

            cp = str(row["control_point"])
            by_control_point[cp] = by_control_point.get(cp, 0) + 1

            raw_cats = json.loads(str(row["categories"]))
            for cat in raw_cats:
                cat_key = str(cat)
                by_category[cat_key] = by_category.get(cat_key, 0) + 1

            latencies.append(float(row["latency_ms"]))

        latencies.sort()
        period_start = datetime.fromisoformat(str(rows[0]["timestamp_utc"]))
        period_end = datetime.fromisoformat(str(rows[-1]["timestamp_utc"]))

        return AuditStats(
            total_requests=len(rows),
            blocked_count=blocked,
            masked_count=masked,
            routed_local_count=routed_local,
            flagged_count=flagged,
            allowed_count=allowed,
            approval_required_count=approval_required,
            by_category=by_category,
            by_control_point=by_control_point,
            p50_latency_ms=_percentile(latencies, 50.0),
            p95_latency_ms=_percentile(latencies, 95.0),
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
