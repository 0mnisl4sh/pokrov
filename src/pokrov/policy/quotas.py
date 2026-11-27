"""Thread-safe per-principal quota enforcement using LRU-bounded token buckets."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from pokrov.core.config import QuotaConfig
from pokrov.core.contracts import QuotaCheckResult

_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_DAY = 86_400.0
_FLOAT_EPSILON = 1e-9


@dataclass(slots=True)
class _PrincipalBucket:
    rpm_tokens: float
    tpm_tokens: float
    last_refill: float
    daily_used: int
    daily_window_start: float


class TokenBucketQuotaManager:
    """Enforces payload-size and rate-limit quotas per principal.

    Implements the ``QuotaManager`` protocol using LRU-bounded token buckets for
    ``rpm`` and ``tpm``, a 24-hour window for ``daily_requests``, and hard
    per-request bounds for ``max_prompt_chars`` and ``max_estimated_tokens``.
    """

    def __init__(
        self,
        quota_config: QuotaConfig,
        time_fn: Callable[[], float] | None = None,
        max_principals: int = 10_000,
    ) -> None:
        if max_principals <= 0:
            raise ValueError(f"max_principals must be positive, got {max_principals}")
        self._config = quota_config
        self._time_fn: Callable[[], float] = time_fn if time_fn is not None else time.monotonic
        self._max_principals = max_principals
        self._lock = threading.RLock()
        self._buckets: OrderedDict[str, _PrincipalBucket] = OrderedDict()

    @property
    def config(self) -> QuotaConfig:
        """Return the active quota configuration."""
        return self._config

    @property
    def max_principals(self) -> int:
        """Return maximum number of principal buckets retained in memory."""
        return self._max_principals

    def _get_and_refill_bucket_locked(self, principal: str, now: float) -> _PrincipalBucket:
        bucket = self._buckets.get(principal)
        if bucket is None:
            while len(self._buckets) >= self._max_principals:
                self._buckets.popitem(last=False)
            bucket = _PrincipalBucket(
                rpm_tokens=float(self._config.rpm),
                tpm_tokens=float(self._config.tpm),
                last_refill=now,
                daily_used=0,
                daily_window_start=now,
            )
            self._buckets[principal] = bucket
            return bucket

        self._buckets.move_to_end(principal)

        elapsed = max(0.0, now - bucket.last_refill)
        if elapsed > 0.0:
            rpm_cap = float(self._config.rpm)
            tpm_cap = float(self._config.tpm)
            bucket.rpm_tokens = min(
                rpm_cap,
                bucket.rpm_tokens + (elapsed * rpm_cap) / _SECONDS_PER_MINUTE,
            )
            bucket.tpm_tokens = min(
                tpm_cap,
                bucket.tpm_tokens + (elapsed * tpm_cap) / _SECONDS_PER_MINUTE,
            )
            bucket.last_refill = now

        if now - bucket.daily_window_start >= _SECONDS_PER_DAY:
            bucket.daily_used = 0
            bucket.daily_window_start = now

        return bucket

    def _remaining_counts(self, bucket: _PrincipalBucket) -> tuple[int, int, int]:
        remaining_rpm = max(0, int(bucket.rpm_tokens + _FLOAT_EPSILON))
        remaining_tpm = max(0, int(bucket.tpm_tokens + _FLOAT_EPSILON))
        remaining_daily = max(0, self._config.daily_requests - bucket.daily_used)
        return remaining_rpm, remaining_tpm, remaining_daily

    def check_and_consume(
        self,
        principal: str,
        prompt_chars: int,
        estimated_tokens: int,
    ) -> QuotaCheckResult:
        """Check all 5 quota dimensions for ``principal`` and consume budget if allowed."""
        if prompt_chars < 0:
            raise ValueError(f"prompt_chars must be non-negative, got {prompt_chars}")
        if estimated_tokens < 0:
            raise ValueError(f"estimated_tokens must be non-negative, got {estimated_tokens}")

        with self._lock:
            now = self._time_fn()
            bucket = self._get_and_refill_bucket_locked(principal, now)
            rem_rpm, rem_tpm, rem_daily = self._remaining_counts(bucket)

            if prompt_chars > self._config.max_prompt_chars:
                return QuotaCheckResult(
                    allowed=False,
                    reason=(
                        f"Prompt length {prompt_chars} exceeds max_prompt_chars "
                        f"({self._config.max_prompt_chars})."
                    ),
                    retry_after_seconds=0,
                    remaining_rpm=rem_rpm,
                    remaining_tpm=rem_tpm,
                    remaining_daily_requests=rem_daily,
                )

            if estimated_tokens > self._config.max_estimated_tokens:
                return QuotaCheckResult(
                    allowed=False,
                    reason=(
                        f"Estimated tokens {estimated_tokens} exceeds max_estimated_tokens "
                        f"({self._config.max_estimated_tokens})."
                    ),
                    retry_after_seconds=0,
                    remaining_rpm=rem_rpm,
                    remaining_tpm=rem_tpm,
                    remaining_daily_requests=rem_daily,
                )

            if estimated_tokens > self._config.tpm:
                return QuotaCheckResult(
                    allowed=False,
                    reason=(
                        f"Estimated tokens {estimated_tokens} exceeds tpm bucket capacity "
                        f"({self._config.tpm})."
                    ),
                    retry_after_seconds=0,
                    remaining_rpm=rem_rpm,
                    remaining_tpm=rem_tpm,
                    remaining_daily_requests=rem_daily,
                )

            reasons: list[str] = []
            retry_candidates: list[int] = []

            if bucket.daily_used >= self._config.daily_requests:
                reasons.append(f"Daily request quota exceeded ({self._config.daily_requests}/day).")
                daily_elapsed = max(0.0, now - bucket.daily_window_start)
                retry_candidates.append(max(1, math.ceil(_SECONDS_PER_DAY - daily_elapsed)))

            if bucket.rpm_tokens + _FLOAT_EPSILON < 1.0:
                reasons.append(f"RPM rate limit exceeded ({self._config.rpm} req/min).")
                deficit = max(0.0, 1.0 - bucket.rpm_tokens)
                wait_seconds = (deficit * _SECONDS_PER_MINUTE) / float(self._config.rpm)
                retry_candidates.append(max(1, math.ceil(wait_seconds)))

            needed_tokens = float(estimated_tokens)
            if bucket.tpm_tokens + _FLOAT_EPSILON < needed_tokens:
                reasons.append(f"TPM rate limit exceeded ({self._config.tpm} tok/min).")
                deficit = max(0.0, needed_tokens - bucket.tpm_tokens)
                wait_seconds = (deficit * _SECONDS_PER_MINUTE) / float(self._config.tpm)
                retry_candidates.append(max(1, math.ceil(wait_seconds)))

            if reasons:
                return QuotaCheckResult(
                    allowed=False,
                    reason=" ".join(reasons),
                    retry_after_seconds=max(retry_candidates),
                    remaining_rpm=rem_rpm,
                    remaining_tpm=rem_tpm,
                    remaining_daily_requests=rem_daily,
                )

            bucket.rpm_tokens = max(0.0, bucket.rpm_tokens - 1.0)
            bucket.tpm_tokens = max(0.0, bucket.tpm_tokens - needed_tokens)
            bucket.daily_used += 1

            post_rpm, post_tpm, post_daily = self._remaining_counts(bucket)
            return QuotaCheckResult(
                allowed=True,
                reason="",
                retry_after_seconds=0,
                remaining_rpm=post_rpm,
                remaining_tpm=post_tpm,
                remaining_daily_requests=post_daily,
            )

    def reset(self, principal: str | None = None) -> None:
        """Clear stored quota buckets for ``principal`` or all principals."""
        with self._lock:
            if principal is None:
                self._buckets.clear()
            else:
                self._buckets.pop(principal, None)


__all__ = ["TokenBucketQuotaManager"]
