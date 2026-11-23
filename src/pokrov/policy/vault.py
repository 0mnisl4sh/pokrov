"""Reversible AES-256-GCM tokenization vaults for Pokrov."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pokrov.core.contracts import (
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    EntityCategory,
)

_PLACEHOLDER_PATTERN = re.compile(
    re.escape(PLACEHOLDER_PREFIX) + r"[A-Z0-9_]+:[0-9a-f]{8}" + re.escape(PLACEHOLDER_SUFFIX)
)
_VAULT_FORMAT_VERSION = 1
_AES_KEY_DOMAIN = b"pokrov:local-vault:aes-256-gcm:v1"
_HMAC_KEY_DOMAIN = b"pokrov:local-vault:tag-hmac:v1"


@dataclass(slots=True)
class _VaultEntry:
    nonce: bytes
    ciphertext: bytes
    expires_at: float


VaultKey = tuple[str, str]


def _validate_vault_params(ttl_seconds: int, max_entries: int) -> None:
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
    if max_entries <= 0:
        raise ValueError(f"max_entries must be positive, got {max_entries}")


def _encode_hmac_payload(session_id: str, category: str, raw_value: str) -> bytes:
    parts = (
        session_id.encode("utf-8"),
        category.encode("utf-8"),
        raw_value.encode("utf-8"),
    )
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def _make_placeholder(
    hmac_key: bytes,
    session_id: str,
    category: EntityCategory,
    raw_value: str,
) -> str:
    payload = _encode_hmac_payload(session_id, category.value, raw_value)
    tag = hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()[:8]
    return f"{PLACEHOLDER_PREFIX}{category.value.upper()}:{tag}{PLACEHOLDER_SUFFIX}"


def _make_aad(session_id: str, placeholder: str) -> bytes:
    return f"{session_id}|{placeholder}".encode()


def _encrypt_entry(
    aesgcm: AESGCM,
    session_id: str,
    placeholder: str,
    raw_value: str,
    expires_at: float,
) -> _VaultEntry:
    nonce = os.urandom(12)
    aad = _make_aad(session_id, placeholder)
    ciphertext = aesgcm.encrypt(nonce, raw_value.encode("utf-8"), aad)
    return _VaultEntry(nonce=nonce, ciphertext=ciphertext, expires_at=expires_at)


def _decrypt_entry(
    aesgcm: AESGCM,
    session_id: str,
    placeholder: str,
    entry: _VaultEntry,
) -> str:
    aad = _make_aad(session_id, placeholder)
    plaintext = aesgcm.decrypt(entry.nonce, entry.ciphertext, aad)
    return plaintext.decode("utf-8")


def _cleanup_expired_entries(entries: OrderedDict[VaultKey, _VaultEntry], now: float) -> int:
    expired_keys = [key for key, entry in entries.items() if entry.expires_at <= now]
    for key in expired_keys:
        del entries[key]
    return len(expired_keys)


def _purge_session_entries(entries: OrderedDict[VaultKey, _VaultEntry], session_id: str) -> int:
    session_keys = [key for key in entries if key[0] == session_id]
    for key in session_keys:
        del entries[key]
    return len(session_keys)


def _store_in_dict(
    entries: OrderedDict[VaultKey, _VaultEntry],
    aesgcm: AESGCM,
    hmac_key: bytes,
    session_id: str,
    category: EntityCategory,
    raw_value: str,
    *,
    now: float,
    ttl_seconds: int,
    max_entries: int,
) -> str:
    _cleanup_expired_entries(entries, now)
    placeholder = _make_placeholder(hmac_key, session_id, category, raw_value)
    key: VaultKey = (session_id, placeholder)
    entry = _encrypt_entry(
        aesgcm,
        session_id,
        placeholder,
        raw_value,
        expires_at=now + ttl_seconds,
    )
    if key in entries:
        entries[key] = entry
        entries.move_to_end(key)
    else:
        while len(entries) >= max_entries:
            entries.popitem(last=False)
        entries[key] = entry
    return placeholder


def _retrieve_from_dict(
    entries: OrderedDict[VaultKey, _VaultEntry],
    aesgcm: AESGCM,
    session_id: str,
    placeholder: str,
    *,
    now: float,
) -> tuple[str | None, bool]:
    """Return ``(plaintext_or_none, dict_mutated)``."""
    key: VaultKey = (session_id, placeholder)
    entry = entries.get(key)
    if entry is None:
        return None, False
    if entry.expires_at <= now:
        del entries[key]
        return None, True

    mutated = False
    if len(entries) > 1:
        last_key = next(reversed(entries))
        if last_key != key:
            entries.move_to_end(key)
            mutated = True

    return _decrypt_entry(aesgcm, session_id, placeholder, entry), mutated


def _restore_text_from_dict(
    entries: OrderedDict[VaultKey, _VaultEntry],
    aesgcm: AESGCM,
    session_id: str,
    text: str,
    *,
    now: float,
) -> tuple[str, int, bool]:
    """Return ``(restored_text, replacement_count, dict_mutated)``."""
    if PLACEHOLDER_PREFIX not in text:
        return text, 0, False

    restored_count = 0
    mutated = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal restored_count, mutated
        placeholder = match.group(0)
        value, did_mutate = _retrieve_from_dict(
            entries,
            aesgcm,
            session_id,
            placeholder,
            now=now,
        )
        if did_mutate:
            mutated = True
        if value is None:
            return placeholder
        restored_count += 1
        return value

    restored = _PLACEHOLDER_PATTERN.sub(_replace, text)
    return restored, restored_count, mutated


class InMemoryAesVault:
    """Thread-safe RAM-only AES-256-GCM vault with TTL and LRU eviction.

    Implements the ``VaultBackend`` protocol without persisting secrets to disk.
    """

    def __init__(
        self,
        ttl_seconds: int = 900,
        max_entries: int = 10_000,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        _validate_vault_params(ttl_seconds, max_entries)
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._time_fn: Callable[[], float] = time_fn if time_fn is not None else time.monotonic
        self._lock = threading.RLock()
        self._aes_key = os.urandom(32)
        self._hmac_key = os.urandom(32)
        self._aesgcm = AESGCM(self._aes_key)
        self._entries: OrderedDict[VaultKey, _VaultEntry] = OrderedDict()

    def store(self, session_id: str, category: EntityCategory, raw_value: str) -> str:
        """Encrypt and store ``raw_value``, returning a deterministic session placeholder."""
        with self._lock:
            return _store_in_dict(
                self._entries,
                self._aesgcm,
                self._hmac_key,
                session_id,
                category,
                raw_value,
                now=self._time_fn(),
                ttl_seconds=self._ttl_seconds,
                max_entries=self._max_entries,
            )

    def retrieve(self, session_id: str, placeholder: str) -> str | None:
        """Decrypt and return the original value for ``(session_id, placeholder)``."""
        with self._lock:
            value, _ = _retrieve_from_dict(
                self._entries,
                self._aesgcm,
                session_id,
                placeholder,
                now=self._time_fn(),
            )
            return value

    def restore_text(self, session_id: str, text: str) -> tuple[str, int]:
        """Replace all valid session placeholders in ``text`` with original values."""
        if PLACEHOLDER_PREFIX not in text:
            return text, 0
        with self._lock:
            restored, count, _ = _restore_text_from_dict(
                self._entries,
                self._aesgcm,
                session_id,
                text,
                now=self._time_fn(),
            )
            return restored, count

    def purge_session(self, session_id: str) -> int:
        """Delete all entries belonging to ``session_id`` and return count removed."""
        with self._lock:
            return _purge_session_entries(self._entries, session_id)

    def cleanup_expired(self) -> int:
        """Purge all expired vault entries and return count removed."""
        with self._lock:
            return _cleanup_expired_entries(self._entries, self._time_fn())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return (
            f"InMemoryAesVault(ttl_seconds={self._ttl_seconds}, "
            f"max_entries={self._max_entries}, entries={count})"
        )


def _derive_keyfile_path(vault_path: Path) -> Path:
    if vault_path.suffix == ".key":
        return Path(f"{vault_path}.master.key")
    return vault_path.with_suffix(".key")


def _chmod_restrictive(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _write_bytes_atomic(target_path: Path, data: bytes) -> None:
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(parent, 0o700)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _chmod_restrictive(tmp_path)
        os.replace(tmp_path, target_path)
        _chmod_restrictive(target_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class LocalEncryptedVault:
    """Opt-in persistent AES-256-GCM vault for short-lived IDE Guard CLI processes.

    Implements the ``VaultBackend`` protocol. Stores encrypted entries in
    ``local_vault_path`` and the 32-byte master key in a sibling ``.key`` file
    with restrictive permissions.
    """

    def __init__(
        self,
        local_vault_path: str | Path = ".pokrov/vault.enc",
        ttl_seconds: int = 900,
        max_entries: int = 10_000,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        _validate_vault_params(ttl_seconds, max_entries)
        self._vault_path = Path(local_vault_path)
        self._key_path = _derive_keyfile_path(self._vault_path)
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._time_fn: Callable[[], float] = time_fn if time_fn is not None else time.time
        self._lock = threading.RLock()

    @property
    def vault_path(self) -> Path:
        """Return path to the encrypted vault file."""
        return self._vault_path

    @property
    def key_path(self) -> Path:
        """Return path to the sibling master key file."""
        return self._key_path

    def _derive_keys(self, master_key: bytes) -> tuple[AESGCM, bytes]:
        if len(master_key) != 32:
            raise ValueError(f"Invalid master key length in {self._key_path}: {len(master_key)}")
        aes_key = hmac.new(master_key, _AES_KEY_DOMAIN, hashlib.sha256).digest()
        hmac_key = hmac.new(master_key, _HMAC_KEY_DOMAIN, hashlib.sha256).digest()
        return AESGCM(aes_key), hmac_key

    def _ensure_keys_locked(self) -> tuple[AESGCM, bytes]:
        if self._key_path.is_file():
            return self._derive_keys(self._key_path.read_bytes())

        master_key = os.urandom(32)
        _write_bytes_atomic(self._key_path, master_key)
        return self._derive_keys(master_key)

    def _load_existing_locked(
        self,
    ) -> tuple[AESGCM, bytes, OrderedDict[VaultKey, _VaultEntry]] | None:
        if not self._vault_path.is_file():
            return None
        if not self._key_path.is_file():
            raise RuntimeError(
                f"Vault keyfile missing for existing encrypted vault: {self._key_path}"
            )

        aesgcm, hmac_key = self._derive_keys(self._key_path.read_bytes())
        raw_json = self._vault_path.read_text(encoding="utf-8")
        payload = json.loads(raw_json)
        if not isinstance(payload, dict) or payload.get("version") != _VAULT_FORMAT_VERSION:
            raise ValueError(f"Unsupported encrypted vault format in {self._vault_path}")

        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError(f"Invalid entries list in {self._vault_path}")

        entries: OrderedDict[VaultKey, _VaultEntry] = OrderedDict()
        for item in raw_entries:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid entry record in {self._vault_path}")
            session_id = item["s"]
            placeholder = item["p"]
            nonce = base64.b64decode(item["n"], validate=True)
            ciphertext = base64.b64decode(item["c"], validate=True)
            expires_at = float(item["e"])
            if not isinstance(session_id, str) or not isinstance(placeholder, str):
                raise ValueError(f"Invalid key fields in {self._vault_path}")
            entries[(session_id, placeholder)] = _VaultEntry(
                nonce=nonce,
                ciphertext=ciphertext,
                expires_at=expires_at,
            )

        return aesgcm, hmac_key, entries

    def _save_entries_locked(self, entries: OrderedDict[VaultKey, _VaultEntry]) -> None:
        serialized_entries = [
            {
                "s": session_id,
                "p": placeholder,
                "n": base64.b64encode(entry.nonce).decode("ascii"),
                "c": base64.b64encode(entry.ciphertext).decode("ascii"),
                "e": entry.expires_at,
            }
            for (session_id, placeholder), entry in entries.items()
        ]
        payload = {
            "version": _VAULT_FORMAT_VERSION,
            "entries": serialized_entries,
        }
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        _write_bytes_atomic(self._vault_path, encoded)

    def store(self, session_id: str, category: EntityCategory, raw_value: str) -> str:
        """Encrypt and persist ``raw_value``, returning a deterministic session placeholder."""
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                aesgcm, hmac_key = self._ensure_keys_locked()
                entries: OrderedDict[VaultKey, _VaultEntry] = OrderedDict()
            else:
                aesgcm, hmac_key, entries = loaded

            placeholder = _store_in_dict(
                entries,
                aesgcm,
                hmac_key,
                session_id,
                category,
                raw_value,
                now=self._time_fn(),
                ttl_seconds=self._ttl_seconds,
                max_entries=self._max_entries,
            )
            self._save_entries_locked(entries)
            return placeholder

    def retrieve(self, session_id: str, placeholder: str) -> str | None:
        """Load vault from disk and decrypt ``(session_id, placeholder)`` if valid."""
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                return None
            aesgcm, _, entries = loaded
            value, mutated = _retrieve_from_dict(
                entries,
                aesgcm,
                session_id,
                placeholder,
                now=self._time_fn(),
            )
            if mutated:
                self._save_entries_locked(entries)
            return value

    def restore_text(self, session_id: str, text: str) -> tuple[str, int]:
        """Load vault from disk and unmask placeholders in ``text``."""
        if PLACEHOLDER_PREFIX not in text:
            return text, 0
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                return text, 0
            aesgcm, _, entries = loaded
            restored, count, mutated = _restore_text_from_dict(
                entries,
                aesgcm,
                session_id,
                text,
                now=self._time_fn(),
            )
            if mutated:
                self._save_entries_locked(entries)
            return restored, count

    def purge_session(self, session_id: str) -> int:
        """Remove all entries for ``session_id`` from persistent encrypted storage."""
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                return 0
            _, _, entries = loaded
            removed = _purge_session_entries(entries, session_id)
            if removed > 0:
                self._save_entries_locked(entries)
            return removed

    def cleanup_expired(self) -> int:
        """Remove expired entries from persistent encrypted storage."""
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                return 0
            _, _, entries = loaded
            removed = _cleanup_expired_entries(entries, self._time_fn())
            if removed > 0:
                self._save_entries_locked(entries)
            return removed

    def __len__(self) -> int:
        with self._lock:
            loaded = self._load_existing_locked()
            if loaded is None:
                return 0
            return len(loaded[2])

    def __repr__(self) -> str:
        return (
            f"LocalEncryptedVault(local_vault_path={str(self._vault_path)!r}, "
            f"ttl_seconds={self._ttl_seconds}, max_entries={self._max_entries})"
        )


__all__ = ["InMemoryAesVault", "LocalEncryptedVault"]
