"""Reversible AES-256-GCM tokenization vaults for Pokrov."""

from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pokrov.core.contracts import (
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    EntityCategory,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_PLACEHOLDER_PATTERN = re.compile(
    re.escape(PLACEHOLDER_PREFIX) + r"[A-Z0-9_]+:[0-9a-f]{8,16}" + re.escape(PLACEHOLDER_SUFFIX)
)
_VAULT_FORMAT_VERSION = 2
_AES_KEY_DOMAIN = b"pokrov:local-vault:aes-256-gcm:v1"
_HMAC_KEY_DOMAIN = b"pokrov:local-vault:tag-hmac:v1"
_OUTER_AES_KEY_DOMAIN = b"pokrov:local-vault:outer-envelope:v2"
_OUTER_ENVELOPE_AAD = b"pokrov:local-vault:envelope:v2"
_DPAPI_MAGIC = b"PKRDPAPI\x01"
_CRYPTPROTECT_UI_FORBIDDEN = 0x01
_LAZY_SWEEP_INTERVAL_SECONDS = 5.0
_LOCK_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class _VaultEntry:
    nonce: bytes
    ciphertext: bytes
    expires_at: float


VaultKey = tuple[str, str]


def _validate_vault_params(
    ttl_seconds: int,
    max_entries: int,
    max_entries_per_session: int = 1024,
) -> None:
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
    if max_entries <= 0:
        raise ValueError(f"max_entries must be positive, got {max_entries}")
    if max_entries_per_session <= 0:
        raise ValueError(f"max_entries_per_session must be positive, got {max_entries_per_session}")


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
    tag = hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()[:12]
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
    max_entries_per_session: int = 1024,
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
        return placeholder

    sess_keys = [k for k in entries if k[0] == session_id]
    per_sess_cap = min(max_entries, max_entries_per_session)
    while len(sess_keys) >= per_sess_cap:
        evict_key = sess_keys.pop(0)
        entries.pop(evict_key, None)

    has_other_sessions = any(k[0] != session_id for k in entries)
    fair_share = max(1, max_entries // 2)
    while len(entries) >= max_entries:
        if has_other_sessions and len(sess_keys) >= fair_share:
            evict_key = sess_keys.pop(0)
            entries.pop(evict_key, None)
        else:
            evicted_k, _ = entries.popitem(last=False)
            if evicted_k[0] == session_id and evicted_k in sess_keys:
                sess_keys.remove(evicted_k)

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
    on_touch: Callable[[VaultKey], None] | None = None,
    on_expire: Callable[[VaultKey], None] | None = None,
) -> tuple[str, int, bool]:
    """Return ``(restored_text, replacement_count, dict_mutated)``."""
    if PLACEHOLDER_PREFIX not in text:
        return text, 0, False

    restored_count = 0
    mutated = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal restored_count, mutated
        placeholder = match.group(0)
        key: VaultKey = (session_id, placeholder)
        existed = key in entries
        value, did_mutate = _retrieve_from_dict(
            entries,
            aesgcm,
            session_id,
            placeholder,
            now=now,
        )
        if did_mutate:
            mutated = True
            if value is None and existed and on_expire is not None:
                on_expire(key)
            elif value is not None and on_touch is not None:
                on_touch(key)
        if value is None:
            return placeholder
        restored_count += 1
        return value

    restored = _PLACEHOLDER_PATTERN.sub(_replace, text)
    return restored, restored_count, mutated


class InMemoryAesVault:
    """Thread-safe RAM-only AES-256-GCM vault with O(1) lazy TTL and per-session LRU.

    Implements the ``VaultBackend`` protocol without persisting secrets to disk.
    """

    def __init__(
        self,
        ttl_seconds: int = 900,
        max_entries: int = 10_000,
        time_fn: Callable[[], float] | None = None,
        *,
        max_entries_per_session: int = 1024,
    ) -> None:
        _validate_vault_params(ttl_seconds, max_entries, max_entries_per_session)
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._max_entries_per_session = max_entries_per_session
        self._time_fn: Callable[[], float] = time_fn if time_fn is not None else time.monotonic
        self._lock = threading.RLock()
        self._aes_key = os.urandom(32)
        self._hmac_key = os.urandom(32)
        self._aesgcm = AESGCM(self._aes_key)
        self._entries: OrderedDict[VaultKey, _VaultEntry] = OrderedDict()
        self._expiry_order: OrderedDict[VaultKey, float] = OrderedDict()
        self._session_index: dict[str, OrderedDict[str, None]] = {}
        self._last_sweep_at: float = 0.0

    def _remove_key_locked(self, key: VaultKey) -> None:
        self._entries.pop(key, None)
        self._expiry_order.pop(key, None)
        sid, ph = key
        sess_od = self._session_index.get(sid)
        if sess_od is not None:
            sess_od.pop(ph, None)
            if not sess_od:
                del self._session_index[sid]

    def _touch_key_locked(self, key: VaultKey) -> None:
        sid, ph = key
        sess_od = self._session_index.get(sid)
        if sess_od is not None and ph in sess_od:
            sess_od.move_to_end(ph)

    def _lazy_evict_expired_locked(self, now: float) -> None:
        """Evict expired entries from the monotonic expiry queue in O(1) amortized time."""
        while self._expiry_order:
            oldest_key, oldest_exp = next(iter(self._expiry_order.items()))
            if oldest_exp <= now:
                self._remove_key_locked(oldest_key)
            else:
                break

        # Guard against external mutation of _entries in white-box unit tests
        if now - self._last_sweep_at >= _LAZY_SWEEP_INTERVAL_SECONDS and len(
            self._expiry_order
        ) != len(self._entries):
            self._last_sweep_at = now
            self.cleanup_expired()

    def store(self, session_id: str, category: EntityCategory, raw_value: str) -> str:
        """Encrypt and store ``raw_value``, returning a deterministic session placeholder."""
        with self._lock:
            now = self._time_fn()
            self._lazy_evict_expired_locked(now)

            placeholder = _make_placeholder(self._hmac_key, session_id, category, raw_value)
            key: VaultKey = (session_id, placeholder)
            expires_at = now + self._ttl_seconds
            entry = _encrypt_entry(
                self._aesgcm,
                session_id,
                placeholder,
                raw_value,
                expires_at=expires_at,
            )

            sess_od = self._session_index.get(session_id)
            if key in self._entries:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                self._expiry_order[key] = expires_at
                self._expiry_order.move_to_end(key)
                if sess_od is None:
                    sess_od = OrderedDict()
                    self._session_index[session_id] = sess_od
                sess_od[placeholder] = None
                sess_od.move_to_end(placeholder)
                return placeholder

            per_sess_cap = min(self._max_entries, self._max_entries_per_session)
            if sess_od is not None:
                while len(sess_od) >= per_sess_cap:
                    evicted_ph, _ = sess_od.popitem(last=False)
                    self._entries.pop((session_id, evicted_ph), None)
                    self._expiry_order.pop((session_id, evicted_ph), None)
                if not sess_od:
                    del self._session_index[session_id]
                    sess_od = None

            fair_share = max(1, self._max_entries // 2)
            while len(self._entries) >= self._max_entries:
                sess_od = self._session_index.get(session_id)
                has_other_sessions = (
                    len(self._session_index) > 1
                    if session_id in self._session_index
                    else len(self._session_index) >= 1
                )
                if sess_od is not None and has_other_sessions and len(sess_od) >= fair_share:
                    evicted_ph, _ = sess_od.popitem(last=False)
                    self._entries.pop((session_id, evicted_ph), None)
                    self._expiry_order.pop((session_id, evicted_ph), None)
                    if not sess_od:
                        del self._session_index[session_id]
                else:
                    oldest_k, _ = self._entries.popitem(last=False)
                    self._remove_key_locked(oldest_k)

            self._entries[key] = entry
            self._expiry_order[key] = expires_at
            sess_od = self._session_index.setdefault(session_id, OrderedDict())
            sess_od[placeholder] = None
            return placeholder

    def retrieve(self, session_id: str, placeholder: str) -> str | None:
        """Decrypt and return the original value for ``(session_id, placeholder)``."""
        with self._lock:
            key: VaultKey = (session_id, placeholder)
            existed = key in self._entries
            value, mutated = _retrieve_from_dict(
                self._entries,
                self._aesgcm,
                session_id,
                placeholder,
                now=self._time_fn(),
            )
            if mutated:
                if value is None and existed:
                    self._remove_key_locked(key)
                elif value is not None:
                    self._touch_key_locked(key)
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
                on_touch=self._touch_key_locked,
                on_expire=self._remove_key_locked,
            )
            return restored, count

    def purge_session(self, session_id: str) -> int:
        """Delete all entries belonging to ``session_id`` and return count removed."""
        with self._lock:
            sess_od = self._session_index.pop(session_id, None)
            removed = 0
            if sess_od is not None:
                for ph in sess_od:
                    k: VaultKey = (session_id, ph)
                    if self._entries.pop(k, None) is not None:
                        removed += 1
                    self._expiry_order.pop(k, None)
            extra = _purge_session_entries(self._entries, session_id)
            if extra:
                for k in list(self._expiry_order):
                    if k[0] == session_id:
                        del self._expiry_order[k]
                removed += extra
            return removed

    def cleanup_expired(self) -> int:
        """Purge all expired vault entries and return count removed."""
        with self._lock:
            now = self._time_fn()
            self._last_sweep_at = now
            expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired_keys:
                self._remove_key_locked(key)
            return len(expired_keys)

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


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_protect(plaintext: bytes) -> bytes | None:
    """Encrypt key bytes using Windows DPAPI ``CryptProtectData`` when available."""
    if sys.platform != "win32":
        return None
    try:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_wchar_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_DataBlob),
        ]
        crypt32.CryptProtectData.restype = ctypes.c_int
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p

        in_buf = ctypes.create_string_buffer(plaintext, len(plaintext))
        in_blob = _DataBlob(len(plaintext), in_buf)
        entropy_bytes = b"pokrov:local-vault:dpapi:v2"
        ent_buf = ctypes.create_string_buffer(entropy_bytes, len(entropy_bytes))
        ent_blob = _DataBlob(len(entropy_bytes), ent_buf)
        out_blob = _DataBlob()

        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "PokrovLocalVaultKey",
            ctypes.byref(ent_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            return None
        try:
            protected = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return _DPAPI_MAGIC + protected
        finally:
            kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))
    except (AttributeError, OSError):
        return None


def _dpapi_unprotect(blob: bytes) -> bytes:
    """Decrypt DPAPI-wrapped master key bytes."""
    if sys.platform != "win32":
        raise ValueError("Cannot unwrap Windows DPAPI keyfile on non-Windows platform")
    payload = blob[len(_DPAPI_MAGIC) :]
    if not payload:
        raise ValueError("Corrupted empty DPAPI keyfile payload")

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    in_buf = ctypes.create_string_buffer(payload, len(payload))
    in_blob = _DataBlob(len(payload), in_buf)
    entropy_bytes = b"pokrov:local-vault:dpapi:v2"
    ent_buf = ctypes.create_string_buffer(entropy_bytes, len(entropy_bytes))
    ent_blob = _DataBlob(len(entropy_bytes), ent_buf)
    out_blob = _DataBlob()

    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        ctypes.byref(ent_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise ValueError("Failed to decrypt Windows DPAPI vault keyfile")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))


def _derive_keyfile_path(vault_path: Path) -> Path:
    if vault_path.suffix == ".key":
        return Path(f"{vault_path}.master.key")
    return vault_path.with_suffix(".key")


def _derive_lockfile_path(vault_path: Path) -> Path:
    return Path(f"{vault_path}.lock")


def _chmod_restrictive(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


@contextlib.contextmanager
def _interprocess_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an OS-level advisory lock across concurrent CLI hook processes."""
    parent = lock_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(parent, 0o700)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if sys.platform == "win32":
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.005)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                with contextlib.suppress(OSError):
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


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

    Implements the ``VaultBackend`` protocol. Encrypts the entire vault metadata
    and entry table inside a single outer AES-256-GCM envelope on disk, protects
    the sibling ``.key`` file with Windows DPAPI (when available) or ``0o600``
    permissions, and coordinates concurrent processes via OS advisory file locks.
    """

    def __init__(
        self,
        local_vault_path: str | Path = ".pokrov/vault.enc",
        ttl_seconds: int = 900,
        max_entries: int = 10_000,
        time_fn: Callable[[], float] | None = None,
        *,
        max_entries_per_session: int = 1024,
    ) -> None:
        _validate_vault_params(ttl_seconds, max_entries, max_entries_per_session)
        self._vault_path = Path(local_vault_path)
        self._key_path = _derive_keyfile_path(self._vault_path)
        self._lock_path = _derive_lockfile_path(self._vault_path)
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._max_entries_per_session = max_entries_per_session
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

    def _read_master_key_locked(self) -> bytes:
        raw_bytes = self._key_path.read_bytes()
        if raw_bytes.startswith(_DPAPI_MAGIC):
            master_key = _dpapi_unprotect(raw_bytes)
        else:
            master_key = raw_bytes
        if len(master_key) != 32:
            raise ValueError(f"Invalid master key length in {self._key_path}: {len(master_key)}")
        return master_key

    def _derive_keys(self, master_key: bytes) -> tuple[AESGCM, AESGCM, bytes]:
        if len(master_key) != 32:
            raise ValueError(f"Invalid master key length in {self._key_path}: {len(master_key)}")
        entry_aes_key = hmac.new(master_key, _AES_KEY_DOMAIN, hashlib.sha256).digest()
        outer_aes_key = hmac.new(master_key, _OUTER_AES_KEY_DOMAIN, hashlib.sha256).digest()
        hmac_key = hmac.new(master_key, _HMAC_KEY_DOMAIN, hashlib.sha256).digest()
        return AESGCM(entry_aes_key), AESGCM(outer_aes_key), hmac_key

    def _ensure_keys_locked(self) -> tuple[AESGCM, AESGCM, bytes]:
        if self._key_path.is_file():
            return self._derive_keys(self._read_master_key_locked())

        master_key = os.urandom(32)
        dpapi_blob = _dpapi_protect(master_key)
        persisted_key = dpapi_blob if dpapi_blob is not None else master_key
        _write_bytes_atomic(self._key_path, persisted_key)
        return self._derive_keys(master_key)

    def _load_existing_locked(
        self,
    ) -> tuple[AESGCM, AESGCM, bytes, OrderedDict[VaultKey, _VaultEntry]] | None:
        if not self._vault_path.is_file():
            return None
        if not self._key_path.is_file():
            raise RuntimeError(
                f"Vault keyfile missing for existing encrypted vault: {self._key_path}"
            )

        entry_aesgcm, outer_aesgcm, hmac_key = self._derive_keys(self._read_master_key_locked())
        raw_json = self._vault_path.read_text(encoding="utf-8")
        try:
            envelope = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unsupported encrypted vault format in {self._vault_path}") from exc

        if not isinstance(envelope, dict) or envelope.get("version") != _VAULT_FORMAT_VERSION:
            raise ValueError(f"Unsupported encrypted vault format in {self._vault_path}")

        raw_outer_nonce = envelope.get("nonce")
        raw_outer_ct = envelope.get("ciphertext")
        if not isinstance(raw_outer_nonce, str) or not isinstance(raw_outer_ct, str):
            raise ValueError(f"Invalid outer envelope fields in {self._vault_path}")

        try:
            outer_nonce = base64.b64decode(raw_outer_nonce, validate=True)
            outer_ct = base64.b64decode(raw_outer_ct, validate=True)
            inner_bytes = outer_aesgcm.decrypt(outer_nonce, outer_ct, _OUTER_ENVELOPE_AAD)
            raw_entries = json.loads(inner_bytes.decode("utf-8"))
        except (ValueError, InvalidTag) as exc:
            raise ValueError(f"Corrupted or tampered vault envelope in {self._vault_path}") from exc

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

        return entry_aesgcm, outer_aesgcm, hmac_key, entries

    def _save_entries_locked(
        self,
        outer_aesgcm: AESGCM,
        entries: OrderedDict[VaultKey, _VaultEntry],
    ) -> None:
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
        inner_bytes = json.dumps(
            serialized_entries,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        outer_nonce = os.urandom(12)
        outer_ct = outer_aesgcm.encrypt(outer_nonce, inner_bytes, _OUTER_ENVELOPE_AAD)
        envelope = {
            "version": _VAULT_FORMAT_VERSION,
            "nonce": base64.b64encode(outer_nonce).decode("ascii"),
            "ciphertext": base64.b64encode(outer_ct).decode("ascii"),
        }
        encoded = json.dumps(envelope, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        _write_bytes_atomic(self._vault_path, encoded)

    def store(self, session_id: str, category: EntityCategory, raw_value: str) -> str:
        """Encrypt and persist ``raw_value``, returning a deterministic session placeholder."""
        with self._lock, _interprocess_file_lock(self._lock_path):
            loaded = self._load_existing_locked()
            if loaded is None:
                entry_aesgcm, outer_aesgcm, hmac_key = self._ensure_keys_locked()
                entries: OrderedDict[VaultKey, _VaultEntry] = OrderedDict()
            else:
                entry_aesgcm, outer_aesgcm, hmac_key, entries = loaded

            placeholder = _store_in_dict(
                entries,
                entry_aesgcm,
                hmac_key,
                session_id,
                category,
                raw_value,
                now=self._time_fn(),
                ttl_seconds=self._ttl_seconds,
                max_entries=self._max_entries,
                max_entries_per_session=self._max_entries_per_session,
            )
            self._save_entries_locked(outer_aesgcm, entries)
            return placeholder

    def retrieve(self, session_id: str, placeholder: str) -> str | None:
        """Load vault from disk and decrypt ``(session_id, placeholder)`` if valid."""
        with self._lock:
            if not self._vault_path.is_file():
                return None
            with _interprocess_file_lock(self._lock_path):
                loaded = self._load_existing_locked()
                if loaded is None:
                    return None
                entry_aesgcm, outer_aesgcm, _, entries = loaded
                value, mutated = _retrieve_from_dict(
                    entries,
                    entry_aesgcm,
                    session_id,
                    placeholder,
                    now=self._time_fn(),
                )
                if mutated:
                    self._save_entries_locked(outer_aesgcm, entries)
                return value

    def restore_text(self, session_id: str, text: str) -> tuple[str, int]:
        """Load vault from disk and unmask placeholders in ``text``."""
        if PLACEHOLDER_PREFIX not in text:
            return text, 0
        with self._lock:
            if not self._vault_path.is_file():
                return text, 0
            with _interprocess_file_lock(self._lock_path):
                loaded = self._load_existing_locked()
                if loaded is None:
                    return text, 0
                entry_aesgcm, outer_aesgcm, _, entries = loaded
                restored, count, mutated = _restore_text_from_dict(
                    entries,
                    entry_aesgcm,
                    session_id,
                    text,
                    now=self._time_fn(),
                )
                if mutated:
                    self._save_entries_locked(outer_aesgcm, entries)
                return restored, count

    def purge_session(self, session_id: str) -> int:
        """Remove all entries for ``session_id`` from persistent encrypted storage."""
        with self._lock:
            if not self._vault_path.is_file():
                return 0
            with _interprocess_file_lock(self._lock_path):
                loaded = self._load_existing_locked()
                if loaded is None:
                    return 0
                _, outer_aesgcm, _, entries = loaded
                removed = _purge_session_entries(entries, session_id)
                if removed > 0:
                    self._save_entries_locked(outer_aesgcm, entries)
                return removed

    def cleanup_expired(self) -> int:
        """Remove expired entries from persistent encrypted storage."""
        with self._lock:
            if not self._vault_path.is_file():
                return 0
            with _interprocess_file_lock(self._lock_path):
                loaded = self._load_existing_locked()
                if loaded is None:
                    return 0
                _, outer_aesgcm, _, entries = loaded
                removed = _cleanup_expired_entries(entries, self._time_fn())
                if removed > 0:
                    self._save_entries_locked(outer_aesgcm, entries)
                return removed

    def __len__(self) -> int:
        with self._lock:
            if not self._vault_path.is_file():
                return 0
            with _interprocess_file_lock(self._lock_path):
                loaded = self._load_existing_locked()
                if loaded is None:
                    return 0
                return len(loaded[3])

    def __repr__(self) -> str:
        return (
            f"LocalEncryptedVault(local_vault_path={str(self._vault_path)!r}, "
            f"ttl_seconds={self._ttl_seconds}, max_entries={self._max_entries})"
        )


__all__ = ["InMemoryAesVault", "LocalEncryptedVault"]
