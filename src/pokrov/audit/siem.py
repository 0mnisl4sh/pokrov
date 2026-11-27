"""SIEM formatters (CEF, RFC 5424 Syslog, JSONL) and async transport exporter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import ssl
import sys
import threading
from datetime import UTC
from pathlib import Path
from typing import Final

from pokrov import __version__
from pokrov.core.config import SiemConfig
from pokrov.core.contracts import AuditEvent, Severity

_CEF_VENDOR: Final[str] = "Pokrov"
_CEF_PRODUCT: Final[str] = "AI Security Gateway"
_NETWORK_TIMEOUT_SECONDS: Final[float] = 5.0

_CEF_SEVERITY: Final[dict[Severity, str]] = {
    Severity.LOW: "3",
    Severity.MEDIUM: "5",
    Severity.HIGH: "8",
    Severity.CRITICAL: "10",
}

_SYSLOG_SEVERITY: Final[dict[Severity, int]] = {
    Severity.CRITICAL: 2,
    Severity.HIGH: 3,
    Severity.MEDIUM: 4,
    Severity.LOW: 6,
}


def _escape_cef_header(value: str) -> str:
    """Escape a CEF header field (``\\``, ``|``, and ASCII/Unicode line breaks)."""
    cleaned = (
        value.replace("\x00", "")
        .replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\x85", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
    )
    return cleaned.replace("\\", "\\\\").replace("|", "\\|")


def _escape_cef_ext(value: str) -> str:
    """Escape a CEF extension value (``\\``, ``=``, and ASCII/Unicode line breaks)."""
    cleaned = value.replace("\x00", "")
    return (
        cleaned.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\x85", "\\x85")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _syslog_token(value: str, *, max_len: int, fallback: str = "-") -> str:
    """Sanitize a value to RFC 5424 PRINTUSASCII token without whitespace."""
    ascii_chars = [ch for ch in value if 33 <= ord(ch) <= 126]
    token = "".join(ascii_chars)[:max_len]
    return token or fallback


def _to_utc(event: AuditEvent) -> tuple[int, str]:
    """Return ``(epoch_ms, rfc5424_utc_str)`` for an ``AuditEvent``."""
    ts = event.timestamp
    ts_utc = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    return int(ts_utc.timestamp() * 1000), ts_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class PokrovSiemExporter:
    """SIEM formatter and async event shipper implementing ``SiemExporter``.

    Produces CEF 0 (compatible with Kaspersky KUMA, MaxPatrol SIEM, and RuSIEM),
    RFC 5424 Syslog wrapping CEF, and compact single-line JSONL.
    """

    def __init__(
        self,
        config: SiemConfig | None = None,
        *,
        ssl_context: ssl.SSLContext | None = None,
        timeout_seconds: float = _NETWORK_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._config = config or SiemConfig()
        self._ssl_context = ssl_context
        self._timeout_seconds = float(timeout_seconds)
        self._io_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._stream_use_tls: bool | None = None
        self._stream_loop: asyncio.AbstractEventLoop | None = None
        try:
            self._hostname = socket.gethostname()
        except OSError:
            self._hostname = "pokrov-host"

    def format_cef(self, event: AuditEvent) -> str:
        """Format ``event`` as a single-line CEF 0 string."""
        sig_id = f"{event.control_point.value}:{event.action.value}"
        name = event.reason.strip() or f"Pokrov {event.action.value} at {event.control_point.value}"
        header = "|".join(
            [
                "CEF:0",
                _escape_cef_header(_CEF_VENDOR),
                _escape_cef_header(_CEF_PRODUCT),
                _escape_cef_header(__version__),
                _escape_cef_header(sig_id),
                _escape_cef_header(name),
                _CEF_SEVERITY[event.severity],
            ]
        )

        rt_ms, _ = _to_utc(event)
        cats = ",".join(cat.value for cat in event.categories) or "none"
        compliance = (
            ";".join(f"{ref.framework} {ref.article}" for ref in event.compliance_refs) or "none"
        )
        hmacs = ",".join(event.value_hmacs) or "none"
        detectors = ",".join(event.detector_names) or "none"
        latency_str = str(round(float(event.latency_ms), 3))

        ext_parts: list[str] = [
            f"rt={rt_ms}",
            f"externalId={_escape_cef_ext(event.event_id)}",
            f"act={_escape_cef_ext(event.action.value)}",
            f"cat={_escape_cef_ext(cats)}",
            f"suser={_escape_cef_ext(event.principal or 'anonymous')}",
            f"dhost={_escape_cef_ext(event.upstream or 'local')}",
            "cs1Label=Compliance",
            f"cs1={_escape_cef_ext(compliance)}",
            "cn1Label=LatencyMs",
            f"cn1={latency_str}",
            "cs2Label=ControlPoint",
            f"cs2={_escape_cef_ext(event.control_point.value)}",
            "cs3Label=RequestId",
            f"cs3={_escape_cef_ext(event.request_id)}",
            "cs4Label=PolicyProfile",
            f"cs4={_escape_cef_ext(event.policy_profile or 'default')}",
            "cs5Label=ValueHmacs",
            f"cs5={_escape_cef_ext(hmacs)}",
            "cs6Label=Detectors",
            f"cs6={_escape_cef_ext(detectors)}",
        ]
        if event.tool_name:
            ext_parts.append(f"sproc={_escape_cef_ext(event.tool_name)}")
        if event.reason:
            ext_parts.append(f"msg={_escape_cef_ext(event.reason)}")

        return f"{header}|{' '.join(ext_parts)}"

    def format_syslog(self, event: AuditEvent) -> str:
        """Format ``event`` as an RFC 5424 Syslog message wrapping CEF."""
        pri = self._config.facility * 8 + _SYSLOG_SEVERITY[event.severity]
        _, ts_str = _to_utc(event)
        hostname = _syslog_token(self._hostname, max_len=255, fallback="pokrov-host")
        app_name = _syslog_token(self._config.app_name, max_len=48, fallback="pokrov")
        procid = _syslog_token(str(os.getpid()), max_len=128, fallback="-")
        msgid = _syslog_token(event.control_point.value, max_len=32, fallback="-")
        cef = self.format_cef(event)
        return f"<{pri}>1 {ts_str} {hostname} {app_name} {procid} {msgid} - {cef}"

    def format_jsonl(self, event: AuditEvent) -> str:
        """Format ``event`` as a compact single-line JSON string escaping Unicode line breaks."""
        raw = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            raw.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085")
        )

    def format_event(self, event: AuditEvent) -> str:
        """Format ``event`` according to ``config.format``."""
        fmt = self._config.format
        if fmt == "cef":
            return self.format_cef(event)
        if fmt == "syslog":
            return self.format_syslog(event)
        if fmt == "jsonl":
            return self.format_jsonl(event)
        raise ValueError(f"Unsupported SIEM format: {fmt!r}")

    async def send(self, event: AuditEvent) -> bool:
        """Ship ``event`` asynchronously via the configured SIEM transport.

        Returns ``False`` when SIEM shipping is disabled or on transport errors;
        never raises uncaught transport exceptions.
        """
        if not self._config.enabled:
            return False

        try:
            payload = self.format_event(event)
            transport = self._config.transport
            if transport == "stdout":
                self._send_stdout(payload)
            elif transport == "file":
                await asyncio.to_thread(self._send_file, payload)
            elif transport == "udp":
                await asyncio.to_thread(self._send_udp, payload)
            elif transport == "tcp":
                await self._send_tcp_or_tls(payload, use_tls=False)
            elif transport == "tls":
                await self._send_tcp_or_tls(payload, use_tls=True)
            else:
                return False
        except (OSError, TimeoutError, ssl.SSLError, ValueError, RuntimeError):
            return False
        return True

    async def aclose(self) -> None:
        """Close any persistent async TCP/TLS connection cleanly."""
        async with self._async_lock:
            await self._close_stream_unlocked()

    def _send_stdout(self, payload: str) -> None:
        with self._io_lock:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()

    def _send_file(self, payload: str) -> None:
        target = Path(self._config.file_path)
        encoded = (payload + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        with self._io_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, flags, 0o600)
            try:
                if os.name == "posix":
                    with contextlib.suppress(OSError):
                        os.chmod(target, 0o600)
                os.write(fd, encoded)
            finally:
                os.close(fd)

    def _send_udp(self, payload: str) -> None:
        data = payload.encode("utf-8")
        addr_infos = socket.getaddrinfo(
            self._config.host,
            self._config.port,
            type=socket.SOCK_DGRAM,
        )
        if not addr_infos:
            raise OSError(f"Could not resolve UDP target {self._config.host}:{self._config.port}")

        last_exc: OSError | None = None
        for family, socktype, proto, _, sockaddr in addr_infos:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(self._timeout_seconds)
                sock.sendto(data, sockaddr)
                return
            except OSError as exc:
                last_exc = exc
            finally:
                sock.close()
        if last_exc is not None:
            raise last_exc

    def _is_stream_usable(
        self,
        *,
        use_tls: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        return (
            self._writer is not None
            and not self._writer.is_closing()
            and self._reader is not None
            and not self._reader.at_eof()
            and self._stream_use_tls == use_tls
            and self._stream_loop is loop
        )

    async def _open_stream_unlocked(
        self,
        *,
        use_tls: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.StreamWriter:
        ctx: ssl.SSLContext | None = None
        if use_tls:
            ctx = self._ssl_context or ssl.create_default_context()

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._config.host, self._config.port, ssl=ctx),
            timeout=self._timeout_seconds,
        )
        self._reader = reader
        self._writer = writer
        self._stream_use_tls = use_tls
        self._stream_loop = loop
        return writer

    async def _close_stream_unlocked(self) -> None:
        writer = self._writer
        stream_loop = self._stream_loop
        self._reader = None
        self._writer = None
        self._stream_use_tls = None
        self._stream_loop = None
        if writer is None:
            return
        with contextlib.suppress(OSError, RuntimeError):
            writer.close()
        if stream_loop is not None and not stream_loop.is_closed():
            with contextlib.suppress(OSError, TimeoutError, ssl.SSLError, RuntimeError):
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)

    async def _write_and_drain(self, writer: asyncio.StreamWriter, data: bytes) -> None:
        writer.write(data)
        await asyncio.wait_for(writer.drain(), timeout=self._timeout_seconds)

    async def _send_tcp_or_tls(self, payload: str, *, use_tls: bool) -> None:
        data = (payload + "\n").encode("utf-8")
        loop = asyncio.get_running_loop()

        async with self._async_lock:
            reused = self._is_stream_usable(use_tls=use_tls, loop=loop)
            if not reused or self._writer is None:
                reused = False
                await self._close_stream_unlocked()
                writer = await self._open_stream_unlocked(use_tls=use_tls, loop=loop)
            else:
                writer = self._writer

            try:
                await self._write_and_drain(writer, data)
                return
            except (OSError, TimeoutError, ssl.SSLError, RuntimeError):
                await self._close_stream_unlocked()
                if not reused:
                    raise

            fresh_writer = await self._open_stream_unlocked(use_tls=use_tls, loop=loop)
            try:
                await self._write_and_drain(fresh_writer, data)
            except (OSError, TimeoutError, ssl.SSLError, RuntimeError):
                await self._close_stream_unlocked()
                raise


__all__ = ["PokrovSiemExporter"]
