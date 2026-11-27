"""Command-line interface entrypoint for Pokrov (Покров)."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pokrov import APP_NAME, __version__
from pokrov.audit.report_docx import DocxComplianceReportGenerator
from pokrov.audit.report_html import HtmlComplianceReportGenerator
from pokrov.audit.siem import PokrovSiemExporter
from pokrov.audit.store import SqliteAuditStore
from pokrov.core.config import PokrovConfig, load_config
from pokrov.proxy.app import create_default_app

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser and subcommands."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=(
            "Покров (Pokrov) — Open-Source AI Security Gateway "
            "(LLM Firewall, MCP Agent Firewall, Universal IDE Guard, AI Infra Auditor)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to policy YAML file (defaults to POKROV_POLICY_PATH or built-in defaults).",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    subparsers.add_parser("version", help="Print Pokrov version and exit.")

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP LLM Security Gateway.")
    serve_parser.add_argument("--host", default=None, help="Bind address override.")
    serve_parser.add_argument("--port", type=int, default=None, help="Bind port override.")
    serve_parser.add_argument("--profile", default=None, help="Active policy profile override.")

    mcp_parser = subparsers.add_parser(
        "mcp", help="[planned] Run the MCP Security Proxy (Phase 4)."
    )
    mcp_parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="MCP transport mode.",
    )
    mcp_parser.add_argument(
        "target",
        nargs=argparse.REMAINDER,
        help="Wrapped MCP server command (after --).",
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="[planned] Scan AI infrastructure configs and model artifacts (Phase 6).",
    )
    audit_parser.add_argument("path", nargs="?", default=".", help="Target directory or file.")
    audit_parser.add_argument(
        "--format",
        choices=("text", "json", "sarif"),
        default="text",
        help="Output report format.",
    )

    bench_parser = subparsers.add_parser(
        "bench", help="[planned] Run synthetic evaluation benchmarks (Phase 7)."
    )
    bench_parser.add_argument("--output", default=None, help="Optional report output path.")

    report_parser = subparsers.add_parser(
        "report", help="Export audit reports (DOCX / HTML / CEF / JSONL)."
    )
    report_parser.add_argument(
        "--format",
        choices=("docx", "html", "cef", "jsonl"),
        default="docx",
        help="Export format.",
    )
    report_parser.add_argument("--output", required=False, default=None, help="Output file path.")

    protect_parser = subparsers.add_parser(
        "protect",
        help="[planned] Sync IDE ignore rules and sanitize sensitive workspace files (Phase 5).",
    )
    protect_parser.add_argument("path", nargs="?", default=".", help="Workspace root directory.")

    watch_parser = subparsers.add_parser(
        "watch",
        help="[planned] Watch workspace files for sensitive data changes (Phase 5).",
    )
    watch_parser.add_argument("path", nargs="?", default=".", help="Workspace root directory.")

    restore_parser = subparsers.add_parser(
        "restore",
        help="[planned] Restore tokenized placeholders from local IDE Guard vault (Phase 5).",
    )
    restore_parser.add_argument("path", help="File containing placeholders to restore.")

    hook_parser = subparsers.add_parser(
        "hook",
        help="[planned] Handle IDE PreToolUse / beforeReadFile JSON hook input (Phase 5).",
    )
    hook_parser.add_argument(
        "--ide",
        choices=("claude", "cursor", "windsurf", "codex"),
        default="claude",
        help="IDE hook adapter dialect.",
    )

    return parser


def _apply_serve_overrides(
    cfg: PokrovConfig,
    *,
    host: str | None,
    port: int | None,
    profile: str | None,
) -> PokrovConfig:
    updates: dict[str, object] = {}
    if profile:
        cfg.get_profile(profile)
        updates["active_profile"] = profile
    if host is not None or port is not None:
        server_updates: dict[str, object] = {}
        if host is not None:
            server_updates["host"] = host
        if port is not None:
            server_updates["port"] = port
        updates["server"] = cfg.server.model_copy(update=server_updates)
    return cfg.model_copy(update=updates) if updates else cfg


def _resolve_serve_runtime_options(cfg: PokrovConfig) -> tuple[int, int]:
    """Resolve effective Uvicorn ``(workers, timeout_keep_alive)`` from ``PokrovConfig``."""
    workers = 1 if cfg.vault.mode == "memory" else cfg.server.workers
    if cfg.vault.mode == "memory" and cfg.server.workers > 1:
        logger.warning(
            "Forcing workers=1 because vault.mode='memory' stores reversible placeholders "
            "per-process in RAM (configured server.workers=%d; switch to "
            "vault.mode='local_encrypted' for multi-worker deployments).",
            cfg.server.workers,
        )
    timeout_keep_alive = max(1, math.ceil(cfg.server.request_timeout_seconds))
    return workers, timeout_keep_alive


def _run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = load_config(args.config)
    cfg = _apply_serve_overrides(
        cfg,
        host=args.host,
        port=args.port,
        profile=args.profile,
    )
    workers, timeout_keep_alive = _resolve_serve_runtime_options(cfg)
    if workers > 1:
        if args.config:
            os.environ["POKROV_POLICY_PATH"] = str(args.config)
        os.environ["POKROV_PROFILE"] = cfg.active_profile
        uvicorn.run(
            "pokrov.proxy.app:create_default_app",
            factory=True,
            host=cfg.server.host,
            port=cfg.server.port,
            workers=workers,
            timeout_keep_alive=timeout_keep_alive,
            log_level=cfg.server.log_level.lower(),
        )
    else:
        app = create_default_app(cfg)
        uvicorn.run(
            app,
            host=cfg.server.host,
            port=cfg.server.port,
            workers=workers,
            timeout_keep_alive=timeout_keep_alive,
            log_level=cfg.server.log_level.lower(),
        )
    return 0


def _run_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = SqliteAuditStore(cfg.audit)
    try:
        events = store.query(limit=10_000)
        stats = store.stats()
    finally:
        store.close()

    fmt: str = args.format
    if fmt == "docx":
        payload = DocxComplianceReportGenerator().render_bytes(events, stats)
        out_path = Path(args.output) if args.output else Path("pokrov-report.docx")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        print(f"Report written to {out_path} ({len(payload)} bytes)")
        return 0

    if fmt == "html":
        payload = HtmlComplianceReportGenerator().render_bytes(events, stats)
    else:
        exporter = PokrovSiemExporter(cfg.siem)
        lines = [
            exporter.format_cef(ev) if fmt == "cef" else exporter.format_jsonl(ev) for ev in events
        ]
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        print(f"Report written to {out_path} ({len(payload)} bytes)")
    else:
        sys.stdout.write(payload.decode("utf-8"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        print(f"{APP_NAME} {__version__}")
        return 0

    if args.command == "serve":
        return _run_serve(args)

    if args.command == "report":
        return _run_report(args)

    print(
        f"Command '{args.command}' is [planned] (Phases 4-7) and will be "
        "wired in its implementation phase.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
