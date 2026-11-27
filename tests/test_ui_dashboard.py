"""Unit, E2E, air-gap, XSS-safety, and Zero-Retention tests for Pokrov /ui dashboard."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from pokrov.audit.store import SqliteAuditStore
from pokrov.core.config import PokrovConfig, RouterConfig, ServerConfig, UpstreamConfig, load_config
from pokrov.core.contracts import Action, EntityCategory, ProviderType
from pokrov.proxy import create_default_app

UI_STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "pokrov" / "ui" / "static"


def _tok(*parts: str) -> str:
    """Assemble synthetic secrets at runtime (avoids GH013 push-protection triggers)."""
    return "".join(parts)


def _make_ui_config(
    tmp_path: Path,
    *,
    enable_ui: bool = True,
    audit_enabled: bool = True,
    allow_profile_override: bool = False,
    default_upstream: str = "mock-local",
    gateway_api_key_env: str = "",
    hmac_salt: str = "pokrov-ui-test-salt-2026",
) -> PokrovConfig:
    base = load_config(Path(__file__).resolve().parent.parent / "policy.example.yaml")
    upstreams = dict(base.router.upstreams)
    return base.model_copy(
        update={
            "active_profile": "balanced-152fz",
            "server": ServerConfig(
                enable_ui=enable_ui,
                allow_profile_override=allow_profile_override,
                gateway_api_key_env=gateway_api_key_env,
            ),
            "audit": base.audit.model_copy(
                update={
                    "enabled": audit_enabled,
                    "db_path": str(tmp_path / "ui_audit.db"),
                    "hmac_salt": hmac_salt,
                }
            ),
            "router": RouterConfig(
                default_upstream=default_upstream,
                local_upstream="mock-local",
                cloud_upstream="openai-cloud",
                upstreams=upstreams,
            ),
        }
    )


def test_ui_static_assets_airgapped_and_xss_safe() -> None:
    """UI HTML/CSS/JS must contain zero external URLs and zero dangerous DOM sinks."""
    index_html = (UI_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles_css = (UI_STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (UI_STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for name, content in (
        ("index.html", index_html),
        ("styles.css", styles_css),
        ("app.js", app_js),
    ):
        lowered = content.lower()
        assert "http://" not in lowered, f"External http:// URL found in {name}"
        assert "https://" not in lowered, f"External https:// URL found in {name}"
        assert "//cdn" not in lowered, f"Protocol-relative CDN found in {name}"
        assert "@import" not in lowered, f"CSS @import found in {name}"

    assert '<script src="/ui/static/app.js"></script>' in index_html
    assert '<link rel="stylesheet" href="/ui/static/styles.css">' in index_html

    for sink in (".innerhtml", ".outerhtml", "document.write", "eval("):
        assert sink not in app_js.lower(), f"Unsafe JS DOM sink {sink!r} found in app.js"


async def test_ui_html_static_security_headers_and_disabled_flag(tmp_path: Path) -> None:
    """Verify /ui, /ui/, static allowlist, CSP/headers, traversal 404, and enable_ui=False."""
    cfg = _make_ui_config(tmp_path, enable_ui=True)
    app = create_default_app(cfg)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://pokrov.local",
        ) as client,
    ):
        for route in ("/ui", "/ui/"):
            resp = await client.get(route)
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "default-src 'self'" in resp.headers["content-security-policy"]
            assert resp.headers["x-content-type-options"] == "nosniff"
            assert resp.headers["x-frame-options"] == "DENY"
            assert resp.headers["referrer-policy"] == "no-referrer"
            assert "Покров" in resp.text

        css_resp = await client.get("/ui/static/styles.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]

        js_resp = await client.get("/ui/static/app.js")
        assert js_resp.status_code == 200
        assert "javascript" in js_resp.headers["content-type"]

        for bad_path in (
            "/ui/static/unknown.js",
            "/ui/static/..%2Froutes.py",
            "/ui/static/index.html",
        ):
            bad_resp = await client.get(bad_path)
            assert bad_resp.status_code == 404

    disabled_cfg = _make_ui_config(tmp_path, enable_ui=False)
    disabled_app = create_default_app(disabled_cfg)
    async with (
        disabled_app.router.lifespan_context(disabled_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=disabled_app),
            base_url="http://pokrov.local",
        ) as disabled_client,
    ):
        assert (await disabled_client.get("/ui")).status_code == 404
        assert (await disabled_client.get("/api/stats")).status_code == 404


async def test_playground_presets_profiles_routing_and_zero_retention_e2e(
    tmp_path: Path,
) -> None:
    """Run all 4 presets, verify profiles & routing, then binary-scan DB and exports."""
    cfg = _make_ui_config(tmp_path, default_upstream="mock-local")
    app = create_default_app(cfg)

    raw_pii_and_secrets = [
        "4512 654321",
        "112-233-445 95",
        "7707083893",
        "+7 (916) 123-45-67",
        "4512 654З21",
        "0KHQndCY0JvQoSAxMTItMjMzLTQ0NSA5NQ==",
        _tok("sk-", "proj-", "PokrovDemoOpenAIKey1234567890abcd"),
        "postgres://demo_user:DemoPass99@db.internal:5432/prod",
        _tok(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            ".eyJzdWIiOiJkZW1vIn0",
            ".PokrovDemoSig12345",
        ),
    ]

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://pokrov.local",
        ) as client,
    ):
        presets_resp = await client.get("/api/playground/presets")
        assert presets_resp.status_code == 200
        presets = presets_resp.json()["presets"]
        assert [p["id"] for p in presets] == [
            "contract_pii",
            "evasion_homoglyph_b64",
            "ru_prompt_injection",
            "secrets_env_leak",
        ]

        # 1. Contract PII -> MASK_REVERSIBLE, masked_text has <PKR:...>, FSTEC/152-FZ refs
        r_contract = await client.post(
            "/api/playground",
            json={"text": presets[0]["text"], "profile": "balanced-152fz"},
        )
        assert r_contract.status_code == 200
        d_contract = r_contract.json()
        assert d_contract["action"] == Action.MASK_REVERSIBLE.value
        assert d_contract["blocked"] is False
        assert d_contract["masking_applied"] is True
        assert d_contract["replacements_count"] >= 4
        assert "<PKR:RU_PASSPORT:" in d_contract["masked_text"]
        assert "<PKR:RU_SNILS:" in d_contract["masked_text"]
        assert "4512 654321" not in d_contract["masked_text"]
        assert d_contract["router"]["decision"] == "local"
        assert d_contract["audit_recorded"] is True
        assert any("152-ФЗ" in ref["framework"] for ref in d_contract["compliance_refs"])
        assert any("ФСТЭК" in ref["framework"] for ref in d_contract["compliance_refs"])

        # 2. Homoglyph + Base64 -> evasion_detected=True, masked_text hides raw payload
        r_evasion = await client.post(
            "/api/playground",
            json={"text": presets[1]["text"], "profile": "balanced-152fz"},
        )
        assert r_evasion.status_code == 200
        d_evasion = r_evasion.json()
        assert d_evasion["evasion_detected"] is True
        assert d_evasion["action"] == Action.MASK_REVERSIBLE.value
        assert "4512 654З21" not in d_evasion["masked_text"]
        assert "0KHQndCY0JvQoSAxMTItMjMzLTQ0NSA5NQ==" not in d_evasion["masked_text"]
        assert any(sp["evasion_detected"] for sp in d_evasion["spans"])

        # 3. Russian Prompt Injection -> BLOCK, router.decision == "block"
        r_pi = await client.post(
            "/api/playground",
            json={"text": presets[2]["text"], "profile": "balanced-152fz"},
        )
        assert r_pi.status_code == 200
        d_pi = r_pi.json()
        assert d_pi["action"] == Action.BLOCK.value
        assert d_pi["blocked"] is True
        assert d_pi["router"]["decision"] == "block"
        assert d_pi["router"]["upstream"] == "none"
        assert d_pi["masked_text"] == ""
        assert EntityCategory.PROMPT_INJECTION.value in d_pi["categories"]

        # 4. Secrets & .env leak -> BLOCK
        r_sec = await client.post(
            "/api/playground",
            json={"text": presets[3]["text"], "profile": "balanced-152fz"},
        )
        assert r_sec.status_code == 200
        d_sec = r_sec.json()
        assert d_sec["action"] == Action.BLOCK.value
        assert d_sec["blocked"] is True
        assert EntityCategory.SECRET_KEY.value in d_sec["categories"]

        # Zero-Retention check on API playground spans JSON
        spans_json = json.dumps(
            [d_contract["spans"], d_evasion["spans"], d_pi["spans"], d_sec["spans"]],
            ensure_ascii=False,
        )
        for sensitive in raw_pii_and_secrets:
            assert sensitive not in spans_json

        # strict-fstec-117 on contract PII -> ROUTE_LOCAL
        r_strict = await client.post(
            "/api/playground",
            json={
                "text": "Паспорт 4512 654321 и СНИЛС 112-233-445 95",
                "profile": "strict-fstec-117",
                "record_audit": False,
            },
        )
        assert r_strict.status_code == 200
        d_strict = r_strict.json()
        assert d_strict["action"] == Action.ROUTE_LOCAL.value
        assert d_strict["router"]["decision"] == "local"
        assert d_strict["masking_applied"] is False
        assert d_strict["audit_recorded"] is False

        # dev-audit -> FLAG
        r_dev = await client.post(
            "/api/playground",
            json={"text": "Паспорт 4512 654321", "profile": "dev-audit", "record_audit": False},
        )
        assert r_dev.status_code == 200
        assert r_dev.json()["action"] == Action.FLAG.value

        # Clean text -> ALLOW
        r_clean = await client.post(
            "/api/playground",
            json={"text": "Обычный вопрос про погоду.", "record_audit": False},
        )
        assert r_clean.status_code == 200
        assert r_clean.json()["action"] == Action.ALLOW.value
        assert r_clean.json()["detections_count"] == 0

        # Unknown profile -> 400
        r_bad = await client.post(
            "/api/playground",
            json={"text": "Привет", "profile": "no-such-profile"},
        )
        assert r_bad.status_code == 400

        await app.state.audit_dispatcher.flush()

        # Stats, Events, and Reports endpoints
        stats_resp = await client.get("/api/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_requests"] == 4
        assert stats["blocked_count"] == 2
        assert stats["masked_count"] == 2
        assert stats["pii_leaks_stopped"] >= 2
        assert stats["injections_detected"] >= 1
        assert stats["secrets_detected"] >= 1
        assert stats["audit_enabled"] is True

        events_all = await client.get("/api/events")
        assert events_all.status_code == 200
        assert events_all.json()["count"] == 4

        events_blocked = await client.get("/api/events?action=block&since_hours=1")
        assert events_blocked.status_code == 200
        assert events_blocked.json()["count"] == 2

        events_cat = await client.get("/api/events?category=ru_passport")
        assert events_cat.status_code == 200
        assert events_cat.json()["count"] >= 2

        # Download DOCX / HTML / CEF / JSONL reports
        docx_resp = await client.get("/api/reports/docx")
        assert docx_resp.status_code == 200
        assert zipfile.is_zipfile(io.BytesIO(docx_resp.content))

        html_resp = await client.get("/api/reports/html")
        assert html_resp.status_code == 200
        assert b"<!DOCTYPE html>" in html_resp.content

        cef_resp = await client.get("/api/reports/cef")
        assert cef_resp.status_code == 200
        assert b"CEF:0|Pokrov|" in cef_resp.content

        jsonl_resp = await client.get("/api/reports/jsonl")
        assert jsonl_resp.status_code == 200
        jsonl_lines = [ln for ln in jsonl_resp.text.splitlines() if ln.strip()]
        assert len(jsonl_lines) == 4

        bad_rep = await client.get("/api/reports/pdf")
        assert bad_rep.status_code == 422

    # Binary Zero-Retention scan across SQLite DB/WAL/SHM and all exported report bytes
    db_path = Path(cfg.audit.db_path)
    check_store = SqliteAuditStore(cfg.audit)
    try:
        with check_store._lock:
            check_store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    finally:
        check_store.close()

    blobs: list[bytes] = [
        docx_resp.content,
        html_resp.content,
        cef_resp.content,
        jsonl_resp.content,
    ]
    with zipfile.ZipFile(io.BytesIO(docx_resp.content)) as zf:
        for name in zf.namelist():
            blobs.append(zf.read(name))
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.is_file():
            blobs.append(path.read_bytes())

    combined = b"\n".join(blobs)
    for sensitive in raw_pii_and_secrets:
        assert sensitive.encode("utf-8") not in combined, f"Leaked {sensitive!r}"


async def test_ui_policy_switching_upstreams_and_auth_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test /api/policy, demo vs production profile switching, router residencies, and auth."""
    secret_env_val = _tok("sk-", "demo-env-secret-should-never-appear-in-policy")
    monkeypatch.setenv("OPENAI_API_KEY", secret_env_val)

    # 1. Demo mode (default_upstream='mock-local'): allows switching freely including downgrades
    demo_cfg = _make_ui_config(
        tmp_path / "demo",
        allow_profile_override=False,
        default_upstream="mock-local",
    )
    demo_app = create_default_app(demo_cfg)

    async with (
        demo_app.router.lifespan_context(demo_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=demo_app),
            base_url="http://pokrov.local",
        ) as client,
    ):
        pol_resp = await client.get("/api/policy")
        assert pol_resp.status_code == 200
        pol = pol_resp.json()
        assert pol["active_profile"] == "balanced-152fz"
        assert pol["demo_mode"] is True
        assert "***REDACTED***" in pol["policy_yaml"]
        assert "pokrov-ui-test-salt-2026" not in pol["policy_yaml"]
        assert secret_env_val not in pol["policy_yaml"]
        assert secret_env_val not in pol_resp.text
        parsed_yaml: dict[str, Any] = yaml.safe_load(pol["policy_yaml"])
        assert parsed_yaml["active_profile"] == "balanced-152fz"

        oai_up = next(u for u in pol["upstreams"] if u["name"] == "openai-cloud")
        assert oai_up["credentials_configured"] is True
        assert oai_up["status"] == "ready"

        # Upgrade to strict-fstec-117
        sw_up = await client.post("/api/policy/profile", json={"profile": "strict-fstec-117"})
        assert sw_up.status_code == 200
        assert sw_up.json()["active_profile"] == "strict-fstec-117"
        assert (await client.get("/healthz")).json()["profile"] == "strict-fstec-117"

        # Downgrade to dev-audit succeeds in demo mode
        sw_down = await client.post("/api/policy/profile", json={"profile": "dev-audit"})
        assert sw_down.status_code == 200
        assert sw_down.json()["active_profile"] == "dev-audit"
        assert (await client.get("/healthz")).json()["profile"] == "dev-audit"

        # Unknown profile -> 400
        bad_prof_resp = await client.post("/api/policy/profile", json={"profile": "nope"})
        assert bad_prof_resp.status_code == 400

    # 2. Production mode (default_upstream='openai-cloud', allow_profile_override=False)
    prod_cfg = _make_ui_config(
        tmp_path / "prod",
        allow_profile_override=False,
        default_upstream="openai-cloud",
        audit_enabled=False,
        gateway_api_key_env="PKR_UI_GW_KEY",
    )
    # Add ru_cloud default test variant
    ru_up = UpstreamConfig(
        name="gigachat-ru",
        provider=ProviderType.GIGACHAT,
        model="GigaChat",
        is_local=False,
        residency="ru_cloud",
    )
    prod_upstreams = dict(prod_cfg.router.upstreams)
    prod_upstreams["gigachat-ru"] = ru_up
    prod_cfg = prod_cfg.model_copy(
        update={"router": prod_cfg.router.model_copy(update={"upstreams": prod_upstreams})}
    )

    monkeypatch.setenv("PKR_UI_GW_KEY", "ui-gateway-secret-42")
    prod_app = create_default_app(prod_cfg)

    async with (
        prod_app.router.lifespan_context(prod_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=prod_app),
            base_url="http://pokrov.local",
        ) as prod_client,
    ):
        # /ui shell loads without auth so operator can enter API key, but /api/* enforces 401
        assert (await prod_client.get("/ui")).status_code == 200
        assert (await prod_client.get("/api/stats")).status_code == 401
        assert (await prod_client.get("/api/policy")).status_code == 401
        unauth_pg = await prod_client.post("/api/playground", json={"text": "hello"})
        assert unauth_pg.status_code == 401

        auth_headers = {"X-Pokrov-Api-Key": "ui-gateway-secret-42"}

        # Audit disabled graceful responses
        st_off = await prod_client.get("/api/stats", headers=auth_headers)
        assert st_off.status_code == 200
        assert st_off.json()["audit_enabled"] is False
        assert st_off.json()["total_requests"] == 0

        ev_off = await prod_client.get("/api/events", headers=auth_headers)
        assert ev_off.status_code == 200
        assert ev_off.json()["events"] == []

        rep_off = await prod_client.get("/api/reports/cef", headers=auth_headers)
        assert rep_off.status_code == 200
        assert rep_off.text == ""

        # Cloud routing decision in playground when default_upstream is openai-cloud
        pg_cloud = await prod_client.post(
            "/api/playground",
            headers=auth_headers,
            json={"text": "Обычный запрос без ПДн"},
        )
        assert pg_cloud.status_code == 200
        assert pg_cloud.json()["router"]["decision"] == "cloud"

        # Upgrade balanced-152fz -> strict-fstec-117 allowed, then downgrade -> 403
        up_ok = await prod_client.post(
            "/api/policy/profile",
            headers=auth_headers,
            json={"profile": "strict-fstec-117"},
        )
        assert up_ok.status_code == 200

        down_err = await prod_client.post(
            "/api/policy/profile",
            headers={"Authorization": "Bearer ui-gateway-secret-42"},
            json={"profile": "balanced-152fz"},
        )
        assert down_err.status_code == 403

    # 3. RU cloud default upstream decision branch
    ru_cfg = _make_ui_config(
        tmp_path / "ru",
        default_upstream="gigachat-ru",
        audit_enabled=False,
    )
    ru_app = create_default_app(ru_cfg)
    async with (
        ru_app.router.lifespan_context(ru_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ru_app),
            base_url="http://pokrov.local",
        ) as ru_client,
    ):
        pg_ru = await ru_client.post(
            "/api/playground",
            json={"text": "Привет! Расскажи о 152-ФЗ."},
        )
        assert pg_ru.status_code == 200
        assert pg_ru.json()["router"]["decision"] == "ru_cloud"
