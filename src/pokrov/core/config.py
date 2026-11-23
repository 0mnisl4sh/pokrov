"""Configuration and declarative policy loading for Pokrov."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pokrov.core.contracts import (
    Action,
    EntityCategory,
    FileGuardMode,
    ProviderType,
)


class ServerConfig(BaseModel):
    """HTTP gateway binding and runtime settings."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    enable_ui: bool = True
    log_level: str = "INFO"


class VaultConfig(BaseModel):
    """Reversible tokenization vault settings.

    ``memory`` keeps secrets strictly in RAM (gateway Zero-Retention default).
    ``local_encrypted`` is opt-in for short-lived IDE Guard CLI hooks.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["memory", "local_encrypted"] = "memory"
    ttl_seconds: int = Field(default=900, ge=1)
    max_entries: int = Field(default=10_000, ge=1)
    local_vault_path: str = ".pokrov/vault.enc"


class QuotaConfig(BaseModel):
    """Payload size and rate-limit quotas (FSTEC Order 117 sec. 3.18)."""

    model_config = ConfigDict(frozen=True)

    max_prompt_chars: int = Field(default=100_000, ge=1)
    max_estimated_tokens: int = Field(default=32_000, ge=1)
    rpm: int = Field(default=120, ge=1)
    tpm: int = Field(default=200_000, ge=1)
    daily_requests: int = Field(default=10_000, ge=1)


class UpstreamConfig(BaseModel):
    """LLM upstream target definition."""

    model_config = ConfigDict(frozen=True)

    name: str
    provider: ProviderType
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    folder_id_env: str = ""
    tls_verify: bool = True
    ca_bundle_path: str = ""
    is_local: bool = False
    timeout_seconds: float = Field(default=60.0, gt=0.0)


def _default_upstreams() -> dict[str, UpstreamConfig]:
    return {
        "mock-local": UpstreamConfig(
            name="mock-local",
            provider=ProviderType.MOCK,
            model="pokrov-mock-llm",
            is_local=True,
        ),
        "ollama-local": UpstreamConfig(
            name="ollama-local",
            provider=ProviderType.OLLAMA,
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5:7b",
            is_local=True,
        ),
        "openai-cloud": UpstreamConfig(
            name="openai-cloud",
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            is_local=False,
        ),
    }


class RouterConfig(BaseModel):
    """Data-aware routing destinations."""

    model_config = ConfigDict(frozen=True)

    default_upstream: str = "mock-local"
    local_upstream: str = "mock-local"
    cloud_upstream: str = "openai-cloud"
    upstreams: dict[str, UpstreamConfig] = Field(default_factory=_default_upstreams)


class AuditConfig(BaseModel):
    """Zero-Retention SQLite audit persistence settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    db_path: str = ".pokrov/audit.db"
    hmac_salt: str = ""
    retention_days: int = Field(default=90, ge=1)
    zero_retention: bool = True


class SiemConfig(BaseModel):
    """SIEM export configuration (KUMA / MaxPatrol / RuSIEM)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    format: Literal["cef", "syslog", "jsonl"] = "cef"
    transport: Literal["stdout", "udp", "tcp", "tls", "file"] = "stdout"
    host: str = "127.0.0.1"
    port: int = Field(default=514, ge=1, le=65535)
    file_path: str = ".pokrov/siem.log"
    facility: int = Field(default=16, ge=0, le=23)
    app_name: str = "pokrov"


class CategoryRule(BaseModel):
    """Per-category enforcement policy."""

    model_config = ConfigDict(frozen=True)

    action: Action = Action.MASK_REVERSIBLE
    output_action: Action | None = None
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    enabled: bool = True


class CustomPatternRule(BaseModel):
    """User-defined regex rule declared in policy YAML."""

    model_config = ConfigDict(frozen=True)

    name: str
    pattern: str
    category: EntityCategory = EntityCategory.CUSTOM_PATTERN
    action: Action = Action.MASK
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class McpPolicyConfig(BaseModel):
    """MCP Security Proxy RBAC and argument inspection rules."""

    model_config = ConfigDict(frozen=True)

    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    approval_required_tools: list[str] = Field(
        default_factory=lambda: ["send_email", "git_push", "http_post"]
    )
    sql_read_only: bool = True
    blocked_paths: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".ssh",
            "id_rsa",
            "id_ed25519",
            ".cursor/mcp.json",
            "mcp.json",
            "/etc/shadow",
            "/etc/passwd",
        ]
    )
    block_ssrf: bool = True
    hitl_timeout_seconds: int = Field(default=60, ge=1)
    sanitize_responses: bool = True


class IdeGuardConfig(BaseModel):
    """Universal IDE Guard file protection settings."""

    model_config = ConfigDict(frozen=True)

    default_mode: FileGuardMode = FileGuardMode.SANITIZE_MIRROR
    deny_globs: list[str] = Field(
        default_factory=lambda: [".env", ".env.*", "*.pem", "*.key", "*id_rsa*", "*secrets*"]
    )
    mirror_globs: list[str] = Field(
        default_factory=lambda: ["*.docx", "*.xlsx", "*.pdf", "*.csv", "*.txt", "*.md", "*.sql"]
    )
    mirror_dir: str = ".pokrov/sanitized"
    auto_ignore: bool = True
    watch_debounce_ms: int = Field(default=40, ge=5)


class ProfileConfig(BaseModel):
    """Named security policy profile (e.g., dev-audit, balanced-152fz, strict-fstec-117)."""

    model_config = ConfigDict(frozen=True)

    description: str = ""
    default_action: Action = Action.ALLOW
    enable_ner: bool = False
    enable_ml: bool = False
    multi_turn_window: int = Field(default=3, ge=1, le=10)
    cache_ttl_seconds: int = Field(default=300, ge=0)
    rules: dict[EntityCategory, CategoryRule] = Field(default_factory=dict)
    principal_overrides: dict[str, dict[EntityCategory, CategoryRule]] = Field(default_factory=dict)
    custom_patterns: list[CustomPatternRule] = Field(default_factory=list)
    quotas: QuotaConfig = Field(default_factory=QuotaConfig)
    mcp: McpPolicyConfig = Field(default_factory=McpPolicyConfig)
    ide_guard: IdeGuardConfig = Field(default_factory=IdeGuardConfig)


def _balanced_rules() -> dict[EntityCategory, CategoryRule]:
    return {
        EntityCategory.RU_PASSPORT: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_SNILS: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_INN: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_OGRN: CategoryRule(action=Action.FLAG),
        EntityCategory.RU_BANK_CARD: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_BIK_ACCOUNT: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_OMS: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_PHONE: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_AUTO_PLATE: CategoryRule(action=Action.MASK),
        EntityCategory.RU_DRIVER_LICENSE: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_PERSON_NAME: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.RU_ADDRESS: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.EMAIL: CategoryRule(action=Action.MASK_REVERSIBLE),
        EntityCategory.SECRET_KEY: CategoryRule(action=Action.BLOCK, output_action=Action.MASK),
        EntityCategory.PROMPT_INJECTION: CategoryRule(
            action=Action.BLOCK, output_action=Action.BLOCK
        ),
        EntityCategory.INDIRECT_INJECTION: CategoryRule(
            action=Action.BLOCK, output_action=Action.BLOCK
        ),
        EntityCategory.MARKDOWN_EXFIL: CategoryRule(
            action=Action.BLOCK, output_action=Action.BLOCK
        ),
        EntityCategory.MCP_POLICY_VIOLATION: CategoryRule(action=Action.BLOCK),
    }


def _dev_audit_rules() -> dict[EntityCategory, CategoryRule]:
    return {
        category: CategoryRule(action=Action.FLAG, output_action=Action.FLAG)
        for category in EntityCategory
    }


def _strict_fstec_rules() -> dict[EntityCategory, CategoryRule]:
    rules: dict[EntityCategory, CategoryRule] = {}
    for category in EntityCategory:
        if category in {
            EntityCategory.SECRET_KEY,
            EntityCategory.PROMPT_INJECTION,
            EntityCategory.INDIRECT_INJECTION,
            EntityCategory.MARKDOWN_EXFIL,
            EntityCategory.MCP_POLICY_VIOLATION,
        }:
            rules[category] = CategoryRule(
                action=Action.BLOCK, output_action=Action.BLOCK, min_confidence=0.5
            )
        elif category in {EntityCategory.RU_OGRN, EntityCategory.RU_ORG_NAME}:
            rules[category] = CategoryRule(action=Action.FLAG, output_action=Action.FLAG)
        else:
            rules[category] = CategoryRule(
                action=Action.ROUTE_LOCAL,
                output_action=Action.MASK,
                min_confidence=0.5,
            )
    return rules


def _default_profiles() -> dict[str, ProfileConfig]:
    return {
        "dev-audit": ProfileConfig(
            description="Shadow audit mode: logs incidents without blocking traffic.",
            rules=_dev_audit_rules(),
            quotas=QuotaConfig(max_prompt_chars=200_000, rpm=300, tpm=500_000),
            ide_guard=IdeGuardConfig(default_mode=FileGuardMode.SANITIZE_MIRROR),
        ),
        "balanced-152fz": ProfileConfig(
            description=(
                "Balanced 152-FZ protection: reversible masking, injection & secret blocking."
            ),
            rules=_balanced_rules(),
            ide_guard=IdeGuardConfig(default_mode=FileGuardMode.SANITIZE_MIRROR),
        ),
        "strict-fstec-117": ProfileConfig(
            description=(
                "Strict FSTEC Order 117 contour: force local LLM on any PII, strict quotas."
            ),
            enable_ner=True,
            rules=_strict_fstec_rules(),
            quotas=QuotaConfig(
                max_prompt_chars=32_000,
                max_estimated_tokens=8_000,
                rpm=60,
                tpm=60_000,
                daily_requests=2_000,
            ),
            ide_guard=IdeGuardConfig(default_mode=FileGuardMode.ROUTE_LOCAL),
        ),
    }


class PokrovConfig(BaseModel):
    """Top-level configuration model for Pokrov."""

    model_config = ConfigDict(frozen=True)

    active_profile: str = "balanced-152fz"
    server: ServerConfig = Field(default_factory=ServerConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    siem: SiemConfig = Field(default_factory=SiemConfig)
    profiles: dict[str, ProfileConfig] = Field(default_factory=_default_profiles)

    def get_profile(self, name: str | None = None) -> ProfileConfig:
        """Return the named policy profile or the active default profile."""
        target = name or self.active_profile
        if target in self.profiles:
            return self.profiles[target]
        if name is None and not self.profiles:
            return ProfileConfig()
        raise ValueError(f"Unknown policy profile: {target}")


def _apply_env_overrides(config: PokrovConfig) -> PokrovConfig:
    """Apply environment variable overrides to a parsed ``PokrovConfig``."""
    updates: dict[str, object] = {}

    profile_env = os.environ.get("POKROV_PROFILE", "").strip()
    if profile_env:
        updates["active_profile"] = profile_env

    host_env = os.environ.get("POKROV_HOST", "").strip()
    port_env = os.environ.get("POKROV_PORT", "").strip()
    if host_env or port_env:
        server_updates: dict[str, object] = {}
        if host_env:
            server_updates["host"] = host_env
        if port_env:
            try:
                server_updates["port"] = int(port_env)
            except ValueError as exc:
                raise ValueError(f"Invalid POKROV_PORT value: {port_env!r}") from exc
        updates["server"] = config.server.model_copy(update=server_updates)

    db_env = os.environ.get("POKROV_DB_PATH", "").strip()
    salt_env = os.environ.get("POKROV_HMAC_SALT", "").strip()
    if db_env or salt_env:
        audit_updates: dict[str, object] = {}
        if db_env:
            audit_updates["db_path"] = db_env
        if salt_env:
            audit_updates["hmac_salt"] = salt_env
        updates["audit"] = config.audit.model_copy(update=audit_updates)

    if not updates:
        return config
    return config.model_copy(update=updates)


def load_config(path: str | Path | None = None) -> PokrovConfig:
    """Load Pokrov configuration from YAML with environment variable overrides."""
    resolved_path: Path | None = None
    if path is not None:
        resolved_path = Path(path)
    else:
        env_path = os.environ.get("POKROV_POLICY_PATH", "").strip()
        if env_path:
            resolved_path = Path(env_path)

    if resolved_path is None:
        return _apply_env_overrides(PokrovConfig())

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Policy configuration file not found: {resolved_path}")

    raw_text = resolved_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text)
    if loaded is None:
        payload: dict[str, object] = {}
    elif isinstance(loaded, dict):
        payload = loaded
    else:
        raise ValueError(f"Policy YAML root must be a mapping, got {type(loaded).__name__}")

    config = PokrovConfig.model_validate(payload)
    return _apply_env_overrides(config)


def policy_hash(config: PokrovConfig, profile_name: str | None = None) -> str:
    """Compute a deterministic SHA-256 digest for a policy profile."""
    effective_name = profile_name or config.active_profile
    profile = config.get_profile(effective_name)
    canonical = json.dumps(
        {"profile": effective_name, "config": profile.model_dump(mode="json")},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "AuditConfig",
    "CategoryRule",
    "CustomPatternRule",
    "IdeGuardConfig",
    "McpPolicyConfig",
    "PokrovConfig",
    "ProfileConfig",
    "QuotaConfig",
    "RouterConfig",
    "ServerConfig",
    "SiemConfig",
    "UpstreamConfig",
    "VaultConfig",
    "load_config",
    "policy_hash",
]
