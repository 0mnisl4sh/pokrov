"""Unit tests for DeclarativePolicyEngine, AES-256-GCM Vaults, and TokenBucketQuotaManager."""

from __future__ import annotations

import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from pokrov.core.config import (
    CategoryRule,
    PokrovConfig,
    ProfileConfig,
    QuotaConfig,
    RouterConfig,
    load_config,
)
from pokrov.core.contracts import (
    Action,
    ComplianceRef,
    Detection,
    EntityCategory,
    InspectionResult,
    PolicyEngine,
    QuotaManager,
    VaultBackend,
)
from pokrov.policy.engine import DeclarativePolicyEngine
from pokrov.policy.quotas import TokenBucketQuotaManager
from pokrov.policy.vault import InMemoryAesVault, LocalEncryptedVault

REPO_ROOT = Path(__file__).resolve().parent.parent


def _det(
    category: EntityCategory,
    *,
    confidence: float = 0.95,
    start: int = 0,
    end: int = 10,
    detector: str = "test_detector",
    compliance_refs: tuple[ComplianceRef, ...] = (),
) -> Detection:
    return Detection(
        category=category,
        start=start,
        end=end,
        confidence=confidence,
        detector=detector,
        compliance_refs=compliance_refs,
    )


class _MutableClock:
    def __init__(self, initial: float = 1_000.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


def test_protocols_runtime_checkable(tmp_path: Path) -> None:
    """All three modules must satisfy their frozen runtime_checkable Protocols."""
    engine = DeclarativePolicyEngine(PokrovConfig())
    mem_vault = InMemoryAesVault()
    local_vault = LocalEncryptedVault(tmp_path / "vault.enc")
    quota_mgr = TokenBucketQuotaManager(QuotaConfig())

    assert isinstance(engine, PolicyEngine)
    assert isinstance(mem_vault, VaultBackend)
    assert isinstance(local_vault, VaultBackend)
    assert isinstance(quota_mgr, QuotaManager)


# ---------------------------------------------------------------------------
# DeclarativePolicyEngine tests
# ---------------------------------------------------------------------------


def test_policy_engine_all_three_profiles_from_example_yaml() -> None:
    """Verify dev-audit, balanced-152fz, and strict-fstec-117 profiles."""
    cfg = load_config(REPO_ROOT / "policy.example.yaml")
    engine = DeclarativePolicyEngine(cfg)

    pii_inspection = InspectionResult(
        detections=(
            _det(EntityCategory.RU_PASSPORT, confidence=0.92),
            _det(EntityCategory.RU_SNILS, confidence=0.99, start=12, end=26),
        )
    )

    # 1. dev-audit: always flags, never blocks or routes local
    dev_decision = engine.evaluate(pii_inspection, profile="dev-audit")
    assert dev_decision.action == Action.FLAG
    assert dev_decision.http_status == 200
    assert dev_decision.target_upstream is None
    assert dev_decision.categories == (EntityCategory.RU_PASSPORT, EntityCategory.RU_SNILS)
    assert len(dev_decision.matched_rules) == 2
    assert len(dev_decision.compliance_refs) >= 3

    # 2. balanced-152fz: reversible masking for PII, block for secret/injection
    bal_pii = engine.evaluate(pii_inspection, profile="balanced-152fz")
    assert bal_pii.action == Action.MASK_REVERSIBLE
    assert bal_pii.http_status == 200
    assert bal_pii.target_upstream is None

    bal_mixed = engine.evaluate(
        InspectionResult(
            detections=(
                _det(EntityCategory.RU_OGRN, confidence=0.9),
                _det(EntityCategory.RU_PASSPORT, confidence=0.9),
                _det(EntityCategory.SECRET_KEY, confidence=0.95),
            )
        ),
        profile="balanced-152fz",
    )
    assert bal_mixed.action == Action.BLOCK
    assert bal_mixed.http_status == 403
    assert bal_mixed.target_upstream is None
    assert bal_mixed.categories == (
        EntityCategory.RU_OGRN,
        EntityCategory.RU_PASSPORT,
        EntityCategory.SECRET_KEY,
    )

    # 3. strict-fstec-117: PII forces ROUTE_LOCAL with configured local_upstream
    strict_pii = engine.evaluate(pii_inspection, profile="strict-fstec-117")
    assert strict_pii.action == Action.ROUTE_LOCAL
    assert strict_pii.target_upstream == "ollama-local"
    assert strict_pii.http_status == 200

    strict_inj = engine.evaluate(
        InspectionResult(
            detections=(
                _det(EntityCategory.RU_PASSPORT, confidence=0.8),
                _det(EntityCategory.PROMPT_INJECTION, confidence=0.85),
            )
        ),
        profile="strict-fstec-117",
    )
    assert strict_inj.action == Action.BLOCK
    assert strict_inj.target_upstream is None
    assert strict_inj.http_status == 403


def test_policy_engine_input_vs_output_and_require_approval() -> None:
    """Output direction must honor output_action when set and fallback to action otherwise."""
    cfg = PokrovConfig(
        active_profile="custom",
        router=RouterConfig(local_upstream="mock-local"),
        profiles={
            "custom": ProfileConfig(
                rules={
                    EntityCategory.SECRET_KEY: CategoryRule(
                        action=Action.BLOCK, output_action=Action.MASK
                    ),
                    EntityCategory.RU_PASSPORT: CategoryRule(
                        action=Action.ROUTE_LOCAL, output_action=None
                    ),
                    EntityCategory.MCP_POLICY_VIOLATION: CategoryRule(
                        action=Action.REQUIRE_APPROVAL
                    ),
                }
            )
        },
    )
    engine = DeclarativePolicyEngine(cfg)

    secret_res = InspectionResult(detections=(_det(EntityCategory.SECRET_KEY),))
    in_dec = engine.evaluate(secret_res, direction="input")
    out_dec = engine.evaluate(secret_res, direction="output")
    assert in_dec.action == Action.BLOCK
    assert in_dec.http_status == 403
    assert out_dec.action == Action.MASK
    assert out_dec.http_status == 200

    # Fallback to rule.action when output_action is None
    passport_res = InspectionResult(detections=(_det(EntityCategory.RU_PASSPORT),))
    passport_out = engine.evaluate(passport_res, direction="output")
    assert passport_out.action == Action.ROUTE_LOCAL
    assert passport_out.target_upstream == "mock-local"

    # REQUIRE_APPROVAL must yield http_status=403
    approval_res = InspectionResult(detections=(_det(EntityCategory.MCP_POLICY_VIOLATION),))
    approval_dec = engine.evaluate(approval_res, direction="input")
    assert approval_dec.action == Action.REQUIRE_APPROVAL
    assert approval_dec.http_status == 403


def test_policy_engine_principal_overrides_and_filtering() -> None:
    """Principal overrides, confidence thresholds, disabled rules, and deduplication."""
    custom_ref = ComplianceRef(
        framework="INTERNAL",
        article="SEC-1",
        title="Internal security policy",
    )
    cfg = PokrovConfig(
        active_profile="corp",
        router=RouterConfig(local_upstream="local-llm"),
        profiles={
            "corp": ProfileConfig(
                default_action=Action.ALLOW,
                rules={
                    EntityCategory.RU_PASSPORT: CategoryRule(
                        action=Action.MASK_REVERSIBLE, min_confidence=0.7
                    ),
                    EntityCategory.EMAIL: CategoryRule(action=Action.FLAG, enabled=False),
                    EntityCategory.RU_SNILS: CategoryRule(action=Action.MASK, min_confidence=0.6),
                },
                principal_overrides={
                    "trusted-bot": {
                        EntityCategory.RU_PASSPORT: CategoryRule(enabled=False),
                    },
                    "intern": {
                        EntityCategory.RU_PASSPORT: CategoryRule(
                            action=Action.ROUTE_LOCAL,
                            output_action=Action.BLOCK,
                            min_confidence=0.5,
                        ),
                    },
                },
            )
        },
    )
    engine = DeclarativePolicyEngine(cfg)

    # Below min_confidence (0.69 < 0.7) and disabled rule (EMAIL) -> skipped -> ALLOW
    low_conf = InspectionResult(
        detections=(
            _det(EntityCategory.RU_PASSPORT, confidence=0.69),
            _det(EntityCategory.EMAIL, confidence=0.99),
        )
    )
    low_dec = engine.evaluate(low_conf, principal="regular-user")
    assert low_dec.action == Action.ALLOW
    assert low_dec.categories == ()
    assert low_dec.matched_rules == ()
    assert low_dec.compliance_refs == ()
    assert low_dec.http_status == 200

    # Exact min_confidence boundary (0.7 == 0.7) -> matched, plus deduplication
    dup_res = InspectionResult(
        detections=(
            _det(EntityCategory.RU_PASSPORT, confidence=0.7, compliance_refs=(custom_ref,)),
            _det(EntityCategory.RU_PASSPORT, confidence=0.95, compliance_refs=(custom_ref,)),
        )
    )
    dup_dec = engine.evaluate(dup_res, principal="regular-user")
    assert dup_dec.action == Action.MASK_REVERSIBLE
    assert dup_dec.categories == (EntityCategory.RU_PASSPORT,)
    assert len(dup_dec.matched_rules) == 1
    assert any(ref.framework == "INTERNAL" for ref in dup_dec.compliance_refs)
    assert len({(r.framework, r.article) for r in dup_dec.compliance_refs}) == len(
        dup_dec.compliance_refs
    )

    # trusted-bot disables RU_PASSPORT via principal_overrides
    bot_dec = engine.evaluate(dup_res, principal="trusted-bot")
    assert bot_dec.action == Action.ALLOW
    assert bot_dec.matched_rules == ()

    # intern override lowers min_confidence to 0.5 and escalates to ROUTE_LOCAL / output BLOCK
    intern_res = InspectionResult(detections=(_det(EntityCategory.RU_PASSPORT, confidence=0.55),))
    intern_in = engine.evaluate(intern_res, direction="input", principal="intern")
    assert intern_in.action == Action.ROUTE_LOCAL
    assert intern_in.target_upstream == "local-llm"
    assert intern_in.http_status == 200

    intern_out = engine.evaluate(intern_res, direction="output", principal="intern")
    assert intern_out.action == Action.BLOCK
    assert intern_out.target_upstream is None
    assert intern_out.http_status == 403

    with pytest.raises(ValueError, match="Unknown policy profile"):
        engine.evaluate(dup_res, profile="does-not-exist")


# ---------------------------------------------------------------------------
# InMemoryAesVault tests
# ---------------------------------------------------------------------------


def test_in_memory_vault_deterministic_session_isolation_and_ciphertext_safety() -> None:
    """Placeholders must be deterministic per session, isolated, and expose zero plaintext."""
    vault = InMemoryAesVault()
    secret = "4512 654321"
    snils = "112-233-445 95"

    p1 = vault.store("sess-a", EntityCategory.RU_PASSPORT, secret)
    p2 = vault.store("sess-a", EntityCategory.RU_PASSPORT, secret)
    p_other_cat = vault.store("sess-a", EntityCategory.RU_INN, secret)
    p_other_sess = vault.store("sess-b", EntityCategory.RU_PASSPORT, secret)
    p_snils = vault.store("sess-a", EntityCategory.RU_SNILS, snils)

    assert re.fullmatch(r"<PKR:RU_PASSPORT:[0-9a-f]{8}>", p1)
    assert p1 == p2
    assert p1 != p_other_cat
    assert p1 != p_other_sess
    assert re.fullmatch(r"<PKR:RU_SNILS:[0-9a-f]{8}>", p_snils)

    # Zero-Retention: raw value must never appear in stored entries or repr
    assert secret not in repr(vault)
    assert snils not in repr(vault)
    for (sid, placeholder), entry in vault._entries.items():
        assert secret not in sid
        assert secret not in placeholder
        assert secret.encode("utf-8") not in entry.ciphertext
        assert secret.encode("utf-8") not in entry.nonce
        assert snils.encode("utf-8") not in entry.ciphertext

    # Session isolation on retrieve & restore_text
    assert vault.retrieve("sess-a", p1) == secret
    assert vault.retrieve("sess-b", p1) is None
    assert vault.retrieve("sess-b", p_other_sess) == secret

    masked = f"Паспорт: {p1}, повтор: {p1}, СНИЛС: {p_snils}, чужой: {p_other_sess}"
    restored, count = vault.restore_text("sess-a", masked)
    assert count == 3
    assert restored == f"Паспорт: {secret}, повтор: {secret}, СНИЛС: {snils}, чужой: {p_other_sess}"

    # AAD binds ciphertext to (session_id, placeholder)
    entry_a = vault._entries[("sess-a", p1)]
    vault._entries[("sess-evil", p1)] = entry_a
    with pytest.raises(InvalidTag):
        vault.retrieve("sess-evil", p1)


def test_in_memory_vault_no_nested_placeholder_injection() -> None:
    """Single-pass restore_text must not re-expand placeholders contained inside raw secrets."""
    vault = InMemoryAesVault()
    p_inner = vault.store("s1", EntityCategory.SECRET_KEY, "REAL_SECRET")
    tricky_value = f"prefix_{p_inner}_suffix"
    p_outer = vault.store("s1", EntityCategory.CUSTOM_PATTERN, tricky_value)

    restored, count = vault.restore_text("s1", f"val={p_outer}")
    assert count == 1
    assert restored == f"val={tricky_value}"


def test_in_memory_vault_ttl_expiry_and_purge() -> None:
    """TTL expiration and session purge must cleanly drop encrypted records."""
    clock = _MutableClock(100.0)
    vault = InMemoryAesVault(ttl_seconds=60, max_entries=100, time_fn=clock)

    p1 = vault.store("s1", EntityCategory.EMAIL, "alice@example.ru")
    clock.advance(30.0)
    p2 = vault.store("s2", EntityCategory.EMAIL, "bob@example.ru")

    # At t=130: both valid
    assert vault.retrieve("s1", p1) == "alice@example.ru"
    assert vault.retrieve("s2", p2) == "bob@example.ru"

    # At t=160: p1 (stored at 100, expires at 160) is expired; p2 (stored at 130) is still valid
    clock.advance(30.0)
    assert vault.cleanup_expired() == 1
    assert vault.retrieve("s1", p1) is None
    assert vault.restore_text("s1", f"email={p1}") == (f"email={p1}", 0)
    assert vault.retrieve("s2", p2) == "bob@example.ru"

    # Purge session s2
    assert vault.purge_session("s2") == 1
    assert vault.purge_session("s2") == 0
    assert len(vault) == 0


def test_in_memory_vault_lru_eviction_and_thread_safety() -> None:
    """LRU eviction must evict least recently used item and survive concurrent access."""
    clock = _MutableClock(0.0)
    vault = InMemoryAesVault(ttl_seconds=900, max_entries=3, time_fn=clock)

    p1 = vault.store("s", EntityCategory.RU_PHONE, "+79990000001")
    p2 = vault.store("s", EntityCategory.RU_PHONE, "+79990000002")
    p3 = vault.store("s", EntityCategory.RU_PHONE, "+79990000003")

    # Touch p1 so p2 becomes least-recently-used
    assert vault.retrieve("s", p1) == "+79990000001"
    p4 = vault.store("s", EntityCategory.RU_PHONE, "+79990000004")

    assert len(vault) == 3
    assert vault.retrieve("s", p2) is None
    assert vault.retrieve("s", p1) == "+79990000001"
    assert vault.retrieve("s", p3) == "+79990000003"
    assert vault.retrieve("s", p4) == "+79990000004"

    # Concurrency test
    concurrent_vault = InMemoryAesVault(ttl_seconds=900, max_entries=500)

    def _worker(idx: int) -> tuple[str, str]:
        val = f"7701000{idx:04d}"
        ph = concurrent_vault.store(f"sess-{idx % 4}", EntityCategory.RU_INN, val)
        text, cnt = concurrent_vault.restore_text(f"sess-{idx % 4}", f"INN={ph}")
        assert cnt == 1
        assert text == f"INN={val}"
        return ph, val

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(80)))
    assert len(results) == 80


# ---------------------------------------------------------------------------
# LocalEncryptedVault tests (strictly tmp_path)
# ---------------------------------------------------------------------------


def test_local_encrypted_vault_cross_instance_restore_and_permissions(tmp_path: Path) -> None:
    """Persisted encrypted vault restores secrets across instances without plaintext exposure."""
    vault_file = tmp_path / "ide_guard" / "vault.enc"
    clock = _MutableClock(1_700_000_000.0)

    writer = LocalEncryptedVault(
        local_vault_path=vault_file,
        ttl_seconds=120,
        max_entries=10,
        time_fn=clock,
    )
    # Lazy init: files should not be created until store() is called
    assert not vault_file.exists()
    assert not writer.key_path.exists()

    secret_passport = "4512 987654"
    secret_phone = "+79161234567"
    p_pass = writer.store("ide-1", EntityCategory.RU_PASSPORT, secret_passport)
    p_phone = writer.store("ide-1", EntityCategory.RU_PHONE, secret_phone)

    assert vault_file.is_file()
    assert writer.key_path.is_file()
    assert writer.key_path.parent == vault_file.parent
    assert writer.key_path.suffix == ".key"
    assert len(writer.key_path.read_bytes()) == 32

    if os.name != "nt":
        mode = stat.S_IMODE(writer.key_path.stat().st_mode)
        assert mode == 0o600

    # Zero-Retention on disk: raw secrets must NOT appear in vault file or key file
    disk_vault_bytes = vault_file.read_bytes()
    disk_key_bytes = writer.key_path.read_bytes()
    assert secret_passport.encode("utf-8") not in disk_vault_bytes
    assert secret_phone.encode("utf-8") not in disk_vault_bytes
    assert secret_passport.encode("utf-8") not in disk_key_bytes

    # Simulate fresh CLI process `pokrov restore`
    clock.advance(30.0)
    reader = LocalEncryptedVault(
        local_vault_path=vault_file,
        ttl_seconds=120,
        max_entries=10,
        time_fn=clock,
    )
    # Deterministic placeholder stability across processes with same keyfile
    assert reader.store("ide-1", EntityCategory.RU_PASSPORT, secret_passport) == p_pass

    restored, count = reader.restore_text(
        "ide-1",
        f"Договор: паспорт {p_pass}, тел. {p_phone}",
    )
    assert count == 2
    assert restored == f"Договор: паспорт {secret_passport}, тел. {secret_phone}"

    # Advance past TTL -> expired across processes
    clock.advance(130.0)
    reader_expired = LocalEncryptedVault(
        local_vault_path=vault_file,
        ttl_seconds=120,
        max_entries=10,
        time_fn=clock,
    )
    assert reader_expired.retrieve("ide-1", p_pass) is None
    assert reader_expired.cleanup_expired() >= 0


def test_local_encrypted_vault_lru_purge_and_missing_key(tmp_path: Path) -> None:
    """Verify cross-instance LRU, purge_session, and missing keyfile error handling."""
    vault_file = tmp_path / "vault.enc"
    v1 = LocalEncryptedVault(vault_file, ttl_seconds=900, max_entries=2)
    p1 = v1.store("s1", EntityCategory.EMAIL, "one@example.com")
    p2 = v1.store("s1", EntityCategory.EMAIL, "two@example.com")

    v2 = LocalEncryptedVault(vault_file, ttl_seconds=900, max_entries=2)
    assert v2.retrieve("s1", p1) == "one@example.com"  # touches p1 -> p2 becomes LRU
    p3 = v2.store("s1", EntityCategory.EMAIL, "three@example.com")

    v3 = LocalEncryptedVault(vault_file, ttl_seconds=900, max_entries=2)
    assert v3.retrieve("s1", p2) is None
    assert v3.retrieve("s1", p1) == "one@example.com"
    assert v3.retrieve("s1", p3) == "three@example.com"
    assert v3.purge_session("s1") == 2
    assert len(v3) == 0

    # Missing keyfile while vault.enc exists must raise RuntimeError
    v3.store("s2", EntityCategory.EMAIL, "keep@example.com")
    v3.key_path.unlink()
    with pytest.raises(RuntimeError, match="keyfile missing"):
        v3.retrieve("s2", "anything")

    # Empty / nonexistent local vault returns safe empty values without creating files
    empty_vault = LocalEncryptedVault(tmp_path / "never_written.enc")
    assert len(empty_vault) == 0
    assert empty_vault.retrieve("s", "<PKR:EMAIL:12345678>") is None
    assert empty_vault.restore_text("s", "plain text") == ("plain text", 0)
    assert empty_vault.restore_text("s", "<PKR:EMAIL:12345678>") == ("<PKR:EMAIL:12345678>", 0)
    assert empty_vault.purge_session("s") == 0
    assert empty_vault.cleanup_expired() == 0
    assert not empty_vault.vault_path.exists()
    assert "LocalEncryptedVault" in repr(empty_vault)


def test_vault_validation_and_corrupted_file_guards(tmp_path: Path) -> None:
    """Vault constructors validate params and reject malformed key/vault files."""
    with pytest.raises(ValueError, match="ttl_seconds"):
        InMemoryAesVault(ttl_seconds=0)
    with pytest.raises(ValueError, match="max_entries"):
        InMemoryAesVault(max_entries=0)
    with pytest.raises(ValueError, match="ttl_seconds"):
        LocalEncryptedVault(tmp_path / "bad.enc", ttl_seconds=-5)

    # Plaintext fast-path on InMemoryAesVault
    mem = InMemoryAesVault()
    assert mem.restore_text("s", "no placeholders here") == ("no placeholders here", 0)

    # Corrupted vault format & wrong key length
    bad_vault_path = tmp_path / "corrupt.enc"
    lv = LocalEncryptedVault(bad_vault_path)
    lv.store("s", EntityCategory.EMAIL, "a@b.ru")
    bad_vault_path.write_text('{"version": 999}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported encrypted vault format"):
        lv.retrieve("s", "<PKR:EMAIL:00000000>")

    lv.key_path.write_bytes(b"short-key")
    with pytest.raises(ValueError, match="Invalid master key length"):
        lv.retrieve("s", "<PKR:EMAIL:00000000>")

    # Vault path with .key suffix gets .master.key sibling
    key_named_vault = LocalEncryptedVault(tmp_path / "custom.key")
    assert key_named_vault.key_path.name == "custom.key.master.key"


# ---------------------------------------------------------------------------
# TokenBucketQuotaManager tests
# ---------------------------------------------------------------------------


def test_quota_manager_payload_size_limits_do_not_consume_buckets() -> None:
    """max_prompt_chars and max_estimated_tokens must reject oversized payloads atomically."""
    clock = _MutableClock(0.0)
    mgr = TokenBucketQuotaManager(
        QuotaConfig(
            max_prompt_chars=100,
            max_estimated_tokens=50,
            rpm=5,
            tpm=200,
            daily_requests=10,
        ),
        time_fn=clock,
    )

    # Dimension 1: max_prompt_chars
    over_chars = mgr.check_and_consume("alice", prompt_chars=101, estimated_tokens=10)
    assert not over_chars.allowed
    assert "max_prompt_chars" in over_chars.reason
    assert over_chars.retry_after_seconds == 0
    assert over_chars.remaining_rpm == 5
    assert over_chars.remaining_tpm == 200
    assert over_chars.remaining_daily_requests == 10

    # Dimension 2: max_estimated_tokens
    over_tokens = mgr.check_and_consume("alice", prompt_chars=100, estimated_tokens=51)
    assert not over_tokens.allowed
    assert "max_estimated_tokens" in over_tokens.reason
    assert over_tokens.retry_after_seconds == 0
    assert over_tokens.remaining_rpm == 5

    # Exact boundaries must pass
    ok = mgr.check_and_consume("alice", prompt_chars=100, estimated_tokens=50)
    assert ok.allowed
    assert ok.remaining_rpm == 4
    assert ok.remaining_tpm == 150
    assert ok.remaining_daily_requests == 9


def test_quota_manager_rpm_tpm_and_daily_requests() -> None:
    """Enforce RPM, TPM, and daily_requests with accurate refills and principal isolation."""
    clock = _MutableClock(100.0)
    mgr = TokenBucketQuotaManager(
        QuotaConfig(
            max_prompt_chars=10_000,
            max_estimated_tokens=1_000,
            rpm=3,
            tpm=100,
            daily_requests=4,
        ),
        time_fn=clock,
    )

    # Request 1: consumes 1 RPM, 60 TPM, 1 daily
    r1 = mgr.check_and_consume("alice", prompt_chars=50, estimated_tokens=60)
    assert r1.allowed
    assert r1.remaining_rpm == 2
    assert r1.remaining_tpm == 40
    assert r1.remaining_daily_requests == 3

    # Dimension 4 (TPM): Request 2 asks for 60 tokens (only 40 left) -> denied, RPM not consumed!
    r2_denied = mgr.check_and_consume("alice", prompt_chars=50, estimated_tokens=60)
    assert not r2_denied.allowed
    assert "TPM" in r2_denied.reason
    # Need 20 more tokens at 100/60 tok/s -> 12s
    assert r2_denied.retry_after_seconds == 12
    assert r2_denied.remaining_rpm == 2
    assert r2_denied.remaining_daily_requests == 3

    # Principal isolation: bob still has full quota
    bob_res = mgr.check_and_consume("bob", prompt_chars=50, estimated_tokens=80)
    assert bob_res.allowed
    assert bob_res.remaining_rpm == 2
    assert bob_res.remaining_tpm == 20

    # Advance 12s -> alice refills 20 TPM (40 -> 60) and 0.6 RPM (2 -> 2.6)
    clock.advance(12.0)
    r2_ok = mgr.check_and_consume("alice", prompt_chars=50, estimated_tokens=60)
    assert r2_ok.allowed
    assert r2_ok.remaining_daily_requests == 2

    # Consume remaining RPM smoothly: advance 60s -> full RPM=3, TPM=100, daily=2 used
    clock.advance(60.0)
    r3 = mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=10)
    r4 = mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=10)
    assert r3.allowed
    assert r4.allowed
    assert r4.remaining_daily_requests == 0

    # Dimension 5 (daily_requests): even after RPM/TPM refill (+120s), daily quota is exhausted
    clock.advance(120.0)
    r5_daily_denied = mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=10)
    assert not r5_daily_denied.allowed
    assert "Daily" in r5_daily_denied.reason
    assert r5_daily_denied.retry_after_seconds > 80_000
    assert r5_daily_denied.remaining_daily_requests == 0

    # Advance past 24h from window start -> daily counter resets
    clock.advance(86_400.0)
    r6_ok = mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=10)
    assert r6_ok.allowed
    assert r6_ok.remaining_daily_requests == 3

    # Dimension 3 (RPM): exhaust 3 RPM in burst (currently 2 left after r6_ok)
    assert mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=5).allowed
    assert mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=5).allowed
    rpm_denied = mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=5)
    assert not rpm_denied.allowed
    assert "RPM" in rpm_denied.reason
    # 1 token needed at 3/60 req/s = 20s
    assert rpm_denied.retry_after_seconds == 20

    # Advance 20s -> 1 RPM token refilled
    clock.advance(20.0)
    assert mgr.check_and_consume("alice", prompt_chars=10, estimated_tokens=5).allowed


def test_quota_manager_validation_and_concurrency() -> None:
    """Negative counts raise ValueError and concurrent calls respect strict limits."""
    mgr = TokenBucketQuotaManager(
        QuotaConfig(
            max_prompt_chars=1000,
            max_estimated_tokens=500,
            rpm=10,
            tpm=5000,
            daily_requests=100,
        ),
        time_fn=lambda: 500.0,
    )

    with pytest.raises(ValueError, match="prompt_chars"):
        mgr.check_and_consume("u", prompt_chars=-1, estimated_tokens=10)
    with pytest.raises(ValueError, match="estimated_tokens"):
        mgr.check_and_consume("u", prompt_chars=10, estimated_tokens=-1)

    # Single request exceeding tpm bucket capacity (when max_estimated_tokens >= tpm)
    tight_mgr = TokenBucketQuotaManager(
        QuotaConfig(max_estimated_tokens=1000, tpm=100, rpm=10, daily_requests=10)
    )
    over_tpm_cap = tight_mgr.check_and_consume("u", prompt_chars=10, estimated_tokens=150)
    assert not over_tpm_cap.allowed
    assert "tpm" in over_tpm_cap.reason

    def _hit(_: int) -> bool:
        return mgr.check_and_consume("shared-user", prompt_chars=10, estimated_tokens=10).allowed

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_hit, range(25)))

    assert sum(1 for allowed in outcomes if allowed) == 10
    mgr.reset("shared-user")
    assert mgr.check_and_consume("shared-user", prompt_chars=10, estimated_tokens=10).allowed
