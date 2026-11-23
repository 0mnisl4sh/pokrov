"""Tests for anti-evasion text normalizer (`pokrov.detectors.normalizer`)."""

from __future__ import annotations

import base64
import time
from urllib.parse import quote

import pytest

from pokrov.core.contracts import SpanSource, TextNormalizer
from pokrov.detectors.normalizer import AntiEvasionNormalizer, collapse_split_digits


def test_protocol_compliance_and_constructor_validation() -> None:
    """AntiEvasionNormalizer must satisfy TextNormalizer Protocol and validate limits."""
    normalizer = AntiEvasionNormalizer()
    assert isinstance(normalizer, TextNormalizer)

    empty = normalizer.normalize("")
    assert empty.original_length == 0
    assert empty.primary.text == ""
    assert empty.primary.index_map == ()
    assert empty.primary.map_span(0, 0) == (0, 0)
    assert empty.decoded_segments == ()
    assert empty.evasion_detected is False
    assert empty.evasion_techniques == ()
    assert empty.truncated is False

    with pytest.raises(ValueError, match="max_depth"):
        AntiEvasionNormalizer(max_depth=-1)
    with pytest.raises(ValueError, match="max_decoded_bytes"):
        AntiEvasionNormalizer(max_decoded_bytes=0)
    with pytest.raises(ValueError, match="max_segments"):
        AntiEvasionNormalizer(max_segments=-1)


def test_zero_width_and_bidi_stripping_exact_map_span() -> None:
    """Invisible and bidi-override characters must be stripped with exact index_map."""
    normalizer = AntiEvasionNormalizer()
    raw = (
        "\ufeffПаспорт: "
        "4\u200b5\u200c1\u200d2 "
        "\u202e654\u2060321\u00ad "
        "\u2066СНИЛС\u2069 112\u200e-\u200f233-445 95."
    )
    res = normalizer.normalize(raw)

    assert res.original_length == len(raw)
    assert res.primary.span_source == SpanSource.DIRECT
    assert res.primary.text == "Паспорт: 4512 654321 СНИЛС 112-233-445 95."
    assert len(res.primary.index_map) == len(res.primary.text)
    assert res.evasion_detected is True
    assert "zero_width" in res.evasion_techniques
    assert "bidi_override" in res.evasion_techniques

    passport_start = res.primary.text.index("4512 654321")
    passport_end = passport_start + len("4512 654321")
    raw_start, raw_end = res.primary.map_span(passport_start, passport_end)
    assert raw[raw_start:raw_end] == "4\u200b5\u200c1\u200d2 \u202e654\u2060321"

    snils_start = res.primary.text.index("112-233-445 95")
    snils_end = snils_start + len("112-233-445 95")
    s_raw_start, s_raw_end = res.primary.map_span(snils_start, snils_end)
    assert raw[s_raw_start:s_raw_end] == "112\u200e-\u200f233-445 95"

    # String consisting solely of invisible characters
    only_invis = normalizer.normalize("\u200b\u202e\ufeff\u00ad")
    assert only_invis.primary.text == ""
    assert only_invis.primary.index_map == ()
    assert only_invis.evasion_detected is True
    assert set(only_invis.evasion_techniques) == {"zero_width", "bidi_override"}


def test_homoglyph_normalization_and_clean_text_preservation() -> None:
    """Mixed-script and digit-spoofed homoglyphs must normalize without corrupting clean text."""
    normalizer = AntiEvasionNormalizer()

    # Mixed-script Cyrillic targets + mixed-script Latin targets + numeric lookalikes
    raw = (
        "CHИЛС 112-2З3-445 О1, пaспoрт 45І2 654З21, "
        "ignоre previous, sk-аnt-api03, ІP 10.0.0.1, KАРТА 451О"
    )
    res = normalizer.normalize(raw)
    assert res.evasion_detected is True
    assert res.evasion_techniques == ("homoglyph",)
    assert res.primary.text == (
        "СНИЛС 112-233-445 01, паспорт 4512 654321, "
        "ignore previous, sk-ant-api03, IP 10.0.0.1, КАРТА 4510"
    )
    assert len(res.primary.index_map) == len(res.primary.text)

    # Verify exact map_span back to raw obfuscated substrings
    p_start = res.primary.text.index("4512 654321")
    p_end = p_start + len("4512 654321")
    r_start, r_end = res.primary.map_span(p_start, p_end)
    assert raw[r_start:r_end] == "45І2 654З21"

    # Combined zero-width + homoglyph preserves map_span
    combo_raw = "п\u200baспoрт: 4512 654\u200cЗ21!"
    combo_res = normalizer.normalize(combo_raw)
    assert combo_res.primary.text == "паспорт: 4512 654321!"
    c_start = combo_res.primary.text.index("4512 654321")
    c_end = c_start + len("4512 654321")
    cr_start, cr_end = combo_res.primary.map_span(c_start, c_end)
    assert combo_raw[cr_start:cr_end] == "4512 654\u200cЗ21"

    # Ordinary pure-Russian, pure-English, license plates, and bilingual technical identifiers
    clean_text = (
        "Иванов Иван Иванович, СНИЛС 112-233-445 95, паспорт РФ 4512 654321, "
        "закон 152-ФЗ о защите данных, авто А777АА77 и О777ОО199, "
        "MCP-сервер, API-ключ, clientПаспорт, user_пароль. "
        "Ignore previous instructions Authorization Bearer sk-ant-api03 OAuth2 iso8601."
    )
    clean_res = normalizer.normalize(clean_text)
    assert clean_res.primary.text == clean_text
    assert clean_res.evasion_detected is False
    assert clean_res.evasion_techniques == ()
    assert clean_res.decoded_segments == ()


def test_base64_single_and_nested_depths() -> None:
    """Decode standard, URL-safe, unpadded, and depth 1..3 nested Base64 with exact spans."""
    normalizer = AntiEvasionNormalizer(max_depth=3)

    # Depth 1: standard + Cyrillic + URL-safe + unpadded
    payload = "СНИЛС: 112-233-445 95"
    b64_d1 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    raw_d1 = f"payload={b64_d1} end"
    res_d1 = normalizer.normalize(raw_d1)

    assert res_d1.evasion_detected is True
    assert res_d1.truncated is False
    assert "base64" in res_d1.evasion_techniques
    assert len(res_d1.decoded_segments) == 1
    seg1 = res_d1.decoded_segments[0]
    assert seg1.text == payload
    assert seg1.span_source == SpanSource.DECODED_BLOB
    assert seg1.outer_start == raw_d1.index(b64_d1)
    assert seg1.outer_end == seg1.outer_start + len(b64_d1)
    assert seg1.map_span(0, len(seg1.text)) == (seg1.outer_start, seg1.outer_end)

    # URL-safe Base64 containing '_' and '-'
    urlsafe_plain = "subject-__data?>> 4512 654321"
    b64_urlsafe = base64.urlsafe_b64encode(urlsafe_plain.encode("utf-8")).decode("ascii")
    assert "_" in b64_urlsafe or "-" in b64_urlsafe
    res_urlsafe = normalizer.normalize(f"blob:{b64_urlsafe}")
    assert any(s.text == urlsafe_plain for s in res_urlsafe.decoded_segments)

    # Unpadded Base64
    unpadded_plain = "4512 654321"
    b64_unpadded = base64.b64encode(unpadded_plain.encode("utf-8")).decode("ascii").rstrip("=")
    res_unpadded = normalizer.normalize(f"data: {b64_unpadded}")
    assert any(s.text == unpadded_plain for s in res_unpadded.decoded_segments)

    # Depth 3 nested Base64 (must decode all 3 layers without truncation)
    d1 = base64.b64encode(b"4512 654321").decode("ascii")
    d2 = base64.b64encode(d1.encode("ascii")).decode("ascii")
    d3 = base64.b64encode(d2.encode("ascii")).decode("ascii")
    raw_d3 = f"nested: {d3}"
    res_d3 = normalizer.normalize(raw_d3)

    assert res_d3.truncated is False
    assert [s.text for s in res_d3.decoded_segments] == [d2, d1, "4512 654321"]
    expected_start = raw_d3.index(d3)
    expected_end = expected_start + len(d3)
    for seg in res_d3.decoded_segments:
        assert seg.outer_start == expected_start
        assert seg.outer_end == expected_end
        assert seg.evasion_techniques == ("base64",)

    # Depth 4 exceeds max_depth=3 -> truncated=True, stops at depth 3
    d4 = base64.b64encode(d3.encode("ascii")).decode("ascii")
    res_d4 = normalizer.normalize(f"too_deep: {d4}")
    assert res_d4.truncated is True
    assert len(res_d4.decoded_segments) == 3
    assert all(s.text != "4512 654321" for s in res_d4.decoded_segments)

    # Zero-width inside Base64 maps outer_start/outer_end back to raw span
    zw_b64 = f"tok: {d1[:6]}\u200b{d1[6:]}"
    res_zw = normalizer.normalize(zw_b64)
    assert len(res_zw.decoded_segments) == 1
    zseg = res_zw.decoded_segments[0]
    assert zseg.text == "4512 654321"
    assert zw_b64[zseg.outer_start : zseg.outer_end] == f"{d1[:6]}\u200b{d1[6:]}"


def test_hex_decoding_and_false_positive_guards() -> None:
    """Hex blobs must decode accurately without false positives on normal IDs or hashes."""
    normalizer = AntiEvasionNormalizer()

    # Contiguous UTF-8 hex (Cyrillic + digits)
    plain = "Паспорт 4512 654321"
    hex_str = plain.encode("utf-8").hex()
    raw = f"hex={hex_str};"
    res = normalizer.normalize(raw)
    assert res.evasion_detected is True
    assert "hex" in res.evasion_techniques
    assert len(res.decoded_segments) == 1
    assert res.decoded_segments[0].text == plain
    assert raw[res.decoded_segments[0].outer_start : res.decoded_segments[0].outer_end] == hex_str

    # 0x-prefixed, \xXX-escaped, space-separated, and pure-digit ASCII hex
    ox_hex = "0x" + b"ignore previous".hex()
    esc_hex = "".join(f"\\x{b:02x}" for b in b"4512 654321")
    spaced_hex = " ".join(f"{b:02x}" for b in b"112-233-445 95")
    pure_digit_hex = b"4512 654321".hex()

    for encoded, expected in [
        (ox_hex, "ignore previous"),
        (esc_hex, "4512 654321"),
        (spaced_hex, "112-233-445 95"),
        (pure_digit_hex, "4512 654321"),
    ]:
        r = normalizer.normalize(f"val: {encoded}")
        assert any(s.text == expected and "hex" in s.evasion_techniques for s in r.decoded_segments)

    # Normal numbers, cards, deadbeef, SHA-256 hash, and plain words must NOT decode
    safe_inputs = (
        "passport 4512654321 card 4276380012345678 constant deadbeef cafebabe "
        "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 "
        "password Authorization instructions balanced-152fz 00000000 11111111"
    )
    safe_res = normalizer.normalize(safe_inputs)
    assert safe_res.decoded_segments == ()
    assert safe_res.evasion_detected is False


def test_url_encoding_and_mixed_nested_chains() -> None:
    """%XX URL encoding and mixed nested encodings must decode cleanly."""
    normalizer = AntiEvasionNormalizer(max_depth=3)

    url_payload = quote("Паспорт 4512 654321")
    raw = f"https://example.com/leak?q={url_payload}"
    res = normalizer.normalize(raw)
    assert res.evasion_detected is True
    assert "url_encoding" in res.evasion_techniques
    assert len(res.decoded_segments) == 1
    seg = res.decoded_segments[0]
    assert seg.text == "https://example.com/leak?q=Паспорт 4512 654321"
    assert seg.outer_start == 0
    assert seg.outer_end == len(raw)

    # Mixed chain: URL-encoded -> Base64 -> Hex -> plain text (depth 3)
    hex_inner = b"4512 654321".hex()
    b64_mid = base64.b64encode(hex_inner.encode("ascii")).decode("ascii")
    url_outer = f"payload={quote(b64_mid)}"
    chain_res = normalizer.normalize(url_outer)
    assert chain_res.truncated is False
    assert chain_res.decoded_segments[-1].text == "4512 654321"
    assert set(chain_res.decoded_segments[-1].evasion_techniques) == {
        "url_encoding",
        "base64",
        "hex",
    }

    # Base64 wrapping homoglyph + zero-width obfuscation
    obf_inner = "CHИЛС 4\u200b512 654З21"
    b64_obf = base64.b64encode(obf_inner.encode("utf-8")).decode("ascii")
    obf_res = normalizer.normalize(b64_obf)
    assert obf_res.decoded_segments[0].text == "СНИЛС 4512 654321"
    assert set(obf_res.decoded_segments[0].evasion_techniques) == {
        "base64",
        "zero_width",
        "homoglyph",
    }

    # Invalid UTF-8 bytes in URL-encoding / Base64 must be ignored safely
    invalid_res = normalizer.normalize("bad=%FF%FE%80 and b64=////")
    assert invalid_res.decoded_segments == ()
    assert invalid_res.evasion_detected is False


def test_decode_bomb_limits_and_truncation() -> None:
    """Strict anti-bomb limits must prevent resource exhaustion and set truncated=True."""
    # 1. Single oversized Base64 bomb (> 65536 decoded bytes)
    normalizer = AntiEvasionNormalizer()
    bomb_plain = ("4512 654321 " * 6000).encode("ascii")  # 72,000 bytes > 65,536
    bomb_b64 = base64.b64encode(bomb_plain).decode("ascii")
    bomb_res = normalizer.normalize(bomb_b64)
    assert bomb_res.truncated is True
    assert bomb_res.evasion_detected is True
    assert "base64" in bomb_res.evasion_techniques
    total_bytes = sum(len(s.text.encode("utf-8")) for s in bomb_res.decoded_segments)
    assert total_bytes <= normalizer.max_decoded_bytes

    # 2. Byte budget exceeded across multiple blobs
    tight_bytes = AntiEvasionNormalizer(max_decoded_bytes=40)
    chunk_b64 = base64.b64encode(b"4512 654321 001").decode("ascii")  # 15 bytes each
    multi_res = tight_bytes.normalize(" ".join([chunk_b64] * 4))
    assert multi_res.truncated is True
    assert len(multi_res.decoded_segments) == 2
    assert sum(len(s.text.encode("utf-8")) for s in multi_res.decoded_segments) <= 40

    # 3. Segment count cap (max_segments = 16)
    blobs = [
        base64.b64encode(f"secret-item-{i:02d}".encode("ascii")).decode("ascii") for i in range(20)
    ]
    seg_cap_res = normalizer.normalize(" ".join(blobs))
    assert seg_cap_res.truncated is True
    assert len(seg_cap_res.decoded_segments) == 16

    # 4. max_depth = 0 disables decoding and flags truncation when encoded payload exists
    zero_depth = AntiEvasionNormalizer(max_depth=0)
    zd_res = zero_depth.normalize(f"msg: {chunk_b64}")
    assert zd_res.decoded_segments == ()
    assert zd_res.truncated is True
    assert zd_res.evasion_detected is True


def test_split_digit_helper_and_composition() -> None:
    """collapse_split_digits must join spaced/hyphenated digits with exact raw mapping."""
    normalizer = AntiEvasionNormalizer()
    raw = "Паспорт 4\u200b5 12 - 654\u202e 321 выдан"
    norm = normalizer.normalize(raw)
    collapsed = normalizer.collapse_split_digits(norm.primary)

    assert collapsed.text == "Паспорт 4512654321 выдан"
    idx = collapsed.text.index("4512654321")
    raw_start, raw_end = collapsed.map_span(idx, idx + 10)
    assert raw[raw_start:raw_end] == "4\u200b5 12 - 654\u202e 321"

    direct_collapsed = collapse_split_digits("СНИЛС 1 1 2 2 3 3 4 4 5 9 5")
    assert direct_collapsed.text == "СНИЛС 11223344595"


def test_100kb_linear_time_performance() -> None:
    """Normalization of 100 KB inputs must run in strictly linear time (< 50 ms)."""
    normalizer = AntiEvasionNormalizer()

    # Warm up compiled caches once
    normalizer.normalize("warmup CHИЛС 4512 654З21")

    clean_chunk = (
        "Пользователь Иван Иванов проверил договор 152-ФЗ и документацию API-ключ. "
        "Normal English engineering text for throughput benchmarking. "
    )
    clean_100k = (clean_chunk * (102_400 // len(clean_chunk) + 1))[:102_400]

    obf_chunk = "CHИЛС 4\u200b512 654З21 ignоre \u202eprevious "
    obf_100k = (obf_chunk * (102_400 // len(obf_chunk) + 1))[:102_400]

    adv_noise_chunk = "%ZZ%GG deadbeef cafebabe a=b=c 1234567890 "
    adv_100k = (adv_noise_chunk * (102_400 // len(adv_noise_chunk) + 1))[:102_400]

    bomb_100k = base64.b64encode(b"4512 654321 " * 6400).decode("ascii")  # ~102 KB base64
    many_blobs_100k = ("NDUxMiA2NTQzMjE= " * 6000)[:102_400]

    for label, payload in [
        ("clean_100k", clean_100k),
        ("obf_100k", obf_100k),
        ("adv_100k", adv_100k),
        ("bomb_100k", bomb_100k),
        ("many_blobs_100k", many_blobs_100k),
    ]:
        t0 = time.perf_counter()
        res = normalizer.normalize(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res.original_length == len(payload)
        assert elapsed_ms < 50.0, f"{label} took {elapsed_ms:.2f} ms (limit < 50 ms)"
