"""
Tests for rankstream. Uses a deterministic mock LM so the full protocol
logic (sync, gating, deferral, framing, crypto) is exercised in milliseconds
without downloading a model. A real-model smoke test is at the bottom
(marked slow).
"""

import hashlib

import pytest
import torch

from rankstream import (DesyncError, FrameError, bits_to_bytes, bytes_to_bits,
                        decode_bits, deframe, encode_bits, frame,
                        position_capacity)


# ---------------------------------------------------------------------------
# Deterministic mock LM: logits are a hash of the context tail, so encoder
# and decoder always agree. `spike_every` simulates peaked contexts.
# ---------------------------------------------------------------------------

class MockLM:
    def __init__(self, vocab=4096, spike_every=0, spike=30.0, seed=b"mock"):
        self.vocab = vocab
        self.spike_every = spike_every
        self.spike = spike
        self.seed = seed

    def logits(self, ctx):
        h = hashlib.sha256(
            self.seed + b"".join(int(t).to_bytes(4, "little")
                                 for t in ctx[-8:])).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))
        lg = torch.randn(self.vocab, generator=g)
        if self.spike_every and len(ctx) % self.spike_every == 0:
            lg[int(h[8]) % self.vocab] += self.spike  # peaked context
        return lg


# ---------------------------------------------------------------------------
# Layer 1: framing / crypto
# ---------------------------------------------------------------------------

def test_bits_bytes_roundtrip():
    data = bytes(range(256))
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_frame_roundtrip_plain_and_encrypted():
    payload = b"\x00binary\x01payload\xff" * 5
    for pw in ("", "hunt3r"):
        assert deframe(frame(payload, pw), pw) == payload


def test_frame_crc_tamper_detected():
    framed = bytearray(frame(b"hello", ""))
    framed[-3] ^= 0xFF
    with pytest.raises(FrameError):
        deframe(bytes(framed), "")


def test_frame_wrong_passphrase_detected():
    framed = frame(b"hello", "right")
    with pytest.raises(FrameError):
        deframe(framed, "wrong")


# ---------------------------------------------------------------------------
# Layer 2: capacity rule
# ---------------------------------------------------------------------------

def test_capacity_flat_context_capped_by_base_max():
    probs = torch.full((1000,), 1 / 1000)
    assert position_capacity(probs, tau=1e-4, base_max=64) == 6   # 2^6=64
    assert position_capacity(probs, tau=1e-4, base_max=50) == 5   # floor: 32


def test_capacity_peaked_context_is_free():
    probs = torch.zeros(1000)
    probs[7] = 0.999
    probs[:7] = 0.001 / 7
    assert position_capacity(probs, tau=0.5, base_max=64) == 0


def test_capacity_tau_zero_means_fixed_base():
    probs = torch.zeros(5000)
    probs[0] = 1.0
    assert position_capacity(probs, tau=0.0, base_max=8) == 3


# ---------------------------------------------------------------------------
# Layer 2+3: full protocol over the mock LM
# ---------------------------------------------------------------------------

def roundtrip(payload, passphrase="", key_ctx=(1, 2, 3), **kw):
    lm = MockLM(spike_every=kw.pop("spike_every", 5))
    bits = bytes_to_bits(frame(payload, passphrase))
    s_ids, written = encode_bits(lm.logits, list(key_ctx), bits, **kw)
    assert written == len(bits), "encoding infeasible under test settings"
    got = decode_bits(lm.logits, list(key_ctx), s_ids, **kw)
    return deframe(bits_to_bytes(got), passphrase), s_ids


def test_mock_roundtrip_binary_payload_encrypted():
    payload = bytes(range(64))
    recovered, _ = roundtrip(payload, passphrase="s3cret",
                             base_max=16, tau=1e-3, temperature=1.0)
    assert recovered == payload


def test_mock_roundtrip_fixed_base():
    payload = b"fixed base four"
    recovered, _ = roundtrip(payload, base_max=4, tau=0.0, temperature=1.0)
    assert recovered == payload


def test_deferral_at_peaked_contexts():
    # every 2nd context peaked: ~half the positions must be free (m=0),
    # so the stegotext is longer than fixed-base would be, but exact.
    payload = b"deferral test payload"
    (recovered, s_ids), lm = roundtrip(payload, base_max=16, tau=1e-3,
                                       temperature=1.0, spike_every=2), None
    assert recovered == payload


def test_wrong_key_fails_loudly():
    payload = b"top secret"
    lm = MockLM(spike_every=5)
    bits = bytes_to_bits(frame(payload, "pw"))
    s_ids, _ = encode_bits(lm.logits, [1, 2, 3], bits,
                           base_max=16, tau=1e-3, temperature=1.0)
    with pytest.raises((DesyncError, FrameError)):
        got = decode_bits(lm.logits, [9, 9, 9], s_ids,
                          base_max=16, tau=1e-3, temperature=1.0)
        deframe(bits_to_bytes(got), "pw")


def test_infeasible_when_everything_gated():
    lm = MockLM(spike_every=1)  # every context peaked
    bits = bytes_to_bits(frame(b"never fits", ""))
    s_ids, written = encode_bits(lm.logits, [0], bits, base_max=16,
                                 tau=0.9, temperature=1.0, max_tokens=64)
    assert written < len(bits)


def test_temperature_widens_capacity_at_peaked_contexts():
    # Peaked context: heating lifts the tail above tau -> more capacity.
    logits = torch.zeros(512)
    logits[0] = 8.0
    cold = position_capacity(torch.softmax(logits / 0.5, -1), 1e-3, 64)
    hot = position_capacity(torch.softmax(logits / 4.0, -1), 1e-3, 64)
    assert cold == 0
    assert hot > cold


def test_temperature_can_reduce_capacity_when_flat():
    # With an ABSOLUTE floor tau, heating a distribution that is already
    # flatter than tau pushes mass below the floor -> less capacity.
    # (Found by a failing test; worth documenting.)
    logits = torch.zeros(4096)  # uniform: p = 2.4e-4 < 1e-3 when hot
    cold = position_capacity(torch.softmax(logits / 0.5, -1), 1e-3, 64)
    hot = position_capacity(torch.softmax(logits / 4.0, -1), 1e-3, 64)
    assert hot <= cold


# ---------------------------------------------------------------------------
# Real model smoke test (slow: downloads/loads Qwen2.5-1.5B on CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available() and
                    __import__("os").environ.get("SLOW") != "1",
                    reason="set SLOW=1 to run the real-model roundtrip")
def test_real_model_roundtrip():
    from calgacus import get_model
    from rankstream import hf_next_logits
    model, tok = get_model("Qwen/Qwen2.5-1.5B")
    ctx = tok.encode("Grandma's soup diary:", add_special_tokens=False)
    payload = b"the eagle has landed"
    bits = bytes_to_bits(frame(payload, "pw"))
    s_ids, written = encode_bits(hf_next_logits(model), ctx, bits,
                                 base_max=16, tau=1e-3, temperature=1.0)
    assert written == len(bits)
    got = decode_bits(hf_next_logits(model), ctx, s_ids,
                      base_max=16, tau=1e-3, temperature=1.0)
    assert deframe(bits_to_bytes(got), "pw") == payload
