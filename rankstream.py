#!/usr/bin/env python3
"""
rankstream — layered steganography: arbitrary bitstream -> rank sequence -> text.

A generalization of Calgacus (arXiv:2510.20075, see calgacus.py) that splits
the pipeline into independent layers:

  Layer 1 (framing/crypto):  payload -> nonce || ENC(len || payload || CRC32)
  Layer 2 (channel coding):  bitstream -> rank sequence, with an ADAPTIVE
                             per-position base:
                                 B_i = #{tokens with p >= tau}  (clamped to
                                 base_max, rounded down to a power of two)
                             capacity at position i is m_i = log2(B_i) bits.
                             At peaked contexts m_i = 0: a free (rank-1) token
                             is emitted and the digit is deferred -- this is
                             plausibility gating, so deep ranks only ever
                             occur at flat contexts, like in natural text.
  Layer 3 (rendering):       rank sequence -> stegotext, by rank-following
                             generation after the steering key k.

Knobs:
  --base-max     cap on the per-position base (fixed base when tau=0)
  --tau          permissibility floor: minimum token probability (at the
                 given temperature) for a position to carry payload bits
  --temperature  flattens the distribution used for gating -> higher T means
                 more positions qualify, i.e. larger effective base
  --passphrase   encrypt-then-embed (HMAC-SHA256 CTR keystream). Without it,
                 framing + CRC integrity only.
  --key          steering text k: sets the topic of the cover text AND is
                 required for decode sync. Still needed (see README).

Commands:
  plan    estimate how many tokens/words a piped payload needs, swept over
          tau x base-max settings (one calibration pass)
  encode  embed a piped file/bitstream into a stegotext
  decode  recover the payload from a stegotext

Examples:
  echo -n "meet at dawn" | python rankstream.py plan --key "A recipe blog:"
  echo -n "meet at dawn" | python rankstream.py encode --key "A recipe blog:" \
      --passphrase hunt3r --out stego.json
  python rankstream.py decode --file stego.json --key "A recipe blog:" \
      --passphrase hunt3r
"""

import base64
import hashlib
import hmac
import json
import math
import os
import sys
import zlib
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from calgacus import get_model, start_context, token_by_rank

app = typer.Typer(help=__doc__, add_completion=False)
console = Console(stderr=True)

NONCE_LEN = 8
LEN_LEN = 4
CRC_LEN = 4
OVERHEAD = NONCE_LEN + LEN_LEN + CRC_LEN  # bytes added to every payload


# ---------------------------------------------------------------------------
# Layer 1: framing + crypto
# ---------------------------------------------------------------------------

def keystream(passphrase: str, nonce: bytes, nbytes: int) -> bytes:
    """HMAC-SHA256 in counter mode. Dependency-free stream cipher."""
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out += hmac.new(passphrase.encode(),
                        nonce + ctr.to_bytes(8, "big"),
                        hashlib.sha256).digest()
        ctr += 1
    return bytes(out[:nbytes])


def frame(payload: bytes, passphrase: str = "") -> bytes:
    """nonce || ENC(len || payload || crc32). Encrypt-then-embed."""
    nonce = os.urandom(NONCE_LEN)
    body = len(payload).to_bytes(LEN_LEN, "big") + payload
    body += zlib.crc32(body).to_bytes(CRC_LEN, "big")
    if passphrase:
        ks = keystream(passphrase, nonce, len(body))
        body = bytes(a ^ b for a, b in zip(body, ks))
    return nonce + body


class FrameError(Exception):
    pass


def deframe(data: bytes, passphrase: str = "") -> bytes:
    if len(data) < OVERHEAD:
        raise FrameError("not enough data for frame header")
    nonce, body = data[:NONCE_LEN], data[NONCE_LEN:]
    if passphrase:
        ks = keystream(passphrase, nonce, len(body))
        body = bytes(a ^ b for a, b in zip(body, ks))
    n = int.from_bytes(body[:LEN_LEN], "big")
    if n > len(body) - LEN_LEN - CRC_LEN:
        raise FrameError(f"bad length field ({n}); wrong key or passphrase?")
    payload = body[LEN_LEN:LEN_LEN + n]
    crc = int.from_bytes(body[LEN_LEN + n:LEN_LEN + n + CRC_LEN], "big")
    if crc != zlib.crc32(body[:LEN_LEN + n]):
        raise FrameError("CRC mismatch; wrong key, passphrase, or model?")
    return payload


# ---------------------------------------------------------------------------
# Bits <-> bytes
# ---------------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> list[int]:
    return [(b >> (7 - j)) & 1 for b in data for j in range(8)]


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | b
        out.append(v)
    return bytes(out)


# ---------------------------------------------------------------------------
# Layer 2: adaptive-base channel coding
# ---------------------------------------------------------------------------

def position_capacity(probs: torch.Tensor, tau: float, base_max: int) -> int:
    """m_i = floor(log2(min(base_max, #{p >= tau}))); 0 => free position."""
    eligible = int((probs >= tau).sum()) if tau > 0 else probs.numel()
    base = max(1, min(base_max, eligible))
    return base.bit_length() - 1  # floor(log2(base)); B_eff = 1 << m


def rank_of(logits: torch.Tensor, tok: int) -> int:
    return int((logits > logits[tok]).sum().item()) + 1


def gate_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=-1)


class DesyncError(Exception):
    pass


# ---------------------------------------------------------------------------
# Layer 3: rendering (encode) / reading (decode)
# ---------------------------------------------------------------------------

def encode_bits(next_logits, ctx, bits, base_max, tau, temperature,
                tail=6, max_tokens=4096):
    """
    Emit stegotext token ids carrying `bits`. `next_logits(ctx) -> tensor`
    abstracts the LM (so tests can plug in a mock). Returns (s_ids, written).
    After the payload, `tail` rank-1 tokens give a graceful ending.
    """
    s_ids, idx = [], 0
    ctx = list(ctx)
    while idx < len(bits) and len(s_ids) < max_tokens:
        logits = next_logits(ctx)
        m = position_capacity(gate_probs(logits, temperature), tau, base_max)
        if m == 0:  # peaked context: free token, digit deferred
            nxt = int(torch.argmax(logits))
        else:
            take = min(m, len(bits) - idx)
            d = 0
            for b in bits[idx:idx + take]:
                d = (d << 1) | b
            d <<= m - take  # zero-pad the final partial digit
            nxt = token_by_rank(logits, d + 1)
            idx += take
        s_ids.append(nxt)
        ctx.append(nxt)
    written = idx
    for _ in range(tail):
        logits = next_logits(ctx)
        nxt = int(torch.argmax(logits))
        s_ids.append(nxt)
        ctx.append(nxt)
    return s_ids, written


def decode_bits(next_logits, ctx, s_ids, base_max, tau, temperature):
    """Read the bitstream back off stegotext token ids."""
    bits = []
    ctx = list(ctx)
    for pos, tok in enumerate(s_ids):
        logits = next_logits(ctx)
        m = position_capacity(gate_probs(logits, temperature), tau, base_max)
        if m > 0:
            d = rank_of(logits, tok) - 1
            if d >= (1 << m):
                raise DesyncError(
                    f"position {pos}: token rank {d + 1} exceeds base "
                    f"{1 << m}; wrong key, model, or parameters?")
            bits += [(d >> j) & 1 for j in range(m - 1, -1, -1)]
        ctx.append(tok)
    return bits


# ---------------------------------------------------------------------------
# HF model adapter
# ---------------------------------------------------------------------------

def hf_next_logits(model):
    def f(ctx):
        with torch.no_grad():
            return model(torch.tensor([ctx])).logits[0, -1]
    return f


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_payload(infile: Optional[str]) -> bytes:
    if infile:
        with open(infile, "rb") as f:
            return f.read()
    return sys.stdin.buffer.read()


def _key_ids(tok, key: str):
    return tok.encode(key, add_special_tokens=False) if key else start_context(tok)


@app.command()
def plan(
    key: str = typer.Option("", "--key", "-k", help="steering/cover text"),
    infile: Optional[str] = typer.Option(None, "--in", "-i",
                                         help="payload file (default: stdin)"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B"),
    temperature: float = typer.Option(1.0, "--temperature", "-T"),
    cal_tokens: int = typer.Option(256, "--calibration-tokens",
                                   help="greedy tokens used to estimate rate"),
):
    """Estimate tokens/words needed to stego the piped payload, swept over
    tau x base-max (one calibration generation after the key)."""
    payload = _read_payload(infile)
    frame_bits = (len(payload) + OVERHEAD) * 8
    mdl, tok = get_model(model)
    ctx = _key_ids(tok, key)
    f = hf_next_logits(mdl)

    taus = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1]
    bmaxs = [2, 4, 8, 16, 32, 64]
    counts = {t: [] for t in taus}
    cal = list(ctx)
    console.print(f"[rankstream] calibrating {cal_tokens} tokens ...")
    for _ in range(cal_tokens):
        logits = f(cal)
        probs = gate_probs(logits, temperature)
        for t in taus:
            counts[t].append(int((probs >= t).sum()) if t > 0
                             else probs.numel())
        cal.append(int(torch.argmax(logits)))

    table = Table(title=f"capacity after key ({len(ctx)} tok) — payload "
                        f"{len(payload)} B + {OVERHEAD} B framing "
                        f"= {frame_bits} bits")
    table.add_column("tau")
    table.add_column("base max")
    table.add_column("bits/token", justify="right")
    table.add_column("tokens needed", justify="right")
    table.add_column("~words", justify="right")
    for t in taus:
        for B in bmaxs:
            rates = [min(B, c).bit_length() - 1 for c in counts[t]]
            rate = sum(rates) / len(rates)
            if rate <= 0:
                table.add_row(f"{t:g}", str(B), "0", "infeasible", "-")
                continue
            need = math.ceil(frame_bits / rate)
            table.add_row(f"{t:g}", str(B), f"{rate:.2f}", str(need),
                          f"{need * 0.75:.0f}")
    console.print(table)
    console.print("note: tau=0 => fixed base; capacity 0 rows need lower "
                  "tau or higher --temperature")


@app.command()
def encode(
    key: str = typer.Option("", "--key", "-k"),
    infile: Optional[str] = typer.Option(None, "--in", "-i"),
    out: str = typer.Option("stego_rs.json", "--out", "-o"),
    passphrase: str = typer.Option("", "--passphrase", "-p"),
    base_max: int = typer.Option(16, "--base-max", "-B"),
    tau: float = typer.Option(1e-3, "--tau"),
    temperature: float = typer.Option(1.0, "--temperature", "-T"),
    tail: int = typer.Option(6, "--tail"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B"),
):
    """Embed the piped payload into a stegotext steered by --key."""
    payload = _read_payload(infile)
    framed = frame(payload, passphrase)
    bits = bytes_to_bits(framed)
    mdl, tok = get_model(model)
    ctx = _key_ids(tok, key)
    console.print(f"[rankstream] {len(payload)} B payload -> {len(bits)} "
                  f"framed bits; embedding ...")
    s_ids, written = encode_bits(hf_next_logits(mdl), ctx, bits,
                                 base_max, tau, temperature, tail=tail)
    if written < len(bits):
        console.print(f"[red]infeasible: only {written}/{len(bits)} bits fit "
                      f"(lower tau / raise temperature / raise base-max)")
        raise typer.Exit(1)
    stegotext = tok.decode(s_ids, skip_special_tokens=False)
    doc = {
        "protocol": "rankstream/1",
        "model": mdl.name_or_path,
        "params": {"base_max": base_max, "tau": tau,
                   "temperature": temperature},
        "payload_bytes": len(payload),
        "framed_bits": len(bits),
        "stegotext": stegotext,
        "stego_token_ids_b64": base64.b64encode(
            b"".join(int(t).to_bytes(4, "little") for t in s_ids)).decode(),
    }
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    rate = len(bits) / max(1, len(s_ids) - tail)
    console.print(f"[rankstream] {len(s_ids)} tokens "
                  f"({rate:.2f} bits/token); saved -> {out}")
    print(stegotext)


@app.command()
def decode(
    key: str = typer.Option("", "--key", "-k"),
    file: str = typer.Option("stego_rs.json", "--file"),
    out: Optional[str] = typer.Option(None, "--out", "-o",
                                      help="payload file (default: stdout)"),
    passphrase: str = typer.Option("", "--passphrase", "-p"),
    model: str = typer.Option("", "--model", help="default: from stego file"),
):
    """Recover the payload from a stegotext."""
    with open(file) as fh:
        doc = json.load(fh)
    p = doc["params"]
    mdl, tok = get_model(model or doc["model"])
    raw = base64.b64decode(doc["stego_token_ids_b64"])
    s_ids = [int.from_bytes(raw[i:i + 4], "little")
             for i in range(0, len(raw), 4)]
    ctx = _key_ids(tok, key)
    console.print(f"[rankstream] reading {len(s_ids)} tokens ...")
    try:
        bits = decode_bits(hf_next_logits(mdl), ctx, s_ids,
                           p["base_max"], p["tau"], p["temperature"])
        payload = deframe(bits_to_bytes(bits), passphrase)
    except (DesyncError, FrameError) as e:
        console.print(f"[red]decode failed: {e}")
        raise typer.Exit(1)
    console.print(f"[rankstream] OK: {len(payload)} bytes "
                  f"(CRC verified)")
    if out:
        with open(out, "wb") as fh:
            fh.write(payload)
        console.print(f"[rankstream] payload -> {out}")
    else:
        sys.stdout.buffer.write(payload + b"\n")


if __name__ == "__main__":
    app()
