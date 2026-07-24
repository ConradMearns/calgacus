#!/usr/bin/env python3
"""
sprite_demo — hide a 32x32, 8-color sprite inside a game-review stegotext.

Payload format (408 bytes): 8 RGB palette entries (24 B) + 1024 pixels
packed at 3 bits/pixel (384 B). Run once; prints stats for the HTML page.
"""

import base64
import json

from calgacus import get_model
from rankstream import (bits_to_bytes, bytes_to_bits, decode_bits, deframe,
                        encode_bits, frame, hf_next_logits)

# 16x16-ish space invader defined as 11x8, scaled to 32x32 with color bands
ROWS = [
    "..X.....X..",
    "...X...X...",
    "..XXXXXXX..",
    ".XX.XXX.XX.",
    "XXXXXXXXXXX",
    "X.XXXXXXX.X",
    "X.X.....X.X",
    "...XX.XX...",
]
PALETTE = [(15, 20, 25), (240, 244, 255), (80, 220, 100), (60, 200, 160),
           (60, 180, 220), (90, 140, 240), (150, 110, 240), (220, 90, 160)]

KEY = "My review of the new retro arcade game:"
PASS = "inv4d3r"
SETTINGS = dict(base_max=64, tau=1e-4, temperature=1.0)


def build_sprite():
    pix = []
    for y in range(32):
        src = ROWS[min(7, y // 4)]
        for x in range(32):
            if 5 <= x < 27 and src[(x - 5) // 2] == "X":
                pix.append(2 + (y // 5) % 6)      # colors 2..7 by band
            else:
                pix.append(0)                     # background
    bits = [b for p in pix for b in ((p >> 2) & 1, (p >> 1) & 1, p & 1)]
    palette = bytes(c for rgb in PALETTE for c in rgb)
    return palette + bits_to_bytes(bits)


def main():
    payload = build_sprite()
    print(f"sprite payload: {len(payload)} bytes "
          f"({len(PALETTE)} colors, 3 bpp)", flush=True)

    model, tok = get_model("Qwen/Qwen2.5-1.5B")
    ctx = tok.encode(KEY, add_special_tokens=False)
    f = hf_next_logits(model)

    bits = bytes_to_bits(frame(payload, PASS))
    print(f"framed: {len(bits)} bits; embedding ...", flush=True)
    s_ids, written = encode_bits(f, ctx, bits, tail=8, **SETTINGS)
    assert written == len(bits), "infeasible!"
    stegotext = tok.decode(s_ids, skip_special_tokens=False)
    print(f"encoded into {len(s_ids)} tokens "
          f"({len(bits) / len(s_ids):.2f} bits/token)", flush=True)

    got = decode_bits(f, ctx, s_ids, **SETTINGS)
    recovered = deframe(bits_to_bytes(got), PASS)
    assert recovered == payload, "MISMATCH"
    print("decoded: byte-identical to original ✓", flush=True)

    doc = {
        "key": KEY, "settings": SETTINGS,
        "payload_bytes": len(payload), "tokens": len(s_ids),
        "bits_per_token": round(len(bits) / len(s_ids), 2),
        "payload_b64": base64.b64encode(payload).decode(),
        "stegotext": stegotext,
    }
    with open("sprite_stego.json", "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print("saved -> sprite_stego.json", flush=True)
    print("\n--- STEGOTEXT ---")
    print(stegotext)


if __name__ == "__main__":
    main()
