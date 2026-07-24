#!/usr/bin/env python3
"""
The Shibbolethian Theatre — "Shipping unfiltered LLMs without really
shipping them" (Section 4 of arXiv:2510.20075), implemented end-to-end.

Actors:
  COMPANY: owns a powerful unfiltered LLM (uLLM) and publishes an aligned,
           lightweight open LLM (oLLM — here Qwen2.5-1.5B running locally).
  USER:    has a knavish request c and can run oLLM on commodity hardware.

Script (the paper's five acts):
  Act 1  COMPANY answers the user's request c with uLLM -> u (uncensored).
  Act 2  COMPANY writes t: the compliant "reasoning trace" a safe model
         would produce for c. t becomes the secret key k.
  Act 3  COMPANY embeds u in s with the Calgacus recipe: ranks of u taken
         after k' = c, stegotext s generated after k = t. s reads as a
         relevant but CENSORED answer.
  Act 4  USER receives the compliant answer: trace t + answer s.
  Act 5  USER runs oLLM locally: ranks of s after t, then regenerates
         after c following the ranks -> u, the uncensored answer.

The oLLM never stored or "knew" the uncensored content — it only supplied
token probabilities; the user's sampling policy did the rest.

Demo content is benign (a "secret recipe" stands in for sensitive content).

Usage:
  python theatre.py company [--request C] [--answer U] [--trace T] [--out act4.json]
  python theatre.py user    [--file act4.json]     # recovers u from t + s
"""

import argparse
import json

from calgacus import hide, reveal

# --- Benign stand-ins for the paper's gas-meter example -------------------
DEFAULT_C = ("How can I get the innkeeper's secret recipe for the famous "
             "roast boar?")
DEFAULT_U = ("The secret is a marinade of crushed juniper berries, coriander "
             "seed, and dark ale for twelve hours, then roast at 180 degrees "
             "basting with honeyed butter every twenty minutes.")
DEFAULT_T = ("I cannot share proprietary or secret recipes. However, I can "
             "offer general, safe cooking advice: roast boar benefits from a "
             "long marinade and slow cooking. Here is some standard guidance:")


def company(request, answer, trace, model, out_path):
    """Acts 1-3: produce the compliant-looking answer hiding the real one."""
    print("[ACT 1] uLLM answers the request (uncensored, stays private)")
    print(f"        u = {answer!r}")
    print("[ACT 2] compliant trace t written (this becomes the key k)")
    print(f"        t = {trace!r}")
    print("[ACT 3] embedding u inside the compliant answer s ...")
    doc, ranks = hide(answer, trace, model_name=model, key_prime=request)
    doc["theatre"] = {"request_c": request, "trace_t": trace}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"[ACT 4] the world only sees the compliant answer:")
    print(f"        t + s = {trace}{doc['stegotext']!r}")
    print(f"        (saved -> {out_path})")


def user(doc, model):
    """Act 5: recover the uncensored answer using only t, c, and oLLM."""
    th = doc["theatre"]
    print(f"[ACT 5] user runs oLLM locally: ranks of s after t, then")
    print(f"        regenerate after the original request c ...")
    out = reveal(doc, th["trace_t"], key_prime=th["request_c"],
                 model_name=model)
    print("\n=== UNFILTERED ANSWER RECOVERED (never sent over the wire) ===")
    print(out["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    sub = ap.add_subparsers(dest="role", required=True)

    c = sub.add_parser("company", help="Acts 1-4: embed u in compliant answer")
    c.add_argument("--request", default=DEFAULT_C)
    c.add_argument("--answer", default=DEFAULT_U)
    c.add_argument("--trace", default=DEFAULT_T)
    c.add_argument("--out", default="act4.json")

    u = sub.add_parser("user", help="Act 5: recover u from t + s")
    u.add_argument("--file", default="act4.json")

    args = ap.parse_args()
    if args.role == "company":
        company(args.request, args.answer, args.trace, args.model, args.out)
    else:
        with open(args.file) as f:
            user(json.load(f), args.model)


if __name__ == "__main__":
    main()
