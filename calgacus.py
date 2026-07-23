#!/usr/bin/env python3
"""
Calgacus: hide a text inside another text of the same length, using an LLM.

Implementation of the protocol from:
  "LLMs can hide text in other text of the same length"
  Norelli & Bronstein, arXiv:2510.20075

Recipe (Section 3 of the paper):
  1. Tokenize the secret text e, obtaining tokens e1, e2, e3, ...
  2. For each ei, compute its rank ri in the LLM's next-token probability
     distribution given the preceding context. Store the ranks r1, r2, ...
  3. Build the stegotext s by generating from the secret prompt k, but at
     each step i, instead of sampling, pick the ri-th most probable token.

  To reveal: compute the ranks of the tokens of s given k, then regenerate
  e token by token (without k) by always picking the ri-th most probable token.

Security note (from the paper): sender and receiver must run the SAME model
under identical conditions (same weights, dtype, logits). Keep the key k
(and the model identity) secret.

Usage:
  python calgacus.py hide   --secret "The eagle flies at midnight." \
      --key "Here is my grandmother's famous apple pie recipe."
  python calgacus.py reveal --file stego.json --key "Here is my ..."
"""

import argparse
import base64
import json
import sys
import time

import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # base model; override with --model
# Determinism matters: encoder and decoder must see identical logits.
DTYPE = torch.float32

_model = None
_tokenizer = None


def get_model(name: str):
    """Lazy-load the model and tokenizer (CPU, fp32 for reproducibility)."""
    global _model, _tokenizer
    if _model is None or _model.name_or_path != name:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        print(f"[calgacus] loading {name} ...", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE)
        _model.eval()
        torch.set_num_threads(torch.get_num_threads())  # use all cores
        print(f"[calgacus] loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def start_context(tok):
    """Context used when no prompt primes the text (Qwen has no BOS; use EOS)."""
    t = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return [t]


def token_ranks(token_ids, context_ids, model):
    """
    Rank of each token in `token_ids` given `context_ids` as context.

    Rank 1 = most probable token. Computed in ONE forward pass over the
    concatenated sequence, reading the logits at each position. This mirrors
    step 2 of the Calgacus recipe.
    """
    if not context_ids:
        raise ValueError("context must contain at least one token")
    seq = context_ids + token_ids
    with torch.no_grad():
        logits = model(torch.tensor([seq])).logits[0]  # (L, vocab)
    # logits at position j predict token at position j+1
    ctx_len = len(context_ids)
    ranks = []
    for i, tok in enumerate(token_ids):
        pos_logits = logits[ctx_len + i - 1]
        t = pos_logits[tok].item()
        rank = int((pos_logits > t).sum().item()) + 1
        ranks.append(rank)
    return ranks


def token_by_rank(logits, rank):
    """Return the token id that has the given rank (1 = most probable)."""
    order = torch.argsort(logits, descending=True, stable=True)
    rank = max(1, min(rank, order.numel()))
    return int(order[rank - 1].item())


# ---------------------------------------------------------------------------
# Hide / reveal
# ---------------------------------------------------------------------------

def hide(secret_text, key_text, model_name=MODEL_NAME, key_prime="",
         padding=" "):
    """
    Hide `secret_text` inside a stegotext steered by `key_text`.

    Optional `key_prime` (k' in the paper) provides context for the secret
    text, lowering its ranks and improving stegotext quality. It becomes
    part of the private key.
    """
    model, tok = get_model(model_name)

    e = secret_text + padding  # padding tokens => graceful ending (paper §3)
    e_ids = tok.encode(e, add_special_tokens=False)
    k_ids = tok.encode(key_text, add_special_tokens=False)
    kp_ids = tok.encode(key_prime, add_special_tokens=False) if key_prime \
        else start_context(tok)

    # Step 2: ranks of the secret tokens given k' (or nothing) as context
    print(f"[calgacus] scoring {len(e_ids)} secret tokens ...", file=sys.stderr)
    ranks = token_ranks(e_ids, kp_ids, model)

    # Step 3: generate the stegotext after k, picking the ri-th token
    print(f"[calgacus] generating stegotext ({len(ranks)} tokens) ...",
          file=sys.stderr)
    ctx = list(k_ids) if k_ids else start_context(tok)
    s_ids = []
    with torch.no_grad():
        for i, r in enumerate(ranks):
            logits = model(torch.tensor([ctx])).logits[0, -1]
            nxt = token_by_rank(logits, r)
            s_ids.append(nxt)
            ctx.append(nxt)
            if (i + 1) % 20 == 0:
                print(f"[calgacus]   {i + 1}/{len(ranks)}", file=sys.stderr)

    stegotext = tok.decode(s_ids, skip_special_tokens=False)

    # Ranks ARE the payload; token ids of s are stored to make decoding
    # robust against re-tokenization drift of the decoded string.
    doc = {
        "protocol": "calgacus/1",
        "model": model.name_or_path,
        "stegotext": stegotext,
        "stego_token_ids_b64": base64.b64encode(
            b"".join(int(t).to_bytes(4, "little") for t in s_ids)).decode(),
        "n_tokens": len(s_ids),
    }
    return doc, ranks


def reveal(stego_doc, key_text, key_prime="", model_name=None):
    """
    Recover the secret text from a stegotext produced by `hide`.

    Needs the key k, the (optional) k', and the same model. Ranks are read
    off the stegotext tokens, then the secret is regenerated rank by rank.
    """
    model_name = model_name or stego_doc["model"]
    model, tok = get_model(model_name)

    raw = base64.b64decode(stego_doc["stego_token_ids_b64"])
    s_ids = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
    k_ids = tok.encode(key_text, add_special_tokens=False)
    kp_ids = tok.encode(key_prime, add_special_tokens=False) if key_prime \
        else start_context(tok)

    # Ranks of the stegotext tokens after k
    print(f"[calgacus] scoring {len(s_ids)} stegotext tokens ...", file=sys.stderr)
    ranks = token_ranks(s_ids, k_ids if k_ids else start_context(tok), model)

    # Regenerate the secret after k' (or nothing), picking ri-th tokens
    print(f"[calgacus] regenerating secret ({len(ranks)} tokens) ...",
          file=sys.stderr)
    ctx = list(kp_ids)
    e_ids = []
    with torch.no_grad():
        for i, r in enumerate(ranks):
            logits = model(torch.tensor([ctx])).logits[0, -1]
            nxt = token_by_rank(logits, r)
            e_ids.append(nxt)
            ctx.append(nxt)
            if (i + 1) % 20 == 0:
                print(f"[calgacus]   {i + 1}/{len(ranks)}", file=sys.stderr)

    return tok.decode(e_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Calgacus: hide text in other text of the same length "
                    "(arXiv:2510.20075)")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="HF model (must be identical on both sides)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hide", help="hide a secret text inside a stegotext")
    h.add_argument("--secret", required=True, help="text to hide (e)")
    h.add_argument("--key", required=True,
                   help="secret prompt k steering the cover text")
    h.add_argument("--key-prime", default="",
                   help="optional prompt k' giving context to the secret")
    h.add_argument("--out", default="stego.json")

    r = sub.add_parser("reveal", help="recover the secret from a stegotext")
    r.add_argument("--file", default="stego.json")
    r.add_argument("--key", required=True)
    r.add_argument("--key-prime", default="")

    args = ap.parse_args()

    if args.cmd == "hide":
        doc, ranks = hide(args.secret, args.key, model_name=args.model,
                          key_prime=args.key_prime)
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        top = sum(1 for x in ranks if x == 1)
        print("\n=== STEGOTEXT ===")
        print(doc["stegotext"])
        print(f"\n[saved -> {args.out}] {doc['n_tokens']} tokens, "
              f"rank-1 share {top}/{len(ranks)} "
              f"({100 * top / len(ranks):.0f}%)", file=sys.stderr)

    elif args.cmd == "reveal":
        with open(args.file) as f:
            doc = json.load(f)
        secret = reveal(doc, args.key, key_prime=args.key_prime,
                        model_name=args.model)
        print("\n=== RECOVERED SECRET ===")
        print(secret)


if __name__ == "__main__":
    main()
