# Calgacus — hide text in other text of the same length

Implementation of **"LLMs can hide text in other text of the same length"**
(Norelli & Bronstein, arXiv:2510.20075). A secret text `e` is hidden inside a
completely different, plausible text `s` of the *same token length*, recoverable
exactly by anyone who knows the key.

## How it works (Section 3 of the paper)

1. Tokenize the secret `e` → `e1, e2, e3, ...`
2. For each `ei`, record its **rank** `ri` in the LLM's next-token distribution
   given the preceding context (optionally primed with a prompt `k'`).
3. Generate the stegotext `s` by prompting the LLM with the secret key `k`,
   but instead of sampling, always pick the `ri`-th most probable token.

To reveal: read the ranks of `s`'s tokens after `k`, then regenerate `e` after
`k'` by picking the `ri`-th token at each step.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install transformers
```

Default model: `Qwen/Qwen2.5-1.5B` (ungated, ~3 GB, runs on CPU). The paper
used Llama-3-8B; any small model works, bigger = more coherent stegotexts.
To use **Gemma** (gated — needs `HF_TOKEN` with license accepted):

```bash
export HF_TOKEN=...   # accept license at huggingface.co/google/gemma-3-1b-pt
python calgacus.py --model google/gemma-3-1b-pt hide ...
```

## Usage

```bash
# Hide: the key steers topic/tone of the cover text
.venv/bin/python calgacus.py hide \
  --secret "Caesar has failed the Republic." \
  --key "Here it is: the infamous British roasted boar with mint sauce." \
  --out stego.json

# Reveal (same machine / same model / same key)
.venv/bin/python calgacus.py reveal --file stego.json \
  --key "Here it is: the infamous British roasted boar with mint sauce."
```

Optional `--key-prime` primes the *secret* side with context (the paper's `k'`
variation): lowers the secret's ranks → more coherent stegotext, at the cost of
a larger private key. `stego.json` contains the stegotext plus its token ids
(re-tokenizing raw text can drift under BPE, so ids are used for decoding).

## Verified round-trips (Qwen2.5-1.5B, CPU)

| Secret | Key (steers cover) | Stegotext (excerpt) |
|---|---|---|
| "Caesar has failed the Republic..." | roasted boar recipe | "...Date this delicious delicvey for four Thursday night... 1. Preheat air" |
| "The current government has repeatedly failed..." | eulogy of Caesar (with `k'`) | "...Hindered ten times from undertaking his great endeavors... through a combination of determination..." |

Both secrets were recovered **exactly**; a wrong key yields garbage
(LaTeX-looking noise), matching the paper's security analysis: the key and the
model identity are the whole secret.

## The Shibbolethian Theatre (Section 4)

`theatre.py` implements the paper's AI-safety scenario end-to-end: a company
ships only a *compliant* answer (trace `t` + answer `s`), but the user locally
recovers the unfiltered answer `u`. The demo is benign (a "secret recipe"
stands in for sensitive content):

```bash
.venv/bin/python theatre.py company   # Acts 1-4 -> act4.json
.venv/bin/python theatre.py user      # Act 5 -> recovers u exactly
```

Verified: the world sees *"...standard guidance: Let your meat stand in the
marinade 24 hours beforehand..."* while the user recovers the full secret
recipe — the open model never stored or generated it; it only supplied token
probabilities and the user's sampling policy did the rest.

## Detectability harness & deniability

```bash
# Reproduces the Fig. 4 effect: s is plausible but shifted below real text
.venv/bin/python calgacus.py score --file act4.json --key "<t>" --key-prime "<c>"
#   recovered secret e : -2.773 | stegotext s : -3.567 | samples: -2.948

# Deniability probe (Fig. 15): what does s decode to under bogus keys?
.venv/bin/python calgacus.py decoy --file stego.json --keys "key1" "key2" ...
```

Findings: wrong keys decode to gibberish (an attacker can't verify guesses);
a near-miss key differing in a few words also diverges completely; and when
the secret was primed with `k'`, even the *correct* `k` fails without `k'` —
the key-prime is an effective second factor.

## A negative result (why no "distribution-preserving Calgacus")

The paper's detectability gap comes from rank-1 tokens being "wasted" on
high-entropy positions. A natural fix is to permute the rank sequence so low
ranks land on low-entropy contexts. This is impossible to do exactly: any
fixed public rule for the permutation transmits only the *multiset* of ranks,
never their *order* — and the order is the payload. A recoverable permutation
must be a deterministic function of the final stegotext (a fixed-point
condition), and value-matching rules uniquely determine the arrangement,
leaving zero channel for the order. Hence full capacity + exact decoding +
distribution preservation cannot coexist; provably secure schemes (Meteor,
Zamir) escape only by giving up fixed capacity (variable-length stegotexts).

## rankstream.py — arbitrary bitstreams, adaptive base (new system)

`calgacus.py` preserves the paper's text-in-text protocol. `rankstream.py`
generalizes it into layers: **framing/crypto** (nonce ‖ ENC(len ‖ payload ‖
CRC32), HMAC-SHA256 CTR keystream) → **channel coding** (bits → ranks with an
adaptive per-position base `B_i = #{p ≥ τ}` capped by `--base-max`, rounded
down to a power of two; peaked contexts carry 0 bits and the digit is
deferred — plausibility gating) → **rendering** (rank-following generation
after key `k`).

```bash
printf 'Meet at the old lighthouse at dawn.' | .venv/bin/python rankstream.py plan   -k "Grandma's soup diary, entry 12:"
printf 'Meet at the old lighthouse at dawn.' | .venv/bin/python rankstream.py encode -k "Grandma's soup diary, entry 12:" -p hunt3r -B 16 --tau 1e-3 -o s.json
.venv/bin/python rankstream.py decode -k "Grandma's soup diary, entry 12:" -p hunt3r --file s.json
```

Verified: 58-byte payload → 159 tokens (3.87 bits/token) of soup-diary prose,
exact recovery, CRC-verified; wrong passphrase fails loudly. `plan` prints a
τ × base-max sweep from one calibration pass: e.g. τ=0.1 → 0.34 bits/token
vs τ=1e-4/B=64 → 4.13 bits/token — the rate/stealth frontier in one table.

**Keys**: still needed, now in two roles — `k` steers/syncs the cover text
(required), `--passphrase` adds real encryption (optional but recommended;
encrypt-then-embed also whitens the digit statistics).

**Temperature nuance** (found by a failing test): with an *absolute* floor τ,
raising `--temperature` only helps at peaked contexts; on already-flat
contexts it pushes mass below τ and *reduces* capacity.

Tests: `.venv/bin/python -m pytest test_rankstream.py` — 14 fast tests on a
deterministic mock LM (framing, CRC tamper, wrong key/passphrase, capacity
rule, gating/deferral, roundtrips), plus a `SLOW=1` real-model roundtrip.

## Caveats (from the paper)

- **Determinism**: both sides must run the identical model, dtype, and logits —
  different GPU/architecture numerics can break decoding.
- Quality of `s` depends on how well the LLM predicts `e` (low ranks). Random
  strings (hashes) produce gibberish; append padding tokens to `e` for a
  graceful ending (`hide` already appends one).
- Stegotexts are plausible but on average less probable than genuine text —
  see the paper's "Low entropy token choices" analysis.
