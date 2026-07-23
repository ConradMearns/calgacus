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

## Caveats (from the paper)

- **Determinism**: both sides must run the identical model, dtype, and logits —
  different GPU/architecture numerics can break decoding.
- Quality of `s` depends on how well the LLM predicts `e` (low ranks). Random
  strings (hashes) produce gibberish; append padding tokens to `e` for a
  graceful ending (`hide` already appends one).
- Stegotexts are plausible but on average less probable than genuine text —
  see the paper's "Low entropy token choices" analysis.
