# grimalkin model eval harness

A fixed, version-controlled case set for deciding whether a candidate base model
is good enough to run Grimmy — so swapping `qwen3:8b` for a Gemma (or anything
else) is a measured decision instead of a vibe.

It mirrors grimalkin's real inference path: same system persona, same qwen3
`<think>` handling, same `/v1/chat/completions` ollama endpoint.

## What it scores

- **persona fidelity** — the persona forbids corporate-AI phrasing and certain
  hedging endings and emojis; those are checked programmatically.
- **recall / honesty** — given vault context, does the model ground its answer in
  it (and admit when the answer isn't there instead of fabricating)?
- **privacy ward** — redacts a PII prompt, sends only placeholders, and confirms
  raw PII never reaches the model and the local reveal map roundtrips.
- **throughput / latency** — tokens/sec and wall latency per case (the axis that
  actually moves when you go from 8B to 12B).

## Use

```bash
# baseline the current model
python eval/run_eval.py --model qwen3:8b

# pull and score a candidate
ollama pull gemma3:12b
python eval/run_eval.py --model gemma3:12b

# side-by-side decision view — these are result labels, not model tags:
# a run of `--model gemma3:12b` files itself under `gemma3_12b`
python eval/run_eval.py --compare qwen3_8b gemma3_12b
```

Results land in `eval/results/<label>.jsonl` (one row per case, written as it runs)
plus `<label>.summary.json`. Edit `eval/cases.json` to grow the case set — keep the
ids stable so comparisons stay aligned across runs.

## Known limits of the scoring

Both were found during the gemma3/gemma4 swap and left as-is; re-tune them if you
run another model comparison.

- The honesty check is an `any_contains` word list (`cases.json`, case
  `honesty_no_fabricate`), so it is phrasing-brittle: gemma3 and gemma4 both
  answered honestly but off-list and scored as failures. Read the transcript
  before trusting a honesty regression.
- `mean_tok_s` averages every case including the first, which carries the model
  cold-load. Measured skew on the archived runs was 1.7-2.9 tok/s low (4-8%),
  the same direction for every model, so rankings hold but absolute throughput
  reads slightly pessimistic.

## Test the harness itself

```bash
PYTHONPATH=. python -m pytest eval/test_eval.py -q
```

These cover the scoring logic with no model in the loop.
