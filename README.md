# FilingDelta

Detects the material changes between two years of a company's SEC filing, then checks every dollar claim in the changed text against the company's structured XBRL data. Output is a brief where each line cites the evidence it came from; a contradicted claim pauses the run for human review.

Not investment advice. A research tool.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                 # unit tests + the eval gate
python -m evals.run       # metrics against labeled fixtures; exit 1 if any drops below its floor

export FILINGDELTA_UA="Your Name you@example.com"   # SEC requires a declared User-Agent
python -m filingdelta 320193                        # latest two 10-Ks for a CIK, Item 1A only
```

## Pipeline (LangGraph)

```
diff -> classify -> [no material change: stop] -> claims -> verify -> brief -> critic -> [contradicted claim: pause for review]
```

| Stage | What it does | Swap-in point |
|---|---|---|
| `diff` | Pairs paragraphs across years: modified / added / removed | `embed=`: any `list[str] -> list[vector]` (dense HF encoder) |
| `classify` | Labels each change; cosmetic ones stop here | `classifier=`: any `dict -> label` (distilled HF classifier) |
| `claims` | Pulls `$` amounts with their metric and year | token-classification model |
| `verify` | Compares to XBRL `companyfacts` within the stated rounding | already structured data |
| `brief` | Template lines, each ending in an evidence citation | LLM synthesizer |
| `critic` | Drops any line without a valid citation | LLM-as-judge on top |

State is plain dicts, so checkpointing works with any LangGraph checkpointer (in-memory here).

## Evals

`evals/fixtures/cases.json` holds hand-labeled filing pairs; `evals/thresholds.json` holds the floors; CI fails the build on a regression.

**The fixtures are small and synthetic. The numbers prove the harness works, not that the system is accurate on real filings.** Do not quote them as results. The next step is a real gold set from EDGAR (see below).

## Not built yet

- Retrieval layer (LlamaIndex, pgvector, hybrid search, reranker) and a pgvector vs Pinecone benchmark
- Real Hugging Face model adapters and the model-selection benchmark
- LLM synthesizer and judge, tracing, caching, cost routing
- Real gold set: label changes from actual 10-K pairs; backtest against 8-K Item 4.02 restatements
- Item 1A only; other sections and 10-Q/8-K are not parsed
- `python -m filingdelta` and the EDGAR client are tested against mocked HTTP only, not live EDGAR
- Known limits are marked `shortcut:` in the code (regex ignores negation, calendar-year matching for fiscal years, all-pairs diff)
