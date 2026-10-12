# Evaluation: retrieval and routing quality (Groww MF FAQ chatbot)

**Snapshot evaluated:** scraped facts and Chroma index dated 2026-05-07 (8 Quant schemes, 449 chunks)
**Reproduce:** `python evals/run_routing_eval.py` (plain Python) and `python evals/run_retrieval_eval.py` (needs torch, transformers, numpy, and the LFS model weights)

## Why this eval exists
The repo's own Phase 5 report shows 100% on every metric over 12 cases. Auditing it showed that number is not informative:
- **All 12 stored answers pad to exactly 3 sentences by repeating one sentence** (3 sentences, 2 unique, in 12 of 12). The "3-sentence contract" metric rewards padding.
- The Phase 3 "retrieval hit" only checks that the top 6 chunks include *some* chunk from the right scheme, not that the chunk contains the answer.

These evals are stricter and use ground truth from the scraped `scheme_facts.jsonl`, not hand-written expectations.

## How the system actually answers (important for reading the numbers)
For a known scheme + field, `backend/server.py` finds the scheme and field with rule-based matching, then looks up a structured fact and passes it to the LLM as a "fact hint". Vector retrieval is supporting context and the fallback. So there are two layers to evaluate separately.

## Eval A: rule-based routing (`run_routing_eval.py`)
720 generated queries: 8 schemes × 6 fields × 3 phrasings (English, Hinglish, paraphrase) × 5 scheme-name variants (canonical, no plan suffix, lowercase, compact "midcap", without "Quant").

| Slice | Scheme resolved | Field resolved | Correct fact returned |
|---|---|---|---|
| All | 100% | 72% | **72%** |
| English phrasing | 100% | 100% | 100% |
| Hinglish phrasing | 100% | 100% | 100% |
| **Paraphrase phrasing** | 100% | **17%** | **17%** |
| Riskometer field | 100% | 100% | 100% |
| Other 5 fields | 100% | 67% | 67% |

**Reading:** scheme resolution is robust, including the Large Cap vs Large and Mid Cap ambiguity. Field detection is a literal keyword list (e.g. "expense ratio", "nav", "aum"), so natural paraphrases such as "how much does X charge as fees", "current price per unit", "fund size in crores" or "which index is X compared against" return no field and skip the structured-fact path.

## Eval B: vector retrieval alone (`run_retrieval_eval.py`)
144 queries (8 schemes × 6 fields × 3 phrasings), top-6 chunks, a hit requires the right scheme **and** the true value in the chunk text.

| Slice | scheme@6 | answer-chunk@1 | @3 | @6 |
|---|---|---|---|---|
| All | 99% | 23% | 51% | **63%** |
| English | 100% | 31% | 58% | 67% |
| Hinglish | 100% | 29% | 56% | 69% |
| Paraphrase | 98% | 8% | 40% | 54% |
| expense ratio / NAV / benchmark | 100% | 42-50% | 88-92% | 100% |
| minimum SIP | 96% | 0% | 38% | 75% |
| **AUM** | 100% | 0% | 0% | **0%** |
| **riskometer** | 100% | 0% | 0% | **4%** |

I verified AUM and riskometer text exists in the chunks (e.g. "Fund size (AUM) ₹2,782.37 Cr"), so these are real retrieval failures, not matcher artefacts. The retriever reliably finds the right scheme but not the chunk that contains the answer.

## Findings
1. **Retrieval finds the scheme, not the answer.** 99% scheme hit vs 63% answer hit. Likely cause: tiny field chunks ("Fund size (AUM) ₹…") carry no scheme name, so the query's scheme name pulls scheme-title and navigation chunks instead. This is a hypothesis from the pattern, not yet tested.
2. **AUM and riskometer are effectively unretrievable by vector search** (0% and 4% at k=6).
3. **Keyword field matching breaks on paraphrases** (17% correct), so paraphrased queries fall through to the weakest layer, retrieval (54% answer@6).
4. **The reported 100% in Phase 5 is a format metric, not a faithfulness metric.**

## Limitations (say these out loud in an interview)
- Queries are **templated**, not real user logs. Real behaviour will be messier.
- Ground truth is the project's own scraped facts, so this measures the system against its source snapshot, not the source's accuracy on the live Groww page.
- The **LLM answer stage was not evaluated** (no Groq key available), so faithfulness and padding of generated text were only audited on the 12 stored responses.
- Retrieval eval uses exact cosine over re-embedded chunks with the same model, not Chroma's HNSW path (the stored index was written by a newer chromadb than `requirements.txt` pins and could not be opened).
- Single snapshot, 8 schemes, one AMC.

## Fixes this suggests (not yet implemented)
- Prefix every chunk with scheme name + field label before embedding (contextual chunk headers), and filter retrieval by scheme metadata.
- Replace the keyword field list with a synonym map or a small intent classifier; add the paraphrases above as a regression set.
- Replace "exactly 3 sentences" with a faithfulness check: every claim must be entailed by the retrieved/fact context, and duplicates fail.
- Re-run both scripts after the changes and report the delta.
