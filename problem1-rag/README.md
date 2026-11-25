# Problem 1 — Cost-Efficient RAG on Postgres + pgvector

A question-answering service over a document corpus (PDF, HTML, Markdown). It stores vectors in
**Postgres + pgvector** instead of a managed vector database, and it is evaluated on
retrieval quality, answer quality, latency and cost.

**Headline results** (27 fixed questions, k = 5; full tables below):

| Layer | Result |
|---|---|
| Retrieval | Hit rate 1.00 · MRR 0.94 · nDCG@5 0.92 · evidence recall 0.98 (multi-document recall 0.75) |
| Answers | Faithfulness 1.00 · relevance 5.0/5 · correctness 5.0/5 (LLM judge) · gold answer contained in 88% · 3/3 unanswerable correctly declined |
| Latency | Vector search p50 3.7 ms / p95 4.7 ms · end-to-end p50 0.76 s / p95 1.19 s (LLM dominates) |
| Scale | HNSW p50 0.82 ms at 100K and 1.15 ms at 250K (measured), recall@5 0.98; about 2.5 ms projected at 10M |
| Cost | 60–83% cheaper than always-on Pinecone pods at every size. Against Pinecone serverless it depends on query volume (see §4.5) |

---

## 1. Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion (rag ingest / POST /ingest)"]
        F["PDF · HTML · MD files"] --> L["Loaders<br/>pypdf · BeautifulSoup · front-matter MD"]
        L --> H{"sha256 + chunk params<br/>same as stored?"}
        H -- yes --> SKIP["skip (unchanged)"]
        H -- no --> C["Sentence-aware token chunker<br/>256 tokens, 32 overlap"]
        C --> E1["bge-small-en-v1.5<br/>384-d, local CPU<br/>embeds 'title + chunk'"]
        E1 --> TX["One transaction: delete doc's old chunks,<br/>upsert document, insert chunks<br/>(deterministic chunk IDs)"]
    end
    TX --> DB[("Postgres 17 + pgvector 0.8<br/>documents · chunks<br/>HNSW cosine index · doc_type index")]

    subgraph Query["Query (rag query / POST /query)"]
        Q["question, k, doc_type filter"] --> E2["Embed query<br/>(bge query instruction)"]
        E2 --> S["Top-k HNSW search<br/>WHERE doc_type = ANY(filter)<br/>iterative index scan"]
        DB --> S
        S --> T{"any chunk with<br/>cosine ≥ MIN_SCORE?"}
        T -- no --> NA1["'No relevant context' answer<br/>LLM not called · 0 tokens"]
        T -- yes --> G["Groq gpt-oss-120b<br/>numbered passages + strict grounding prompt"]
        G --> V{"model replied<br/>INSUFFICIENT_CONTEXT?"}
        V -- yes --> NA2["'No relevant context' answer"]
        V -- no --> CIT["Normalise and parse [n] citations,<br/>map to chunk IDs"]
        CIT --> R["JSON: answer, citations, retrieved chunks,<br/>token usage, latency breakdown"]
    end
    R --> LOG["logs/queries.jsonl + log line:<br/>latency per stage, chunk counts, tokens"]
```

The "no relevant context" path has two layers. A similarity threshold stops clearly off-topic
questions before any LLM call. For questions that look on-topic but have no answer in the
corpus, the generator is told to reply with a fixed sentinel instead of guessing.

## 2. Setup and run (about 10 minutes)

**Prerequisites:** Python 3.11, [uv](https://docs.astral.sh/uv/), Docker Desktop, and a Groq API
key (free tier). The service was built on Windows 11 and is OS-independent. Embeddings run on CPU,
so no GPU is needed. The first run downloads the embedding model (about 130 MB) from Hugging Face.

**Environment variables** (all in `.env`; see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | `groq`, `openai/gpt-oss-120b` | Answer generator |
| `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_API_KEY` | `groq`, `qwen/qwen3.8-27b` | Evaluation judge (different model family). `gemini` and `mistral` also work |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5544/rag` | pgvector container from `docker-compose.yml` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `256`, `32` | Chunking, in tokens |
| `TOP_K`, `MIN_SCORE` | `5`, `0.5` | Retrieval depth, and the similarity below which a chunk is ignored |

No secrets are in the code or in git. Keys are read only from `.env`, which git ignores.

**Install, start the database, and ingest:**

```powershell
cd problem1-rag
uv sync                                  # creates .venv with Python 3.11 and installs everything
Copy-Item .env.example .env              # then put your Groq key in LLM_API_KEY and JUDGE_API_KEY
docker compose up -d --wait              # Postgres 17 + pgvector on localhost:5544
.\.venv\Scripts\rag.exe ingest           # ingests data/corpus (256-token chunks, 32 overlap)
.\.venv\Scripts\rag.exe ingest           # run again: every file "unchanged", vector count unchanged
.\.venv\Scripts\rag.exe stats
```

On macOS or Linux, use `.venv/bin/rag` instead of `.\.venv\Scripts\rag.exe`.

**Ask a question from the CLI:**

```powershell
.\.venv\Scripts\rag.exe query "What is the nightly hotel cap in London?" --doc-type policies
.\.venv\Scripts\rag.exe query "How long does the KX-200 fly with a full payload?" --full   # full JSON
```

**Or over HTTP:**

```powershell
.\.venv\Scripts\uvicorn.exe rag_app.api:app --port 8000
```
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/query -ContentType application/json `
  -Body '{"question": "How many concurrent streams does the Team plan allow?", "k": 5, "doc_type": ["engineering"]}'
```

Endpoints: `POST /query`, `POST /ingest`, `GET /stats`, `GET /health`. Each response contains the
answer, the citations (chunk ID, document and similarity score), every retrieved chunk, the token
usage, and a latency breakdown (embedding, retrieval, generation, total).

**Reproduce every number in this README:**

```powershell
.\.venv\Scripts\python.exe -m pytest                      # 16 unit tests: metrics, chunking, loaders, judge parsing
.\.venv\Scripts\python.exe eval\validate_questions.py     # every evidence span exists in its document
.\.venv\Scripts\rag.exe eval                              # 3-layer eval -> results/eval_report.md
.\.venv\Scripts\python.exe eval\sweep.py                  # chunk-size and k sweeps -> results/sweep_report.md
.\.venv\Scripts\python.exe scripts\bench_scale.py --sizes 100000 250000      # -> results/scale_benchmark.md
.\.venv\Scripts\python.exe scripts\bench_scale.py --sizes 250000 --halfvec   # adds the halfvec run
.\.venv\Scripts\python.exe scripts\cost_model.py          # -> results/cost_comparison.md
```

**Evidence screenshots** (`docs/screenshots/`, rendered from the real output of
`scripts/demo_query.ps1` and `scripts/demo_reingest.ps1`):

| End-to-end query: latency per stage, chunk count, tokens | Idempotent re-ingest: 28 vectors before and after |
|---|---|
| ![query](docs/screenshots/p1_end_to_end_query.png) | ![reingest](docs/screenshots/p1_idempotent_reingest.png) |

## 3. Corpus and evaluation set

- **Corpus** (`data/corpus/`): 12 documents about a fictional drone company, "Kestrel Labs". There
  are 4 Markdown runbooks, 4 HTML product and support pages, and 4 PDF policies and reports (the
  PDFs are generated reproducibly by `scripts/build_pdf_corpus.py`). Because the company is made up,
  **the LLM cannot answer from what it learned in training**, so faithfulness and abstention are
  real tests. The metadata filter `doc_type` comes from the folder name (`engineering`, `product`,
  `support`, `policies`, `finance`).
- **Questions** (`eval/questions.yaml`): 27 fixed questions. 22 need one document, 2 need two
  documents, and 3 are not answerable from the corpus. Two of the questions use a metadata filter.
  Each answerable question has a short gold answer and one or more **evidence spans**.
- **Relevance definition:** a retrieved chunk is relevant when it comes from the evidence document
  and contains the evidence span (after lower-casing, removing markdown emphasis and collapsing
  whitespace). Defining relevance by text span rather than chunk ID keeps the labels valid when the
  chunk size changes, which the sweep relies on. `eval/validate_questions.py` checks that all 26
  spans exist.

## 4. Evaluation results

All numbers come from `results/eval_results.json` (per-question detail) and
`results/eval_report.md`. Settings: k = 5, chunks of 256 tokens with 32 overlap, bge-small (384-d),
gpt-oss-120b as generator, and Qwen3 as judge.

### 4.1 Retrieval (24 answerable questions)

| Metric | Value | How it is computed |
|---|---|---|
| Hit rate@5 | **1.000** | At least one relevant chunk in the top 5 |
| Recall@5 (evidence) | **0.979** | Share of a question's evidence spans found in the top 5 (multi-document questions have 2) |
| MRR | **0.938** | Mean of 1 / rank of the first relevant chunk |
| nDCG@5 | **0.922** | Binary gains; the ideal ranking uses every relevant chunk in the index (capped at k) |
| Precision@5 | 0.217 | Relevant chunks / 5. Usually only 1–2 relevant chunks exist, so the maximum is about 0.3 |
| Context precision (rank-aware AP) | **0.896** | Mean precision@i at each rank holding a relevant chunk (RAGAS style) |
| Multi-document recall | 0.750 | q16 never retrieves the gateway page (see Discussion) |

The metric functions are unit-tested against hand-computed values (`tests/test_metrics.py`).

**Chunk-size sweep** (`results/sweep_report.md`, retrieval only, no LLM):

| Chunk / overlap | Vectors | Hit rate | MRR | nDCG@5 |
|---|---|---|---|---|
| 96 / 16 | 71 | 0.875 | 0.812 | 0.809 |
| 128 / 16 | 53 | 0.958 | 0.917 | 0.917 |
| 256 / 0 | 27 | 0.958 | 0.917 | 0.905 |
| **256 / 32** (default) | 28 | **1.000** | 0.938 | 0.922 |
| 256 / 64 | 30 | 1.000 | 0.958 | 0.937 |
| 384 / 48 | 20 | 1.000 | 0.927 | 0.935 |
| 480 / 64 | 16 | 0.958 | 0.896 | 0.902 |

**k sweep:** k = 1 gives hit rate 0.875 and multi-document recall 0.25. k = 3 gives 1.00 and 0.50.
k = 5 gives 1.00 and 0.75. k = 10 adds nothing, at twice the prompt tokens.

### 4.2 Answer quality (judge: Qwen3 on Groq, a different model family from the generator)

| Metric | Score | Method |
|---|---|---|
| Faithfulness | **1.00** | The judge lists every atomic claim and labels it supported or not by the passages; the score is supported / total, **computed by us from the labels** |
| Answers with any unsupported claim | 0 / 24 | Same claim labels |
| Answer relevance | **5.0 / 5** | Anchored 1–5 scale (5 = directly and completely answers) |
| Correctness vs gold | **5.0 / 5** | Anchored 1–5 scale against the reference answer |
| Exact match | 0.00 | SQuAD normalisation. The model answers in full sentences, so this is always 0 and is reported for completeness |
| Gold answer contained | **0.875** | The normalised gold string appears in the answer. Misses: "three months" vs "3 months", and paraphrases of long gold answers |
| Token F1 | 0.307 | SQuAD token F1, kept low by sentence-length answers |
| Citation accuracy | **1.00** | The answer cites at least one relevant chunk |
| Abstention | 3/3 correct, 0 false | Unanswerable questions declined; no answerable question declined |

**Is the judge credible?** Two things were checked:
1. **Planted error:** given "31 minutes … and it is waterproof" with a passage that only says IP54,
   the judge marked the waterproof claim unsupported (faithfulness 0.5).
2. **Cross-family agreement:** the first run used Gemini 2.5 Flash as the judge until its free
   quota ran out. Gemini gave identical scores (1.00 / 5 / 5) on the 19 answers it judged.

Every judge prompt and raw response is in `logs/judge_calls.jsonl`. `rag eval --rejudge` re-scores
a saved run without regenerating the answers.

### 4.3 Latency

Measured on 27 questions × 10 repetitions after a warm-up, on a laptop (i7-13650HX, Docker with 3.9 GB).

| Stage | p50 ms | p95 ms |
|---|---|---|
| Vector search (SQL, 28 vectors) | 3.7 | 4.7 |
| Query embedding (CPU) | 31.1 | 39.7 |
| Retrieval total | 34.8 | 43.3 |
| LLM generation (Groq) | 693 | 909 |
| **End-to-end** | **756** | **1,187** |

On average an LLM call uses 1,356 prompt tokens and 63 completion tokens. Generation is about
92% of end-to-end latency.

### 4.4 Latency at scale

The real corpus is only 28 vectors, so `scripts/bench_scale.py` loads synthetic 384-d vectors into a
separate table and builds the same HNSW index the application uses (cosine, m = 16,
ef_construction = 64). The vectors are **clustered**, not uniform-random: each is a noisy copy of a
cluster centre, and each query is a noisy copy of a stored vector, so every query has real near
neighbours. Recall is measured against exact search, run on a separate connection with index scans
disabled. Full results: `results/scale_benchmark.md`.

| Vectors | Index | Index build | Table + index | p50 ms | p95 ms | Recall@5 (ef 40) | Recall@5 (ef 100) | Exact scan p50 |
|---|---|---|---|---|---|---|---|---|
| 100,000 | float32 | 36 s | 355 MB | 0.82 | 1.35 | 0.98 | 1.00 | 20 ms |
| 250,000 | float32 | 94 s | 886 MB | 1.15 | 2.22 | 0.98 | 1.00 | 40 ms |
| 250,000 | halfvec | 62 s | 677 MB | 1.23 | 2.30 | 0.92 | 0.98 | 42 ms |
| 1,000,000 | float32 | — | ~3.5 GB | ~1.7 | ~3.5 | — | — | — (projected) |
| 10,000,000 | float32 | — | ~35 GB | ~2.5 | ~5.7 | — | — | — (projected) |

- **Storage is linear and measured:** 3,720 bytes per vector (2,048 for the index) at both 100K and
  250K, a 0.0% difference. HNSW node size depends on the dimension and m, not on N, so the 1M and
  10M sizes are a direct multiplication.
- **Latency at 1M and 10M is a projection:** a fit of p = a + b·ln(N) through the two measured
  sizes, because an HNSW search visits O(log N) nodes. It assumes the index stays in RAM, which is
  how the cost model sizes the server.
- **The 20%-selective `doc_type` filter costs nothing** (1.07 ms p50 at 250K) and keeps recall at
  0.98, thanks to pgvector 0.8's iterative index scans.
- **Why 1M was not measured:** the attempt was stopped. A 1M index (about 2 GB) exceeds the 1 GB
  build memory available in this laptop's 3.9 GB Docker VM, so pgvector falls back to a much slower
  on-disk build (still about 50% done after 13 minutes). On a server sized as below, the index fits
  in memory. This is itself a practical trade-off of self-hosting: **build time depends on RAM**.
- Two benchmark bugs were found and fixed along the way. First, a cached prepared-statement plan
  kept a sequential scan after index scans were re-enabled, so "HNSW" latency silently equalled
  exact-scan latency. Second, an "idle in transaction" lock deadlocked the `DROP TABLE`.

### 4.5 Monthly cost: pgvector vs a managed vector database

Generated by `scripts/cost_model.py`. Every price and workload assumption is in
[`eval/pricing.yaml`](eval/pricing.yaml), with its source URL, so it can be checked and changed in
one place. Storage per vector is the **measured** 3,720 bytes (or 2,840 bytes with a halfvec index)
plus 1,100 bytes of chunk text.

| Vectors | pgvector, self-hosted (infrastructure) | pgvector, halfvec index | Pinecone serverless | Pinecone pods (p1, always-on) | pgvector vs pods |
|---|---|---|---|---|---|
| 100K | **$14** (t4g.small) | $14 | $50 (plan minimum) | $81 (1 pod) | −83% |
| 1M | **$26** (t4g.medium) | $26 | $50 (plan minimum) | $81 (1 pod) | −67% |
| 10M | **$164** (r7g.xlarge, 32 GB) | **$84** (r7g.large, 16 GB) | $63 | $405 (5 pods) | −60% |

Adding operating time (2 h/month at $75/h, so +$150) gives $164, $176 and $314.

**Sensitivity to query volume:** Pinecone serverless charges about 1 read unit per GB of index for
every query, while a pgvector node's cost does not depend on traffic. This decides the comparison
against serverless:

| Queries / month | 100K vectors: pgvector / serverless | 1M vectors | 10M vectors |
|---|---|---|---|
| 100K (the assumed light load) | $14 / $50 | $26 / $50 | $164 / **$63** |
| 1M | $14 / $50 | $26 / $50 | $164 / $443 |
| 10M | $14 / $50 | $26 / $425 | $164 / $4,247 |

**Assumptions:**
- **Pricing basis:** AWS us-east-1 on-demand prices, no reserved or savings-plan discounts, and a
  single node with no replica. Add a second node for high availability, which doubles compute.
- **pgvector sizing:** the instance's RAM must hold the HNSW index plus 25% headroom and 1.5 GB for
  Postgres and the OS. Storage is gp3 at 1.5× the data size, plus daily snapshots.
- **Pinecone serverless:** Standard plan rates ($50 minimum, $0.33 per GB-month, $16 per million
  read units, $4 per million write units), with 10% of vectors rewritten each month.
- **Pinecone pods:** p1.x1 at $0.111/h, about 1.67M vectors per pod at 384 dimensions. Pods are
  closed to new customers since August 2025, but they are the "always-on pods" model the assignment
  describes.
- **Excluded:** LLM and embedding costs, which are the same whichever store is used.

## 5. Design decisions

**Why pgvector?** It runs inside Postgres, which most teams already operate, back up and monitor, so
there is no new database to learn. Chunks, document metadata and vectors live in one transactional
store, so idempotent re-ingest is a single SQL transaction rather than a cross-system consistency
problem, and metadata filters are ordinary `WHERE` clauses with a B-tree index. HNSW gives
millisecond search with recall above 0.98, as measured above. It runs on any VM or managed Postgres
(RDS, Cloud SQL, Supabase, Neon), so there is no lock-in. The alternatives considered:
- **FAISS:** a library, not a database. There is no persistence, filtering or concurrent writes
  without building them.
- **ChromaDB, LanceDB, sqlite-vec:** good embedded options, but single-process, and a weaker story
  for concurrent writers and backups.
- **Qdrant:** a strong choice, but it is a second database to run next to the metadata store.

**Chunking.** Chunks are packed sentence by sentence up to 256 tokens of the embedding model's own
tokenizer, with the last ~32 tokens carried into the next chunk. A sentence is split only if it is
longer than a whole chunk. The sweep shows small chunks (96 tokens) losing context (hit rate 0.875),
and very large ones (480 tokens) diluting the embedding and approaching bge-small's 512-token limit.
256/32 was the smallest setting with hit rate 1.0. 256/64 is marginally better, by one question, at
about 7% more vectors. The document title is prepended to each chunk **only for embedding**, which
gives otherwise context-free chunks (a table row, say) their document's context.

**Embedding model.** `BAAI/bge-small-en-v1.5`: 384 dimensions, runs locally on CPU at about 30 ms per
query, with no API cost and no data leaving the machine. Compared with a 768-d or 1536-d API model,
it halves or quarters the storage and RAM per vector, which drives the cost at 10M vectors. The
risk is lower recall on harder, real-world queries; the model is a single setting, and a change
re-embeds the corpus automatically because the model name is part of the re-ingest check.

**No relevant context.** First, if no retrieved chunk reaches cosine 0.5, the service returns a
fixed "not found" answer without calling the LLM. "What is the capital of France?" took 36 ms and
0 tokens. Second, the prompt tells the model to answer only from the numbered passages and to reply
`INSUFFICIENT_CONTEXT` otherwise, which also becomes the fixed answer. The evaluation shows **the
threshold alone is not enough**: the top chunk's mean similarity was 0.73 for unanswerable questions
vs 0.75 for answerable ones, so all 3 unanswerable questions were caught by the second layer.

**Idempotent re-ingest.** Each document row stores the file's sha256, the chunk size and overlap,
and the embedding model. A file whose hash and settings match is skipped. A changed file has all its
old chunks deleted and its new chunks written **in one transaction**. Chunk IDs are a hash of
(document, file hash, chunk settings, chunk index), and inserts use `ON CONFLICT DO NOTHING`, so a
duplicate vector cannot be written even if two ingests race. `--prune` removes documents whose files
are gone. Re-running `rag ingest` on the corpus reports 12 unchanged files, 0 chunks written, and
28 vectors before and after.

**Metadata filter.** `doc_type` is stored on every chunk with a B-tree index. pgvector 0.8's
`hnsw.iterative_scan = relaxed_order` keeps searching the graph when the filter removes candidates,
so filtered queries still return k results instead of fewer.

## 6. Discussion

**Is pgvector a credible alternative?** Yes, with caveats. On quality it gives up nothing measurable
here: HNSW returns the same neighbours as exact search 98–100% of the time, in 1–2 ms, which is
under 1% of end-to-end latency. On cost:
- **Against always-on pods** (the cost problem the assignment describes), it is 60–83% cheaper at
  every size.
- **Against usage-priced serverless,** the answer depends on traffic. pgvector wins at 100K–1M
  vectors, where serverless's $50 minimum dominates, and it wins by 2.7× at 10M vectors once traffic
  reaches 1M queries a month. Serverless is cheaper only for a large **and** rarely queried index
  ($63 vs $164 at 10M vectors and 100K queries a month), and a halfvec index narrows that gap to
  $84.

**When would I switch back to a managed vector database?**
1. **No one to operate it.** With operating time counted (about $150 a month), self-hosting does not
   pay for itself below a few million vectors unless the Postgres is already run for something else.
2. **The index outgrows one node's RAM.** Beyond roughly 50–100M vectors (about 200–400 GB of
   index) it needs sharding or very large instances, which managed services handle for you.
3. **Large and rarely queried.** For a large index with light, bursty traffic, serverless's pricing
   by storage plus reads is cheaper, as the 10M / 100K-queries row shows.
4. **Strict requirements:** an SLA, multi-region replication, or zero-downtime index rebuilds
   without in-house expertise.

**Was retrieval or generation the weak link?** **Retrieval.**
- **Generation has no measured failures:** 24/24 answers were faithful and correct, every citation
  pointed to a relevant chunk, and all 3 unanswerable questions were declined.
- **Retrieval's misses:**
  1. **Multi-document recall is 0.75.** For q16, single-vector retrieval found the runbook but never
     the gateway page with the 6-hour buffer. The answer was right only because the runbook repeats
     that figure.
  2. **The similarity threshold cannot separate unanswerable questions.** Their top chunks scored
     0.73, against 0.75 for answerable ones, so abstention relied entirely on the generator.
- **Next steps:** hybrid BM25 + vector search, or query decomposition for multi-hop questions, plus
  a cross-encoder reranker whose scores are better calibrated for an abstention threshold.

**Trade-offs knowingly accepted:**
- **Operations work:** you run the backups, upgrades and monitoring yourself.
- **Single-node scaling ceiling.**
- **Build time depends on RAM:** shown when the 1M build fell back to disk on a 3.9 GB VM.
- **Speed vs recall is a setting you own:** ef_search 40 gives recall 0.98, ef_search 100 gives 1.00
  for about 0.2 ms more.
- **halfvec quantisation:** about −6 recall points at ef_search 40 in exchange for a 43% smaller
  index.
- **A small local embedding model:** cheaper than a hosted one, possibly lower recall on harder data.

**Limitations of this evaluation (stated honestly):**
- **Easy, small corpus:** the corpus is synthetic and only 28 chunks, so the retrieval and answer
  scores are near their ceiling and the question set does not separate strong systems from very
  strong ones.
- **Synthetic scale vectors:** the scale benchmark uses clustered synthetic vectors. Real embeddings
  have different neighbourhood structure, so recall at scale should be re-checked on real data.
- **Single judge family:** the LLM judge is one model family (Qwen3), cross-checked against Gemini
  on 19 answers. There is no human-labelled agreement study; that is Problem 2's focus.
- **Partly projected latency:** latency beyond 250K vectors is projected, not measured.

## 7. Repository layout

```
problem1-rag/
├── src/rag_app/
│   ├── loaders.py      PDF / HTML / MD -> clean text
│   ├── chunking.py     sentence-aware token chunker
│   ├── embeddings.py   bge-small (local)
│   ├── store.py        Postgres schema, idempotent replace, filtered HNSW search
│   ├── llm.py          Groq / Gemini / Mistral client with usage, latency, retries
│   ├── pipeline.py     ingest + answer (thresholds, prompt, citations, logging)
│   ├── api.py          FastAPI
│   ├── cli.py          rag ingest | query | eval | stats | reset
│   └── evaluation/     metrics.py · judge.py · harness.py
├── eval/               questions.yaml · validate_questions.py · sweep.py · pricing.yaml
├── scripts/            build_pdf_corpus.py · bench_scale.py · cost_model.py · check_mistral.py
├── data/corpus/        12 documents
├── results/            every evaluation, sweep, benchmark and cost output
├── tests/              unit tests
└── docker-compose.yml  pgvector/pgvector:pg17
```
