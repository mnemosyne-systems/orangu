# orangu-search-bench

A small retrieval benchmark for the hybrid lexical/semantic search proof of
concept. It runs the committed query fixture (`queries.json`) against:

- **lexical-only** — BM25 over the lexical sidecar
- **semantic-only** — cosine similarity over the embedding index
- **hybrid** — Reciprocal Rank Fusion of both

and reports MRR, Recall@5, mean latency, indexing time, and the lexical
sidecar's on-disk size.

```sh
# Lexical-only (no embeddings server required)
cargo run --release --bin orangu-search-bench -- --workspace .

# Semantic-only and hybrid
cargo run --release --bin orangu-search-bench -- --workspace . \
    --endpoint http://127.0.0.1:8080 --model embeddings
```

`--verbose` prints the top lexical, semantic and hybrid hits per query. The fixture marks
standalone constants that the current Rust extractor does not emit (for example
`MAX_CHUNK_CHARS`) as `coverage_gap`; those are reported separately rather than
counted as ranking failures.

Queries are evaluated against the workspace as it exists, so test code that
contains the same identifiers or strings also participates in the ranking.
The fixture avoids phrases that only exist in test code.

## Results

Run on the `hybrid-search-phases` branch, 2026-11-13: 22,492 indexed symbols,
18 scored queries (10 conceptual, 7 exact, 1 phrase). Embeddings:
`ggml-org/embeddinggemma-300M-GGUF:Q8_0` (768 dims, CPU).

| metric | semantic-only | lexical-only | hybrid |
|---|---|---|---|
| overall MRR | 0.294 | 0.536 | 0.483 |
| overall Recall@5 | 0.444 | 0.667 | 0.667 |
| conceptual MRR (n=10) | 0.278 | 0.165 | 0.220 |
| conceptual Recall@5 (n=10) | 0.500 | 0.400 | 0.500 |
| exact MRR / Recall@5 (n=7) | 0.357 / 0.429 | 1.000 / 1.000 | 0.857 / 1.000 |
| phrase rank | miss | 1 (verified exact) | 2 |
| mean query latency | 230 ms | 17 ms | 276 ms |

Semantic and hybrid latency include one query embedding request (~200 ms).

| index | cold build | warm build | on disk |
|---|---|---|---|
| lexical sidecar | 10.8 s | 2.8 s | 43,921 KiB (42.9 MiB) |
| semantic index | ~4.8 h | 26 s | 210 MB |

Standalone constants (`MAX_CHUNK_CHARS`, `EMBED_BATCH_TOKEN_BUDGET`, `RRF_K`)
are reported as extraction coverage gaps and excluded from the metrics.

