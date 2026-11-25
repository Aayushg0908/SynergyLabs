# pgvector scale benchmark

Synthetic clustered 384-d vectors, hnsw m=16 ef_construction=64, cosine, k=5. Postgres 17.12 (Debian 17.12-1.pgdg12+1) + pgvector 0.8.7 in Docker (20 vCPU, 3.9 GB RAM (Docker Desktop, WSL2)); shared_buffers=655368kB. Latency is client-measured per SQL query.

| Vectors | Index precision | Load s | Index build s | Table+index | Bytes/vector | Exact scan p50 ms |
|---|---|---|---|---|---|---|
| 100,000 | float32 | 8.8 | 35.6 | 355 MB (index 195 MB) | 3,720 | 20.1 |
| 250,000 | float32 | 7.9 | 94.0 | 886 MB (index 488 MB) | 3,718 | 39.8 |
| 250,000 | halfvec | 6.7 | 62.0 | 677 MB (index 279 MB) | 2,840 | 41.8 |

Steady-state latency (after one untimed warm-up pass over the query set):

| Vectors | Index precision | ef_search | Query | p50 ms | p95 ms | Recall@5 vs exact |
|---|---|---|---|---|---|---|
| 100,000 | float32 | 40 | cold first pass (pages read from disk) | 0.89 | 1.44 | — |
| 100,000 | float32 | 40 | unfiltered | 0.82 | 1.35 | 0.98 |
| 100,000 | float32 | 40 | filtered 20pct | 0.83 | 1.27 | 0.984 |
| 100,000 | float32 | 100 | unfiltered | 1.02 | 1.55 | 1.0 |
| 100,000 | float32 | 100 | filtered 20pct | 1.09 | 1.79 | 1.0 |
| 250,000 | float32 | 40 | cold first pass (pages read from disk) | 1.76 | 3.54 | — |
| 250,000 | float32 | 40 | unfiltered | 1.15 | 2.22 | 0.98 |
| 250,000 | float32 | 40 | filtered 20pct | 1.07 | 1.62 | 0.98 |
| 250,000 | float32 | 100 | unfiltered | 1.36 | 2.34 | 1.0 |
| 250,000 | float32 | 100 | filtered 20pct | 1.43 | 2.63 | 1.0 |
| 250,000 | halfvec | 40 | cold first pass (pages read from disk) | 2.35 | 8.3 | — |
| 250,000 | halfvec | 40 | unfiltered | 1.23 | 2.3 | 0.92 |
| 250,000 | halfvec | 40 | filtered 20pct | 1.15 | 2.26 | 0.92 |
| 250,000 | halfvec | 100 | unfiltered | 1.37 | 2.21 | 0.98 |
| 250,000 | halfvec | 100 | filtered 20pct | 1.36 | 2.26 | 0.98 |
