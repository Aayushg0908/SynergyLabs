# Monthly vector-store cost: self-hosted pgvector vs Pinecone

Storage per vector measured at 100,000, 250,000 vectors (float32 index): HNSW index 2,048 B, heap 1,640 B, plus 1,100 B of chunk text. Bytes/vector differ by 0.0% between the measured sizes. halfvec index: 1,170 B/vector.

Workload: 384-d vectors, 100,000 queries/month, 10% of vectors rewritten per month. All prices: `eval/pricing.yaml`.

| Vectors | pgvector (EC2 + EBS) | pgvector, halfvec index | pgvector + ops time | Pinecone serverless | Pinecone pods (p1, always-on) | pgvector vs serverless | pgvector vs pods |
|---|---|---|---|---|---|---|---|
| 100,000 | $14 (t4g.small (2 GB RAM)) | $14 (t4g.small (2 GB RAM)) | $164 | $50 | $81 (1 pod) | 72% | 83% |
| 1,000,000 | $26 (t4g.medium (4 GB RAM)) | $26 (t4g.medium (4 GB RAM)) | $176 | $50 | $81 (1 pod) | 47% | 67% |
| 10,000,000 | $164 (r7g.xlarge (32 GB RAM)) | $84 (r7g.large (16 GB RAM)) | $314 | $63 | $405 (5 pods) | -160% | 60% |

The comparison columns compare pgvector infrastructure only; a negative value means pgvector costs more. Operating time is shown separately: 2 h/month at $75/h.

## Breakdown

| Vectors | pgvector index GB | RAM needed GB | disk GB | compute $ | storage $ | Pinecone storage GB | RU / query | Pinecone usage $ (before $50 minimum) |
|---|---|---|---|---|---|---|---|---|
| 100,000 | 0.19 | 1.7 | 20 | 12.26 | 1.62 | 0.26 | 0.26 | 0.63 |
| 1,000,000 | 1.91 | 3.9 | 20 | 24.53 | 1.82 | 2.64 | 2.64 | 6.3 |
| 10,000,000 | 19.07 | 25.3 | 68 | 156.37 | 7.68 | 26.41 | 26.41 | 62.98 |

## Sensitivity to query volume (monthly $, pgvector infrastructure vs Pinecone serverless)

| Queries / month | 100,000 vectors: pgvector / serverless | 1,000,000 vectors: pgvector / serverless | 10,000,000 vectors: pgvector / serverless |
|---|---|---|---|
| 100,000 | $14 / $50 | $26 / $50 | $164 / $63 |
| 1,000,000 | $14 / $50 | $26 / $50 | $164 / $443 |
| 10,000,000 | $14 / $50 | $26 / $425 | $164 / $4,247 |

## Retrieval latency vs size (HNSW, k=5, ef_search=40, unfiltered)

| Vectors | p50 ms | p95 ms | Source |
|---|---|---|---|
| 100,000 | 0.82 | 1.35 | measured |
| 250,000 | 1.15 | 2.22 | measured |
| 1,000,000 | 1.65 | 3.54 | projected (a + b*ln N fit) |
| 10,000,000 | 2.48 | 5.72 | projected (a + b*ln N fit) |

Fit for p50: -3.326 + 0.36*ln(N) ms. This is a projection from 2 measured sizes, not a measurement.
