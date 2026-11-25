"""Scale benchmark for pgvector: latency, recall and storage at 100K / 1M vectors.

The real corpus is only 28 chunks, so this loads synthetic 384-d vectors (same
dimension as bge-small) into a separate table `bench_vectors` and builds the same
HNSW index (cosine, m=16, ef_construction=64) the application uses.

Synthetic data is CLUSTERED, not uniform-random: uniform random high-dimensional
vectors have no near neighbours, which makes ANN search look unrealistically hard
(or trivially easy, depending on the metric). Each vector is a noisy copy of one of
N/100 cluster centres; queries are noisy copies of stored vectors, so every query
has genuine near neighbours, like a real question about an indexed document.

Measured per size:
  * load time (binary COPY) and HNSW build time
  * on-disk size of the table and the index (bytes / vector feeds the cost model)
  * query latency p50/p95 for k=5 at ef_search 40 (pgvector default) and 100
  * recall@5 of HNSW vs exact search (sequential scan) on a query sample
  * the same for a metadata-filtered query (doc_type, 20% selectivity)

Usage:  python scripts/bench_scale.py --sizes 100000 1000000
Out:    results/scale_benchmark.json, results/scale_benchmark.md
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from rag_app.config import PROJECT_ROOT, get_settings
from rag_app.evaluation.metrics import percentile

DIM = 384
K = 5
DOC_TYPES = ["engineering", "product", "support", "policies", "finance"]
RESULTS = PROJECT_ROOT / "results"
CLUSTER_NOISE = 0.8  # norm of the noise added to a cluster centre
QUERY_NOISE = 0.3    # norm of the noise added to a stored vector to form a query


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def generate(n: int, rng: np.random.Generator, batch: int = 50_000):
    """Yield (ids, doc_types, vectors) batches of clustered unit vectors."""
    centres = unit(rng.standard_normal((max(n // 100, 10), DIM)))
    for start in range(0, n, batch):
        m = min(batch, n - start)
        c = rng.integers(0, len(centres), m)
        # noise of norm ~0.8 -> cos(vector, centre) ~0.78, cos between cluster mates ~0.6
        vecs = unit(centres[c] + CLUSTER_NOISE * rng.standard_normal((m, DIM)) / np.sqrt(DIM))
        types = rng.integers(0, len(DOC_TYPES), m)
        yield np.arange(start, start + m), types, vecs


def load(conn: psycopg.Connection, n: int, seed: int) -> tuple[float, np.ndarray]:
    conn.execute("DROP TABLE IF EXISTS bench_vectors")
    conn.execute(f"CREATE TABLE bench_vectors (id bigint PRIMARY KEY, doc_type text NOT NULL, embedding vector({DIM}) NOT NULL)")
    rng = np.random.default_rng(seed)
    sample_rng = np.random.default_rng(seed + 1)
    samples = []  # keep a few stored vectors to derive queries from
    t0 = time.perf_counter()
    with conn.cursor() as cur, cur.copy("COPY bench_vectors (id, doc_type, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["int8", "text", "vector"])
        for ids, types, vecs in generate(n, rng):
            for i, t, v in zip(ids, types, vecs):
                copy.write_row((int(i), DOC_TYPES[t], v))
            samples.append(vecs[sample_rng.integers(0, len(vecs), 20)])
            print(f"  loaded {ids[-1] + 1:,}/{n:,}", end="\r", flush=True)
    conn.commit()
    print()
    return time.perf_counter() - t0, np.concatenate(samples)


def build_index(conn: psycopg.Connection) -> float:
    t0 = time.perf_counter()
    conn.execute("SET maintenance_work_mem = '1GB'")
    conn.execute("SET max_parallel_maintenance_workers = 4")
    conn.execute("CREATE INDEX bench_hnsw ON bench_vectors USING hnsw (embedding vector_cosine_ops) "
                 "WITH (m = 16, ef_construction = 64)")
    conn.execute("CREATE INDEX bench_doc_type ON bench_vectors (doc_type)")
    conn.execute("ANALYZE bench_vectors")
    conn.commit()
    return time.perf_counter() - t0


def search(conn, q: np.ndarray, doc_type: str | None) -> tuple[list[int], float]:
    where = "WHERE doc_type = %(t)s" if doc_type else ""
    sql = f"SELECT id FROM bench_vectors {where} ORDER BY embedding <=> %(q)s LIMIT {K}"
    t0 = time.perf_counter()
    ids = [r[0] for r in conn.execute(sql, {"q": q, "t": doc_type}).fetchall()]
    return ids, (time.perf_counter() - t0) * 1000


def measure(conn, exact_conn, queries: np.ndarray, n_recall: int) -> dict:
    out = {}
    qtypes = [DOC_TYPES[i % len(DOC_TYPES)] for i in range(len(queries))]

    # Ground truth: exact search with index scans disabled, on a SEPARATE connection.
    # (Toggling enable_indexscan on the measured connection is not enough: a cached
    # prepared-statement plan keeps the sequential scan after the setting is restored.)
    exact_conn.execute("SET enable_indexscan = off")
    truth, truth_f, exact_ms = [], [], []
    for q, t in zip(queries[:n_recall], qtypes):
        ids, ms = search(exact_conn, q, None)
        truth.append(set(ids))
        exact_ms.append(ms)
        truth_f.append(set(search(exact_conn, q, t)[0]))
    out["exact_scan_ms"] = {"p50": round(percentile(exact_ms, 50), 1), "p95": round(percentile(exact_ms, 95), 1)}

    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    # Cold pass: the first queries after a restart read HNSW pages from disk. Report it,
    # then measure steady state (all queries once more after a full warm-up pass).
    cold = [search(conn, q, None)[1] for q in queries]
    out["cold_first_pass"] = {"p50_ms": round(percentile(cold, 50), 2), "p95_ms": round(percentile(cold, 95), 2)}
    for ef in (40, 100):
        conn.execute(f"SET hnsw.ef_search = {ef}")
        for i, q in enumerate(queries):  # warm-up pass, untimed
            search(conn, q, None)
            search(conn, q, qtypes[i])
        for label, filt in (("unfiltered", False), ("filtered_20pct", True)):
            lat, rec = [], []
            for i, q in enumerate(queries):
                ids, ms = search(conn, q, qtypes[i] if filt else None)
                lat.append(ms)
                if i < n_recall:
                    gt = truth_f[i] if filt else truth[i]
                    rec.append(len(set(ids) & gt) / K)
            out[f"ef{ef}_{label}"] = {"p50_ms": round(percentile(lat, 50), 2), "p95_ms": round(percentile(lat, 95), 2),
                                      "recall_at_5": round(float(np.mean(rec)), 4), "n_queries": len(lat)}
    return out


def sizes(conn) -> dict:
    row = conn.execute("SELECT pg_relation_size('bench_vectors'), pg_total_relation_size('bench_vectors'), "
                       "pg_relation_size('bench_hnsw') FROM bench_vectors LIMIT 1").fetchone()
    return {"heap_bytes": row[0], "total_bytes": row[1], "hnsw_index_bytes": row[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--recall-queries", type=int, default=50)
    ap.add_argument("--keep", action="store_true", help="keep the bench table after the last size")
    args = ap.parse_args()

    dsn = get_settings().database_url
    # autocommit: no connection sits "idle in transaction" holding a lock that would block
    # the next DROP TABLE; prepare_threshold=None: every query is planned fresh.
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    exact_conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    register_vector(conn)
    register_vector(exact_conn)
    server = {r[0]: r[1] for r in conn.execute(
        "SELECT name, setting || coalesce(unit, '') FROM pg_settings WHERE name IN "
        "('shared_buffers','maintenance_work_mem','max_parallel_maintenance_workers','server_version')")}
    server["pgvector"] = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]

    runs = []
    for n in args.sizes:
        print(f"== {n:,} vectors")
        load_s, samples = load(conn, n, seed=42)
        build_s = build_index(conn)
        rng = np.random.default_rng(7)
        picks = samples[rng.integers(0, len(samples), args.queries)]
        queries = unit(picks + QUERY_NOISE * rng.standard_normal(picks.shape) / np.sqrt(DIM))  # cos ~0.96 to source
        sz = sizes(conn)
        res = {"vectors": n, "load_seconds": round(load_s, 1), "index_build_seconds": round(build_s, 1), **sz,
               "bytes_per_vector_total": round(sz["total_bytes"] / n, 1),
               **measure(conn, exact_conn, queries, args.recall_queries)}
        runs.append(res)
        print(json.dumps(res, indent=2))

    if not args.keep:
        conn.execute("DROP TABLE IF EXISTS bench_vectors")
        conn.commit()

    report = {"hardware": {"host_cpu": platform.processor(), "docker_vm": "20 vCPU, 3.9 GB RAM (Docker Desktop, WSL2)",
                           "os": platform.platform()},
              "postgres": server, "dim": DIM, "k": K, "index": "hnsw m=16 ef_construction=64, cosine", "runs": runs}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scale_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "scale_benchmark.md").write_text(render(report), encoding="utf-8")


def render(r: dict) -> str:
    mb = lambda b: f"{b / 1024**2:,.0f} MB"  # noqa: E731
    lines = ["# pgvector scale benchmark", "",
             f"Synthetic clustered {r['dim']}-d vectors, {r['index']}, k={r['k']}. Postgres {r['postgres'].get('server_version')} "
             f"+ pgvector {r['postgres']['pgvector']} in Docker ({r['hardware']['docker_vm']}); "
             f"shared_buffers={r['postgres'].get('shared_buffers')}. Latency is client-measured per SQL query.", "",
             "| Vectors | Load s | Index build s | Table+index | Bytes/vector | Exact scan p50 ms |", "|---|---|---|---|---|---|"]
    for x in r["runs"]:
        lines.append(f"| {x['vectors']:,} | {x['load_seconds']} | {x['index_build_seconds']} | {mb(x['total_bytes'])} "
                     f"(index {mb(x['hnsw_index_bytes'])}) | {x['bytes_per_vector_total']:,.0f} | {x['exact_scan_ms']['p50']} |")
    lines += ["", "Steady-state latency (after one untimed warm-up pass over the query set):", "",
              "| Vectors | ef_search | Query | p50 ms | p95 ms | Recall@5 vs exact |", "|---|---|---|---|---|---|"]
    for x in r["runs"]:
        lines.append(f"| {x['vectors']:,} | 40 | cold first pass (pages read from disk) | {x['cold_first_pass']['p50_ms']} "
                     f"| {x['cold_first_pass']['p95_ms']} | — |")
        for ef in (40, 100):
            for label in ("unfiltered", "filtered_20pct"):
                m = x[f"ef{ef}_{label}"]
                lines.append(f"| {x['vectors']:,} | {ef} | {label.replace('_', ' ')} | {m['p50_ms']} | {m['p95_ms']} | {m['recall_at_5']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
