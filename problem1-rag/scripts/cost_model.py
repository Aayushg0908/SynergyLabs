"""Monthly cost of the vector store at 100K / 1M / 10M vectors: self-hosted pgvector vs Pinecone.

Storage per vector comes from the MEASURED scale benchmark (results/scale_benchmark.json:
heap + HNSW index bytes per vector at the largest size run), not from a formula. Prices
and workload assumptions live in eval/pricing.yaml. LLM and embedding costs are excluded
because they are identical whichever store is used (embeddings run locally on CPU).

Usage:  python scripts/cost_model.py
Out:    results/cost_comparison.json, results/cost_comparison.md
"""

import json
import math

import yaml

from rag_app.config import PROJECT_ROOT

SIZES = [100_000, 1_000_000, 10_000_000]
GB = 1024**3


def measured_bytes() -> dict:
    bench = json.loads((PROJECT_ROOT / "results" / "scale_benchmark.json").read_text(encoding="utf-8"))
    run = max(bench["runs"], key=lambda r: r["vectors"])
    return {"from_vectors": run["vectors"],
            "index_bytes_per_vector": run["hnsw_index_bytes"] / run["vectors"],
            "heap_bytes_per_vector": run["heap_bytes"] / run["vectors"],
            "total_bytes_per_vector": run["total_bytes"] / run["vectors"]}


def pgvector_cost(n: int, p: dict, w: dict, m: dict, hours: int) -> dict:
    s = p["self_hosted_pgvector"]
    index_gb = n * m["index_bytes_per_vector"] / GB
    # heap from the benchmark holds id + doc_type + vector; add the chunk text the app stores
    data_gb = n * (m["total_bytes_per_vector"] + w["avg_chunk_text_bytes"]) / GB
    ram_needed = index_gb * s["ram_headroom"] + s["ram_overhead_gb"]
    inst = next((i for i in s["instances"] if i["ram_gb"] >= ram_needed), s["instances"][-1])
    disk_gb = max(20, math.ceil(data_gb * s["disk_headroom"]))
    compute = inst["usd_per_hour"] * hours
    storage = disk_gb * s["ebs_gp3_per_gb_month"] + data_gb * s["snapshot_per_gb_month"]
    infra = compute + storage
    ops = s["ops_hours_per_month"] * s["ops_usd_per_hour"]
    return {"instance": f"{inst['name']} ({inst['ram_gb']} GB RAM)", "index_gb": round(index_gb, 2),
            "data_gb": round(data_gb, 2), "ram_needed_gb": round(ram_needed, 1), "disk_gb": disk_gb,
            "compute_usd": round(compute, 2), "storage_usd": round(storage, 2), "infra_usd": round(infra, 2),
            "ops_usd": ops, "total_with_ops_usd": round(infra + ops, 2)}


def pinecone_serverless_cost(n: int, p: dict, w: dict) -> dict:
    c = p["pinecone_serverless"]
    record_bytes = w["dim"] * 4 + w["avg_chunk_text_bytes"] + c["bytes_per_record_overhead"]
    storage_gb = n * record_bytes / GB
    ru_per_query = max(c["ru_min_per_query"], storage_gb * c["ru_per_gb_queried"])
    reads = w["queries_per_month"] * ru_per_query
    writes = n * w["monthly_rewrite_fraction"] * math.ceil(record_bytes / 1024) * c["wu_per_kb"]
    usage = (storage_gb * c["storage_per_gb_month"] + reads / 1e6 * c["read_unit_usd_per_million"]
             + writes / 1e6 * c["write_unit_usd_per_million"])
    return {"storage_gb": round(storage_gb, 2), "ru_per_query": round(ru_per_query, 2),
            "read_units": round(reads), "write_units": round(writes),
            "usage_usd": round(usage, 2), "total_usd": round(max(usage, c["plan_minimum_usd"]), 2)}


def pinecone_pods_cost(n: int, p: dict, w: dict, hours: int) -> dict:
    c = p["pinecone_pods"]
    capacity = c["p1_capacity_at_768d"] * 768 / w["dim"]
    pods = max(1, math.ceil(n / capacity)) * c["replicas"]
    return {"pods": pods, "pod_type": "p1.x1", "capacity_per_pod": round(capacity),
            "total_usd": round(pods * c["p1_x1_usd_per_hour"] * hours, 2)}


def main() -> None:
    p = yaml.safe_load((PROJECT_ROOT / "eval" / "pricing.yaml").read_text(encoding="utf-8"))
    w, hours = p["workload"], p["hours_per_month"]
    m = measured_bytes()
    rows = []
    for n in SIZES:
        pg = pgvector_cost(n, p, w, m, hours)
        sv = pinecone_serverless_cost(n, p, w)
        pod = pinecone_pods_cost(n, p, w, hours)
        rows.append({"vectors": n, "pgvector": pg, "pinecone_serverless": sv, "pinecone_pods": pod})
    out = {"measured_storage": m, "assumptions": p, "rows": rows}
    res = PROJECT_ROOT / "results"
    (res / "cost_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (res / "cost_comparison.md").write_text(render(out), encoding="utf-8")
    print(render(out))


def usd(x: float) -> str:
    return f"${x:,.0f}"


def render(o: dict) -> str:
    m, w = o["measured_storage"], o["assumptions"]["workload"]
    lines = [
        "# Monthly vector-store cost: self-hosted pgvector vs Pinecone", "",
        f"Storage per vector measured at {m['from_vectors']:,} vectors: HNSW index {m['index_bytes_per_vector']:,.0f} B, "
        f"heap {m['heap_bytes_per_vector']:,.0f} B (+{w['avg_chunk_text_bytes']:,} B chunk text). "
        f"Workload: {w['dim']}-d vectors, {w['queries_per_month']:,} queries/month, "
        f"{w['monthly_rewrite_fraction']:.0%} of vectors rewritten per month. Prices: eval/pricing.yaml.", "",
        "| Vectors | pgvector (EC2 + EBS) | + ops time | Pinecone serverless | Pinecone pods (p1, always-on) | pgvector saves vs serverless | vs pods |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in o["rows"]:
        pg, sv, pod = r["pgvector"], r["pinecone_serverless"], r["pinecone_pods"]
        save_sv = 1 - pg["infra_usd"] / sv["total_usd"]
        save_pod = 1 - pg["infra_usd"] / pod["total_usd"]
        lines.append(f"| {r['vectors']:,} | {usd(pg['infra_usd'])} ({pg['instance']}) | {usd(pg['total_with_ops_usd'])} | "
                     f"{usd(sv['total_usd'])} | {usd(pod['total_usd'])} ({pod['pods']} pod{'s' if pod['pods'] > 1 else ''}) | "
                     f"{save_sv:.0%} | {save_pod:.0%} |")
    lines += ["", "## Breakdown", "",
              "| Vectors | pgvector index GB | RAM needed GB | disk GB | compute $ | storage $ | Pinecone storage GB | RU/query | Pinecone usage $ (before $50 min) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in o["rows"]:
        pg, sv = r["pgvector"], r["pinecone_serverless"]
        lines.append(f"| {r['vectors']:,} | {pg['index_gb']} | {pg['ram_needed_gb']} | {pg['disk_gb']} | {pg['compute_usd']} | "
                     f"{pg['storage_usd']} | {sv['storage_gb']} | {sv['ru_per_query']} | {sv['usage_usd']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
