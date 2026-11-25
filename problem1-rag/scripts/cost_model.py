"""Monthly cost of the vector store at 100K / 1M / 10M vectors: self-hosted pgvector vs Pinecone.

Storage per vector comes from the MEASURED scale benchmark (results/scale_benchmark.json:
heap + HNSW index bytes per vector at the largest measured size), not from a formula. HNSW
node size depends on the dimension and m, not on N, so bytes/vector is constant across
sizes; the script checks this across the measured sizes and reports the spread.

Retrieval latency at sizes that were not measured is projected with a least-squares fit
p = a + b*ln(N) (HNSW search visits O(log N) nodes) and is labelled as a projection.

Prices and workload assumptions live in eval/pricing.yaml. LLM and embedding costs are
excluded because they are the same whichever store is used (embeddings run locally on CPU).

Usage:  python scripts/cost_model.py
Out:    results/cost_comparison.json, results/cost_comparison.md
"""

import json
import math

import yaml

from rag_app.config import PROJECT_ROOT

SIZES = [100_000, 1_000_000, 10_000_000]
QUERY_VOLUMES = [100_000, 1_000_000, 10_000_000]
GB = 1024**3


def load_bench() -> list[dict]:
    return json.loads((PROJECT_ROOT / "results" / "scale_benchmark.json").read_text(encoding="utf-8"))["runs"]


def precision(run: dict) -> str:
    return run.get("precision", "float32")


def measured_bytes(runs: list[dict], prec: str) -> dict | None:
    rs = [r for r in runs if precision(r) == prec]
    if not rs:
        return None
    run = max(rs, key=lambda r: r["vectors"])
    per = [r["total_bytes"] / r["vectors"] for r in rs]
    return {"precision": prec, "from_vectors": run["vectors"], "sizes_measured": sorted(r["vectors"] for r in rs),
            "index_bytes_per_vector": run["hnsw_index_bytes"] / run["vectors"],
            "heap_bytes_per_vector": run["heap_bytes"] / run["vectors"],
            "total_bytes_per_vector": run["total_bytes"] / run["vectors"],
            "bytes_per_vector_spread_across_sizes": round((max(per) - min(per)) / min(per), 4)}


def latency_projection(runs: list[dict]) -> dict:
    """Fit p50 / p95 = a + b*ln(N) on the measured float32 runs (ef_search=40, unfiltered, k=5)."""
    rs = sorted((r for r in runs if precision(r) == "float32"), key=lambda r: r["vectors"])
    out = {"measured": {str(r["vectors"]): r["ef40_unfiltered"] for r in rs}, "projected": {}, "fit": {}}
    if len(rs) < 2:
        return out
    xs = [math.log(r["vectors"]) for r in rs]
    mx = sum(xs) / len(xs)
    for stat in ("p50_ms", "p95_ms"):
        ys = [r["ef40_unfiltered"][stat] for r in rs]
        my = sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        out["fit"][stat] = {"a": round(a, 3), "b_per_ln_n": round(b, 3)}
        for n in SIZES:
            out["projected"].setdefault(str(n), {})[stat] = round(max(a + b * math.log(n), 0.0), 2)
    return out


def pgvector_cost(n: int, p: dict, w: dict, m: dict, hours: int) -> dict:
    s = p["self_hosted_pgvector"]
    index_gb = n * m["index_bytes_per_vector"] / GB
    # the benchmark heap holds id + doc_type + vector; add the chunk text the app stores
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
    runs = load_bench()
    m = measured_bytes(runs, "float32")
    mh = measured_bytes(runs, "halfvec")
    rows = []
    for n in SIZES:
        rows.append({
            "vectors": n,
            "pgvector": pgvector_cost(n, p, w, m, hours),
            "pgvector_halfvec": pgvector_cost(n, p, w, mh, hours) if mh else None,
            "pinecone_serverless": pinecone_serverless_cost(n, p, w),
            "pinecone_pods": pinecone_pods_cost(n, p, w, hours),
        })
    # Sensitivity: serverless reads scale with queries x index size; pgvector does not (a single
    # node serves these rates easily: 10M queries/month is ~4 QPS at ~2 ms per search).
    sensitivity = []
    for q in QUERY_VOLUMES:
        wq = {**w, "queries_per_month": q}
        sensitivity.append({"queries_per_month": q, **{
            str(n): {"pinecone_serverless": pinecone_serverless_cost(n, p, wq)["total_usd"],
                     "pgvector": pgvector_cost(n, p, w, m, hours)["infra_usd"]} for n in SIZES}})
    out = {"measured_storage": m, "measured_storage_halfvec": mh, "latency": latency_projection(runs),
           "query_volume_sensitivity": sensitivity, "assumptions": p, "rows": rows}
    res = PROJECT_ROOT / "results"
    (res / "cost_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (res / "cost_comparison.md").write_text(render(out), encoding="utf-8")
    print(render(out))


def usd(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x:+.0%}".replace("+", "−" if x < 0 else "")


def render(o: dict) -> str:
    m, mh, w = o["measured_storage"], o["measured_storage_halfvec"], o["assumptions"]["workload"]
    ops = o["assumptions"]["self_hosted_pgvector"]
    sizes = ", ".join(f"{n:,}" for n in m["sizes_measured"])
    lines = [
        "# Monthly vector-store cost: self-hosted pgvector vs Pinecone", "",
        f"Storage per vector measured at {sizes} vectors (float32 index): HNSW index "
        f"{m['index_bytes_per_vector']:,.0f} B, heap {m['heap_bytes_per_vector']:,.0f} B, plus "
        f"{w['avg_chunk_text_bytes']:,} B of chunk text. Bytes/vector differ by "
        f"{m['bytes_per_vector_spread_across_sizes']:.1%} between the measured sizes."
        + (f" halfvec index: {mh['index_bytes_per_vector']:,.0f} B/vector." if mh else ""), "",
        f"Workload: {w['dim']}-d vectors, {w['queries_per_month']:,} queries/month, "
        f"{w['monthly_rewrite_fraction']:.0%} of vectors rewritten per month. All prices: `eval/pricing.yaml`.", "",
        "| Vectors | pgvector (EC2 + EBS) | pgvector, halfvec index | pgvector + ops time | Pinecone serverless "
        "| Pinecone pods (p1, always-on) | pgvector vs serverless | pgvector vs pods |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in o["rows"]:
        pg, pgh, sv, pod = r["pgvector"], r["pgvector_halfvec"], r["pinecone_serverless"], r["pinecone_pods"]
        half = f"{usd(pgh['infra_usd'])} ({pgh['instance']})" if pgh else "n/a"
        lines.append(
            f"| {r['vectors']:,} | {usd(pg['infra_usd'])} ({pg['instance']}) | {half} | {usd(pg['total_with_ops_usd'])} | "
            f"{usd(sv['total_usd'])} | {usd(pod['total_usd'])} ({pod['pods']} pod{'s' if pod['pods'] > 1 else ''}) | "
            f"{pct(1 - pg['infra_usd'] / sv['total_usd'])} | {pct(1 - pg['infra_usd'] / pod['total_usd'])} |")
    lines += ["", f"The comparison columns compare pgvector infrastructure only; a negative value means pgvector "
              f"costs more. Operating time is shown separately: {ops['ops_hours_per_month']} h/month at "
              f"${ops['ops_usd_per_hour']}/h."]

    lines += ["", "## Breakdown", "",
              "| Vectors | pgvector index GB | RAM needed GB | disk GB | compute $ | storage $ | Pinecone storage GB "
              "| RU / query | Pinecone usage $ (before $50 minimum) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in o["rows"]:
        pg, sv = r["pgvector"], r["pinecone_serverless"]
        lines.append(f"| {r['vectors']:,} | {pg['index_gb']} | {pg['ram_needed_gb']} | {pg['disk_gb']} | "
                     f"{pg['compute_usd']} | {pg['storage_usd']} | {sv['storage_gb']} | {sv['ru_per_query']} | "
                     f"{sv['usage_usd']} |")

    lines += ["", "## Sensitivity to query volume (monthly $, pgvector infrastructure vs Pinecone serverless)", "",
              "| Queries / month | " + " | ".join(f"{n:,} vectors: pgvector / serverless" for n in SIZES) + " |",
              "|---|" + "---|" * len(SIZES)]
    for s in o["query_volume_sensitivity"]:
        cells = [f"{usd(s[str(n)]['pgvector'])} / {usd(s[str(n)]['pinecone_serverless'])}" for n in SIZES]
        lines.append(f"| {s['queries_per_month']:,} | " + " | ".join(cells) + " |")

    lat = o["latency"]
    lines += ["", "## Retrieval latency vs size (HNSW, k=5, ef_search=40, unfiltered)", "",
              "| Vectors | p50 ms | p95 ms | Source |", "|---|---|---|---|"]
    measured = {int(k): v for k, v in lat["measured"].items()}
    rows = [(n, v["p50_ms"], v["p95_ms"], "measured") for n, v in measured.items()]
    rows += [(int(n), v["p50_ms"], v["p95_ms"], "projected (a + b*ln N fit)")
             for n, v in lat["projected"].items() if int(n) not in measured]
    for n, p50, p95, src in sorted(rows):
        lines.append(f"| {n:,} | {p50} | {p95} | {src} |")
    if lat["fit"]:
        f50 = lat["fit"]["p50_ms"]
        lines += ["", f"Fit for p50: {f50['a']} + {f50['b_per_ln_n']}*ln(N) ms. This is a projection from "
                  f"{len(measured)} measured sizes, not a measurement."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
