"""Print submission evidence straight from the committed call log and result files.

  python scripts/show_evidence.py prompt   [case_id]   raw judge prompt + raw response (pointwise)
  python scripts/show_evidence.py position [case_id]   one pair judged in both orders + flip rate
  python scripts/show_evidence.py probe    [probe_id]  verdict on an adversarial probe, naive vs mitigated
"""

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def calls():
    for line in (ROOT / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def show_prompt(case_id: str = "rm03") -> None:
    rec = next(r for r in calls() if r["tag"] == "suite_mitigated" and r["meta"].get("case_id") == case_id
               or r["tag"] == "suite" and r["meta"].get("case_id") == case_id)
    print(f"=== logs/calls.jsonl  call_id={rec['call_id']}  judge={rec['provider']}/{rec['model']}  "
          f"T={rec['temperature']}  json_mode={rec['json_mode']}")
    print(f"=== tokens: prompt {rec['usage']['prompt_tokens']}, completion {rec['usage']['completion_tokens']} "
          f"(reasoning {rec['usage']['reasoning_tokens']}), latency {rec['latency_ms'] / 1000:.1f} s")
    for m in rec["messages"]:
        print(f"\n--- {m['role'].upper()} PROMPT ---")
        print(m["content"])
    print("\n--- RAW RESPONSE (unparsed) ---")
    print(rec["raw_response"])


def show_position(case_id: str | None = None) -> None:
    rows = [json.loads(l) for l in (ROOT / "results" / "ab" / "pairwise.jsonl").read_text(encoding="utf-8").splitlines()]
    comp = json.loads((ROOT / "results" / "ab" / "comparison.json").read_text(encoding="utf-8"))
    row = next(r for r in rows if (r["case_id"] == case_id if case_id else r["flip"]))
    print(f"=== Position-bias check: case {row['case_id']}, A = {comp['config_a']}, B = {comp['config_b']}")
    for key, label in (("ab", "Order AB (A shown as Response 1)"), ("ba", "Order BA (B shown as Response 1)")):
        v = row[key]
        print(f"\n{label}: winner = {v['winner']}")
        print(textwrap.fill("  rationale: " + v["rationale"], 120, subsequent_indent="    "))
    print(f"\nFinal verdict (both orders must agree): {row['final']}   flip = {row['flip']}")
    pw = comp["pairwise"]
    print(f"\nWhole A/B run: {comp['n_pairs']} pairs, position flip rate = {pw['position_flip_rate']:.2f} "
          f"-> flipped pairs are scored as ties")
    bias = ROOT / "results" / "bias" / "bias_results.json"
    if bias.exists():
        b = json.loads(bias.read_text(encoding="utf-8")).get("position")
        if b:
            print(f"Flip rate naive prompt {b['naive']['flip_rate']:.2f} -> mitigated prompt {b['mitigated']['flip_rate']:.2f}; "
                  f"first-position win share {b['naive']['first_position_win_share']:.2f} -> {b['mitigated']['first_position_win_share']:.2f}")


def show_probe(probe_id: str = "p_fq01_verbose_wrong") -> None:
    b = json.loads((ROOT / "results" / "bias" / "bias_results.json").read_text(encoding="utf-8"))["probes"]
    import yaml
    probes = {p["id"]: p for p in yaml.safe_load((ROOT / "probes" / "adversarial.yaml").read_text(encoding="utf-8"))["probes"]}
    p = probes[probe_id]
    print(f"=== Adversarial probe {probe_id}  (family: {p['family']}, an unbiased judge must: {p['expect']})")
    print("\n--- probe output shown to the judge ---")
    print(p["model_output"])
    for variant in ("naive", "mitigated"):
        row = next(r for r in b[variant]["probes"] if r["id"] == probe_id)
        print(f"\n[{variant} prompt] overall = {row['overall']}, correctness = {row['correctness']}, "
              f"{'PASS' if row['passed'] else 'FAIL'} -> {'FOOLED' if row['fooled'] else 'not fooled'}")
    rec = next(r for r in reversed(list(calls())) if r["tag"] == "bias:probe:mitigated" and r["meta"].get("case_id") == probe_id)
    print("\n--- mitigated judge raw response ---")
    print(rec["raw_response"])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["prompt"]
    {"prompt": show_prompt, "position": show_position, "probe": show_probe}[cmd](*rest)
