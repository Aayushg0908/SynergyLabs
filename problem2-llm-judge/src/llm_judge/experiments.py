"""A/B comparison, bias experiments and judge validation (registered as CLI subcommands)."""

import sys
from pathlib import Path

import yaml

from .config import PROJECT_ROOT, get_settings
from .llm import ChatClient, Usage, usage_dict
from .report import results_dir, write_json, write_jsonl
from .schema import TestCase
from .suite import DEFAULT_SUITE, load_suite

CONFIGS = PROJECT_ROOT / "configs"


# ===================================================================== generation
def load_gen_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def generate_outputs(cfg: dict, cases: list[TestCase], usage: Usage) -> list[TestCase]:
    from .judge import run_parallel

    client = ChatClient(cfg["provider"], cfg["model"], cfg.get("temperature", 0.0), cfg.get("max_tokens", 1024),
                        role="generator", usage=usage)

    def one(case: TestCase) -> TestCase:
        system = cfg["system_template"].format(system_prompt=case.system_prompt or cfg["default_system_prompt"]).strip()
        r = client.chat([{"role": "system", "content": system}, {"role": "user", "content": case.input}],
                        tag=f"generate:{cfg['name']}", meta={"case_id": case.id, "config": cfg["name"]})
        return case.model_copy(update={"model_output": r.text.strip(), "gold_label": None})

    return run_parallel(one, cases, 3, f"generate {cfg['name']}")


def cmd_generate(args) -> None:
    cfg = load_gen_config(args.config)
    cases = load_suite(Path(args.suite))
    usage = Usage()
    outs = generate_outputs(cfg, cases, usage)
    path = results_dir("ab") / f"outputs_{cfg['name']}.yaml"
    path.write_text(yaml.safe_dump({"generator": {k: cfg[k] for k in ("name", "provider", "model", "temperature")},
                                    "cases": [c.model_dump(exclude_none=True) for c in outs]},
                                   allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
    print(f"wrote {len(outs)} outputs -> {path}  ({usage.calls} generator calls, "
          f"{usage.prompt_tokens + usage.completion_tokens:,} tokens)")


# ===================================================================== A/B comparison
def cmd_compare(args) -> None:
    from .aggregate import compare_configs, suite_report
    from .judge import Judge, run_parallel

    a_cases, b_cases = load_suite(Path(args.a)), load_suite(Path(args.b))
    name_a, name_b = args.name_a or Path(args.a).stem.replace("outputs_", ""), args.name_b or Path(args.b).stem.replace("outputs_", "")
    assert [c.id for c in a_cases] == [c.id for c in b_cases], "A and B must cover the same cases"
    usage = Usage()
    judge = Judge(_judge_client(args.judge, usage), args.variant)
    w = get_settings().max_concurrency
    print(f"A/B {name_a} vs {name_b}: {len(a_cases)} cases, judge {judge.name}", file=sys.stderr)
    point_a = run_parallel(lambda c: judge.pointwise(c, tag="ab:pointwise"), a_cases, w, f"pointwise {name_a}")
    point_b = run_parallel(lambda c: judge.pointwise(c, tag="ab:pointwise"), b_cases, w, f"pointwise {name_b}")
    pairs = run_parallel(lambda ab: judge.pairwise(ab[0], ab[0].model_output, ab[1].model_output, tag="ab:pairwise"),
                         list(zip(a_cases, b_cases)), w, "pairwise (both orders)")
    result = compare_configs(name_a, name_b, a_cases, point_a, point_b, pairs)
    result["judge"], result["usage"] = judge.name, usage_dict(usage)
    out = results_dir(args.tag)
    write_json(out / "comparison.json", result)
    write_jsonl(out / "pairwise.jsonl", [{"case_id": p["case_id"], "final": p["final"], "flip": p["flip"],
                                           "ab": p["ab"].model_dump(), "ba": p["ba"].model_dump()} for p in pairs])
    write_jsonl(out / f"pointwise_{name_a}.jsonl", point_a)
    write_jsonl(out / f"pointwise_{name_b}.jsonl", point_b)
    (out / "comparison.md").write_text(comparison_markdown(result, a_cases, point_a, point_b, pairs), encoding="utf-8")
    print(result["decision"])


def comparison_markdown(r: dict, cases, point_a, point_b, pairs) -> str:
    from .report import f
    pw, pt = r["pairwise"], r["pointwise"]
    ci = pt["bootstrap_95ci"]
    lines = [f"# A/B comparison: {r['config_a']} vs {r['config_b']}", "",
             f"Judge `{r['judge']}`, {r['n_pairs']} cases. Pairwise verdicts run in both orders; a pair counts as a win "
             "only if both orders agree, otherwise it is a tie.", "",
             f"**Decision: {r['decision']}**", "",
             "| Config | Pass rate | Mean overall | Pairwise wins | Win rate (ties = 0.5) |", "|---|---|---|---|---|",
             f"| A: {r['config_a']} | {f(pt['a']['pass_rate'])} | {f(pt['a']['mean_overall'])} | {pw['wins_a']} | {f(pw['win_rate_a'])} |",
             f"| B: {r['config_b']} | {f(pt['b']['pass_rate'])} | {f(pt['b']['mean_overall'])} | {pw['wins_b']} | {f(pw['win_rate_b'])} |", "",
             f"Ties {pw['ties']} · sign test p = {pw['sign_test_p']} · position flip rate {f(pw['position_flip_rate'])} · "
             f"mean overall diff (B - A) {f(pt['mean_overall_diff_b_minus_a'])}, 95% bootstrap CI "
             f"[{f(ci[0]) if ci else 'n/a'}, {f(ci[1]) if ci else 'n/a'}]", "",
             "| Criterion | A mean | B mean |", "|---|---|---|"]
    for k in pt["a"]["mean_by_criterion"]:
        lines.append(f"| {k} | {f(pt['a']['mean_by_criterion'][k])} | {f(pt['b']['mean_by_criterion'][k])} |")
    lines += ["", "| case | A overall | B overall | AB order | BA order | final |", "|---|---|---|---|---|---|"]
    for c, va, vb, p in zip(cases, point_a, point_b, pairs):
        lines.append(f"| {c.id} | {f(va.overall)} | {f(vb.overall)} | {p['ab'].winner} | {p['ba'].winner} | {p['final']} |")
    return "\n".join(lines) + "\n"


def _judge_client(spec: str, usage: Usage, temperature: float | None = None) -> ChatClient:
    from .cli import make_judge_client
    return make_judge_client(spec, usage, temperature)


def add_commands(sub, make_judge_client) -> None:
    g = sub.add_parser("generate", help="generate outputs for a suite with a generator config")
    g.add_argument("--config", required=True, help="configs/gen_*.yaml")
    g.add_argument("--suite", default=str(DEFAULT_SUITE))
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("compare", help="A/B compare two output sets and declare a winner")
    c.add_argument("--a", default=str(get_settings().results_dir / "ab" / "outputs_prompt_v1.yaml"))
    c.add_argument("--b", default=str(get_settings().results_dir / "ab" / "outputs_prompt_v2.yaml"))
    c.add_argument("--name-a")
    c.add_argument("--name-b")
    c.add_argument("--judge", default="primary")
    c.add_argument("--variant", choices=["mitigated", "naive"], default="mitigated")
    c.add_argument("--tag", default="ab")
    c.set_defaults(fn=cmd_compare)

    from .bias import cmd_bias
    b = sub.add_parser("bias", help="bias experiments: naive vs mitigated prompt")
    b.add_argument("--only", nargs="+", choices=["position", "probes", "spread", "selfpref"])
    b.add_argument("--selfpref-n", type=int, default=20, help="pairs used for the self-preference test")
    b.set_defaults(fn=cmd_bias)

    from .validate import cmd_validate
    v = sub.add_parser("validate", help="judge validation: gold agreement, test-retest, cross-family, probes")
    v.add_argument("--repeats", type=int, default=2, help="re-runs per temperature for test-retest")
    v.add_argument("--skip-secondary", action="store_true")
    v.set_defaults(fn=cmd_validate)
