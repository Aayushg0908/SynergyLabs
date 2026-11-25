"""judge — LLM-as-judge pipeline CLI.

  judge run        suite in -> structured verdicts + suite report out
  judge generate   produce model outputs for a suite with a generator config (for A/B)
  judge compare    A/B comparison of two output sets -> declared winner
  judge bias       bias experiments (position, verbosity, sycophancy, clustering, self-preference)
  judge validate   judge validation (gold-label agreement, test-retest, adversarial probes)
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings
from .llm import ChatClient, Usage, usage_dict


def make_judge_client(spec: str, usage: Usage | None = None, temperature: float | None = None) -> ChatClient:
    """spec: 'primary' | 'secondary' | '<provider>:<model>'."""
    s = get_settings()
    if spec == "primary":
        provider, model = s.judge_provider, s.judge_model
    elif spec == "secondary":
        provider, model = s.judge2_provider, s.judge2_model
    else:
        provider, _, model = spec.partition(":")
    t = s.judge_temperature if temperature is None else temperature
    # 8192: Nemotron's hidden reasoning occasionally used the whole 4096-token budget, leaving an
    # empty answer (1 of 112 A/B calls). max_tokens is not part of the replay-cache key.
    return ChatClient(provider, model, temperature=t, max_tokens=8192, role="judge", usage=usage)


def cmd_run(args) -> None:
    from .aggregate import suite_report
    from .judge import Judge, run_parallel
    from .report import results_dir, suite_markdown, write_json, write_jsonl
    from .suite import load_suite

    cases = load_suite(Path(args.suite))
    usage = Usage()
    judge = Judge(make_judge_client(args.judge, usage), args.variant)
    tag = args.tag or f"suite_{args.variant}"
    print(f"judging {len(cases)} cases with {judge.name}", file=sys.stderr)
    verdicts = run_parallel(lambda c: judge.pointwise(c, tag=tag), cases, get_settings().max_concurrency, "cases")
    rep = suite_report(cases, verdicts)
    out = results_dir(tag)
    write_jsonl(out / "verdicts.jsonl", verdicts)
    write_json(out / "report.json", {"judge": judge.name, "suite": str(args.suite), "report": rep, "usage": usage_dict(usage)})
    (out / "report.md").write_text(suite_markdown(f"Suite report — {tag}", judge.name, rep, usage_dict(usage), cases, verdicts),
                                   encoding="utf-8")
    print(f"pass rate {rep['pass_rate']:.2f}, mean overall {rep['mean_overall']:.2f}, parse {rep['parse_status']} -> {out}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="judge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="judge a suite and write a report")
    r.add_argument("--suite", default=str(get_settings().results_dir.parent / "suites" / "core_suite.yaml"))
    r.add_argument("--variant", choices=["mitigated", "naive"], default="mitigated")
    r.add_argument("--judge", default="primary", help="primary | secondary | provider:model")
    r.add_argument("--tag", help="results/<tag>/ output folder")
    r.set_defaults(fn=cmd_run)

    from . import experiments
    experiments.add_commands(sub, make_judge_client)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
