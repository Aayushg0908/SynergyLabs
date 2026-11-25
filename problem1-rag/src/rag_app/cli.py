"""Command-line interface:  rag ingest | query | stats | reset"""

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Kestrel RAG service CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="ingest a file or folder of PDF/HTML/MD")
    p.add_argument("path", nargs="?", help="default: CORPUS_DIR")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--prune", action="store_true", help="delete documents whose files no longer exist")

    q = sub.add_parser("query", help="ask a question")
    q.add_argument("question")
    q.add_argument("-k", type=int)
    q.add_argument("--doc-type", action="append", help="metadata filter; repeatable")
    q.add_argument("--full", action="store_true", help="print the full JSON response")

    e = sub.add_parser("eval", help="run the 3-layer evaluation on eval/questions.yaml")
    e.add_argument("-k", type=int)
    e.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    e.add_argument("--retrieval-only", action="store_true", help="retrieval + latency layers only (no LLM calls)")
    e.add_argument("--latency-reps", type=int, default=10)
    e.add_argument("--gen-interval", type=float, default=2.0, help="seconds between generator calls (rate limits)")
    e.add_argument("--judge-interval", type=float, default=7.0, help="seconds between judge calls (rate limits)")
    e.add_argument("--tag", default="eval", help="output file prefix in results/")

    sub.add_parser("stats", help="show document/vector counts")
    r = sub.add_parser("reset", help="drop all tables")
    r.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from .config import get_settings
    from .store import VectorStore

    if args.cmd == "reset":
        if not args.yes:
            sys.exit("refusing to drop tables without --yes")
        VectorStore.reset(get_settings().database_url)
        print("dropped tables documents, chunks")
        return

    from .pipeline import RAGService

    svc = RAGService()
    if args.cmd == "ingest":
        result = svc.ingest(Path(args.path) if args.path else None, args.chunk_size, args.chunk_overlap, args.prune)
        result.pop("documents") if len(result["documents"]) > 30 else None
        print(json.dumps(result, indent=2))
    elif args.cmd == "eval":
        from .evaluation.harness import run

        result = run(svc, args.k, use_judge=not args.no_judge, retrieval_only=args.retrieval_only,
                     latency_reps=args.latency_reps, gen_interval=args.gen_interval,
                     judge_interval=args.judge_interval, tag=args.tag)
        print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2, default=float))
    elif args.cmd == "stats":
        print(json.dumps(svc.store.stats(), indent=2))
    elif args.cmd == "query":
        resp = svc.answer(args.question, args.k, args.doc_type)
        if args.full:
            print(json.dumps(resp, indent=2))
        else:
            print("\nANSWER:", resp["answer"])
            for c in resp["citations"]:
                print(f"  [{c['ref']}] {c['doc_id']}  (score {c['score']})")
            print("usage:", resp["usage"], "| latency_ms:", resp["latency_ms"])


if __name__ == "__main__":
    main()
