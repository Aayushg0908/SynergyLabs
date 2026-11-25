"""Check that every evidence span in questions.yaml really occurs in its source document.

Run:  python eval/validate_questions.py
"""

import sys
from pathlib import Path

import yaml

from rag_app.config import get_settings
from rag_app.evaluation.metrics import contains_span
from rag_app.loaders import load_document

QUESTIONS = Path(__file__).with_name("questions.yaml")


def main() -> int:
    corpus = get_settings().corpus_dir
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    missing = [(q["id"], e["doc"], e["span"]) for q in questions for e in q["evidence"]
               if not contains_span(load_document(corpus / e["doc"]).text, e["span"])]
    n_spans = sum(len(q["evidence"]) for q in questions)
    print(f"{len(questions)} questions, {n_spans} evidence spans")
    for m in missing:
        print("MISSING:", *m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
