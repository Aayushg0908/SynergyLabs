"""Fast unit tests that need neither the database nor an API key."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from rag_app.chunking import Chunker
from rag_app.loaders import load_document
from rag_app.pipeline import chunk_id, normalize_citations

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def test_normalize_citations():
    assert normalize_citations("31 minutes【1】.") == "31 minutes[1]."
    assert normalize_citations("x 【2†source】 y [1, 3]") == "x [2] y [1][3]"
    assert normalize_citations("already [4]") == "already [4]"


def test_chunk_id_is_deterministic():
    a = chunk_id("a.md", "h", 256, 32, 0)
    assert a == chunk_id("a.md", "h", 256, 32, 0)
    assert a != chunk_id("a.md", "h", 256, 32, 1)
    assert a != chunk_id("a.md", "h2", 256, 32, 0)


def test_chunks_respect_size_and_overlap(tokenizer):
    text = load_document(CORPUS / "policies" / "travel-and-expense-policy.pdf").text
    chunker = Chunker(tokenizer, chunk_size=64, chunk_overlap=16)
    chunks = chunker.split(text)
    assert len(chunks) > 3
    # allow a few tokens of slack for the separators added when joining units
    assert all(c.token_count <= 64 + 4 for c in chunks)
    # consecutive chunks share text (overlap) and together cover the document
    assert any(chunks[i].text.split()[0] in chunks[i - 1].text for i in range(1, len(chunks)))
    assert "$0.67 per mile" in " ".join(c.text for c in chunks)


@pytest.mark.parametrize("rel,needle", [
    ("product/kx200-specifications.html", "Maximum payload | 1.5 kg"),
    ("engineering/api-rate-limits.md", "Retry-After"),
    ("policies/leave-policy.pdf", "20 weeks of fully paid parental leave"),
])
def test_loaders_extract_text(rel, needle):
    doc = load_document(CORPUS / rel)
    assert needle in doc.text
    assert doc.title and len(doc.content_hash) == 64
