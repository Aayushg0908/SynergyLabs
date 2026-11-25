"""Turn PDF / HTML / Markdown files into clean plain text plus a title."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

FORMATS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md", ".markdown": "md"}

_BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "div", "section", "br"]


@dataclass
class LoadedDocument:
    path: Path
    title: str
    text: str
    source_format: str
    content_hash: str  # sha256 of the raw file bytes


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_lines(text: str) -> str:
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def load_markdown(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    title = None
    front = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.S)
    if front:
        for line in front.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "title":
                title = value.strip()
        raw = raw[front.end():]
    if not title:
        heading = re.search(r"^#\s+(.+)$", raw, re.M)
        title = heading.group(1).strip() if heading else path.stem
    return title, _clean_lines(raw)


def load_html(path: Path) -> tuple[str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "head"]):
        tag.decompose()
    body = soup.body or soup
    # 1) collapse source-code line wraps inside text nodes
    for node in body.find_all(string=True):
        node.replace_with(re.sub(r"\s+", " ", str(node)))
    # 2) table rows -> "cell | cell | cell" lines
    for tr in body.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(soup.new_string(" | ".join(cells) + "\n"))
    # 3) block elements -> line breaks
    for tag in body.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")
    return title, _clean_lines(body.get_text())


def load_pdf(path: Path) -> tuple[str, str]:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = _clean_lines(text)
    # Re-flow hard line wraps: join a line onto the previous one when the previous
    # line does not end a sentence and the next line continues it.
    text = re.sub(r"(?<![.:!?])\n(?=[a-z0-9$(%])", " ", text)
    meta_title = reader.metadata.title if reader.metadata else None
    title = meta_title or (text.splitlines()[0] if text else path.stem)
    return title, text


def load_document(path: Path) -> LoadedDocument:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported file type: {path}")
    loader = {"pdf": load_pdf, "html": load_html, "md": load_markdown}[fmt]
    title, text = loader(path)
    return LoadedDocument(path=path, title=title, text=text, source_format=fmt, content_hash=file_sha256(path))


def discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS)
