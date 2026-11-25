"""Structure-aware, token-based chunking.

Text is split into units (lines, then sentences). Units are packed greedily into chunks
of at most `chunk_size` tokens; each new chunk starts with the trailing units of the
previous chunk, up to `chunk_overlap` tokens. Chunks therefore never cut a sentence in
half unless a single sentence is longer than `chunk_size` (then it is split by tokens).

Tokens are counted with the embedding model's own tokenizer, so `chunk_size` maps
directly onto the model's 512-token input limit.
"""

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"'(])")


@dataclass
class _Unit:
    text: str
    tokens: int
    new_line: bool  # unit started a new line in the source


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int


class Chunker:
    def __init__(self, tokenizer, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("need chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for i, sentence in enumerate(_SENTENCE_SPLIT.split(line)):
                ids = self.tokenizer.encode(sentence, add_special_tokens=False)
                if len(ids) <= self.chunk_size:
                    units.append(_Unit(sentence, len(ids), new_line=(i == 0)))
                    continue
                step = self.chunk_size - self.chunk_overlap
                for j, start in enumerate(range(0, len(ids), step)):
                    piece = ids[start:start + self.chunk_size]
                    units.append(_Unit(self.tokenizer.decode(piece), len(piece), new_line=(i == 0 and j == 0)))
                    if start + self.chunk_size >= len(ids):
                        break
        return units

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        out = []
        for i, u in enumerate(units):
            if i > 0:
                out.append("\n" if u.new_line else " ")
            out.append(u.text)
        return "".join(out)

    def split(self, text: str) -> list[Chunk]:
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        for unit in self._units(text):
            if current and current_tokens + unit.tokens > self.chunk_size:
                groups.append(current)
                # carry trailing units forward as overlap
                carry: list[_Unit] = []
                carried = 0
                for prev in reversed(current):
                    if carried + prev.tokens > self.chunk_overlap:
                        break
                    carry.insert(0, prev)
                    carried += prev.tokens
                while carry and carried + unit.tokens > self.chunk_size:
                    carried -= carry.pop(0).tokens
                current, current_tokens = carry, carried
            current.append(unit)
            current_tokens += unit.tokens
        if current:
            groups.append(current)

        chunks = []
        for i, group in enumerate(groups):
            body = self._join(group)
            chunks.append(Chunk(index=i, text=body, token_count=self.count_tokens(body)))
        return chunks
