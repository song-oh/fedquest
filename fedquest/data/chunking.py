from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORDS_PER_TOKEN = 0.75


def chunk_text(text: str, chunk_tokens: int = 256, overlap: int = 32) -> list[str]:
    max_words = max(8, int(chunk_tokens * WORDS_PER_TOKEN))
    overlap_words = max(0, int(overlap * WORDS_PER_TOKEN))
    words: list[list[str]] = []
    for sentence in _SENTENCE.split(text.strip()):
        toks = sentence.split()
        for i in range(0, len(toks), max_words):
            words.append(toks[i : i + max_words])
    chunks: list[str] = []
    current: list[str] = []
    for sent in words:
        if current and len(current) + len(sent) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words else []
            if len(current) + len(sent) > max_words:
                current = []
        current.extend(sent)
    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks or ([text.strip()] if text.strip() else [])
