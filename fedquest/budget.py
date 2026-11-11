from __future__ import annotations

import glob
import os
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

# a generated token costs as much as four prompt tokens
OUT_WEIGHT = 4
FRAME_TOKENS = 10
MESSAGE_TOKENS = 10
IMAGE_TOKENS = 280
EXPECTED_OUT = {"typed_graph": 200, "typed_fix": 200, "typed_repair": 200, "read_index": 60, "ircot": 48,
                "planner": 160, "extract": 300, "ground": 40, "route": 40,
                "rewoo_plan": 300, "rewoo_worker": 24}
DEFAULT_OUT = 64


class BudgetExhausted(RuntimeError):
    pass


class TokenCounter:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or _tokenizer_path()
        self._tok = None
        self._lock = threading.Lock()
        if self.path:
            try:
                from tokenizers import Tokenizer

                self._tok = Tokenizer.from_file(self.path)
            except Exception:
                self._tok = None

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def text(self, s: str) -> int:
        if not s:
            return 0
        if self._tok is None:
            return max(1, len(s) // 4)
        with self._lock:
            return len(self._tok.encode(s, add_special_tokens=False).ids)

    def messages(self, messages: Sequence[dict[str, Any]]) -> int:
        n, images = FRAME_TOKENS, 0
        for m in messages:
            n += MESSAGE_TOKENS
            content = m.get("content")
            if isinstance(content, str):
                n += self.text(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        n += self.text(str(part.get("text", "")))
                    elif part.get("type") in ("image_url", "image"):
                        images += 1
        return n + IMAGE_TOKENS * images


def _tokenizer_path() -> str | None:
    explicit = os.environ.get("FEDQUEST_TOKENIZER")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface")]
    for root in roots:
        if not root:
            continue
        hits = sorted(glob.glob(os.path.join(root, "hub", "models--google--gemma-4-31B-it", "snapshots", "*",
                                             "tokenizer.json")))
        if hits:
            return hits[-1]
    return None


@lru_cache(maxsize=1)
def counter() -> TokenCounter:
    return TokenCounter()


def count_messages(messages: Sequence[dict[str, Any]]) -> int:
    return counter().messages(messages)


def count_text(s: str) -> int:
    return counter().text(s)


def expected_out(task: str, max_tokens: int) -> int:
    return min(int(max_tokens), EXPECTED_OUT.get(task, DEFAULT_OUT))


def call_cost(messages: Sequence[dict[str, Any]], task: str, max_tokens: int) -> int:
    return count_messages(messages) + OUT_WEIGHT * expected_out(task, max_tokens)


def check(ledger: Any, messages: Sequence[dict[str, Any]], *, task: str, max_tokens: int, charge: str | None) -> None:
    limit = getattr(ledger, "budget", None)
    # the answer call is never refused; ledger.reserve keeps room for it
    if limit is None or charge == "answer":
        return
    need = call_cost(messages, task, max_tokens)
    if ledger.total_tokens + need + ledger.reserve > limit:
        ledger.budget_refusals += 1
        raise BudgetExhausted(f"{task}: {need} more cost units would overrun the budget of {limit} "
                              f"({ledger.total_tokens} spent, {ledger.reserve} reserved for the answer)")


def remaining(ledger: Any) -> int | None:
    limit = getattr(ledger, "budget", None)
    return None if limit is None else limit - ledger.total_tokens


def fit_indices(texts: Sequence[str], room: int, items: int | None = None) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    used = 0
    for i, t in enumerate(texts):
        if items is not None and len(out) >= items:
            break
        n = count_text(t) + 2
        if used + n > room:
            continue
        out.append((i, t))
        used += n
    if not out and texts and room > 32:
        out.append((0, texts[0][: max(64, (room - 2) * 4)]))
    return out


def fit_texts(texts: Sequence[str], room: int, items: int | None = None) -> list[str]:
    return [t for _, t in fit_indices(texts, room, items)]
