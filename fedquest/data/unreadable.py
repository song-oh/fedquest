from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MMQA_UNREADABLE = frozenset({
    "17ae0616ac745e70781203267f3a382d",
    "bf201cbbd058ef51aef89b1be4158c2a",
    "ef457a7b3ab437cd78ab9f82dc083048",
    "225c3db49d60b5ef30ed0bfc649ebf78",
    "b413cc1dc4969dcbe4cb6a55c0f2e359",
})


def unreadable(obj: Any) -> bool:
    get = obj.get if isinstance(obj, dict) else lambda k, d=None: getattr(obj, k, d)
    return get("modality") == "image" and get("doc_id") in MMQA_UNREADABLE


def skip_rows(objects: Iterable[Any]) -> set[int]:
    return {i for i, o in enumerate(objects) if unreadable(o)}
