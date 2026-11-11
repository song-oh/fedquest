# fact extraction per object, query-conditioned, or several objects in one call
from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from fedquest.llm import prompts
from fedquest.llm.client import LLMClient
from fedquest.llm.parsing import parse_json, parse_triples
from fedquest.types import ExtractionResult, Fact, ObjectRecord, Usage

log = logging.getLogger(__name__)

BLOCK_MAX_TOKENS = 4096


def load_image_data_uri(path: Path, max_side: int = 768, quality: int = 85) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
    except (OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def uri_bytes(uri: str | None) -> int:
    if not uri:
        return 0
    b64 = uri.split(",", 1)[-1]
    return len(b64) * 3 // 4 - (len(b64) - len(b64.rstrip("=")))


def content_bytes(text: str, uri: str | None = None) -> int:
    return len(text.encode()) + uri_bytes(uri)


def shown_text(obj: ObjectRecord, mode: str) -> str:
    if obj.modality != "image":
        return f"{obj.title}\n{obj.text}"
    if mode == "query":
        return ""
    if mode in ("query_t", "query_block"):
        return prompts.image_of(obj.title)
    return obj.title if obj.text == obj.title else f"{obj.title}\n{obj.text}"


def _live(live: bool) -> dict[str, bool]:
    return {"live": True} if live else {}


class Extractor:
    def __init__(self, llm: LLMClient, *, raw_root: Path, version: str = prompts.EXTRACT_VERSION) -> None:
        self.llm = llm
        self.raw_root = raw_root
        self.version = version

    def _image(self, obj: ObjectRecord) -> str | None:
        return load_image_data_uri(self.raw_root / obj.image_path) if obj.image_path else None

    def inference_bytes(self, obj: ObjectRecord, mode: str) -> int:
        uri = self._image(obj) if obj.modality == "image" else None
        return content_bytes(shown_text(obj, mode), uri)

    def _result(self, obj: ObjectRecord, site: str, version: str, triples: list[tuple[str, str, str]],
                usage: Usage, error: str | None = None, shown: int = 0) -> ExtractionResult:
        facts = [Fact(h, r, t, site, obj.oid, version, obj.modality) for h, r, t in triples]
        return ExtractionResult(site, obj.oid, obj.modality, version, facts, usage, doc_id=obj.doc_id, error=error,
                                inference_bytes=shown, trust=self.llm.trust)

    def _failed(self, obj: ObjectRecord, site: str, version: str, error: str) -> ExtractionResult:
        return self._result(obj, site, version, [], Usage(), error)

    async def extract(self, obj: ObjectRecord, site: str, live: bool = False) -> ExtractionResult:
        uri = None
        if obj.modality == "image":
            uri = self._image(obj)
            if uri is None:
                return self._failed(obj, site, self.version, "image missing")
            caption = obj.text if obj.text != obj.title else None
            messages = prompts.extract_image_messages(obj.title, caption, uri)
            role, task = "extract_image", "extract_image"
        elif obj.modality == "table":
            messages, role, task = prompts.extract_table_messages(obj.title, obj.text), "extract_text", "extract_table"
        else:
            messages, role, task = prompts.extract_text_messages(obj.title, obj.text), "extract_text", "extract_text"
        try:
            reply = await self.llm.chat(messages, role=role, task=task, max_tokens=1024, **_live(live))
        except Exception as e:
            log.warning("extraction of %s failed: %s", obj.oid, e)
            return self._failed(obj, site, self.version, f"{type(e).__name__}: {e}")
        out = self._result(obj, site, self.version, parse_triples(parse_json(reply.text)), reply.usage,
                           shown=content_bytes(shown_text(obj, "kg"), uri))
        out.call_wall = reply.wall  # type: ignore[attr-defined]
        return out

    async def extract_query(self, obj: ObjectRecord, site: str, query: str, image_title: bool = False,
                            live: bool = False) -> ExtractionResult:
        titled = image_title and obj.modality == "image"
        version = prompts.QUERY_EXTRACT_T_VERSION if titled else prompts.QUERY_EXTRACT_VERSION
        uri = self._image(obj) if obj.modality == "image" else None
        if obj.modality == "image" and uri is None:
            return self._failed(obj, site, version, "image missing")
        text = shown_text(obj, "query_t" if titled else "query")
        role = "extract_image" if obj.modality == "image" else "extract_text"
        try:
            reply = await self.llm.chat(prompts.extract_query_messages(query, text, uri), role=role,
                                        task="extract_query", max_tokens=1024, **_live(live))
        except Exception as e:
            return self._failed(obj, site, version, f"{type(e).__name__}: {e}")
        entities, triples = prompts.parse_query_records(reply.text)
        result = self._result(obj, site, version, triples, reply.usage, shown=content_bytes(text, uri))
        result.entities = [[name, description] for name, description in entities]
        result.call_wall = reply.wall  # type: ignore[attr-defined]
        return result

    async def extract_query_block(self, objs: Sequence[ObjectRecord], site: str, query: str,
                                  image_title: bool = True, live: bool = False) -> list[ExtractionResult]:
        version = prompts.QUERY_EXTRACT_BLOCK_VERSION
        if not objs:
            return []
        modality = objs[0].modality
        if any(o.modality != modality for o in objs) or len(objs) > prompts.PER_CALL.get(modality, 1):
            raise ValueError(f"a block call takes at most {prompts.PER_CALL.get(modality, 1)} objects of one modality")
        uris = [self._image(o) if modality == "image" else None for o in objs]
        failed = {o.oid: self._failed(o, site, version, "image missing")
                  for o, u in zip(objs, uris, strict=True) if modality == "image" and u is None}
        shown = [(o, u) for o, u in zip(objs, uris, strict=True) if o.oid not in failed]
        if not shown:
            return [failed[o.oid] for o in objs]
        texts = [shown_text(o, "query_block" if image_title else "query") for o, _ in shown]
        images = [u for _, u in shown if u] if modality == "image" else None
        role = "extract_image" if modality == "image" else "extract_text"
        try:
            reply = await self.llm.chat(prompts.extract_query_block_messages(query, texts, images), role=role,
                                        task="extract_query", max_tokens=min(BLOCK_MAX_TOKENS, 1024 * len(shown)),
                                        **_live(live))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            return [failed.get(o.oid) or self._failed(o, site, version, error) for o in objs]
        entities, triples = prompts.parse_query_records(reply.text, limit=40 * len(shown))

        # a record goes to the objects that name it (head or tail); if none do, to every object
        def owners(*names: str) -> list[int]:
            hit = [i for i, t in enumerate(texts) if any(n and prompts.mentions(t, n) for n in names)]
            return hit or list(range(len(shown)))

        mine: list[list[tuple[str, str, str]]] = [[] for _ in shown]
        records: list[list[list[str]]] = [[] for _ in shown]
        for h, r, t in triples:
            for i in owners(h, t):
                mine[i].append((h, r, t))
        for name, description in entities:
            for i in owners(name):
                records[i].append([name, description])
        out = dict(failed)
        for i, ((o, u), text) in enumerate(zip(shown, texts, strict=True)):
            out[o.oid] = self._result(o, site, version, mine[i], reply.usage if i == 0 else Usage(),
                                      shown=content_bytes(text, u))
            out[o.oid].entities = records[i]
            out[o.oid].call_wall = reply.wall if i == 0 else None  # type: ignore[attr-defined]
        return [out[o.oid] for o in objs]
