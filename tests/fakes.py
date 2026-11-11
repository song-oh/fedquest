# scripted databases (FakeSite) and the running example the offline tests share
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fedquest.config import SITE_ROLES, load_models
from fedquest.federation.base import WIRE_VERSION, Federation, share_slack, vocab_filters
from fedquest.llm import prompts
from fedquest.llm.fake import fake_read, message_text
from fedquest.paths import CONFIGS
from fedquest.site.extractor import content_bytes
from fedquest.site.service import (
    TitleIndex,
    _bind,
    _calls,
    _picked,
    _shown,
    candidate,
    check_probe,
    mention_text,
    names_any,
    titled,
)
from fedquest.types import (
    Candidate,
    CollectionInfo,
    ExtractionResult,
    Fact,
    HippoRequest,
    ProbeCounts,
    ProbeRequest,
    RawObject,
    ReadResult,
    SiteInfo,
    Usage,
)


@dataclass
class FakeObject:
    oid: str
    collection: str
    modality: str
    title: str
    text: str
    triples: list[tuple[str, str, str]]
    prompt_tokens: int = 400
    completion_tokens: int = 100
    doc_id: str = ""


def version_of(o: FakeObject) -> str:
    from fedquest.utils import stable_hash

    return stable_hash(o.oid, o.title, o.text, n=12)


Ranker = Callable[[str, str], list[str]]


def rules(table: list[tuple[str, str, list[str]]], default: dict[str, list[str]] | None = None) -> Ranker:
    def rank(key: str, collection: str) -> list[str]:
        low = key.lower()
        for needle, coll, oids in table:
            if coll == collection and needle.lower() in low:
                return oids
        return (default or {}).get(collection, [])

    return rank


FAKE_MODELS = {r: load_models(CONFIGS / "models.yaml").roles[r] for r in SITE_ROLES}


class FakeSite:
    def __init__(self, name: str, objects: list[FakeObject], ranker: Ranker, policy: str = "snippets",
                 trust: str = "alcf", models: dict[str, str] | None = None):
        self.name = name
        self.policy = policy
        self.objects = {o.oid: o for o in objects}
        self.overlays: dict[str, dict[str, FakeObject]] = {}
        self.ranker = ranker
        self.trust = trust
        self.models = dict(FAKE_MODELS if models is None else models)
        self.extract_calls: list[tuple[str, str]] = []
        self.probes: list[str] = []
        self.probe_requests: list[ProbeRequest] = []
        self.read_calls: list[list[str]] = []
        self.read_requests: list[dict[str, Any]] = []
        self.reader: Callable[[str], Any] | None = None
        self.hippo_ranker: Ranker | None = None
        self.hippo_index: set[str] | None = None
        self.hippo_requests: list[HippoRequest] = []

    async def info(self) -> dict[str, Any]:
        colls = sorted({(o.collection, o.modality) for o in self.objects.values()})
        return SiteInfo(
            name=self.name,
            machine=self.name,
            policy=self.policy,  # type: ignore[arg-type]
            collections=[
                CollectionInfo(c, m, "fake", sum(o.collection == c for o in self.objects.values()))  # type: ignore[arg-type]
                for c, m in colls
            ],
            extractor_version=prompts.EXTRACT_VERSION,
            reader_version=prompts.READ_VERSION,
            models=dict(self.models),
            trust=self.trust,
            wire=WIRE_VERSION,
        ).to_dict()

    async def blooms(self, vocab: bool = False) -> dict[str, dict[str, Any]]:
        from fedquest.bloom import BloomFilter, title_entries
        from fedquest.site.service import _title_key

        cols = sorted({o.collection for o in self.objects.values()})
        if vocab:
            return {} if self.policy == "sealed" else vocab_filters(
                {c: {w for o in self.objects.values() if o.collection == c and o.modality != "image"
                     for w in prompts._norm_text(mention_text(o)).split()}
                 for c in cols if not self._pictures(c)})
        return {c: BloomFilter.of(title_entries(_title_key(o.title) for o in self.objects.values()
                                                if o.collection == c)).to_dict() for c in cols}

    async def probe(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]] | dict[str, Any]]:
        out: list[list[dict[str, Any]] | dict[str, Any]] = []
        every = sorted({o.collection for o in self.objects.values()})
        for r in map(ProbeRequest.from_dict, requests):
            check_probe(r)
            self.probes.append(r.key)
            self.probe_requests.append(r)
            saved = self.objects
            if r.epoch is not None:
                self.objects = {**saved, **self.overlays.get(r.epoch, {})}
            try:
                with prompts.mention_scope(r.mention):
                    reply = self._serve(r, every)
                if r.versions:
                    for c in (reply["c"] if isinstance(reply, dict) else reply):
                        if c["oid"] in self.objects:
                            c["ver"] = version_of(self.objects[c["oid"]])
                out.append(reply)
            finally:
                self.objects = saved
        return out

    def _serve(self, r: ProbeRequest, every: list[str]) -> list[dict[str, Any]] | dict[str, Any]:
        if r.semijoin and self.policy != "sealed":
            cands, counts = self._semijoin(r, every)
            return {"c": [c.to_dict() for c in cands], "n": [n.to_dict() for n in counts]}
        cands = []
        for coll in every:
            if r.collections and coll not in r.collections:
                continue
            ranked = [oid for oid in self.ranker(r.key, coll) if oid in self.objects]
            if r.kind == "title":
                cands += self._titled(r, coll, ranked)
                continue
            cands += [candidate(self.name, self.policy, coll, self.objects[oid], r.offset + i + 1,  # type: ignore[arg-type]
                                mentions=r.mentions)
                      for i, oid in enumerate(ranked[r.offset : r.offset + self._limit(r, coll)])]
        dicts = [c.to_dict() for c in cands]
        return {"c": dicts, "n": []} if r.semijoin else dicts

    def _pictures(self, coll: str) -> bool:
        return all(o.modality == "image" for o in self.objects.values() if o.collection == coll)

    async def hippo(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        out = []
        for r in map(HippoRequest.from_dict, requests):
            self.hippo_requests.append(r)
            cands: list[Candidate] = []
            for coll in sorted({o.collection for o in self.objects.values()}):
                texts = all(o.modality == "text" for o in self.objects.values() if o.collection == coll)
                if not texts or (r.collections and coll not in r.collections) or \
                        (self.hippo_index is not None and coll not in self.hippo_index):
                    continue
                ranked = [oid for oid in (self.hippo_ranker or self.ranker)(r.question, coll) if oid in self.objects]
                cands += [candidate(self.name, self.policy, coll, self.objects[oid], i + 1)  # type: ignore[arg-type]
                          for i, oid in enumerate(ranked[: r.k])]
            out.append([c.to_dict() for c in cands])
        return out

    def _limit(self, r: ProbeRequest, coll: str) -> int:
        if r.k_group is None or not r.group_size or r.encoder not in (None, "fake"):
            return r.k
        m = sum(o.collection == coll for o in self.objects.values())
        return -(-r.k_group * m // r.group_size) + share_slack(r.k_group, m / r.group_size)

    def _semijoin(self, r: ProbeRequest, every: list[str]) -> tuple[list[Candidate], list[ProbeCounts]]:
        cands: list[Candidate] = []
        counts: list[ProbeCounts] = []
        for coll in every:
            if (r.collections and coll not in r.collections) or self._pictures(coll):
                continue
            first = {oid: i for i, oid in enumerate(self.ranker(r.key, coll))}
            joined = sorted((o for o in self.objects.values() if o.collection == coll and o.modality != "image"
                             and names_any(o, r.mentions)), key=lambda o: first.get(o.oid, len(first)))
            page = joined[r.offset : r.offset + self._limit(r, coll)]
            for j, o in enumerate(page, r.offset + 1):
                c = candidate(self.name, self.policy, coll, o, j, mentions=r.mentions)  # type: ignore[arg-type]
                c.jrank = j
                cands.append(c)
            counts.append(ProbeCounts(coll, len(joined), len(page)))
        return cands, counts

    def _titled(self, r: ProbeRequest, coll: str, ranked: list[str]) -> list[Candidate]:
        objs = [o for o in self.objects.values() if o.collection == coll]
        index = TitleIndex([o.title for o in objs])
        pages = any(o.modality != "text" for o in objs)
        found = {t: {objs[i].oid for i in hits} for t, hits in titled(index, r.titles, contain=pages).items()}
        first = {oid: i for i, oid in enumerate(ranked)}
        among = sorted((o.oid for o in objs if any(o.oid in hits for hits in found.values())),
                       key=lambda oid: first.get(oid, len(first)))
        kept = {oid for hits in found.values() for oid in [o for o in among if o in hits][: r.k]}
        rank = {oid: i for i, oid in enumerate((o for o in among if o in kept), 1)}
        out: list[Candidate] = []
        for title, hits in found.items():
            out += [candidate(self.name, self.policy, coll, self.objects[oid], rank[oid], title, r.mentions)  # type: ignore[arg-type]
                    for oid in sorted(hits & kept, key=rank.__getitem__)]
        return out

    def _result(self, o: FakeObject, triples: list[tuple[str, str, str]], version: str, raw: bool) -> dict[str, Any]:
        facts = [Fact(h, r, t, self.name, o.oid, version, o.modality) for h, r, t in triples]
        usage = Usage(o.prompt_tokens, o.completion_tokens, 0, 1.0, 1)
        raw_text = o.text if raw and self.policy == "snippets" and o.modality != "image" else None
        return ExtractionResult(
            self.name, o.oid, o.modality, version, facts, usage, raw_text=raw_text, doc_id=o.doc_id,  # type: ignore[arg-type]
            inference_bytes=content_bytes(f"{o.title}\n{o.text}"), trust=self.trust,
        ).to_dict()

    async def extract(self, oids: list[str], include_raw: bool = False, mode: str = "kg",
                      query: str | None = None, blocks: list[list[str]] | None = None,
                      live: bool = False) -> list[dict[str, Any]]:
        out = []
        for oid in list(dict.fromkeys([*oids, *(o for block in blocks or [] for o in block)])):
            self.extract_calls.append((oid, mode))
            o = self.objects[oid]
            version = prompts.EXTRACT_VERSION if mode == "kg" else prompts.QUERY_EXTRACT_VERSION
            out.append(self._result(o, o.triples, version, include_raw))
        return out

    def _seen_as(self, o: FakeObject) -> prompts.ReadObject:
        if o.modality == "image":
            return prompts.ReadObject("image", f"{o.title}: " + "; ".join(f"({h} | {r} | {t})" for h, r, t in o.triples))
        return prompts.ReadObject("table row" if o.modality == "table" else "text", _shown(o))  # type: ignore[arg-type]

    async def read(self, question: str, unknowns: list[dict[str, Any]], blocks: list[list[str]],
                   prefilter: list[str] | None = None, exempt: list[str] | None = None,
                   labels: dict[str, str] | None = None, include_raw: bool = False,
                   extract: bool = False, live: bool = False, mention: int = 1, versions: bool = False,
                   harvest: int = 0, epoch: str | None = None) -> dict[str, Any]:
        self.read_requests.append({"unknowns": unknowns, "blocks": blocks, "prefilter": prefilter, "exempt": exempt,
                                   "labels": labels, "include_raw": include_raw, "extract": extract,
                                   **({"mention": mention} if mention != 1 else {}),
                                   **({"versions": True} if versions else {}), **({"harvest": harvest} if harvest
                                                                                  else {}),
                                   **({"epoch": epoch} if epoch is not None else {})})
        saved = self.objects
        if epoch is not None:
            self.objects = {**saved, **self.overlays.get(epoch, {})}
        try:
            with prompts.mention_scope(mention):
                out = await self._read(question, unknowns, blocks, prefilter, exempt, labels, include_raw, extract)
            if versions or harvest:
                read = [self.objects[oid] for oid in out.get("read", []) if oid in self.objects]
                if versions:
                    out["versions"] = {o.oid: version_of(o) for o in read}
                if harvest:
                    known = {v.lower() for u in unknowns for v in u.get("values") or [] if v}
                    out["harvest"] = [{"entity": h, "relation": r, "value": t, "oid": o.oid}
                                      for o in read if o.modality != "image" for h, r, t in o.triples
                                      if h.lower() in known][: harvest * max(1, len(read))]
            return out
        finally:
            self.objects = saved

    async def versions(self, oids: list[str], epoch: str | None = None) -> dict[str, str]:
        objs = {**self.objects, **(self.overlays.get(epoch, {}) if epoch is not None else {})}
        return {oid: version_of(objs[oid]) for oid in oids if oid in objs}

    async def witnesses(self, items: list[dict[str, Any]], epoch: str | None = None) -> list[bool]:
        objs = {**self.objects, **(self.overlays.get(epoch, {}) if epoch is not None else {})}
        out = []
        for it in items:
            o = objs.get(str(it.get("oid", "")))
            out.append(o is not None and o.modality != "image" and prompts.span_holds(
                o.text, str(it.get("w") or ""), it.get("known") or None, o.title))
        return out

    async def update(self, epoch: str, edits: list[dict[str, Any]]) -> dict[str, str]:
        space = self.overlays.setdefault(epoch, {})
        out = {}
        for e in edits:
            o = self.objects.get(e["oid"])
            if o is not None:
                space[o.oid] = dataclasses.replace(o, text=e["text"])
                out[o.oid] = version_of(space[o.oid])
        return out

    async def _read(self, question: str, unknowns: list[dict[str, Any]], blocks: list[list[str]],
                    prefilter: list[str] | None, exempt: list[str] | None, labels: dict[str, str] | None,
                    include_raw: bool, extract: bool) -> dict[str, Any]:
        if not unknowns:
            return ReadResult(self.name, [], {}, Usage()).to_dict()
        labels = labels or {}
        picked, skipped = _picked(blocks, self.objects.get, prefilter or [], set(exempt or []))
        if extract:
            objs = [o for block in picked for o in block]
            self.extract_calls += [(o.oid, "query") for o in objs]
            done = [ExtractionResult.from_dict(self._result(o, o.triples, prompts.QUERY_EXTRACT_VERSION, False))
                    for o in objs]
            out = ReadResult(self.name, [o.oid for o in objs], {}, Usage(), skipped=skipped, extractions=done,
                             modalities={o.oid: o.modality for o in objs}, doc_ids={o.oid: o.doc_id for o in objs},
                             titles={o.oid: labels.get(o.oid, o.title) for o in objs}, trust=self.trust)
            for x in done:
                out.usage.add(x.usage)
                out.usage_by_modality.setdefault(x.modality, Usage()).add(x.usage)
            return out.to_dict()
        calls = [group for block in picked for group in _calls([(o, self._seen_as(o)) for o in block])]
        objs = [o for group in calls for o, _ in group]
        result = ReadResult(self.name, [o.oid for o in objs], {}, Usage(), skipped=skipped,
                            modalities={o.oid: o.modality for o in objs}, doc_ids={o.oid: o.doc_id for o in objs},
                            titles={o.oid: labels.get(o.oid, o.title) for o in objs},
                            inference_bytes=sum(content_bytes(ro.text, ro.image) for g in calls for _, ro in g),
                            trust=self.trust)
        unk = [prompts.ReadUnknown.from_dict(u) for u in unknowns]
        for group in calls:
            self.read_calls.append([o.oid for o, _ in group])
            text = message_text(prompts.read_messages(question, [ro for _, ro in group], unk))
            reply = self.reader(text) if self.reader else fake_read(text)
            written = reply if isinstance(reply, str) else json.dumps(reply)
            usage = Usage(len(text) // 4, max(1, len(written) // 4), 0, 0.1, 1)
            result.usage.add(usage)
            result.usage_by_modality.setdefault(group[0][0].modality, Usage()).add(usage)
            _bind(result, [o for o, _ in group], prompts.read_reply(written), unk, labels)
        if include_raw and self.policy == "snippets":
            cited = {h["oid"] for hits in result.bindings.values() for h in hits}
            result.raw = {o.oid: o.text for o in objs if o.oid in cited and o.modality != "image"}
        return result.to_dict()

    async def fetch_raw(self, oids: list[str], max_px: int = 384, quality: int = 75,
                        epoch: str | None = None) -> list[dict[str, Any]]:
        objs = {**self.objects, **(self.overlays.get(epoch, {}) if epoch is not None else {})}
        out = []
        for oid in oids:
            o = objs[oid]
            if self.policy in ("facts", "sealed"):
                out.append(RawObject(self.name, oid, o.modality, o.title, denied=True, doc_id=o.doc_id).to_dict())  # type: ignore[arg-type]
            elif o.modality == "image":
                out.append(
                    RawObject(self.name, oid, "image", o.title, image_b64="AAAA", nbytes=4, doc_id=o.doc_id).to_dict()
                )
            else:
                out.append(
                    RawObject(
                        self.name, oid, o.modality, o.title, text=o.text, nbytes=len(o.text.encode()), doc_id=o.doc_id  # type: ignore[arg-type]
                    ).to_dict()
                )
        return out

    async def stats(self, busy_only: bool = False) -> dict[str, Any]:
        return {"objects": len(self.objects), "extract_calls": len(self.extract_calls)}


def _wire_check(action: str, kwargs: dict[str, Any]) -> None:
    import inspect

    from fedquest.remote.host import ACTIONS
    from fedquest.site.service import SiteService

    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    sent = {k: v for k, v in kwargs.items() if k not in ("profile", "rid")}
    inspect.signature(getattr(SiteService, action)).bind(None, **sent)


class FakeEndpoint:
    def __init__(self, site: FakeSite):
        self.name = site.name
        self.site = site

    async def call(self, action: str, **kwargs: Any) -> Any:
        _wire_check(action, kwargs)
        kwargs.pop("rid", None)
        return await getattr(self.site, action)(**kwargs)

    async def close(self) -> None:
        return None


def fake_federation(*sites: FakeSite) -> Federation:
    return Federation([FakeEndpoint(s) for s in sites])


def running_example(*, image_first: bool = True) -> tuple[Federation, dict[str, FakeSite]]:
    row = FakeObject(
        "mmqa/row/t1#6", "row", "table", "Ben Piazza — Filmography",
        "Ben Piazza | Filmography | Year: 1976; Title: The Bad News Bears; Role: Bob Whitewood",
        [("Ben Piazza", "acted in", "The Bad News Bears"), ("The Bad News Bears", "released in", "1976")],
        prompt_tokens=300, completion_tokens=60, doc_id="t1",
    )
    row2 = FakeObject(
        "mmqa/row/t1#2", "row", "table", "Ben Piazza — Filmography",
        "Ben Piazza | Filmography | Year: 1962; Title: No Exit; Role: Camarero",
        [("Ben Piazza", "acted in", "No Exit"), ("No Exit", "released in", "1962")],
        prompt_tokens=300, completion_tokens=60, doc_id="t1",
    )
    poster = FakeObject(
        "mmqa/image/p1", "image", "image", "The Bad News Bears", "The Bad News Bears",
        [("The Bad News Bears", "poster depicts", "baseball"), ("The Bad News Bears", "poster shows", "children")],
        prompt_tokens=1500, completion_tokens=120, doc_id="p1",
    )
    decoy = FakeObject(
        "mmqa/image/p9", "image", "image", "Ben Piazza Theatre", "Ben Piazza Theatre",
        [("Ben Piazza Theatre", "depicts", "a building")],
        prompt_tokens=1500, completion_tokens=120, doc_id="p9",
    )
    bio = FakeObject(
        "mmqa/text/b1#0", "text", "text", "Ben Piazza",
        "Ben Piazza (1933-1991) was an American actor and writer.",
        [("Ben Piazza", "was", "American actor")],
        prompt_tokens=200, completion_tokens=40, doc_id="b1",
    )
    first = [decoy.oid, poster.oid] if image_first else [poster.oid, decoy.oid]
    aurora = FakeSite(
        "aurora",
        [poster, decoy],
        rules([("bad news bears", "image", [poster.oid, decoy.oid]), ("sport", "image", first)],
              {"image": [decoy.oid, poster.oid]}),
    )
    midway3 = FakeSite(
        "midway3",
        [row, row2],
        rules([("ben piazza", "row", [row.oid, row2.oid])], {"row": [row2.oid, row.oid]}),
    )
    edge = FakeSite(
        "edge", [bio], rules([("ben piazza", "text", [bio.oid])], {"text": [bio.oid]}), policy="facts"
    )
    sites = {"aurora": aurora, "midway3": midway3, "edge": edge}
    return fake_federation(aurora, midway3, edge), sites


RUNNING_QUESTION = "What sport is on the poster of the 1976 film Ben Piazza acted in?"

RUNNING_PATTERN = (
    '("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]sport1)[REC]'
    '("answer_entity"[SEP]sport1)[REC]'
    '("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in 1976 film)[REC]'
    '("relation"[SEP]film1[SEP]sport1[SEP]poster depicts sport)[END]'
)


def running_responder(task: str, messages: list[dict[str, Any]]) -> str:
    text = " ".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for m in messages
        for part in (m["content"] if isinstance(m["content"], list) else [m["content"]])
    )
    if task == "pattern":
        return RUNNING_PATTERN
    if task == "verify":
        return '{"answerable": %s, "answer": ""}' % ("true" if "baseball" in text else "false")
    if task in ("read", "read_index", "read_facts"):
        from fedquest.llm.fake import fake_read

        return json.dumps(fake_read(text))
    if task == "closed_book":
        return "<ANS>basketball</ANS>"
    if task == "answer":
        return "<ANS>baseball</ANS>" if "baseball" in text else "<ANS>unknown</ANS>"
    if task == "judge":
        return "4" if "baseball" in text.split("Reference Answer")[0] or "baseball" in text else "0"
    if task == "ircot":
        return "So the answer is: baseball." if "baseball" in text else "The film is The Bad News Bears."
    if task == "planner":
        if "OBSERVATION: [1]" not in text:
            return '{"thought": "find the film", "action": "search", "input": "Ben Piazza 1976 film"}'
        if "OBSERVATION: (" not in text:
            return '{"thought": "read the row", "action": "extract", "input": "1"}'
        return '{"thought": "done", "action": "finish", "input": "baseball"}'
    return '{"triples": []}'
