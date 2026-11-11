# per-question ledger: model usage, rounds, bytes moved and what each site was told
from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fedquest.timeline import row_fields
from fedquest.types import ExtractionResult, Fact, ReadResult, Record, Usage


def _nbytes(obj: Any) -> int:
    return len(json.dumps(obj, default=str, ensure_ascii=False).encode())


def _norm(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def common_prefix(a: str, b: str, chunk: int = 4096) -> int:
    n, i = min(len(a), len(b)), 0
    while i + chunk <= n and a[i : i + chunk] == b[i : i + chunk]:
        i += chunk
    while i < n and a[i] == b[i]:
        i += 1
    return i


def fact_bytes(facts: list[Fact]) -> int:
    return sum(len(f.head) + len(f.relation) + len(f.tail) + 6 for f in facts)


EVIDENCE_PURPOSES = ("bind", "bind_index", "reader", "planner", "extract_entry")


@dataclass
class Ledger(Record):
    extraction: dict[str, Usage] = field(default_factory=dict)
    coordinator: dict[str, Usage] = field(default_factory=dict)
    extractions: int = 0
    extraction_errors: int = 0
    replays: int = 0
    reused: int = 0
    reuse_counts: dict[str, int] = field(default_factory=dict)
    repair_counts: dict[str, int] = field(default_factory=dict)
    ground_counts: dict[str, int] = field(default_factory=dict)
    depth_counts: dict[str, int] = field(default_factory=dict)
    compile_counts: dict[str, int] = field(default_factory=dict)
    invariant_counts: dict[str, int] = field(default_factory=dict)
    fired_counts: dict[str, int] = field(default_factory=dict)
    reads: int = 0
    probes: int = 0
    rounds: int = 0
    bytes_control: int = 0
    bytes_facts: int = 0
    bytes_raw: int = 0
    bytes_summary: int = 0
    wall_seconds: float = 0.0
    critical_seconds: float = 0.0
    reembed_seconds: float = 0.0
    plan_seconds: float = 0.0
    inference: dict[str, int] = field(default_factory=dict)
    inference_cross: int = 0
    keys_seen: dict[str, set[str]] = field(default_factory=dict)
    values_seen: dict[str, set[str]] = field(default_factory=dict)
    question_seen: set[str] = field(default_factory=set)
    messages: int = 0
    bytes_reply: int = 0
    fanout_max_sum: float = 0.0
    fanout_med_sum: float = 0.0
    deadline_misses: int = 0
    wan_seconds: float = 0.0
    coord_replays: int = 0
    live: dict[str, float] = field(default_factory=dict)
    llm: dict[str, float] = field(default_factory=dict)
    prefix_ideal: int = 0
    prompts_seen: dict[str, list[str]] = field(default_factory=dict)
    to_coord: dict[str, int] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    told_cross: set[tuple[str, str]] = field(default_factory=set)
    timeline: dict[str, float] = field(default_factory=dict)
    skipped_sites: int = 0
    joinable: int = 0
    site_ms: float = 0.0
    reply_entries: int = 0
    budget: int | None = None
    reserve: int = 0
    budget_refusals: int = 0
    gold: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_log: dict[str, dict[str, Any]] = field(default_factory=dict)
    returned_by_site: dict[str, int] = field(default_factory=dict)
    returned_total: int = 0
    probe_rounds: int = 0
    context_logged: bool = False

    def note_returned(self, request: Any, cands: Iterable[Any], round_no: int) -> None:
        for c in cands:
            self.returned_total += 1
            self.returned_by_site[c.site] = self.returned_by_site.get(c.site, 0) + 1
            if not self.gold or c.doc_id not in self.gold:
                continue
            e = self.evidence_log.setdefault(c.doc_id, dict(self.gold[c.doc_id]))
            e["returned"] = e.get("returned", 0) + 1
            if "first_round" not in e:
                e.update({"first_round": round_no, "kind": getattr(request, "kind", ""),
                          "semijoin": bool(getattr(request, "semijoin", False)), "site": c.site,
                          "collection": c.collection, "modality": c.modality, "rank": c.rank,
                          "z": c.z, "complete": bool(c.complete)})
            e["best_rank"] = min(e.get("best_rank", c.rank), c.rank)

    def note_read(self, doc_ids: Iterable[str], where: str) -> None:
        for d in doc_ids:
            if d in self.gold:
                e = self.evidence_log.setdefault(d, dict(self.gold[d]))
                e[f"read_{where}"] = e.get(f"read_{where}", 0) + 1

    def note_context(self, doc_ids: Iterable[str]) -> None:
        self.context_logged = True
        for d in doc_ids:
            if d in self.gold:
                self.evidence_log.setdefault(d, dict(self.gold[d]))["in_context"] = True

    def charge_extraction(self, r: ExtractionResult, domain: str | None = None) -> None:
        self.extraction.setdefault(r.modality, Usage()).add(r.usage)
        self.extractions += 1
        self.replays += int(r.replay)
        released = fact_bytes(r.facts) + (len(r.raw_text.encode()) if r.raw_text else 0)
        self.bytes_facts += fact_bytes(r.facts)
        if r.raw_text:
            self.bytes_raw += len(r.raw_text.encode())
        self.charge_to_coord(r.site, released)
        self._inference(r.site, r.inference_bytes, r.trust, domain)

    def charge_read(self, r: ReadResult, domain: str | None = None) -> None:
        for mod, u in (r.usage_by_modality or {"text": r.usage}).items():
            self.extraction.setdefault(f"read_{mod}", Usage()).add(u)
        self.extractions += len(r.read)
        self.reads += r.usage.calls
        self.replays += int(r.replay)
        hits = sum(len(b.get("entity", "")) + 8 for hits in r.bindings.values() for b in hits)
        raw = sum(len(t.encode()) for t in r.raw.values())
        self.bytes_facts += hits
        self.bytes_raw += raw
        self.charge_to_coord(r.site, hits + raw)
        self._inference(r.site, r.inference_bytes, r.trust, domain)

    def _inference(self, site: str, nbytes: int, trust: str, domain: str | None) -> None:
        if not nbytes:
            return
        self.inference[site] = self.inference.get(site, 0) + nbytes
        if trust and domain and trust != domain:
            self.inference_cross += nbytes

    def disclose(self, site: str, keys: Iterable[str] = (), values: Iterable[str] = (), question: bool = False,
                 ) -> None:
        self.keys_seen.setdefault(site, set()).update(_norm(k) for k in keys if k)
        told = {_norm(v) for v in values if v}
        self.values_seen.setdefault(site, set()).update(told)
        # a value told to a site crosses when it first came from another site
        self.told_cross.update((site, v) for v in told if self.origins.get(v, site) != site)
        if question:
            self.question_seen.add(site)

    def note_origin(self, value: str, site: str) -> None:
        if value and site:
            self.origins.setdefault(_norm(value), site)

    def charge_to_coord(self, site: str, nbytes: int) -> None:
        if nbytes:
            self.to_coord[site] = self.to_coord.get(site, 0) + nbytes

    def charge_model_path(self, wall: float = 0.0, retries: int = 0, backoff: float = 0.0, http429: int = 0,
                          http5xx: int = 0) -> None:
        for key, value in (("wall", wall), ("retries", retries), ("backoff", backoff), ("http429", http429),
                           ("http5xx", http5xx)):
            if value:
                self.llm[key] = self.llm.get(key, 0) + value

    def charge_live(self, purpose: str, seconds: float) -> None:
        self.live[purpose] = self.live.get(purpose, 0.0) + seconds

    def note_prompt(self, role: str, text: str, prompt_tokens: int) -> None:
        earlier = self.prompts_seen.setdefault(role, [])
        # prefix_ideal: tokens an ideal prefix cache would reuse, against the role's last 8 prompts
        if text and earlier and prompt_tokens:
            shared = max(common_prefix(text, e) for e in earlier)
            self.prefix_ideal += round(prompt_tokens * shared / len(text))
        if text:
            earlier.append(text)
            del earlier[:-8]

    def charge_llm(self, purpose: str, usage: Usage) -> None:
        self.coordinator.setdefault(purpose, Usage()).add(usage)

    def charge_round(self, seconds_by_site: dict[str, float], *, control: Any = None, reply: Any = None,
                     retries: dict[str, int] | None = None) -> None:
        self.rounds += 1
        self.messages += len(seconds_by_site) + sum((retries or {}).values())
        if seconds_by_site:
            self.critical_seconds += max(seconds_by_site.values())
            self.fanout_max_sum += max(seconds_by_site.values())
            self.fanout_med_sum += statistics.median(seconds_by_site.values())
        if control is not None:
            self.bytes_control += _nbytes(control)
            if retries and isinstance(control, dict):
                self.bytes_control += sum(n * _nbytes(control[s]) for s, n in retries.items() if s in control)
        if reply is not None:
            self.bytes_reply += _nbytes(reply)

    def charge_counted(self, joinable: int = 0, site_ms: float = 0.0, skipped: int = 0) -> None:
        self.joinable += joinable
        self.site_ms += site_ms
        self.skipped_sites += skipped

    def charge_raw(self, nbytes: int) -> None:
        self.bytes_raw += nbytes

    def charge_summaries(self, summaries: list[str]) -> None:
        self.bytes_summary += sum(len(s.encode()) for s in summaries)

    def charge_reuse(self, facts: list[Fact], n_objects: int = 1) -> None:
        self.reused += n_objects
        self.bytes_facts += fact_bytes(facts)

    def count_reuse(self, what: str, n: int = 1) -> None:
        if n:
            self.reuse_counts[what] = self.reuse_counts.get(what, 0) + n

    def count_repair(self, what: str, n: int = 1) -> None:
        if n:
            self.repair_counts[what] = self.repair_counts.get(what, 0) + n

    def count_compile(self, what: str, n: int = 1) -> None:
        if n:
            self.compile_counts[what] = self.compile_counts.get(what, 0) + n

    def count_invariant(self, what: str, n: int = 1) -> None:
        if n:
            self.invariant_counts[what] = self.invariant_counts.get(what, 0) + n

    def count_fired(self, what: str, n: int = 1) -> None:
        if n:
            self.fired_counts[what] = self.fired_counts.get(what, 0) + n

    def count_ground(self, what: str, n: int = 1) -> None:
        if n:
            self.ground_counts[what] = self.ground_counts.get(what, 0) + n

    def count_depth(self, what: str, n: int = 1) -> None:
        if not n:
            return
        if what == "depth_max":
            self.depth_counts[what] = max(self.depth_counts.get(what, 0), n)
        else:
            self.depth_counts[what] = self.depth_counts.get(what, 0) + n

    @property
    def extraction_usage(self) -> Usage:
        total = Usage()
        for u in self.extraction.values():
            total.add(u)
        return total

    @property
    def coordinator_usage(self) -> Usage:
        total = Usage()
        for u in self.coordinator.values():
            total.add(u)
        return total

    @property
    def extraction_tokens(self) -> int:
        return self.extraction_usage.cost_units

    @property
    def evidence_usage(self) -> Usage:
        total = self.extraction_usage
        for purpose in EVIDENCE_PURPOSES:
            if purpose in self.coordinator:
                total.add(self.coordinator[purpose])
        return total

    @property
    def evidence_tokens(self) -> int:
        return self.evidence_usage.cost_units

    @property
    def total_tokens(self) -> int:
        return self.extraction_tokens + self.coordinator_usage.cost_units

    def summary(self) -> dict[str, Any]:
        ext, coord, evidence = self.extraction_usage, self.coordinator_usage, self.evidence_usage
        out: dict[str, Any] = {
            "evidence_tokens": evidence.cost_units,
            "evidence_seconds": round(evidence.seconds, 3),
            "evidence_calls": evidence.calls,
            "extraction_tokens": ext.cost_units,
            "extraction_errors": self.extraction_errors,
            "extraction_replays": self.replays,
            "extraction_seconds": round(ext.seconds, 3),
            "coordinator_tokens": coord.cost_units,
            "coordinator_calls": coord.calls,
            "coordinator_seconds": round(coord.seconds, 3),
            "plan_seconds": round(self.plan_seconds, 4),
            "total_tokens": ext.cost_units + coord.cost_units,
            "evidence_in": evidence.prompt_tokens,
            "evidence_out": evidence.completion_tokens,
            "total_in": ext.prompt_tokens + coord.prompt_tokens,
            "total_out": ext.completion_tokens + coord.completion_tokens,
            "reused_objects": self.reused,
            "read_calls": self.reads,
            "objects_read": self.extractions,
            "probes": self.probes,
            "rounds": self.rounds,
            "bytes_control": self.bytes_control,
            "bytes_facts": self.bytes_facts,
            "bytes_raw": self.bytes_raw,
            "bytes_summary": self.bytes_summary,
            "wall_seconds": round(self.wall_seconds, 3),
            "critical_seconds": round(self.critical_seconds, 3),
            "reembed_seconds": round(self.reembed_seconds, 3),
            "bytes_inference_cross": self.inference_cross,
            "bytes_inference_coordinator": coord.bytes_in,
            "question_sites": len(self.question_seen),
        }
        out.update({k: n for k, n in sorted(self.reuse_counts.items()) if n})
        out.update({k: n for k, n in sorted(self.repair_counts.items()) if n})
        out.update({k: n for k, n in sorted(self.ground_counts.items()) if n})
        out.update({k: n for k, n in sorted(self.depth_counts.items()) if n})
        for counts in (self.compile_counts, self.invariant_counts, self.fired_counts):
            out.update({k: n for k, n in sorted(counts.items()) if n})
        out.update({k: n for k, n in (("skipped_sites", self.skipped_sites), ("joinable", self.joinable),
                                      ("site_ms", round(self.site_ms, 3))) if n})
        if self.probes:
            out["reply_entries"] = self.reply_entries
        if self.budget is not None:
            out["budget"] = self.budget
            out["budget_left"] = self.budget - (ext.cost_units + coord.cost_units)
            out["budget_refusals"] = self.budget_refusals
            out["tokens_unweighted"] = ext.prompt_tokens + coord.prompt_tokens + ext.completion_tokens \
                + coord.completion_tokens
        if self.gold:
            out["evidence_log"] = {d: {**dict(self.gold.get(d, {})), **dict(self.evidence_log.get(d, {}))}
                                   for d in sorted(set(self.gold) | set(self.evidence_log))}
            out["returned_total"] = self.returned_total
            out["context_logged"] = self.context_logged
            out["returned_by_site"] = dict(sorted(self.returned_by_site.items()))
        for site in sorted(set(self.inference) | set(self.keys_seen) | set(self.values_seen) | self.question_seen):
            out[f"bytes_inference_{site}"] = self.inference.get(site, 0)
            out[f"disclosed_keys_{site}"] = len(self.keys_seen.get(site, ()))
            out[f"disclosed_values_{site}"] = len(self.values_seen.get(site, ()))
        for modality, u in sorted(self.extraction.items()):
            out[f"extraction_tokens_{modality}"] = u.cost_units
            out[f"extraction_calls_{modality}"] = u.calls
        for purpose, u in sorted(self.coordinator.items()):
            out[f"coordinator_tokens_{purpose}"] = u.cost_units
        out.update({
            "messages": self.messages,
            "bytes_reply": self.bytes_reply,
            "fanout_max_sum": round(self.fanout_max_sum, 3),
            "fanout_med_sum": round(self.fanout_med_sum, 3),
            "deadline_misses": self.deadline_misses,
            "wan_seconds": round(self.wan_seconds, 3),
            "coord_replays": self.coord_replays,
            "live_seconds": round(sum(self.live.values()), 3),
            "llm_wall_seconds": round(self.llm.get("wall", 0.0), 3),
            "llm_retries": int(self.llm.get("retries", 0)),
            "llm_backoff_seconds": round(self.llm.get("backoff", 0.0), 3),
            "llm_429": int(self.llm.get("http429", 0)),
            "llm_5xx": int(self.llm.get("http5xx", 0)),
            "prefix_cached_ideal": self.prefix_ideal,
            "values_cross": len(self.told_cross),
        })
        for site, nbytes in sorted(self.to_coord.items()):
            out[f"bytes_to_coord_{site}"] = nbytes
        for origin, n in sorted(Counter(self.origins.get(v, "") for _, v in self.told_cross).items()):
            out[f"values_out_{origin}"] = n
        for purpose, u in sorted(self.coordinator.items()):
            out[f"coordinator_seconds_{purpose}"] = round(u.seconds, 3)
            out[f"coordinator_in_{purpose}"] = u.prompt_tokens
            out[f"coordinator_out_{purpose}"] = u.completion_tokens
            out[f"coordinator_calls_{purpose}"] = u.calls
        for modality, u in sorted(self.extraction.items()):
            out[f"extraction_in_{modality}"] = u.prompt_tokens
            out[f"extraction_out_{modality}"] = u.completion_tokens
            out[f"extraction_seconds_{modality}"] = round(u.seconds, 3)
        for purpose, seconds in sorted(self.live.items()):
            out[f"live_seconds_{purpose}"] = round(seconds, 3)
        if self.timeline:
            out.update(row_fields(self.timeline, out["wall_seconds"]))
        return out
