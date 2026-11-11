from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Any

from fedquest.kg.normalize import norm_entity
from fedquest.kg.view import AskRecord, Ref, default_writer, materializable
from fedquest.types import Candidate, ExtractionResult, Fact
from fedquest.utils import stable_hash

_KEYED = (("node", "FQNode"), ("version", "FQVersion"), ("ask", "FQAsk"), ("obj", "FQObj"), ("probe", "FQProbe"),
          ("alias", "FQAlias"), ("istat", "FQIndexStat"), ("tuple", "FQTuple"))
_SPACED = ("FQObject", "FQVersion", "FQNode", "FQAsk", "FQObj", "FQProbe", "FQAlias", "FQIndexStat", "FQTuple",
           "FQSpace")

_SCHEMA = (
    "CREATE CONSTRAINT fq6_object_uid IF NOT EXISTS FOR (o:FQObject) REQUIRE o.uid IS UNIQUE",
    "CREATE CONSTRAINT fq6_fact_fid IF NOT EXISTS FOR ()-[r:FQ_FACT]-() REQUIRE r.fid IS UNIQUE",
    "CREATE INDEX fq6_fact_ns IF NOT EXISTS FOR ()-[r:FQ_FACT]-() ON (r.ns)",
    "CREATE INDEX fq6_object_ns IF NOT EXISTS FOR (o:FQObject) ON (o.ns)",
    "CREATE INDEX fq6_fact_ref IF NOT EXISTS FOR ()-[r:FQ_FACT]-() ON (r.ref)",
    "CREATE INDEX fq6_fact_src IF NOT EXISTS FOR ()-[r:FQ_FACT]-() ON (r.src)",
    "CREATE CONSTRAINT fq6_space_ns IF NOT EXISTS FOR (s:FQSpace) REQUIRE s.ns IS UNIQUE",
    *(f"CREATE CONSTRAINT fq6_{name}_uid IF NOT EXISTS FOR (x:{label}) REQUIRE x.uid IS UNIQUE"
      for name, label in _KEYED),
    *(f"CREATE INDEX fq6_{name}_ns IF NOT EXISTS FOR (x:{label}) ON (x.ns)" for name, label in _KEYED),
    "CREATE INDEX fq6_ask_value IF NOT EXISTS FOR (a:FQAsk) ON (a.ns, a.value)",
    "CREATE INDEX fq6_alias_canon IF NOT EXISTS FOR (l:FQAlias) ON (l.ns, l.canon)",
)
_META = "CREATE CONSTRAINT fq6_meta_project IF NOT EXISTS FOR (m:FQMeta) REQUIRE m.project IS UNIQUE"

_IN = "x.ns = $ns OR x.ns STARTS WITH $sub"
_CLEAR = (f"MATCH ()-[x:FQ_FACT]->() USING INDEX SEEK x:FQ_FACT(ns) WHERE {_IN} WITH x LIMIT 20000 DELETE x "
          "RETURN count(*) AS n",
          *(f"MATCH (x:{label}) USING INDEX SEEK x:{label}(ns) WHERE {_IN} WITH x LIMIT 20000 DETACH DELETE x "
            "RETURN count(*) AS n" for label in _SPACED))
_ORPHANS = ("MATCH (e:FQEntity) WHERE NOT EXISTS { (e)-[:FQ_FACT]-() } WITH e LIMIT 20000 DELETE e "
            "RETURN count(*) AS n")

_RESERVE = """
MERGE (s:FQSpace {ns: $ns}) ON CREATE SET s.n = 0
SET s.n = s.n + $k
RETURN s.n - $k AS base
"""
_ADD_NODES = """
UNWIND $rows AS row
MERGE (e:FQNode {uid: row.uid}) ON CREATE SET e.ns = $ns, e.key = row.key, e.name = row.name
"""
_ADD_OBJECTS = """
UNWIND $rows AS row
MERGE (o:FQObject {uid: row.uid})
ON CREATE SET o.ns = $ns, o.site = row.site, o.oid = row.oid, o.seq = row.seq
SET o.version = row.version, o.modality = row.modality
"""
_ADD_VERSIONS = """
UNWIND $rows AS row
MERGE (x:FQVersion {uid: row.uid}) ON CREATE SET x.ns = $ns, x.site = row.site, x.oid = row.oid
WITH x, row WHERE NOT row.overs IN coalesce(x.dead, [])
SET x.overs = row.overs
"""
_ADD_FACTS = """
UNWIND $rows AS row
MATCH (h:FQNode {uid: row.hu}) USING INDEX SEEK h:FQNode(uid)
MATCH (t:FQNode {uid: row.tu}) USING INDEX SEEK t:FQNode(uid)
OPTIONAL MATCH (x:FQVersion {uid: row.src}) USING INDEX SEEK x:FQVersion(uid)
MERGE (h)-[r:FQ_FACT {fid: row.fid}]->(t)
ON CREATE SET r.ns = $ns, r.ref = row.ref, r.src = row.src, r.seq = row.seq, r.head = row.head,
              r.relation = row.relation, r.tail = row.tail, r.site = row.site, r.oid = row.oid, r.version = row.version,
              r.modality = row.modality, r.overs = row.overs, r.wspan = row.wspan, r.writer = $writer,
              r.at = timestamp(), r.qid = row.qid,
              r.stale = CASE WHEN row.overs <> '' AND row.overs IN coalesce(x.dead, []) THEN row.seq END
"""
PROJECT = "fedquest"
_FACT_FIELDS = "r.head AS head, r.relation AS relation, r.tail AS tail, r.site AS site, r.oid AS oid, " \
               "r.version AS version, r.modality AS modality, coalesce(r.overs, '') AS overs, " \
               "coalesce(r.wspan, '') AS wspan"

_COVERED = f"""
UNWIND $uids AS uid
MATCH (o:FQObject {{uid: uid}}) USING INDEX SEEK o:FQObject(uid) WHERE o.version = $version
OPTIONAL MATCH ()-[r:FQ_FACT {{ref: uid}}]->() USING INDEX SEEK r:FQ_FACT(ref)
RETURN uid, r.seq AS seq, {_FACT_FIELDS} ORDER BY seq
"""
_LOOKUP = """
MATCH (e:FQNode)-[r:FQ_FACT]-() USING INDEX SEEK e:FQNode(uid)
WHERE e.uid IN $uids AND {live}
WITH r.head AS head, r.relation AS relation, r.tail AS tail, r.site AS site, r.oid AS oid, r.version AS version,
     r.modality AS modality, coalesce(r.overs, '') AS overs, coalesce(r.wspan, '') AS wspan, min($at[e.uid]) AS first,
     min(r.seq) AS seq
RETURN head, relation, tail, site, oid, version, modality, overs, wspan ORDER BY first, seq LIMIT $limit
"""
_FACTS = f"""
MATCH ()-[r:FQ_FACT]->() USING INDEX SEEK r:FQ_FACT(ns) WHERE r.ns = $ns
OPTIONAL MATCH (o:FQObject {{uid: r.ref}})
RETURN {_FACT_FIELDS} ORDER BY o.seq, r.seq
"""
_EXISTS = "RETURN EXISTS { MATCH (s:FQSpace {ns: $ns}) } OR EXISTS { MATCH (o:FQObject {ns: $ns}) } AS found"
_NAMESPACES = ("CALL { MATCH (x:FQSpace) RETURN x.ns AS ns UNION MATCH (x:FQObject) RETURN x.ns AS ns } "
               "RETURN count(ns) AS n")
_COUNTS = """
RETURN COUNT { MATCH (x:FQObject) WHERE {scope} } AS objects, COUNT { MATCH (x:FQAsk) WHERE {scope} } AS asks,
       COUNT { MATCH (x:FQAsk)-[:FQ_WITNESS]->() WHERE {scope} } AS witnesses,
       COUNT { MATCH (x:FQAsk)-[:FQ_SILENT]->() WHERE {scope} } AS negatives,
       COUNT { MATCH (x:FQAlias) WHERE {scope} } AS aliases, COUNT { MATCH (x:FQProbe) WHERE {scope} } AS probes,
       COUNT { MATCH (x:FQTuple) WHERE {scope} } AS tuples
"""
_PER_SITE = """
MATCH ()-[x:FQ_FACT]->() {seek} WHERE {scope}
WITH DISTINCT x.head AS head, x.relation AS relation, x.tail AS tail, x.site AS site, x.oid AS oid,
     x.version AS version, x.modality AS modality, coalesce(x.overs, '') AS overs, coalesce(x.wspan, '') AS wspan
RETURN site, count(*) AS n
"""
_ENTITIES = """
MATCH (e)-[x:FQ_FACT]-() {seek} WHERE {scope}
WITH e.key AS key, count(DISTINCT x.site) AS sites
RETURN count(key) AS entities, sum(CASE WHEN sites >= 2 THEN 1 ELSE 0 END) AS cross_site
"""

_MARK_STALE = """
UNWIND $rows AS row
MATCH ()-[r:FQ_FACT {src: row.uid}]->() USING INDEX SEEK r:FQ_FACT(src)
WHERE r.stale IS NULL AND r.overs <> '' AND r.overs <> row.overs
SET r.stale = $seq
RETURN count(DISTINCT [r.src, r.head, r.relation, r.tail, r.version, r.modality, r.overs,
                        coalesce(r.wspan, '')]) AS n
"""
_KILL_VERSIONS = """
UNWIND $rows AS row
MERGE (x:FQVersion {uid: row.uid}) ON CREATE SET x.ns = $ns, x.site = row.site, x.oid = row.oid
WITH x, row, coalesce(x.dead, []) AS dead,
     COLLECT { MATCH ()-[r:FQ_FACT {src: row.uid}]->() USING INDEX SEEK r:FQ_FACT(src) WHERE r.overs <> ''
               RETURN DISTINCT r.overs } AS seen
WITH x, row, dead,
     [w IN seen + [x.overs] WHERE w IS NOT NULL AND w <> '' AND w <> row.overs AND NOT w IN dead] AS newly
SET x.dead = dead + reduce(acc = [], w IN newly | CASE WHEN w IN acc THEN acc ELSE acc + [w] END),
    x.overs = row.overs
"""
_VERSIONS = "MATCH (x:FQVersion) WHERE x.uid IN $uids AND x.overs IS NOT NULL RETURN x.uid AS uid, x.overs AS overs"
_PROVENANCE = """
UNWIND $rows AS row
MATCH ()-[r:FQ_FACT {src: row.src}]->() USING INDEX SEEK r:FQ_FACT(src)
WHERE r.head = row.head AND r.relation = row.relation AND r.tail = row.tail AND r.version = row.version
      AND r.modality = row.modality AND coalesce(r.overs, '') = row.overs AND coalesce(r.wspan, '') = row.wspan
WITH row.i AS i, r ORDER BY r.seq
RETURN i, collect(coalesce(r.writer, ''))[0] AS writer, collect(coalesce(r.qid, ''))[0] AS qid,
       count(r.stale) > 0 AS stale
"""
_FACT_STATS = """
MATCH ()-[r:FQ_FACT]->() USING INDEX SEEK r:FQ_FACT(ns) WHERE r.ns = $ns
WITH r ORDER BY r.seq
WITH r.head AS head, r.relation AS relation, r.tail AS tail, r.site AS site, r.oid AS oid, r.version AS version,
     r.modality AS modality, coalesce(r.overs, '') AS overs, coalesce(r.wspan, '') AS wspan,
     collect(coalesce(r.writer, '')) AS writers,
     collect(coalesce(r.qid, '')) AS qids, count(r.stale) AS marks
RETURN writers[0] AS writer, count(*) AS facts, sum(CASE WHEN marks > 0 THEN 1 ELSE 0 END) AS stale,
       sum(CASE WHEN version ENDS WITH '+harvest' THEN 1 ELSE 0 END) AS harvested,
       collect(DISTINCT qids[0]) AS questions
"""
_FACT_ENTITIES = ("MATCH (e)-[x:FQ_FACT]-() USING INDEX SEEK x:FQ_FACT(ns) WHERE x.ns = $ns "
                  "RETURN count(DISTINCT e.key) AS n")
_TUPLE_COUNT = "RETURN COUNT { MATCH (x:FQTuple) WHERE x.ns = $ns } AS n"
_TUPLES = """
UNWIND $uids AS uid
MATCH (x:FQTuple {uid: uid}) USING INDEX SEEK x:FQTuple(uid)
RETURN x.uid AS uid, x.payload AS payload
"""
_PUT_TUPLES = """
UNWIND $rows AS row
MERGE (x:FQTuple {uid: row.uid}) ON CREATE SET x.ns = $ns, x.key = row.key, x.payload = row.payload
"""

_ASK_FIELDS = """
OPTIONAL MATCH (a)-[e:FQ_WITNESS|FQ_SILENT]->(o:FQObj)
RETURN a.uid AS uid, a.seq AS seq, a.value AS value, a.relation AS relation, a.head AS head, type(e) AS kind,
       e.values AS stated, e.chance AS chance, e.seq AS at, o.site AS site, o.oid AS oid
"""
_ASKS = "UNWIND $values AS v MATCH (a:FQAsk {ns: $ns, value: v}) USING INDEX SEEK a:FQAsk(ns, value)" + _ASK_FIELDS
_HELD = "UNWIND $uids AS uid MATCH (a:FQAsk {uid: uid}) USING INDEX SEEK a:FQAsk(uid)" + _ASK_FIELDS
_NODES = "MATCH (a:FQAsk {ns: $ns}) RETURN DISTINCT a.value AS value"
_INDEX_STATS = "MATCH (s:FQIndexStat {ns: $ns}) RETURN s.cls AS cls, s.n AS n, s.w AS w, s.e AS e"
_ADD_INDEX_STATS = """
UNWIND $rows AS row
MERGE (s:FQIndexStat {uid: row.uid})
ON CREATE SET s.ns = $ns, s.cls = row.cls, s.n = 0, s.w = 0, s.e = 0.0
SET s.n = s.n + row.n, s.w = s.w + row.w, s.e = coalesce(s.e, 0.0) + row.e
"""
_PUT_ASKS = """
UNWIND $rows AS row
MERGE (a:FQAsk {uid: row.uid})
ON CREATE SET a.ns = $ns, a.value = row.value, a.relation = row.relation, a.head = row.head, a.seq = row.seq
WITH a
MATCH (a)-[e:FQ_WITNESS|FQ_SILENT]->()
DELETE e
"""
_OBJ = "MERGE (o:FQObj {uid: row.uid}) ON CREATE SET o.ns = $ns, o.site = row.site, o.oid = row.oid"
_ASK = "UNWIND $rows AS row MATCH (a:FQAsk {uid: row.ask}) USING INDEX SEEK a:FQAsk(uid) "
_PUT_WITNESSES = (_ASK + _OBJ +
                  " CREATE (a)-[:FQ_WITNESS {values: row.stated, chance: row.chance, seq: row.at}]->(o)")
_PUT_SILENT = _ASK + _OBJ + " CREATE (a)-[:FQ_SILENT]->(o)"
_PUT_ENTRIES = "UNWIND $rows AS row " + _OBJ + " SET o.entry = coalesce(o.entry, row.entry)"
_ENTRIES = "MATCH (o:FQObj) WHERE o.uid IN $uids AND o.entry IS NOT NULL RETURN o.uid AS uid, o.entry AS entry"
_PUT_PROBES = """
UNWIND $rows AS row
MERGE (p:FQProbe {uid: row.uid}) ON CREATE SET p.ns = $ns, p.key = row.key, p.result = row.result
"""
_PROBES = "MATCH (p:FQProbe) WHERE p.uid IN $uids RETURN p.uid AS uid, p.result AS result"
_PUT_ALIASES = """
UNWIND $rows AS row
MERGE (l:FQAlias {uid: row.uid})
ON CREATE SET l.ns = $ns, l.key = row.key, l.canonical = row.canonical, l.canon = row.canon
"""
_ALIASES = "MATCH (l:FQAlias) WHERE l.uid IN $uids RETURN l.key AS key, l.canonical AS canonical"
_KNOWN = """
UNWIND $rows AS row
WITH row WHERE EXISTS { MATCH (e:FQNode {uid: row.uid})-[:FQ_FACT]-() USING INDEX SEEK e:FQNode(uid) }
            OR EXISTS { MATCH (a:FQAsk {ns: $ns, value: row.key})-[:FQ_WITNESS]->()
                        USING INDEX SEEK a:FQAsk(ns, value) }
            OR EXISTS { MATCH (l:FQAlias {ns: $ns, canon: row.key}) USING INDEX SEEK l:FQAlias(ns, canon) }
RETURN row.key AS key
"""


def _fact(rec: Any) -> Fact:
    return Fact(rec["head"], rec["relation"], rec["tail"], rec["site"], rec["oid"], rec["version"], rec["modality"],
                rec["overs"] or "", rec.get("wspan") or "")


def _object_uid(ns: str, site: str, oid: str) -> str:
    return f"{ns}|{site}|{oid}"


def _node_uid(ns: str, key: str) -> str:
    return f"{ns}|{key}"


def _fid(ns: str, r: ExtractionResult, f: Fact) -> str:
    return stable_hash(ns, r.site, r.oid, f.head, f.relation, f.tail, f.site, f.oid, f.version, f.modality,
                       *([f.overs] if f.overs else []), *([f"w:{f.wspan}"] if f.wspan else []), n=24)


def _tuple_uid(ns: str, key: str) -> str:
    return f"{ns}|{key}"


def _uid(kind: str, ns: str, *key: Any) -> str:
    return stable_hash(kind, ns, *key, n=32)


def _ask_uid(ns: str, rec: AskRecord) -> str:
    return _uid("ask", ns, rec.value, rec.relation, rec.head)


def _chunks(rows: list[Any], n: int = 1000) -> Iterator[list[Any]]:
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _records(rows: Sequence[Any]) -> dict[str, tuple[int, AskRecord]]:
    held: dict[str, tuple[int, AskRecord]] = {}
    stated: dict[str, list[tuple[int, Ref, list[str], float | None]]] = defaultdict(list)
    for r in rows:
        if r["uid"] not in held:
            held[r["uid"]] = (r["seq"], AskRecord(r["value"], r["relation"], r["head"]))
        if r["kind"] == "FQ_WITNESS":
            stated[r["uid"]].append((r["at"], (r["site"], r["oid"]), list(r["stated"]), r["chance"]))
        elif r["kind"] == "FQ_SILENT":
            held[r["uid"]][1].negatives.add((r["site"], r["oid"]))
    for uid, edges in stated.items():
        rec = held[uid][1]
        for _, ref, values, chance in sorted(edges, key=lambda e: e[0]):
            rec.witnesses[ref] = values
            if chance is not None:
                rec.chance[ref] = chance
    return held


async def _rows(tx: Any, query: str, **params: Any) -> list[Any]:
    return [record async for record in await tx.run(query, params)]


async def _reserve(tx: Any, ns: str, k: int = 0) -> int:
    return (await _rows(tx, _RESERVE, ns=ns, k=k))[0]["base"]


class Neo4jKG:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None,
                 writer: str | None = None) -> None:
        from neo4j import AsyncGraphDatabase, NotificationMinimumSeverity

        self.uri = uri
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password),
                                                notifications_min_severity=NotificationMinimumSeverity.OFF)
        self.database = database
        self._writer = writer
        self._write = asyncio.Lock()
        self._schema = asyncio.Lock()
        self._schema_ready = False

    @property
    def writer(self) -> str:
        return self._writer or default_writer()

    @writer.setter
    def writer(self, value: str | None) -> None:
        self._writer = value

    async def _run(self, query: str, **params: Any) -> list[Any]:
        result = await self.driver.execute_query(query, params, database_=self.database)
        return list(result.records)

    async def _transact(self, work: Callable[[Any], Awaitable[Any]]) -> Any:
        await self.ensure_schema()
        async with self._write, self.driver.session(
                database=self.database, bookmark_manager=self.driver.execute_query_bookmark_manager) as session:
            return await session.execute_write(work)

    async def _create(self, statement: str) -> None:
        from neo4j.exceptions import ClientError

        try:
            await self._run(statement)
        except ClientError as e:
            if "AlreadyExists" not in (e.code or ""):
                raise

    async def _claim(self) -> None:
        async def marker() -> str | None:
            rows = await self._run("MATCH (m:FQMeta) RETURN m.project AS project LIMIT 1")
            return rows[0]["project"] if rows else None

        project = await marker()
        if project == PROJECT:
            return
        other = await self._run("MATCH (n) WHERE NOT n:FQMeta RETURN DISTINCT labels(n) AS labels LIMIT 5")
        if project is None and not other:
            await self._create(_META)
            await self._run("MERGE (m:FQMeta {project: $p}) ON CREATE SET m.created = datetime()", p=PROJECT)
            return
        if project is None and await marker() == PROJECT:
            return
        found = project or sorted({lb for r in other for lb in r["labels"]})
        raise RuntimeError(
            f"the Neo4j database at {self.uri} already holds other data ({found}); FedQuest only uses its "
            "own database: run `fedquest kg up` and use the NEO4J_URI/NEO4J_PASSWORD it writes to .fedquest.env"
        )

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema:
            if not self._schema_ready:
                await self._claim()
                held = {r["name"] for q in ("SHOW CONSTRAINTS YIELD name", "SHOW INDEXES YIELD name")
                        for r in await self._run(q)}
                for statement in _SCHEMA:
                    if statement.split()[2] not in held:
                        await self._create(statement)
                await self._run("CALL db.awaitIndexes(300)")
                self._schema_ready = True

    async def covered(self, ns: str, refs: Sequence[Ref], version: str) -> dict[Ref, list[Fact]]:
        if not refs:
            return {}
        await self.ensure_schema()
        uids = {_object_uid(ns, site, oid): (site, oid) for site, oid in refs}
        rows = await self._run(_COVERED, uids=list(uids), version=version)
        hit = {r["uid"] for r in rows}
        out: dict[Ref, list[Fact]] = {ref: [] for uid, ref in uids.items() if uid in hit}
        for r in rows:
            if r["seq"] is not None:
                out[uids[r["uid"]]].append(_fact(r))
        return out

    def _lookup_args(self, ns: str, entities: Sequence[str], limit: int, include_stale: bool
                     ) -> tuple[str, dict[str, Any]] | None:
        uids = list(dict.fromkeys(_node_uid(ns, norm_entity(e)) for e in entities))
        if not uids:
            return None
        query = _LOOKUP.replace("{live}", "true" if include_stale else "r.stale IS NULL")
        return query, {"uids": uids, "at": {u: i for i, u in enumerate(uids)}, "limit": max(1, limit)}

    async def lookup(self, ns: str, entities: Sequence[str], limit: int = 200, *,
                     include_stale: bool = False) -> list[Fact]:
        args = self._lookup_args(ns, entities, limit, include_stale)
        if args is None:
            return []
        await self.ensure_schema()
        return [_fact(r) for r in await self._run(args[0], **args[1])]

    async def add(self, ns: str, results: Sequence[ExtractionResult]) -> None:
        results = [r for r in results if materializable(r)]
        objects = [
            {"uid": _object_uid(ns, r.site, r.oid), "site": r.site, "oid": r.oid,
             "version": r.version, "modality": r.modality}
            for r in results
        ]
        nodes: dict[str, dict[str, Any]] = {}
        facts, versions = [], []
        said: dict[str, str] = {}
        for r in results:
            for f in r.facts:
                ends = []
                for name in (f.head, f.tail):
                    key = norm_entity(name)
                    ends.append(uid := _node_uid(ns, key))
                    nodes.setdefault(uid, {"uid": uid, "key": key, "name": name})
                src = _object_uid(ns, f.site, f.oid)
                facts.append({"fid": _fid(ns, r, f), "ref": _object_uid(ns, r.site, r.oid), "src": src,
                              "hu": ends[0], "tu": ends[1], "head": f.head, "relation": f.relation, "tail": f.tail,
                              "site": f.site, "oid": f.oid, "version": f.version, "modality": f.modality,
                              "overs": f.overs, "wspan": f.wspan, "qid": r.qid})
                if f.overs and said.get(src) != f.overs:
                    said[src] = f.overs
                    versions.append({"uid": src, "site": f.site, "oid": f.oid, "overs": f.overs})
        writer = self.writer
        node_rows = sorted(nodes.values(), key=lambda n: n["uid"])
        version_rows = sorted(versions, key=lambda v: v["uid"])

        async def work(tx: Any) -> None:
            base = await _reserve(tx, ns, len(objects) + len(facts))
            for rows in _chunks(node_rows):
                await _rows(tx, _ADD_NODES, ns=ns, rows=rows)
            if objects:
                rows = sorted(({**o, "seq": base + i} for i, o in enumerate(objects)), key=lambda o: o["uid"])
                await _rows(tx, _ADD_OBJECTS, ns=ns, rows=rows)
            for rows in _chunks(version_rows):
                await _rows(tx, _ADD_VERSIONS, ns=ns, rows=rows)
            base += len(objects)
            fact_rows = sorted(({**f, "seq": base + i} for i, f in enumerate(facts)), key=lambda f: f["fid"])
            for rows in _chunks(fact_rows):
                await _rows(tx, _ADD_FACTS, ns=ns, writer=writer, rows=rows)

        await self._transact(work)

    async def facts(self, ns: str) -> list[Fact]:
        await self.ensure_schema()
        return list(dict.fromkeys(_fact(r) for r in await self._run(_FACTS, ns=ns)))

    async def stats(self, ns: str | None = None) -> dict[str, Any]:
        await self.ensure_schema()
        found = ns is not None and bool((await self._run(_EXISTS, ns=ns))[0]["found"])
        one = found or bool(ns)
        scope, seek = ("x.ns = $ns", "USING INDEX SEEK x:FQ_FACT(ns)") if one else ("true", "")
        namespaces = int(found) if one else (await self._run(_NAMESPACES))[0]["n"]
        counts = (await self._run(_COUNTS.replace("{scope}", scope), ns=ns))[0]
        per_site = await self._run(_PER_SITE.replace("{scope}", scope).replace("{seek}", seek), ns=ns)
        entities = await self._run(_ENTITIES.replace("{scope}", scope).replace("{seek}", seek), ns=ns)
        return {
            "namespaces": namespaces,
            "objects": counts["objects"],
            "facts": sum(r["n"] for r in per_site),
            "entities": entities[0]["entities"] if entities else 0,
            "facts_per_site": {r["site"]: r["n"] for r in per_site},
            "cross_site_entities": entities[0]["cross_site"] if entities else 0,
            **{k: counts[k] for k in ("asks", "witnesses", "negatives", "aliases", "probes")},
            **({"tuples": counts["tuples"]} if counts["tuples"] else {}),
        }

    async def export_graph(self, ns: str, limit: int = 20000) -> dict[str, Any]:
        await self.ensure_schema()
        rows = await self._run(
            "MATCH (h)-[r:FQ_FACT]->(t) USING INDEX SEEK r:FQ_FACT(ns) WHERE r.ns = $ns "
            "RETURN h.key AS hk, h.name AS hn, t.key AS tk, t.name AS tn, "
            f"{_FACT_FIELDS} ORDER BY r.seq LIMIT $limit",
            ns=ns, limit=limit,
        )
        edges: dict[Fact, Any] = {}
        for r in rows:
            edges.setdefault(_fact(r), r)
        nodes = {r["hk"]: r["hn"] for r in rows} | {r["tk"]: r["tn"] for r in rows}
        return {
            "nodes": [{"id": k, "name": v} for k, v in nodes.items()],
            "edges": [{"source": r["hk"], "target": r["tk"], **f.to_dict()} for f, r in edges.items()],
        }

    async def clear(self, ns: str) -> None:
        await self.ensure_schema()
        async with self._write:
            for query in (*_CLEAR, _ORPHANS):
                while (await self._run(query, ns=ns, sub=ns + "/"))[0]["n"]:
                    pass

    async def close(self) -> None:
        await self.driver.close()

    async def stale(self, ns: str, refs_versions: Mapping[Ref, str]) -> int:
        rows = sorted(({"uid": _object_uid(ns, site, oid), "site": site, "oid": oid, "overs": v}
                       for (site, oid), v in refs_versions.items() if v), key=lambda row: row["uid"])
        if not rows:
            return 0

        async def work(tx: Any) -> int:
            seq = await _reserve(tx, ns, 1)
            n = (await _rows(tx, _MARK_STALE, ns=ns, seq=seq, rows=rows))[0]["n"]
            await _rows(tx, _KILL_VERSIONS, ns=ns, rows=rows)
            return int(n)

        return await self._transact(work)

    async def versions(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, str]:
        uids = {_object_uid(ns, site, oid): (site, oid) for site, oid in refs}
        if not uids:
            return {}
        await self.ensure_schema()
        held = {r["uid"]: r["overs"] for r in await self._run(_VERSIONS, uids=list(uids))}
        return {ref: held[uid] for uid, ref in uids.items() if uid in held}

    async def writers(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        return {f: p["writer"] for f, p in (await self.provenance(ns, facts)).items()}

    async def qids(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        return {f: p["qid"] for f, p in (await self.provenance(ns, facts)).items()}

    async def provenance(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, dict[str, Any]]:
        asked = list(dict.fromkeys(facts))
        if not asked:
            return {}
        await self.ensure_schema()
        rows = [{"i": i, "src": _object_uid(ns, f.site, f.oid), "head": f.head, "relation": f.relation,
                 "tail": f.tail, "version": f.version, "modality": f.modality, "overs": f.overs, "wspan": f.wspan}
                for i, f in enumerate(asked)]
        held = {r["i"]: r for chunk in _chunks(rows) for r in await self._run(_PROVENANCE, rows=chunk)}
        return {f: {"writer": held[i]["writer"], "qid": held[i]["qid"], "stale": bool(held[i]["stale"])}
                for i, f in enumerate(asked) if i in held}

    async def fact_stats(self, ns: str) -> dict[str, Any]:
        await self.ensure_schema()
        rows = await self._run(_FACT_STATS, ns=ns)
        return {
            "facts": sum(r["facts"] for r in rows),
            "stale": sum(r["stale"] for r in rows),
            "entities": (await self._run(_FACT_ENTITIES, ns=ns))[0]["n"],
            "harvested": sum(r["harvested"] for r in rows),
            "questions": len({q for r in rows for q in r["questions"] if q}),
            "by_writer": {r["writer"]: r["facts"] for r in rows},
            "tuples": (await self._run(_TUPLE_COUNT, ns=ns))[0]["n"],
        }

    async def aliases(self, ns: str, names: Sequence[str]) -> dict[str, str]:
        keys = list(dict.fromkeys(norm_entity(n) for n in names))
        if not keys:
            return {}
        await self.ensure_schema()
        rows = await self._run(_ALIASES, uids=[_uid("alias", ns, k) for k in keys])
        linked = {r["key"]: r["canonical"] for r in rows}
        return {k: linked[k] for k in keys if k in linked}

    async def add_aliases(self, ns: str, links: Mapping[str, str]) -> None:
        rows: dict[str, dict[str, Any]] = {}
        for name, canonical in links.items():
            key = norm_entity(name)
            if key and key != norm_entity(canonical) and key not in rows:
                rows[key] = {"uid": _uid("alias", ns, key), "key": key, "canonical": canonical,
                             "canon": norm_entity(canonical)}

        async def work(tx: Any) -> None:
            await _reserve(tx, ns)
            if rows:
                await _rows(tx, _PUT_ALIASES, ns=ns, rows=sorted(rows.values(), key=lambda r: r["uid"]))

        await self._transact(work)

    async def known(self, ns: str, names: Sequence[str]) -> set[str]:
        keys = [k for k in dict.fromkeys(norm_entity(n) for n in names) if k]
        if not keys:
            return set()
        await self.ensure_schema()
        rows = [{"uid": _node_uid(ns, k), "key": k} for k in keys]
        return {r["key"] for r in await self._run(_KNOWN, ns=ns, rows=rows)}

    async def asks(self, ns: str, values: Sequence[str]) -> list[AskRecord]:
        keys = list(dict.fromkeys(norm_entity(x) for x in values))
        if not keys:
            return []
        await self.ensure_schema()
        at = {v: i for i, v in enumerate(keys)}
        held = _records(await self._run(_ASKS, ns=ns, values=keys))
        return [rec for _, rec in sorted(held.values(), key=lambda h: (at[h[1].value], h[0]))]

    async def nodes(self, ns: str) -> list[str]:
        await self.ensure_schema()
        return [r["value"] for r in await self._run(_NODES, ns=ns) if r["value"]]

    async def index_stats(self, ns: str) -> dict[str, tuple[int, int, float]]:
        await self.ensure_schema()
        return {r["cls"]: (int(r["n"] or 0), int(r["w"] or 0), float(r["e"] or 0.0))
                for r in await self._run(_INDEX_STATS, ns=ns)}

    async def add_index_stats(self, ns: str, deltas: Mapping[str, tuple[int, int] | tuple[int, int, float]]) -> None:
        rows = sorted(({"uid": _uid("istat", ns, cls), "cls": cls, "n": int(n), "w": int(w), "e": float(e)}
                       for cls, (n, w, e) in ((c, (*d, 0.0)[:3]) for c, d in deltas.items()) if n or w),
                      key=lambda r: r["uid"])
        if not rows:
            return

        async def work(tx: Any) -> None:
            await _reserve(tx, ns)
            await _rows(tx, _ADD_INDEX_STATS, ns=ns, rows=rows)

        await self._transact(work)

    async def add_asks(self, ns: str, records: Sequence[AskRecord], entries: Sequence[Candidate]) -> None:
        firsts: dict[Ref, Candidate] = {}
        for c in entries:
            firsts.setdefault(c.ref, c)
        put_entries = sorted(({"uid": _object_uid(ns, *ref), "site": ref[0], "oid": ref[1],
                               "entry": json.dumps(c.to_dict(), ensure_ascii=False)} for ref, c in firsts.items()),
                             key=lambda r: r["uid"])

        async def work(tx: Any) -> None:
            base = await _reserve(tx, ns, len(records))
            uids = list(dict.fromkeys(_ask_uid(ns, rec) for rec in records))
            held = {uid: rec for uid, (_, rec) in _records(await _rows(tx, _HELD, uids=uids)).items()} if uids else {}
            merged: dict[str, AskRecord] = {}
            fresh: dict[str, int] = {}
            for rec in records:
                uid = _ask_uid(ns, rec)
                old = merged.get(uid, held.get(uid))
                if old is None:
                    fresh[uid] = base + len(fresh)
                merged[uid] = rec if old is None else old.merged(rec)
            asks = [{"uid": uid, "value": rec.value, "relation": rec.relation, "head": rec.head, "seq": fresh.get(uid)}
                    for uid, rec in merged.items()]
            witnesses = [{"ask": uid, "uid": _object_uid(ns, *ref), "site": ref[0], "oid": ref[1], "stated": values,
                          "chance": rec.chance.get(ref), "at": i}
                         for uid, rec in merged.items() for i, (ref, values) in enumerate(rec.witnesses.items())]
            silent = [{"ask": uid, "uid": _object_uid(ns, *ref), "site": ref[0], "oid": ref[1]}
                      for uid, rec in merged.items() for ref in sorted(rec.negatives)]
            for query, rows in ((_PUT_ASKS, asks), (_PUT_WITNESSES, witnesses), (_PUT_SILENT, silent),
                                (_PUT_ENTRIES, put_entries)):
                if rows:
                    await _rows(tx, query, ns=ns, rows=rows)

        await self._transact(work)

    async def entries(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, Candidate]:
        if not refs:
            return {}
        await self.ensure_schema()
        uids = {_object_uid(ns, site, oid): (site, oid) for site, oid in refs}
        held = {r["uid"]: r["entry"] for r in await self._run(_ENTRIES, uids=list(uids))}
        return {ref: Candidate.from_dict(json.loads(held[uid])) for uid, ref in uids.items() if uid in held}

    async def probes(self, ns: str, keys: Sequence[str]) -> dict[str, list[Candidate]]:
        uids = {_uid("probe", ns, k): k for k in keys}
        if not uids:
            return {}
        await self.ensure_schema()
        held = {r["uid"]: r["result"] for r in await self._run(_PROBES, uids=list(uids))}
        return {k: [Candidate.from_dict(d) for d in json.loads(held[uid])] for uid, k in uids.items() if uid in held}

    async def add_probes(self, ns: str, results: dict[str, list[Candidate]]) -> None:
        rows = sorted(({"uid": _uid("probe", ns, k), "key": k,
                        "result": json.dumps([c.to_dict() for c in cands], ensure_ascii=False)}
                       for k, cands in results.items()), key=lambda r: r["uid"])

        async def work(tx: Any) -> None:
            await _reserve(tx, ns)
            if rows:
                await _rows(tx, _PUT_PROBES, ns=ns, rows=rows)

        await self._transact(work)

    async def tuples(self, ns: str, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        uids = {_tuple_uid(ns, k): k for k in keys}
        if not uids:
            return {}
        await self.ensure_schema()
        held = {r["uid"]: r["payload"] for chunk in _chunks(list(uids)) for r in await self._run(_TUPLES, uids=chunk)}
        return {k: json.loads(held[uid]) for uid, k in uids.items() if uid in held}

    async def add_tuples(self, ns: str, items: Mapping[str, dict[str, Any]]) -> None:
        rows = sorted(({"uid": _tuple_uid(ns, k), "key": k, "payload": json.dumps(payload, ensure_ascii=False)}
                       for k, payload in items.items()), key=lambda r: r["uid"])

        async def work(tx: Any) -> None:
            await _reserve(tx, ns)
            for chunk in _chunks(rows):
                await _rows(tx, _PUT_TUPLES, ns=ns, rows=chunk)

        await self._transact(work)
