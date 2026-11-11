from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.config import FederationCfg, load_federation
from fedquest.data.unreadable import skip_rows
from fedquest.paths import collection_dir, raw_dir
from fedquest.types import ObjectRecord
from fedquest.utils import read_jsonl

log = logging.getLogger(__name__)

UNREADABLE_DOT = 0.999


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def doc_text(obj: ObjectRecord) -> str:
    if obj.modality == "image" or obj.text.startswith(obj.title.split(" — ")[0]):
        return obj.text
    return f"{obj.title}\n{obj.text}"


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def pixel_vectors(fused: np.ndarray, titles: np.ndarray, threshold: float = UNREADABLE_DOT,
                  ) -> tuple[np.ndarray, list[int]]:
    f, t = _unit(fused), _unit(titles)
    dot = np.einsum("ij,ij->i", f, t)
    v = _unit(2.0 * dot[:, None] * f - t)
    missing = np.flatnonzero(dot > threshold)
    v[missing] = 0.0
    return v, [int(i) for i in missing]


def embed_collection(
    cfg: FederationCfg, site: str, collection: str, device: str = "auto", batch_size: int = 256, force: bool = False
) -> Path:
    from fedquest.vdb.encoders import load_encoder

    coll = cfg.own_site(site, "embed").collection(collection)
    if coll.vectors_from:
        derive_collection(cfg, site, collection, device, batch_size, force)
        return collection_dir(cfg.name, site, collection) / "emb.npy"
    cdir = collection_dir(cfg.name, site, collection)
    objects_path = cdir / "objects.jsonl"
    meta_path = cdir / "emb.json"
    sha = _sha(objects_path)
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("objects_sha256") == sha and meta.get("encoder") == coll.encoder
                and meta.get("vectors", "fused") == coll.vectors):
            log.info("%s/%s/%s: embeddings up to date", cfg.name, site, collection)
            return cdir / "emb.npy"
    objs = [ObjectRecord.from_dict(d) for d in read_jsonl(objects_path)]
    enc = load_encoder(coll.encoder, device)
    t0 = time.perf_counter()
    unread: list[int] = []
    missing: list[int] = []
    if enc.spec.kind == "clip":
        from PIL import Image

        titles = enc.encode_queries([o.title for o in objs], batch_size)
        vecs = np.zeros((len(objs), enc.dim), dtype=np.float32) if coll.vectors == "pixels" else titles.copy()
        root = raw_dir(cfg.dataset)
        for start in range(0, len(objs) if coll.vectors != "titles" else 0, batch_size):
            chunk = objs[start : start + batch_size]
            pics, rows = [], []
            for i, o in enumerate(chunk):
                try:
                    with Image.open(root / (o.image_path or "")) as im:
                        pics.append(im.convert("RGB"))
                    rows.append(start + i)
                except (OSError, ValueError):
                    unread.append(start + i)
            img = enc.encode_images(pics, batch_size) if pics else np.zeros((0, enc.dim), dtype=np.float32)
            for row, v in zip(rows, img, strict=True):
                if coll.vectors == "pixels":
                    vecs[row] = v
                else:
                    fused = v + titles[row]
                    vecs[row] = fused / (np.linalg.norm(fused) or 1.0)
        if coll.vectors == "pixels":
            missing = list(unread)
    else:
        vecs = enc.encode_docs([doc_text(o) for o in objs], batch_size)
    if unread:
        log.warning("%s/%s/%s: %d images unreadable, %s", cfg.name, site, collection, len(unread),
                    "left out of the index" if coll.vectors == "pixels" else "embedded by title only")
    np.save(cdir / "emb.npy", vecs.astype(np.float16))
    meta = {"encoder": coll.encoder, "dim": int(vecs.shape[1]) if len(vecs) else enc.dim, "count": len(objs),
            "objects_sha256": sha, "device": enc.device, "seconds": round(time.perf_counter() - t0, 1),
            "missing_images": len(unread), "vectors": coll.vectors, "missing": missing, "unreadable": unread}
    meta_path.write_text(json.dumps(meta, indent=1))
    log.info("%s/%s/%s: %d vectors in %.1fs on %s", cfg.name, site, collection, len(objs), meta["seconds"], enc.device)
    return cdir / "emb.npy"


def embed_federation(cfg: FederationCfg, sites: list[str] | None = None, device: str = "auto",
                     batch_size: int = 256, force: bool = False) -> None:
    own = cfg.own_sites(sites, "embed")
    for site in cfg.sites:
        if site.name not in own:
            continue
        for coll in site.collections:
            embed_collection(cfg, site.name, coll.name, device, batch_size, force)


def _digest(d: dict[str, Any]) -> str:
    o = ObjectRecord.from_dict(d)
    return hashlib.sha256(f"{doc_text(o)}\x1f{o.image_path or ''}".encode()).hexdigest()[:16]


def _catalog(path: Path) -> dict[str, tuple[int, str]]:
    return {d["oid"]: (i, _digest(d)) for i, d in enumerate(read_jsonl(path))}


def derive_collection(cfg: FederationCfg, site: str, collection: str, device: str = "auto", batch_size: int = 256,
                      force: bool = False, catalogs: dict[Path, dict[str, tuple[int, str]]] | None = None,
                      ) -> dict[str, Any]:
    coll = cfg.own_site(site, "derive").collection(collection)
    if coll.parent is None:
        raise ValueError(f"{cfg.name}/{site}/{collection} sets no vectors_from")
    pfed, psite, pcoll = coll.parent
    parent = load_federation(pfed).site(psite).collection(pcoll)
    cdir, pdir = collection_dir(cfg.name, site, collection), collection_dir(pfed, psite, pcoll)
    if not (pdir / "emb.json").exists():
        raise FileNotFoundError(f"{pdir}/emb.json: derive {cfg.name}/{site}/{collection} where {coll.vectors_from} "
                                "has its vectors (prepare and embed it there first)")
    pmeta = json.loads((pdir / "emb.json").read_text())
    if pmeta["objects_sha256"] != _sha(pdir / "objects.jsonl"):
        raise RuntimeError(f"{pdir}/emb.npy was computed for other objects than its objects.jsonl: re-embed it first")
    if not (cdir / "objects.jsonl").exists():
        raise FileNotFoundError(f"{cdir}/objects.jsonl: run `fedquest data prepare {cfg.name}` first")
    sha = _sha(cdir / "objects.jsonl")
    meta_path = cdir / "emb.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("objects_sha256") == sha and meta.get("encoder") == coll.encoder
                and meta.get("vectors") == coll.vectors and meta.get("vectors_from") == coll.vectors_from
                and meta.get("parent_sha256") == pmeta["objects_sha256"]):
            log.info("%s/%s/%s: derived vectors up to date", cfg.name, site, collection)
            return meta
    pmode = pmeta.get("vectors", "fused")
    clip = parent.encoder == coll.encoder and _kind(coll.encoder) == "clip"
    if pmeta.get("encoder") != coll.encoder or parent.encoder != coll.encoder:
        raise RuntimeError(f"{cfg.name}/{site}/{collection} ({coll.encoder}) cannot copy the vectors of "
                           f"{coll.vectors_from} ({pmeta.get('encoder')}): another encoder's space")
    if pmode != coll.vectors and not (clip and pmode == "fused") and not (clip and coll.vectors == "titles"):
        raise RuntimeError(f"{cfg.name}/{site}/{collection} wants {coll.vectors} vectors; {coll.vectors_from} holds "
                           f"{pmode} ones, which they cannot be recovered from")
    t0 = time.perf_counter()
    catalogs = {} if catalogs is None else catalogs
    if pdir not in catalogs:
        catalogs[pdir] = _catalog(pdir / "objects.jsonl")
    pcat = catalogs[pdir]
    objs = list(read_jsonl(cdir / "objects.jsonl"))
    rows = [pcat.get(d["oid"], (-1, ""))[0] for d in objs]
    absent = [d["oid"] for d, r in zip(objs, rows, strict=True) if r < 0]
    if absent:
        raise RuntimeError(f"{len(absent)} objects of {cfg.name}/{site}/{collection} are not in {coll.vectors_from} "
                           f"(e.g. {absent[0]}): derive needs the parent to hold every object")
    changed = [d["oid"] for d, r in zip(objs, rows, strict=True) if _digest(d) != pcat[d["oid"]][1]]
    if changed:
        raise RuntimeError(f"{len(changed)} objects of {cfg.name}/{site}/{collection} differ from their copy in "
                           f"{coll.vectors_from} (e.g. {changed[0]}): their vectors would not be theirs")
    parent_vecs = np.load(pdir / "emb.npy", mmap_mode="r")
    pmissing = set(pmeta.get("missing") or [])
    missing = [i for i, r in enumerate(rows) if r in pmissing]
    device_used = "copy"
    if pmode == coll.vectors:
        vecs = np.asarray(parent_vecs[rows])
    else:
        from fedquest.vdb.encoders import load_encoder

        enc = load_encoder(coll.encoder, device)
        device_used = enc.device
        titles = enc.encode_queries([d["title"] for d in objs], batch_size)
        if coll.vectors == "titles":
            vecs, missing = titles, []
        else:
            vecs, unread = pixel_vectors(np.asarray(parent_vecs[rows], dtype=np.float32), titles)
            missing = sorted(set(missing) | set(unread))
    np.save(cdir / "emb.npy", vecs.astype(np.float16))
    meta = {"encoder": coll.encoder, "dim": int(vecs.shape[1]) if len(vecs) else int(parent_vecs.shape[1]),
            "count": len(objs), "objects_sha256": sha, "vectors": coll.vectors, "vectors_from": coll.vectors_from,
            "parent_sha256": pmeta["objects_sha256"], "parent_vectors": pmode, "missing": missing,
            "missing_images": len(missing), "device": device_used, "seconds": round(time.perf_counter() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=1))
    log.info("%s/%s/%s: %d %s vectors derived from %s (%d missing)", cfg.name, site, collection, len(objs),
             coll.vectors, coll.vectors_from, len(missing))
    return meta


def derive_federation(name: str, sites: list[str] | None = None, *, device: str = "auto", batch_size: int = 256,
                      force: bool = False) -> dict[str, Any]:
    cfg = load_federation(name)
    own = cfg.own_sites(sites, "derive")
    catalogs: dict[Path, dict[str, tuple[int, str]]] = {}
    out: dict[str, Any] = {}
    for site in cfg.sites:
        if site.name not in own:
            continue
        for coll in site.collections:
            if coll.vectors_from:
                out[f"{site.name}/{coll.name}"] = derive_collection(cfg, site.name, coll.name, device, batch_size,
                                                                    force, catalogs)
    return out


def _kind(encoder: str) -> str:
    from fedquest.vdb.encoders import ENCODERS

    spec = ENCODERS.get(encoder)
    return spec.kind if spec else ""


def stale(cfg: FederationCfg, site: str, collection: str) -> bool:
    if cfg.site(site).data_from:
        return False
    if cfg.data_from:
        return stale(load_federation(cfg.data_from), site, collection)
    coll = cfg.site(site).collection(collection)
    cdir = collection_dir(cfg.name, site, collection)
    if not (cdir / "objects.jsonl").exists() or not (cdir / "emb.json").exists():
        return True
    meta = json.loads((cdir / "emb.json").read_text())
    same = meta.get("objects_sha256") == _sha(cdir / "objects.jsonl") and meta.get("encoder") == coll.encoder
    if coll.parent is None:
        return not (same and meta.get("vectors", "fused") == coll.vectors)
    pdir = collection_dir(*coll.parent)
    if not (pdir / "objects.jsonl").exists() or not (pdir / "emb.json").exists():
        return True
    psha = json.loads((pdir / "emb.json").read_text()).get("objects_sha256")
    return not (psha == _sha(pdir / "objects.jsonl") and same and meta.get("vectors") == coll.vectors
                and meta.get("vectors_from") == coll.vectors_from and meta.get("parent_sha256") == psha)


def stale_collections(federations: Iterable[str], sites: list[str] | None = None) -> list[str]:
    out = []
    for name in federations:
        cfg = load_federation(name)
        out += [f"{cfg.name}/{job}" for job in _jobs(cfg, sites) if stale(cfg, *job.split("/"))]
    return out


def index_skip(cdir: Path, objects: Iterable[Any]) -> set[int]:
    meta = cdir / "emb.json"
    missing = (json.loads(meta.read_text()).get("missing") or []) if meta.exists() else []
    return set(missing) | skip_rows(objects)


INDEXED = "index.json"


def index_stale(cdir: Path, namespace: str) -> bool:
    path = cdir / INDEXED
    built = (json.loads(path.read_text()).get(namespace) or {}).get("emb") if path.exists() else None
    return built is not None and built != _sha(cdir / "emb.json")


def _record_index(cdir: Path, namespace: str, points: int) -> None:
    path = cdir / INDEXED
    marks = json.loads(path.read_text()) if path.exists() else {}
    marks[namespace] = {"emb": _sha(cdir / "emb.json"), "points": points}
    tmp = path.with_name(f"{INDEXED}.part")
    tmp.write_text(json.dumps(marks, indent=1, sort_keys=True))
    tmp.replace(path)


def check_vectors(cdir: Path, vectors: str) -> None:
    meta = json.loads((cdir / "emb.json").read_text())
    if meta["objects_sha256"] != _sha(cdir / "objects.jsonl"):
        raise RuntimeError(f"{cdir}/emb.npy was computed for different objects: re-run `fedquest data "
                           f"prepare` with the same config on both machines, or re-embed here")
    if meta.get("vectors", "fused") != vectors:
        raise RuntimeError(f"{cdir}/emb.npy holds {meta.get('vectors', 'fused')} vectors, the config asks for "
                           f"{vectors}: run `fedquest data derive` (or embed) again")


def build_index(cfg: FederationCfg, site: str, record: bool = True, grpc: bool = False) -> dict[str, int]:
    from fedquest.vdb.qdrant_index import SiteIndex

    if cfg.data_from and not cfg.qdrant.approximate:
        raise ValueError(f"{cfg.name} reads {cfg.data_from}'s index (a policy variant): index {cfg.data_from} instead")
    collections = cfg.own_site(site, "index").collections
    index = SiteIndex(cfg.site_data_dir(site), cfg.qdrant, namespace=cfg.index_namespace(site), grpc=grpc)
    counts = {}
    try:
        for coll in collections:
            cdir = cfg.collection_data_dir(site, coll.name)
            check_vectors(cdir, coll.vectors)
            skip = index_skip(cdir, read_jsonl(cdir / "objects.jsonl"))
            counts[coll.name] = index.build(coll.name, np.load(cdir / "emb.npy").astype(np.float32), skip=skip)
            if record:
                _record_index(cdir, cfg.index_namespace(site), counts[coll.name])
            log.info("%s/%s/%s: indexed %d points (%d rows left out)%s", cfg.name, site, coll.name, counts[coll.name],
                     len(skip), f" as {cfg.qdrant.index.kind}" if cfg.qdrant.approximate and cfg.qdrant.index else "")
    finally:
        index.close()
    return counts


def _jobs(cfg: FederationCfg, sites: list[str] | None) -> list[str]:
    return [f"{s.name}/{c.name}" for s in cfg.sites if (not sites or s.name in sites) and not s.data_from
            for c in s.collections]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("federation", nargs="?")
    ap.add_argument("--sites", default="", help="comma-separated site names (default: all)")
    ap.add_argument("--only", default="", help="one job, <site>/<collection>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", metavar="FED", help="print the <site>/<collection> jobs of FED and exit")
    ap.add_argument("--summary", nargs="*", metavar="FED", help="print embedding metadata of these federations")
    ap.add_argument("--stale", nargs="+", metavar="FED",
                    help="print <federation>/<site>/<collection> of every collection of these federations whose "
                         "vectors are not up to date (what embedding them would compute) and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sites = [s for s in args.sites.split(",") if s] or None
    if args.list:
        print("\n".join(_jobs(load_federation(args.list), sites)))
        return
    if args.stale:
        for line in stale_collections(args.stale, sites):
            print(line)
        return
    if args.summary is not None:
        for fed in args.summary:
            cfg = load_federation(fed)
            for job in _jobs(cfg, sites):
                meta = collection_dir(cfg.name, *job.split("/")) / "emb.json"
                print(fed, job, meta.read_text().replace("\n", " ") if meta.exists() else "MISSING")
        return
    cfg = load_federation(args.federation)
    if args.only:
        site, coll = args.only.split("/")
        embed_collection(cfg, site, coll, args.device, args.batch_size, args.force)
    else:
        embed_federation(cfg, sites, args.device, args.batch_size, args.force)


if __name__ == "__main__":
    main()
