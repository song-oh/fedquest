# anchor-normalized scores, with a key's cosines to fixed anchors as background
from __future__ import annotations

import functools
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import REPO

log = logging.getLogger(__name__)

ANCHORS_ENV = "FEDQUEST_ANCHORS"
DEFAULT_ANCHORS = REPO / "configs" / "anchors" / "wiki.jsonl"
IMAGE_ANCHORS_ENV = "FEDQUEST_IMAGE_ANCHORS"
DEFAULT_IMAGE_ANCHORS = REPO / "configs" / "anchors" / "images.jsonl"
MIN_IMAGES = 128
Z_DECIMALS = 2
_EPS = 1e-6


def anchors_path() -> Path | None:
    env = os.environ.get(ANCHORS_ENV)
    if env is not None and env.strip().lower() in ("off", "none", ""):
        return None
    path = Path(env).expanduser() if env else DEFAULT_ANCHORS
    return path if path.exists() else None


@functools.lru_cache(maxsize=4)
def _texts(path: str) -> tuple[str, ...]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(d["text"] if isinstance(d, dict) else str(d))
    return tuple(out)


def anchor_texts() -> tuple[str, ...]:
    path = anchors_path()
    return _texts(str(path)) if path is not None else ()


@functools.lru_cache(maxsize=4)
def _sources(path: str) -> tuple[str, ...]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(str(d.get("source", "")) if isinstance(d, dict) else "")
    return tuple(out)


VARIANTS = ("n256", "n512", "n1024", "text", "mmqa")


@functools.lru_cache(maxsize=4)
def variant_rows(path: str) -> dict[str, np.ndarray]:
    src = _sources(path)
    n = len(src)
    out = {f"n{k}": np.arange(min(k, n)) for k in (256, 512, 1024)}
    out["text"] = np.array([i for i, s in enumerate(src) if s in ("musique", "twowiki", "hotpot")], dtype=np.int64)
    out["mmqa"] = np.array([i for i, s in enumerate(src) if s.startswith("mmqa")], dtype=np.int64)
    return {k: v for k, v in out.items() if len(v) >= 32}


TAIL_Q = 0.95


# -log10 p of a cosine: empirical below the 0.95 quantile, exponential tail above it
class TailBackground:
    def __init__(self, sims: np.ndarray) -> None:
        self.sims = np.sort(np.asarray(sims, dtype=np.float64))
        n = len(self.sims)
        self.n = n
        self.u = float(np.quantile(self.sims, TAIL_Q)) if n else 0.0
        over = self.sims[self.sims > self.u] - self.u
        self.p_u = (len(over) + 1) / (n + 1)
        self.sigma = float(over.mean()) if len(over) else 1e-3

    def score(self, cos: float) -> float:
        import bisect

        s = float(cos)
        if s <= self.u or not self.n:
            above = self.n - bisect.bisect_left(self.sims.tolist(), s)
            p = (above + 1) / (self.n + 1)
        else:
            p = self.p_u * np.exp(-(s - self.u) / max(self.sigma, 1e-6))
        return round(float(-np.log10(max(p, 1e-300))), Z_DECIMALS)


def _tail(sims: np.ndarray) -> TailBackground:
    return TailBackground(sims)


def variant_backgrounds(anchors: np.ndarray, key: np.ndarray) -> dict[str, Any]:
    path = anchors_path()
    if path is None:
        return {}
    q = np.asarray(key, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(q))
    sims = anchors @ (q / norm if norm > 0 else q)
    out: dict[str, Any] = {name: (float(sims[rows].mean()), float(sims[rows].std()))
                           for name, rows in variant_rows(str(path)).items()}
    out["tail"] = _tail(sims)
    return out


_MATRICES: dict[tuple[str, str], np.ndarray] = {}
_MATRICES_LOCK = threading.Lock()
_EMBED_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def anchor_matrix(encoder_name: str, encoder: Any) -> np.ndarray | None:
    path = anchors_path()
    if path is None:
        return None
    key = (str(path), encoder_name)
    with _MATRICES_LOCK:
        hit = _MATRICES.get(key)
        if hit is not None:
            return hit
        embed_lock = _EMBED_LOCKS.setdefault(key, threading.Lock())
    with embed_lock:
        with _MATRICES_LOCK:
            hit = _MATRICES.get(key)
        if hit is not None:
            return hit
        texts = list(_texts(str(path)))
        disk = _disk_file(str(path), encoder_name)
        rows = _load_rows(disk, len(texts))
        if rows is None:
            rows = np.asarray(encoder.encode_docs(texts), dtype=np.float32)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            rows = rows / np.where(norms > 0, norms, 1.0)
            log.info("anchors: %d texts embedded with %s", len(texts), encoder_name)
            _save_rows(disk, rows)
        else:
            log.info("anchors: %d texts under %s loaded from %s", len(texts), encoder_name, disk)
        with _MATRICES_LOCK:
            _MATRICES.setdefault(key, rows)
            return _MATRICES[key]


def _disk_file(path: str, encoder_name: str) -> Path | None:
    try:
        import hashlib

        from fedquest.paths import runs_root

        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        return runs_root() / "cache" / "anchors" / f"{encoder_name}-{digest}.npy"
    except Exception:
        return None


def _load_rows(disk: Path | None, n: int) -> np.ndarray | None:
    if disk is None or not disk.exists():
        return None
    try:
        rows = np.load(disk)
    except (OSError, ValueError):
        return None
    return rows.astype(np.float32, copy=False) if rows.ndim == 2 and len(rows) == n else None


def _save_rows(disk: Path | None, rows: np.ndarray) -> None:
    if disk is None:
        return
    try:
        disk.parent.mkdir(parents=True, exist_ok=True)
        tmp = disk.with_name(f"{disk.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, rows)
        tmp.replace(disk)
    except OSError:
        pass


def image_anchors_path() -> Path | None:
    if anchors_path() is None:
        return None
    env = os.environ.get(IMAGE_ANCHORS_ENV)
    if env is not None and env.strip().lower() in ("off", "none", ""):
        return None
    path = Path(env).expanduser() if env else DEFAULT_IMAGE_ANCHORS
    return path if path.exists() else None


@functools.lru_cache(maxsize=4)
def _image_entries(path: str) -> tuple[tuple[str, str], ...]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append((str(d["image_path"]), str(d.get("title") or "")))
    return tuple(out)


def _image_paths(path: str) -> tuple[str, ...]:
    return tuple(p for p, _ in _image_entries(path))


_NO_IMAGES = np.zeros((0, 0), dtype=np.float32)


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms > 0, norms, 1.0)


def image_anchor_matrix(encoder_name: str, encoder: Any, raw_root: Path | str, vectors: str = "fused",
                        ) -> np.ndarray | None:
    path = image_anchors_path()
    if path is None:
        return None
    mode = "pixels" if vectors == "pixels" else "fused"
    key = (str(path), encoder_name, f"image-{mode}", str(raw_root))
    with _MATRICES_LOCK:
        hit = _MATRICES.get(key)
    if hit is None:
        from PIL import Image

        images, titles = [], []
        for rel, title in _image_entries(str(path)):
            try:
                with Image.open(Path(raw_root) / rel) as im:
                    images.append(im.convert("RGB"))
                titles.append(title)
            except (OSError, ValueError):
                continue
        if len(images) < MIN_IMAGES:
            log.warning("image anchors: %d of %d load under %s: image collections use the text anchors", len(images),
                        len(_image_paths(str(path))), raw_root)
            hit = _NO_IMAGES
        else:
            rows = _unit_rows(encoder.encode_images(images))
            if mode == "fused":
                rows = _unit_rows(rows + _unit_rows(encoder.encode_queries(titles)))
            hit = rows
            log.info("image anchors: %d images embedded with %s (%s)", len(images), encoder_name, mode)
        with _MATRICES_LOCK:
            hit = _MATRICES.setdefault(key, hit)
    return None if hit.size == 0 else hit


def reference(encoder_name: str, encoder: Any, modality: str, raw_root: Path | str | None = None,
              vectors: str = "fused") -> tuple[np.ndarray | None, bool]:
    if modality == "image" and raw_root is not None and getattr(getattr(encoder, "spec", None), "kind", "") == "clip":
        mat = image_anchor_matrix(encoder_name, encoder, raw_root, vectors)
        if mat is not None:
            return mat, True
    return anchor_matrix(encoder_name, encoder), False


def image_variant_backgrounds(images: np.ndarray, texts: np.ndarray | None, key: np.ndarray,
                              ) -> dict[str, Any]:
    q = np.asarray(key, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(q))
    q = q / norm if norm > 0 else q
    sims = images @ q
    out = {f"n{k}": (float(sims[:k].mean()), float(sims[:k].std())) for k in (256, 512, 1024) if len(sims) >= 32}
    out["mmqa"] = (float(sims.mean()), float(sims.std()))
    out["tail"] = _tail(sims)
    path = anchors_path()
    if texts is not None and path is not None:
        rows = variant_rows(str(path)).get("text")
        if rows is not None:
            t = texts @ q
            out["text"] = (float(t[rows].mean()), float(t[rows].std()))
    return out


def background(anchors: np.ndarray, key: np.ndarray) -> tuple[float, float]:
    q = np.asarray(key, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    s = anchors @ (q / n if n > 0 else q)
    return float(s.mean()), float(s.std())


def zscore(cos: float, stats: tuple[float, float]) -> float:
    mean, std = stats
    return round((float(cos) - mean) / max(std, _EPS), Z_DECIMALS)
