from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fedquest.utils import stable_int

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model_id: str
    dim: int
    kind: Literal["text", "clip", "hash"]
    query_prefix: str = ""
    doc_prefix: str = ""


_BGE_QUERY = "Represent this sentence for searching relevant passages: "

ENCODERS: dict[str, EncoderSpec] = {
    s.name: s
    for s in (
        EncoderSpec("bge-base", "BAAI/bge-base-en-v1.5", 768, "text", query_prefix=_BGE_QUERY),
        EncoderSpec("bge-small", "BAAI/bge-small-en-v1.5", 384, "text", query_prefix=_BGE_QUERY),
        EncoderSpec("e5-base", "intfloat/e5-base-v2", 768, "text", query_prefix="query: ", doc_prefix="passage: "),
        EncoderSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, "text"),
        EncoderSpec("clip", "sentence-transformers/clip-ViT-B-32", 512, "clip"),
        EncoderSpec("hash", "", 256, "hash"),
        EncoderSpec("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", 1024, "text",
                    query_prefix="Instruct: Given a question, retrieve passages that answer the question\nQuery:"),
    )
}


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


class Encoder:
    def __init__(self, spec: EncoderSpec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self._model: Any = None
        if spec.kind != "hash":
            from sentence_transformers import SentenceTransformer

            threads = os.environ.get("FEDQUEST_TORCH_THREADS")
            if threads and device == "cpu":
                import torch

                torch.set_num_threads(int(threads))
            self._model = SentenceTransformer(spec.model_id, device=device)

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _encode(self, items: list[Any], batch_size: int) -> np.ndarray:
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(items, batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def _hash(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in "".join(c.lower() if c.isalnum() else " " for c in t).split():
                rng = np.random.default_rng(stable_int("hash-encoder", tok))
                out[i] += rng.choice((-1.0, 1.0), size=self.dim)
        return _normalize(out)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.spec.kind == "hash":
            return self._hash(texts)
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.spec.kind == "hash":
            return self._hash(texts)
        return self._encode([self.spec.doc_prefix + t for t in texts], batch_size)

    def encode_images(self, images: list[Any], batch_size: int = 32) -> np.ndarray:
        if self.spec.kind == "hash":
            return self._hash([Path(str(i)).name if not hasattr(i, "size") else "image" for i in images])
        if self.spec.kind != "clip":
            raise ValueError(f"encoder {self.spec.name} cannot embed images")
        from PIL import Image

        pils = [i if hasattr(i, "convert") else Image.open(i) for i in images]
        return self._encode([p.convert("RGB") for p in pils], batch_size)


@cache
def load_encoder(name: str, device: str = "cpu") -> Encoder:
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)}")
    device = resolve_device(device) if ENCODERS[name].kind != "hash" else "cpu"
    log.info("loading encoder %s on %s", name, device)
    return Encoder(ENCODERS[name], device)


_UNUSED_FORMATS = ["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"]


def fetch_models(names: list[str]) -> None:
    from huggingface_hub import HfApi, snapshot_download

    for name in names:
        spec = ENCODERS[name]
        if spec.model_id:
            files = set(HfApi().list_repo_files(spec.model_id))
            duplicates = [f for f in files if f.endswith("pytorch_model.bin")
                          and f.removesuffix("pytorch_model.bin") + "model.safetensors" in files]
            path = snapshot_download(spec.model_id, ignore_patterns=[*_UNUSED_FORMATS, *duplicates])
            log.info("cached %s (%s) in %s", name, spec.model_id, path)
