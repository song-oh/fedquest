from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from fedquest.paths import raw_dir

log = logging.getLogger(__name__)

MMQA_BASE = "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset"
MMQA_FILES = {
    "MMQA_train.jsonl.gz": "2d7c8f6f1659df69f8dcdb79e4701c48e53e8674ab8573244b3ffe2e4b1da565",
    "MMQA_dev.jsonl.gz": "2e348ca574b2dc368e84671709689070f943a59f2e08e6b6e374705deb712d31",
    "MMQA_test.jsonl.gz": "7a89ae5b0ce9df4e430ad7d114c7c357f1fe9ff1660a809eff7ad6cd399aa6ae",
    "MMQA_texts.jsonl.gz": "cae3808ccc6c258e91131a3ca3dce43e629936e1496c97a288c4abb04a6ef905",
    "MMQA_tables.jsonl.gz": "8d082d254fd0bfa19bca7e2da20369c15ccb25867db3edfe46b33d59e8dbf7b1",
    "MMQA_images.jsonl.gz": "691227e1758140c51a5617a904b5e363cd44b2582a0c0e9f9fe9153d4e62d5e7",
}
MMQA_IMAGES_ZIP = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"
MGRAG_CAPTIONS = "https://raw.githubusercontent.com/Wyb0627/MGRAG/main/caption/mmqa_caption.json"

MRAMG_REVISION = "308347a7cde9a2649d7a1fef1d3c39032d806d87"
MRAMG_BASE = f"https://huggingface.co/datasets/MRAMG/MRAMG-Bench/resolve/{MRAMG_REVISION}"
MRAMG_FILES = {
    "arxiv_mqa.jsonl": "ff2c22c648320b245da57dfd51582fffbc601eb124d7124da219dfc97c9584ca",
    "doc_arxiv.jsonl": "51060920c361e6afac4b457ac1b8f189d9594ebae1701ad8927d19aec632618a",
    "manual_mqa.jsonl": "51893036c85f8be66026285181b2b23421eb4921d4afbb68467dcafd50e40d9e",
    "doc_manual.jsonl": "71dd5d2caabaf05274d4413345e2c37a7d79d7807ca9a0a062104df51c07efc1",
    "recipe_mqa.jsonl": "c9f196e592ab5042c389eee98ceeb6c4f8705a0a2dc666839571db80aae045e3",
    "doc_recipe.jsonl": "4f59ced46a06c0c229be3d67bf505d469d444d5baed20908d75941395e69ef4c",
    "web_mqa.jsonl": "aebfbc6e067a126fb713f3aed3162eae95a75c2e587feea6be2e51419c71b04c",
    "doc_web.jsonl": "54db12bc148fbda26696bf296506dca5a69a1e600d0094682cbf050a8876ab88",
    "wiki_mqa.jsonl": "f9f87145b30de66da4a382f043169ce0cca2d78b0ea4f73a0f659f1705df73bc",
    "doc_wiki.jsonl": "35a39ed37da9a22b6269dca560378fe853630ecaf2ecd9249fc05a479628dd52",
    "wit_mqa.jsonl": "97bef31f805649a3d29fa5e32544c2d49178ca888b71657531c6a6b122464022",
    "doc_wit.jsonl": "526e5b6baa0328a4abd2a13ff25482d11f7396a10ec8345edebf1bdc47250e09",
}

MUSIQUE_BASE = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main"
MUSIQUE_FILES = {
    "musique.json": "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
    "musique_corpus.json": "73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
}
MUSIQUE_ZIP = ("https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download"
               "&confirm=t")
MUSIQUE_ZIP_SHA = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
MUSIQUE_TRAIN = ("musique_ans_v1.0_train.jsonl", "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str | None = None, retries: int = 5) -> Path:
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = int(r.headers.get("content-length", 0)) + (have if mode == "ab" else 0)
                with open(part, mode) as f:
                    for i, block in enumerate(r.iter_content(1 << 20)):
                        f.write(block)
                        if i % 512 == 0 and total:
                            log.info("%s: %.0f%%", dest.name, 100 * f.tell() / total)
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("download of %s failed (%s); retrying", url, e)
            time.sleep(2**attempt)
    if sha256 is not None and _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {url}")
    part.replace(dest)
    log.info("downloaded %s", dest)
    return dest


def _range_extract(zip_url: Callable[[], str], members: list[tuple[str, Path]], workers: int = 8) -> int:
    from remotezip import RemoteZip

    todo = [(m, d) for m, d in members if not d.exists()]
    log.info("%d of %d archive members to fetch", len(todo), len(members))
    local = threading.local()

    def one(job: tuple[str, Path]) -> None:
        member, dest = job
        for attempt in range(8):
            try:
                if not hasattr(local, "zip"):
                    local.zip = RemoteZip(zip_url())
                dest.parent.mkdir(parents=True, exist_ok=True)
                with local.zip.open(member) as src, open(dest.with_name(dest.name + ".part"), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                dest.with_name(dest.name + ".part").replace(dest)
                return
            except KeyError:
                raise
            except Exception:
                if hasattr(local, "zip"):
                    del local.zip
                if attempt == 7:
                    raise
                time.sleep(min(60, 2**attempt))

    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(one, todo), 1):
            if n % 500 == 0 or n == len(todo):
                log.info("  %d/%d", n, len(todo))
    return len(todo)


def download_mmqa(images: bool = True, image_scope: str = "all") -> Path:
    root = raw_dir("mmqa")
    for name, sha in MMQA_FILES.items():
        fetch(f"{MMQA_BASE}/{name}", root / name, sha)
    if images:
        if image_scope == "all":
            archive = fetch(MMQA_IMAGES_ZIP, root / "final_dataset_images.zip")
            if not (root / "final_dataset_images").is_dir():
                with zipfile.ZipFile(archive) as z:
                    z.extractall(root, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
        else:
            with gzip.open(root / "MMQA_images.jsonl.gz", "rt", encoding="utf-8") as f:
                path_of = {d["id"]: d["path"] for d in map(json.loads, f)}
            with gzip.open(root / f"MMQA_{image_scope}.jsonl.gz", "rt", encoding="utf-8") as f:
                ids = sorted({i for q in map(json.loads, f) for i in q["metadata"]["image_doc_ids"]})
            members = [(f"final_dataset_images/{path_of[i]}", root / "final_dataset_images" / path_of[i]) for i in ids]
            _range_extract(lambda: MMQA_IMAGES_ZIP, members)
    return root


def download_musique() -> Path:
    root = raw_dir("musique")
    for name, sha in MUSIQUE_FILES.items():
        fetch(f"{MUSIQUE_BASE}/{name}", root / name, sha)
    name, sha = MUSIQUE_TRAIN
    train = root / name
    if not (train.exists() and _sha256(train) == sha):
        archive = fetch(MUSIQUE_ZIP, root / "musique_data_v1.0.zip", MUSIQUE_ZIP_SHA)
        part = train.with_name(train.name + ".part")
        with zipfile.ZipFile(archive) as z, z.open(f"data/{name}") as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst)
        if _sha256(part) != sha:
            part.unlink()
            raise RuntimeError(f"checksum mismatch for {name} in {archive}")
        part.replace(train)
        log.info("extracted %s", train)
    return root


HIPPORAG_FILES = {
    "hotpotqa": ("hotpotqa", {
        "hotpotqa.json": "3ad9c0bcbf93f41d7004ca6007049c904d2605046b314a2f7ecfb379c64cba6d",
        "hotpotqa_corpus.json": "9333647b922382776cd2cb02893b390d77984df85a91bfa8be411284978aca7d",
    }),
    "twowiki": ("2wikimultihopqa", {
        "2wikimultihopqa.json": "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4",
        "2wikimultihopqa_corpus.json": "9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a",
    }),
}


def download_hipporag(dataset: str) -> Path:
    folder, files = HIPPORAG_FILES[dataset]
    root = raw_dir(folder)
    for name, sha in files.items():
        fetch(f"{MUSIQUE_BASE}/{name}", root / name, sha)
    return root


def download_msmarco() -> Path:
    from fedquest.data import msmarco

    root = raw_dir(msmarco.FOLDER)
    for split, (_, sha) in msmarco.FILES.items():
        fetch(msmarco.url(split), root / f"{split}.parquet", sha)
    return root


def download_m3sciqa(images: bool = True) -> Path:
    from fedquest.data import m3sciqa

    root = raw_dir(m3sciqa.FOLDER)
    for split, (_, sha) in m3sciqa.FILES.items():
        fetch(m3sciqa.url(split), root / f"{split}.parquet", sha)
    fetch(m3sciqa.PAPERS_URL, root / m3sciqa.PAPERS[0], m3sciqa.PAPERS[1])
    if images:
        found = list(m3sciqa.figures())
        if len(found) != m3sciqa.IMAGES:
            raise RuntimeError(f"M3SciQA: {len(found)} distinct images in the parquets, not {m3sciqa.IMAGES}")
        written = 0
        for anchor, figure, data in found:
            dest = root / m3sciqa.figure_path(anchor, figure)
            if dest.exists() and dest.read_bytes() == data:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(data)
            part.replace(dest)
            written += 1
        log.info("M3SciQA: %d of %d images written to %s", written, len(found), root / m3sciqa.FIGURES)
    return root


def download_hotpot_wiki() -> Path:
    from fedquest.data.distractors import HOTPOT_WIKI, HOTPOT_WIKI_URL

    path = fetch(HOTPOT_WIKI_URL, raw_dir("hotpotqa") / HOTPOT_WIKI)
    log.info("%s sha256 %s", path.name, _sha256(path))
    return path.parent


def download_mgrag_captions() -> Path:
    return fetch(MGRAG_CAPTIONS, raw_dir("mgrag") / "mmqa_caption.json")


def _hf_cdn_url(path: str) -> str:
    for attempt in range(8):
        r = requests.head(f"{MRAMG_BASE}/{path}", allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308):
            return r.headers["location"]
        if r.status_code == 429:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        r.raise_for_status()
        return f"{MRAMG_BASE}/{path}"
    raise RuntimeError("Hugging Face resolver keeps rate-limiting; retry later")


def download_mramg(subsets: list[str], images: bool = True) -> Path:
    root = raw_dir("mramg")
    for name, sha in MRAMG_FILES.items():
        fetch(f"{MRAMG_BASE}/{name}", root / name, sha)
    if images and subsets:
        from remotezip import RemoteZip

        with RemoteZip(_hf_cdn_url("IMAGE.zip")) as z:
            in_zip = set(z.namelist())
            for s in subsets:
                info = f"IMAGE/images_info/{s}_imgs_collection.json"
                if not (root / info).exists():
                    z.extract(info, root)
        members = []
        for s in subsets:
            meta = json.loads((root / f"IMAGE/images_info/{s}_imgs_collection.json").read_text())
            rels = [f"IMAGE/images/{s.upper()}/{v['image_path']}" for v in meta.values()]
            kept = [rel for rel in rels if rel in in_zip]
            skipped = len(rels) - len(kept)
            if skipped:
                log.warning("%s: %d of %d listed images are not in IMAGE.zip; skipped", s, skipped, len(rels))
            members += [(rel, root / rel) for rel in kept]
        _range_extract(lambda: _hf_cdn_url("IMAGE.zip"), members)
    return root
