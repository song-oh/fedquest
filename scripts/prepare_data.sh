#!/usr/bin/env bash
# this facility's databases: download, prepare and embed whatever is missing
set -euo pipefail
SITE=${1:?usage: prepare_data.sh aurora|midway3}
cd "$(dirname "$0")/.."
source .venv/bin/activate
PY=.venv/bin/python

one() {
  local fed=$1 site=$2 dataset missing
  dataset=$($PY -c "from fedquest.config import load_federation; print(load_federation('$fed').dataset)")
  if ! $PY - "$dataset" <<'PY'
import sys
from fedquest.paths import raw_dir
d = raw_dir(sys.argv[1])
sys.exit(0 if d.exists() and any(d.iterdir()) else 1)
PY
  then
    fedquest data download "$dataset"
  fi
  # a collection without objects copies them from a sibling with the same source
  $PY - "$fed" "$site" <<'PY'
import shutil
import sys
from fedquest.config import load_federation
cfg = load_federation(sys.argv[1])
site = cfg.site(sys.argv[2])
for c in site.collections:
    d = cfg.collection_data_dir(site.name, c.name)
    if (d / "objects.jsonl").exists():
        continue
    for other in site.collections:
        src = cfg.collection_data_dir(site.name, other.name) / "objects.jsonl"
        if other.name != c.name and other.source == c.source and src.exists():
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d / "objects.jsonl")
            break
PY
  missing=$($PY - "$fed" "$site" <<'PY'
import sys
from fedquest.config import load_federation
cfg = load_federation(sys.argv[1])
site = cfg.site(sys.argv[2])
out = []
for c in site.collections:
    d = cfg.collection_data_dir(site.name, c.name)
    if not (d / "objects.jsonl").exists():
        out.append("objects")
    if not (d / "emb.npy").exists() or not (d / "emb.json").exists():
        out.append("derive" if c.vectors_from else "embed")
print(" ".join(sorted(set(out))))
PY
)
  if [[ -z $missing ]]; then
    echo "== $fed/$site: objects and vectors present"
    return
  fi
  echo "== $fed/$site: missing $missing"
  if [[ $missing == *objects* ]]; then fedquest data prepare "$fed" --sites "$site"; fi
  if [[ $missing == *derive* ]]; then fedquest data derive "$fed" --sites "$site"; fi
  if [[ $missing == *embed* || $missing == *objects* ]]; then fedquest data embed "$fed" --sites "$site"; fi
}

if [[ $SITE == aurora ]]; then
  for home in mmqa_home musique_next_home twowiki_home hotpot_home; do one "$home" home; done
fi
for fed in mmqa_fed musique_next twowiki_fed hotpot_fed; do one "$fed" "$SITE"; done
for fed in mmqa_fed musique_next twowiki_fed hotpot_fed; do fedquest data status "$fed"; done
