"""Turn the batches a stopped sampler already wrote into a samples file, with the same cleaning as `sample_baskets`.

    python -m mybasket.keep_samples <checkpoint dir> <partial .jsonl> <batches to keep> <minutes spent>
"""
import json
import sys
from pathlib import Path

import numpy as np

from thilimarketbasket.dataset import generate_dataset


def keep(checkpoint, partial, batches, minutes):
    names = {item.name for item in generate_dataset(n_baskets=100_000, seed=0).catalog}
    rows = [json.loads(line) for line in Path(partial).read_text().splitlines() if line.strip()][:batches]
    baskets, unknown, repeats, cut, written = [], 0, 0, 0, 0
    for batch in rows:
        for r in batch:
            pieces = r["pieces"]
            cut += r["cut"]
            written += len(pieces)
            unknown += sum(p not in names for p in pieces)
            pieces = [p for p in pieces if p in names]
            repeats += len(pieces) - len(set(pieces))
            baskets.append(list(dict.fromkeys(pieces)))
    stats = {"baskets": len(baskets), "items written": written, "share not in catalogue": unknown / max(1, written),
             "share repeated": repeats / max(1, written), "baskets cut off": cut / max(1, len(baskets)),
             "mean items per basket": float(np.mean([len(b) for b in baskets])), "minutes": minutes}
    out = Path(checkpoint)
    (out / "samples-kept.json").write_text(json.dumps(baskets))
    (out / "samples-kept-stats.json").write_text(json.dumps(stats))
    print(stats)


if __name__ == "__main__":
    keep(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
