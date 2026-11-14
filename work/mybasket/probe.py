"""Sample baskets from a trained model with PyTorch: a port of `thilimarketbasket.probe.sample_baskets` (MLX only).

Same procedure as the reference: every basket starts from the bare prompt "Basket:", a batch is written together one
token per step from the model's full distribution at temperature 1.0 (no top-k), a basket ends at the end-of-text
token, names not in the catalogue are dropped, repeats are kept once, and a basket cut off at `max_tokens` loses its
last (possibly half-written) item. Progress is appended to a .jsonl file batch by batch so a run can resume.
"""
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from thilimarketbasket.verbalize import PREFIX, SEPARATOR


@torch.no_grad()
def sample_baskets(model, tokenizer, names: Sequence[str], n: int, batch: int = 100, max_tokens: int = 240,
                   temperature: float = 1.0, seed: int = 0, progress_file=None, log=print, return_raw=False):
    known = set(names)
    prompt = list(tokenizer.encode(PREFIX.rstrip()))
    eos = tokenizer.eos_token_id
    model.eval()

    done_rows = []
    path = Path(progress_file) if progress_file else None
    if path and path.exists():
        done_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        log(f"resuming: {len(done_rows)} batches already written")

    for number, start in enumerate(range(0, n, batch)):
        if number < len(done_rows):
            continue
        size = min(batch, n - start)
        generator = torch.Generator().manual_seed(seed * 100_003 + number)   # one seed per batch: resumable
        out = model(input_ids=torch.tensor([prompt] * size), use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        written = [[] for _ in range(size)]
        done = np.zeros(size, dtype=bool)
        for _ in range(max_tokens):
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            token = torch.multinomial(probs, 1, generator=generator)[:, 0]
            chosen = token.numpy()
            for row in np.flatnonzero(~done):
                if chosen[row] == eos:
                    done[row] = True
                else:
                    written[row].append(int(chosen[row]))
            if done.all():
                break
            out = model(input_ids=token[:, None], past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1, :]
        rows = []
        for row, ids in enumerate(written):
            finished = bool(done[row])
            pieces = [p.strip() for p in tokenizer.decode(ids).split(SEPARATOR.strip()) if p.strip()]
            cut = (not finished) and bool(pieces)
            if cut:
                pieces = pieces[:-1]
            rows.append({"pieces": pieces, "cut": cut})
        done_rows.append(rows)
        if path:
            with path.open("a") as f:
                f.write(json.dumps(rows) + "\n")
        log(f"batch {number + 1}/{-(-n // batch)} written")

    baskets, unknown, repeats, cut, items_written = [], 0, 0, 0, 0
    for rows in done_rows:
        for r in rows:
            pieces = r["pieces"]
            cut += r["cut"]
            items_written += len(pieces)
            unknown += sum(p not in known for p in pieces)
            pieces = [p for p in pieces if p in known]
            repeats += len(pieces) - len(set(pieces))
            baskets.append(list(dict.fromkeys(pieces)))
    stats = {"baskets": len(baskets), "items written": items_written,
             "share not in catalogue": unknown / max(1, items_written),
             "share repeated": repeats / max(1, items_written),
             "baskets cut off": cut / max(1, len(baskets)),
             "mean items per basket": float(np.mean([len(b) for b in baskets]))}
    if return_raw:   # the items as written (before cleaning), one list per basket
        stats["raw"] = [r["pieces"] for rows in done_rows for r in rows]
    return baskets, stats
