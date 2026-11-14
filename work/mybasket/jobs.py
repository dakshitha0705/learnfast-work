"""Long CPU jobs for sections 8-11, run in the background and resumable. Results go to work/checkpoints/.

    python -m mybasket.jobs score  <model name or dir> <out folder> [--reshuffle40]
    python -m mybasket.jobs train  [--steps 800 --batch 16]
    python -m mybasket.jobs sample <model dir> <n>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from thilimarketbasket.dataset import generate_dataset
from thilimarketbasket.evaluate import candidate_token_ids, make_cases
from thilimarketbasket.verbalize import verbalize_prefix

ROOT = Path(__file__).resolve().parent.parent / "checkpoints"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def setup():
    data = generate_dataset(n_baskets=100_000, seed=0)
    names = [item.name for item in data.catalog]
    is_val = np.random.default_rng(0).random(len(data.baskets)) < 0.05      # the split from section 7
    train = [b for b, v in zip(data.baskets, is_val) if not v]
    val = [b for b, v in zip(data.baskets, is_val) if v]
    return data, names, train, val


def score_job(model_ref, out_dir, reshuffle40=False, n_cases=200):
    """Score the first `n_cases` of the 200 test cases (always built as 200, so the first 100 are the same cases)."""
    from .score import load, score_candidates
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, names, train, val = setup()
    cases = make_cases(val, names, n_cases=200, seed=0)[:n_cases]
    model, tokenizer = load(model_ref)
    candidates = candidate_token_ids(tokenizer, names)

    if not (out / "scores-blank.npy").exists():
        np.save(out / "scores-blank.npy", score_candidates(model, tokenizer.encode("Basket:"), candidates))
        log("blank scored")

    jobs = [(f"scores-{n_cases}", lambda k: k, len(cases))]
    if reshuffle40:
        jobs.append(("scores-40-reshuffled", lambda k: 1000 + k, 40))
    for name, seed_of, count in jobs:
        final, part = out / f"{name}.npy", out / f"{name}.partial.npy"
        if final.exists():
            log(f"{name} already done")
            continue
        scores = list(np.load(part)) if part.exists() else []
        if scores:
            log(f"{name}: resuming at case {len(scores)}")
        started = time.time()
        for k in range(len(scores), count):
            prefix = verbalize_prefix(cases[k]["visible"], seed=seed_of(k))
            scores.append(score_candidates(model, tokenizer.encode(prefix), candidates))
            if (k + 1) % 5 == 0 or k + 1 == count:
                np.save(part, np.stack(scores))
                log(f"{name}: {k + 1}/{count} cases, {(time.time() - started) / 60:.1f} min this run")
        np.save(final, np.stack(scores))
        part.unlink(missing_ok=True)
        (out / f"{name}.json").write_text(json.dumps({"cases": count, "minutes_last_run": (time.time() - started) / 60}))
    log("DONE")


def train_job(steps, batch):
    from .pretrain import pretrain
    data, names, train, val = setup()
    out = ROOT / f"baskets-{steps}x{batch}"
    if (out / "model.safetensors").exists():
        log("checkpoint already there")
    else:
        pretrain("HuggingFaceTB/SmolLM2-135M", train, str(out), val_baskets=val[:300], steps=steps,
                 batch_size=batch, eval_every=50, log=log)
    log("DONE")


def sample_job(model_dir, n):
    from .probe import sample_baskets
    from .score import load
    data, names, train, val = setup()
    model, tokenizer = load(model_dir)
    out = Path(model_dir)
    final = out / f"samples-{n}.json"
    if final.exists():
        log("samples already there")
    else:
        started = time.time()
        sampled, stats = sample_baskets(model, tokenizer, names, n=n, batch=100, seed=0,
                                        progress_file=out / f"samples-{n}.partial.jsonl", log=log)
        stats["minutes_last_run"] = (time.time() - started) / 60
        final.write_text(json.dumps(sampled))
        (out / f"samples-{n}-stats.json").write_text(json.dumps(stats))
        log(f"stats {stats}")
    log("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--reshuffle40", action="store_true")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cases", type=int, default=200)
    a = parser.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    log(f"torch threads {torch.get_num_threads()}")
    if a.job == "score":
        score_job(a.args[0], a.args[1], reshuffle40=a.reshuffle40, n_cases=a.cases)
    elif a.job == "train":
        train_job(a.steps, a.batch)
    elif a.job == "sample":
        sample_job(a.args[0], int(a.args[1]))
