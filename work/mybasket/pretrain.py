"""Continued pretraining with PyTorch on a CPU: a port of `thilimarketbasket.pretrain.pretrain` (which needs MLX).

Same objective, data order logic and schedule as the reference: length-bucketed batches drawn with a seeded rng, every
basket in a fresh random order (`_encode`), AdamW (weight decay 0.01), gradient clipping at 1.0, linear warmup then a
cosine down to a tenth of the peak, float32 training, held-out loss every `eval_every` steps.

Differences, because this runs for an hour or more on a CPU:
- the batches for all steps are planned up front from the seed, so a run can stop and resume at any saved step
  with exactly the same data;
- a resumable state (model, optimizer, step, history) is saved every `eval_every` steps;
- no compiled step (MLX's `mx.compile` has no direct equivalent used here).
"""
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thilimarketbasket.pretrain import _encode   # tokenizing needs no MLX


def pad(examples):
    """Token ids padded with 0 to a multiple of 16, and a 0/1 mask that is 1 on real tokens (as the reference `_pad`)."""
    width = -(-max(len(e) for e in examples) // 16) * 16
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    mask = torch.zeros(len(examples), width)
    for row, e in enumerate(examples):
        ids[row, :len(e)] = torch.tensor(e)
        mask[row, :len(e)] = 1
    return ids, mask


def next_token_loss(model, ids, mask):
    """Average cross-entropy of predicting each token from the ones before it, ignoring padding."""
    logits = model(input_ids=ids[:, :-1]).logits.float()
    per_token = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
    real = mask[:, 1:]
    return (per_token * real).sum() / real.sum()


@torch.no_grad()
def validation_loss(model, tokenizer, baskets, batch_size: int = 32, seed: int = 123) -> float:
    """Loss on held-out baskets, each written once in a fixed order (as the reference)."""
    examples = _encode(tokenizer, baskets, np.random.default_rng(seed).integers(2**31, size=len(baskets)))
    total = count = 0.0
    model.eval()
    for start in range(0, len(examples), batch_size):
        ids, mask = pad(examples[start:start + batch_size])
        n = float(mask[:, 1:].sum())
        total += float(next_token_loss(model, ids, mask)) * n
        count += n
    return total / count


def plan_batches(tokenizer, baskets, steps: int, batch_size: int, seed: int = 0) -> List[List[List[int]]]:
    """The token ids of every batch for `steps` steps, made the same way as the reference's `batches()`."""
    rng = np.random.default_rng(seed)
    bucket = batch_size * 20
    planned = []
    while len(planned) < steps:
        order = rng.permutation(len(baskets))
        for start in range(0, len(order) - bucket + 1, bucket):
            chosen = order[start:start + bucket]
            examples = _encode(tokenizer, [baskets[i] for i in chosen], rng.integers(2**31, size=len(chosen)))
            by_length = sorted(range(len(examples)), key=lambda i: len(examples[i]))
            groups = [by_length[i:i + batch_size] for i in range(0, len(by_length), batch_size)]
            for g in rng.permutation(len(groups)):
                planned.append([examples[i] for i in groups[g]])
                if len(planned) == steps:
                    return planned
    return planned


def pretrain(model_name: str, baskets, out_dir: str, val_baskets=None, steps: int = 800, batch_size: int = 16,
             learning_rate: float = 1e-4, warmup: int = 50, eval_every: int = 100, seed: int = 0,
             log=print) -> List[Dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "resume.pt"
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    def rate(step):
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    history, done_steps, tokens, seconds_before = [], 0, 0, 0.0
    if state_file.exists():
        saved = torch.load(state_file, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, done_steps, tokens, seconds_before = saved["history"], saved["step"], saved["tokens"], saved["seconds"]
        log(f"resuming at step {done_steps}")
    elif val_baskets is not None:
        history.append({"step": 0, "loss": None, "val_loss": validation_loss(model, tokenizer, val_baskets),
                        "tokens": 0, "seconds": 0.0})
        log(f"step 0  val loss {history[-1]['val_loss']:.3f}")

    planned = plan_batches(tokenizer, baskets, steps, batch_size, seed)
    started, recent = time.time(), []
    for step in range(done_steps + 1, steps + 1):
        model.train()
        ids, mask = pad(planned[step - 1])
        for group in optimizer.param_groups:
            group["lr"] = rate(step - 1)
        loss = next_token_loss(model, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += int(mask.sum())
        recent.append(float(loss))
        if step % 10 == 0:
            elapsed = seconds_before + time.time() - started
            log(f"  step {step}  loss {np.mean(recent[-10:]):.3f}  {elapsed / 60:.1f} min")
        if step % eval_every == 0 or step == steps:
            val = validation_loss(model, tokenizer, val_baskets) if val_baskets is not None else None
            elapsed = seconds_before + time.time() - started
            history.append({"step": step, "loss": float(np.mean(recent)), "val_loss": val, "tokens": tokens,
                            "seconds": elapsed})
            recent = []
            log(f"step {step}  train loss {history[-1]['loss']:.3f}"
                + (f"  val loss {val:.3f}" if val is not None else "")
                + f"  {tokens / elapsed:,.0f} tokens/s")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                        "step": step, "tokens": tokens, "seconds": elapsed}, state_file)

    model.save_pretrained(out)          # float32 safetensors; loads with AutoModelForCausalLM.from_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "history.json").write_text(json.dumps(history))
    state_file.unlink(missing_ok=True)
    return history
