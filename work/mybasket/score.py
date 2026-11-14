"""Score every catalogue item as the next item of a basket, with PyTorch on a CPU.

PyTorch version of `thilimarketbasket.evaluate.score_candidates` (which needs MLX). Like the MLX one, and unlike the
course's untested appendix, it reads the basket once and reuses its key/value cache for every candidate, so a
thousand candidates cost a thousand short sequences, not a thousand whole baskets.
"""
import copy
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def load(name_or_dir: str = MODEL_NAME, threads: int = None):
    """A model in eval mode (float32 on CPU) and its tokenizer."""
    if threads:
        torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    model = AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=torch.float32).eval()
    return model, tokenizer


@torch.no_grad()
def score_candidates(model, prefix_ids: Sequence[int], candidate_ids: Sequence[Sequence[int]], batch: int = 100) -> np.ndarray:
    """log P(candidate | prefix) for every candidate, summed over the candidate's tokens."""
    out = model(input_ids=torch.tensor([list(prefix_ids)]), use_cache=True)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1).numpy()   # how likely each token is to start an item
    prefix_cache = out.past_key_values

    scores = np.zeros(len(candidate_ids))
    order = sorted(range(len(candidate_ids)), key=lambda i: len(candidate_ids[i]))
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        width = max(len(candidate_ids[i]) for i in chunk)
        chosen = None
        if width > 1:
            tokens = torch.tensor([list(candidate_ids[i]) + [0] * (width - len(candidate_ids[i])) for i in chunk])
            cache = copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(len(chunk))
            # right padding needs no mask: a real token never attends to the padding that comes after it
            logits = model(input_ids=tokens[:, :-1], past_key_values=cache, use_cache=True).logits
            # log-prob of the chosen token = its logit - logsumexp; avoids a second vocabulary-sized array
            chosen = (torch.gather(logits, 2, tokens[:, 1:, None])[..., 0] - torch.logsumexp(logits, dim=-1)).numpy()
            del logits, cache
        for row, i in enumerate(chunk):
            length = len(candidate_ids[i])
            scores[i] = first[candidate_ids[i][0]] + (chosen[row, :length - 1].sum() if length > 1 else 0.0)
    return scores


@torch.no_grad()
def slow_score(model, prefix_ids, candidate_ids):
    """The obvious way, for checking: run basket + item as one sequence and add up the item's token log-probs."""
    ids = torch.tensor([list(prefix_ids) + list(candidate_ids)])
    logp = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
    return sum(float(logp[0, len(prefix_ids) + k - 1, t]) for k, t in enumerate(candidate_ids))
