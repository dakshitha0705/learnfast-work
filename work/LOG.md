# Log

One entry per section: date, what was built, key numbers, what I learned.

## Section 0 — Teach a Language Model What Sells Together (2026-11-14)
- Built: `00_intro.ipynb` — support of an itemset, brute-force frequent itemsets on 6 baskets, combinatorial blow-up for 1,000 items.
- Key numbers (all match the course): tea 0.67, tea+sugar 0.50, bread+butter 0.33; frequent at ≥0.5: bread, milk, sugar, tea, sugar+tea; C(1000,5) = 8,250,291,250,200.
- Learned: Support can only stay the same or drop as you add items to a set, and that one rule is what lets Apriori skip most of the trillions of combinations.

## Section 1 — A Shop's Catalogue (2026-11-14)
- Built: `01_catalogue.ipynb` (already scaffolded; ran and verified) — `all_skus` crosses products × picked brands × pack sizes; `my_make_catalog` keeps one SKU per product, then fills to 1,000 at random (seed 0).
- Key numbers (all match the course): 127 products in 12 categories; pool of 1,127 SKUs at 4 brands/product; 1,000 in the catalogue; SKUs per product min 1 / median 8 / max 12; 28 loose SKUs; 116 sachet/Rs-pack SKUs.
- Reference check: `identical, item for item: True`.
- Learned: The catalogue is the model's whole vocabulary, so it's built from SKUs with readable names, not IDs. I was surprised that the seed alone fixes which 1,000 of the 1,127 possible SKUs the shop stocks.

## Section 2 — How Much Each Item Sells (2026-11-14)
- Built: `02_popularity.ipynb` — `level_shares` (category × product × brand × size, each normalised over what is stocked) and `my_popularity` (one share per SKU, sums to 1).
- Key numbers (match the course): Kaveri Valley Toor Dal 1 kg = 0.0792 × 0.2899 × 0.3226 × 0.5455 = 0.40% of sales; best/slowest = 541:1; top 1% / 20% / 50% of SKUs make 12% / 64% / 88% of sales; 114 SKUs make half, 365 make 80%; 116 sachets = 11.8% of sales; doubling Surya Grains: 3.66% → 4.78%, Toor Dal stays 2.30%.
- Reference check: `same as the reference: True`, largest difference 0.0.
- Only difference from the course text: the order of exactly tied SKUs in the top-10/slowest-5 lists (Tomato/Potato swapped; 5th slowest shows Kids Korner Baby Cereal 400 g instead of Fresh Fort Phenyl 2 L — both 0.0032%). pandas' default sort isn't stable, so ties can come out in any order.
- Learned: A SKU's share is four fractions multiplied together, each normalised only over what's stocked, so adding brands splits Toor Dal's slice instead of growing it. Surprising: a sachet sells about like an average SKU; there are just a lot of them.

## Section 3 — Background Baskets (2026-11-14)
- Built: `03_background_baskets.ipynb` — `my_basket_sizes` (2 + negative binomial, mean 8, dispersion 4), `pick_products` (by product share, no repeats), `pick_sku`, quantities 80/14/4/2%, shuffled; `my_background_baskets`. Checked the Gumbel top-k shortcut against one-at-a-time drawing.
- Key numbers (all match the course): sizes mean 8.01 / median 7 / p95 15 / max 40; direct-SKU drawing gives 26% double-product baskets; Gumbel vs loop correlation 0.9989; mine vs reference (5k): 10.20 vs 10.20 items, 8.00 vs 7.96 distinct, 77% vs 77% with a repeat; 153 accidental pairs at ≥0.5% (top: Onion 1 kg + Potato 1 kg, 1.31%); lift median 1.12 overall vs 0.96 in same-size baskets; 0 / 156 / 2,825 accidental pairs at mean basket size 4 / 8 / 16.
- Reference check: statistics match the reference within noise (this section has no exact item-for-item check, since the draws differ).
- Learned: Even with nothing planted, 153 pairs reach 0.5% support just because both items are popular, so support alone can't tell a real pattern from luck. Surprising: lift isn't 1 for independent items here; mixing small and big baskets pushes it to about 1.12.

## Section 4 — Plant One Itemset (2026-11-14)
- Built: `04_plant_one_itemset.ipynb` — `my_plant_itemset`: count chance holders, pick random baskets to top up, swap out clashing SKUs of the same product, add with quantities, reshuffle; raises on impossible requests.
- Key numbers (all match the course): Chai Chaska Tea Powder 250 g + Chai Chaska Sugar 1 kg at 2%: 237 by chance, 1,763 added → exactly 2,000 (2.0000%); tea 3.95% → 5.64%, sugar 5.31% → 6.99%, lift 1.13 → 5.08; itemsets containing the pair at 0.1%: 1 → 21; rare surprising triple: 0 → 50 baskets.
- Reference check: `identical to the reference: True`.
- Learned: Planting is exact top-up: count what chance already gives, then add the pair to just enough random baskets to hit 2,000. Surprising: one planted pair created 20 extra frequent triples I never planted.

## Section 5 — Plant Many (2026-11-14)
- Built: `05_plant_many.ipynb` — `check_plan` (one SKU per product, no nesting, chance ≤30% of target), `my_plant_all` (largest first, `allow_excess=True`), `my_report` (asked-for vs realised support).
- Key numbers (all match the course): 32 itemsets, 79 distinct items, 24 shared; 29 of 32 within 1% of target; worst: Brew Bazaar Tea Powder 250 g + Chai Chaska Sugar 1 kg, 0.20% → 0.66% (+232%) because its items sit in two different 5% sets (both together in 0.269% of baskets); order: large-first 3 misses, rarest-first 6, as listed 3; 40% of baskets rewritten, 16% hold no planted item, 3.1 vs 1.4 planted items per basket; seeds 0–4: 3 / 0 / 2 / no valid plan / 0 misses.
- Reference checks: `identical to the reference: True`; my report's realised supports match `planted_report`: True. (`planted_report` returns a list of dicts, so I wrapped it in a DataFrame.)
- Learned: Planting 32 overlapping itemsets mostly lands exactly (29 of 32), but items shared between sets push some rare ones over target. Surprising: a 0.2% pair came out at 0.66% purely because its two items each sat in a different 5% set.

## Section 6 — The Exact Answer (2026-11-14)
- Built: `06_exact_answer.ipynb` — `plain_apriori` (join/prune/count by scanning), `bitsets` + `my_frequent_itemsets` (prefix join, one AND per candidate), `my_classify` (single / planted / part / extension / chance).
- Key numbers (all counts match the course): 2,000 baskets at 1%: 471 itemsets, plain 2.8 s vs bitset 0.014 s; full truth at 0.05%: 56,470 itemsets (977 / 18,237 / 16,528 / 13,052 / 7,676 by size 1–5) in 4.5 s (course 2.9 s); 491 / 1,747 / 8,317 / 19,766 / 56,470 at 1% / 0.5% / 0.2% / 0.1% / 0.05%; labels: 977 single, 32 planted, 303 part, 31,902 extension, 23,256 chance (planted = 0.06%); 1,128 rules at confidence ≥ 0.3; ≤3 items at 0.1%: 13,650 itemsets, 12 planted.
- Reference checks: identical to the reference miner: True; identical to the dataset's truth: True; same labels as the reference: True.
- Only differences from the course: timings (CPU in Docker is a bit slower).
- Learned: Apriori stays exact while skipping almost everything, and the bitset trick makes 100,000 baskets take seconds instead of hours. Surprising: only 32 of the 56,470 true itemsets are the ones we planted; the rest are extensions and chance.

## Section 7 — Verbalize a Basket (2026-11-14)
- Built: `07_verbalize.ipynb` — `my_verbalize_basket` ("Basket: a; b; c", each item once, seeded shuffle, max 30 items), SmolLM2-135M token lengths, 95/5 train/validation split (seed 0).
- Key numbers (all match the course): sorted order 100% "next is later in alphabet" vs shuffled 50%; tokens per basket mean 108 / median 102 / p95 198 / p99 250 / max 356, ~11.3 per item; context 128 / 192 / 256 cuts 30.1% / 6.1% / 0.8% → chose 256; split 95,002 train / 4,998 held out; separators all 107.8 tokens.
- Reference check: `matches the reference: True` (2,000 baskets).
- Learned: Writing a basket as text is a real design choice: each item once, reshuffled every time, so the model can only learn which items are together. Surprising: an average basket is already about 108 tokens, around 11 per item, because invented brand names split into tiny pieces.

## Section 8 — The Off-the-Shelf Baseline (2026-11-14)
- Built: `08_baseline.ipynb` — `my_rank_of`, `my_summary`, 200 held-out cases (one item hidden), random and popularity baselines, SmolLM2-135M scoring of all 1,000 items per case, bootstrap ranges, no-basket and per-token checks.
- CPU / no-MLX changes: wrote `work/mybasket/score.py` (PyTorch port of the MLX `score_candidates`: run the basket once, copy its key/value cache with `DynamicCache.batch_repeat_interleave`, score only the candidate tokens, float32). The course's appendix re-reads the whole basket per batch, which is far too slow on CPU. Fixes: `dtype=` instead of the deprecated `torch_dtype=`; batches of 100 and `logit − logsumexp` instead of a full `log_softmax` (my first try at 250 ran out of memory in Docker). The 200-case scoring ran as a background job (`python -m mybasket.jobs score ...`, 46 min) and the notebook loads the saved scores.
- Reduction (to save time): skipped the "Try it" reshuffle of 40 cases (~9 min more of scoring). The main test is the full 200 cases, as in the course.
- Key numbers vs the course: popularity MRR 0.052 (same), hits@10 0.12, median rank 105.5 (same); off-the-shelf model MRR **0.021** (course 0.020), hits@50 0.195 (0.19), median rank 342 (343); 95% ranges popularity 0.037–0.069 (same), model 0.013–0.030 (0.013–0.029); no-basket MRR 0.029 (0.031); length/score correlation −0.60 (same); score per token MRR 0.014 (same); first case: eggs at rank 699 (701). Small differences come from float32 on CPU vs half precision on MLX. Timing: 429 ms per candidate the slow way and 25 s per case cached (course on M3: 30 ms and 2.6 s).
- Checks: fast and slow scoring agree exactly (−29.38 / −42.36 / −61.07); the saved scores reproduce the live case-0 scores.
- Learned: An untrained model that never saw our shop scores MRR 0.021, well below just suggesting best sellers (0.052), and it does no better when shown the basket. Surprising: the obvious one-candidate-at-a-time scoring would take 7 minutes per basket on my CPU; reusing the basket's cache brings it to about 25 seconds.
