# Teach a Language Model What Sells Together

My work for the Learn Fast / askThili drill. Phase 1 of an LLM recommender in miniature: build an invented Indian grocery
shop with itemsets planted in 100,000 baskets, mine the exact answer with Apriori, then continue pretraining a small open
language model (SmolLM2-135M) on the baskets and score what it learned against the exact answer and a counting baseline.

Runs in the `lfretail` workbench (`docker compose -f lfretail/docker-compose.yml up`). Sections 8–11 use the PyTorch
path on CPU (Windows), not MLX, so timings and some numbers differ from the course.

## Progress

| # | Section | Status | Date | Notebook |
|---|---------|--------|------|----------|
| 0 | Teach a Language Model What Sells Together | ✅ | 2026-11-14 | 00_intro.ipynb |
| 1 | A Shop's Catalogue | ✅ | 2026-11-14 | 01_catalogue.ipynb |
| 2 | How Much Each Item Sells | ✅ | 2026-11-14 | 02_popularity.ipynb |
| 3 | Background Baskets | ✅ | 2026-11-14 | 03_background_baskets.ipynb |
| 4 | Plant One Itemset | ✅ | 2026-11-14 | 04_plant_one_itemset.ipynb |
| 5 | Plant Many | ✅ | 2026-11-14 | 05_plant_many.ipynb |
| 6 | The Exact Answer | ✅ | 2026-11-14 | 06_exact_answer.ipynb |
| 7 | Verbalize a Basket | ✅ | 2026-11-14 | 07_verbalize.ipynb |
| 8 | The Off-the-Shelf Baseline | ✅ | 2026-11-14 | 08_baseline.ipynb |
| 9 | Continued Pretraining | ✅ | 2026-11-15 | 09_continued_pretraining.ipynb |
| 10 | Probe the Model | ✅ | 2026-11-15 | 10_probe.ipynb |
| 11 | From Itemsets to Cross-sell and Upsell | ✅ | 2026-11-15 | 11_cross_sell.ipynb |

See LOG.md for what each section found and what I learned.
