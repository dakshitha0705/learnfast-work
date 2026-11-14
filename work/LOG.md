

## Section 0 — Teach a Language Model What Sells Together (2026-11-14)
- Built: `00_intro.ipynb` — support of an itemset, brute-force frequent itemsets on 6 baskets, combinatorial blow-up for 1,000 items.
- Key numbers (all match the course): tea 0.67, tea+sugar 0.50, bread+butter 0.33; frequent at ≥0.5: bread, milk, sugar, tea, sugar+tea; C(1000,5) = 8,250,291,250,200.
- Learned: 1. Why can’t tea+sugar have higher support than tea alone?
Every basket that holds tea and sugar also holds tea. So the tea+sugar baskets are a subset of the tea baskets, and a subset can’t be bigger than the whole. Adding an item to an itemset can only keep its support the same or lower it.

2. If a 3-item set is frequent, what about its 2-item subsets?
Each of them must be frequent too. By the same logic as answer 1, every pair inside the triple has support at least as high as the triple, so if the triple clears the threshold, every pair does. This is exactly what Apriori uses: if a pair isn’t frequent, it never builds any triple containing that pair.

3. Why an invented shop instead of real data?
There are two reasons:

You know the right answer. We plant itemsets at exact frequencies, so we can check precisely what the model found and what it missed. Real sales data has no answer key.
The model can’t cheat. A model trained on internet text already “knows” that tea goes with sugar. With made-up brands and deliberately odd combinations, anything it gets right must have come from our baskets, not from what it read before.

## Section 1 — A Shop's Catalogue (2026-11-14)
- Built: `01_catalogue.ipynb` (already scaffolded; ran and verified) — `all_skus` crosses products × picked brands × pack sizes; `my_make_catalog` keeps one SKU per product, then fills to 1,000 at random (seed 0).
- Key numbers (all match the course): 127 products in 12 categories; pool of 1,127 SKUs at 4 brands/product; 1,000 in the catalogue; SKUs per product min 1 / median 8 / max 12; 28 loose SKUs; 116 sachet/Rs-pack SKUs.
- Reference check: `identical, item for item: True`.
- Learned: (Dakshitha to fill)
