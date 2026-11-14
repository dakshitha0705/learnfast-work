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
- Learned: (Dakshitha to fill)
