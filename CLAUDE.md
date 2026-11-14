# CLAUDE.md — Learn Fast drill: "Teach a Language Model What Sells Together"

I'm Dakshitha. This repo is my template copy of `thili-ai/learnfast-workbench`, and it is what I submit.
The drill has 12 sections (0–11). The full text of each is in `course/`. Build them in order, ONE AT A TIME,
and stop after each one so I can mark it complete on askThili. My commit history is assessed, so make one clean
commit per section.

## The 12 sections, in order
| # | Title | Course text | Notebook |
|---|-------|-------------|----------|
| 0 | Teach a Language Model What Sells Together (intro) | course/00_intro.txt | work/00_intro.ipynb |
| 1 | A Shop's Catalogue | course/01_catalogue.txt | work/01_catalogue.ipynb (already exists — run and verify it) |
| 2 | How Much Each Item Sells | course/02_popularity.txt | work/02_popularity.ipynb |
| 3 | Background Baskets | course/03_background_baskets.txt | work/03_background_baskets.ipynb |
| 4 | Plant One Itemset | course/04_plant_one_itemset.txt | work/04_plant_one_itemset.ipynb |
| 5 | Plant Many | course/05_plant_many.txt | work/05_plant_many.ipynb |
| 6 | The Exact Answer | course/06_exact_answer.txt | work/06_exact_answer.ipynb |
| 7 | Verbalize a Basket | course/07_verbalize.txt | work/07_verbalize.ipynb |
| 8 | The Off-the-Shelf Baseline | course/08_baseline.txt | work/08_baseline.ipynb |
| 9 | Continued Pretraining | course/09_continued_pretraining.txt | work/09_continued_pretraining.ipynb |
| 10 | Probe the Model | course/10_probe.txt | work/10_probe.ipynb |
| 11 | From Itemsets to Cross-sell and Upsell | course/11_cross_sell.txt | work/11_cross_sell.ipynb |

Priority: sections 0–3 first (these are the ones released to me so far). Do not start 4+ until I say so.

## Environment
- All code runs inside the Docker workbench, never my Windows Python.
  - Start (usually already running in another window): `docker compose -f lfretail/docker-compose.yml up`
  - Container: `lfretail-workbench`. Jupyter: http://127.0.0.1:8889/?token=learnfast
  - The course package `thilimarketbasket` exists only inside the container.
- Find where `work/` is mounted first: `docker exec lfretail-workbench ls /home/learnfast` (expect `work`).
  If it is somewhere else, adjust the commands below and tell me.
- Run a notebook (keeps outputs in the file, which is what I submit):
  `docker exec -w /home/learnfast lfretail-workbench jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 work/<file>.ipynb`
- Quick checks: `docker exec -w /home/learnfast lfretail-workbench python -c "..."`
- Never `pip install` on Windows. If the container lacks a library, tell me; the fix is a line in
  `lfretail/requirements.txt` + `docker compose -f lfretail/docker-compose.yml build`.

## One-time setup (before section 0)
1. Add these lines to the repo-root `.gitignore` (create it if missing) and commit as "Ignore course text and model checkpoints":
   ```
   course/
   work/checkpoints/
   *.safetensors
   ```
   `course/` is askThili's lesson text and my repo is public — never commit it. Checkpoints are hundreds of MB.
2. Commit `CLAUDE.md`, `work/README.md`, `work/LOG.md` in that same setup commit.

## How to build each section
1. Read the section's text in `course/`. Build the notebook from it: the course's code in small cells, short markdown
   notes in plain words between them, in my own wording (don't paste the lesson prose). Keep the course's
   function names (`my_...`), seeds and parameters so results match.
2. Execute it in the container until it runs top to bottom.
3. Every "check against the reference" cell must pass (`True` / `allclose` / "same numbers"). If one fails, explain
   why before changing anything.
4. Compare the key printed numbers with those in the course text; list any that differ.
5. End the notebook with a `## Test yourself` markdown cell (the 3 questions with short answers, see step 8), then
   `## My notes`, which you write yourself (see below).
6. Update `work/LOG.md` (date, what was built, key numbers, `- Learned: ...`) and tick the section in
   the progress table of `work/README.md` (⬜ → ✅, add the date).
   Write the `Learned:` line and the `## My notes` cell yourself (not "(Dakshitha to fill)"): first person, plain and
   short (1–2 lines), as a student would — the one key idea from the section, plus anything surprising.
7. Commit `work/` (only) and push:
   `git add work/ && git commit -m "Section N: <title> — <one-line result>" && git push`
8. STOP and print exactly this block, then wait for me to say `next`:
   ```
   ✅ SECTION N COMPLETE — <title>
   Mark it complete on askThili.
   Result: <key check + 1–2 key numbers>
   Summary: <5–8 plain-English lines: what it does and why>
   Test yourself: <3 questions>
   Answers: <a short plain-English answer (2–4 lines) to each of the 3 questions>
   Try it: <one change to make and what to expect>
   ```
   The same 3 questions and answers also go at the end of the section's notebook, as a markdown cell
   `## Test yourself` just above `## My notes`.
9. When I say `next`: first, if I edited any notebook notes or LOG.md, commit them as
   `Section N: my notes` and push. Then start the next section.

## Sections 8–11 on Windows (no Apple Mac)
The course runs these with MLX, which only works on Apple silicon. Here:
- Use the PyTorch code in each section's "Appendix: ... without MLX" instead (CPU-only PyTorch is in the image;
  check with `python -c "import torch, transformers"` and tell me if missing). The appendix code is untested by the
  course — fix small things as needed and say what you changed.
- Some `thilimarketbasket` modules (`evaluate`, `pretrain`, `probe`) may import MLX at the top and fail on Linux.
  If so, import what works; for the rest, read the package source inside the container
  (`python -c "import thilimarketbasket, os; print(os.path.dirname(thilimarketbasket.__file__))"`) and write
  PyTorch / pure-Python equivalents in `work/mybasket/`. Do not edit the installed package.
- Save checkpoints, scores and samples under `work/checkpoints/` (ignored by git) so they survive restarts.
- CPU is MUCH slower than the course's M3 timings (scoring 200×1,000 candidates, 800 training steps, sampling
  10,000 baskets). Before any step, time a small slice, estimate the full run, and if it's over ~30 minutes STOP
  and ask me: run it in full (I may leave it overnight), or shrink it (fewer cases/steps/samples). If we shrink,
  write the change clearly in the notebook and LOG.md, and keep the ceiling/baseline at the same size so the
  comparison stays fair. Our numbers will differ from the course's; that's expected — report ours honestly.
- Docker needs ≥ 8 GB memory for training. If the kernel dies, tell me to raise it.

## Don'ts
- Don't edit anything outside `work/`, `.gitignore`, `CLAUDE.md` unless I ask.
- Don't commit `course/` or checkpoints.
- Don't move on to the next section without my `next`.
