Here's the plan. Steps are ordered so each one builds on the last — don't reorder them, and don't skip ahead.

## Step 1 — Lock down the methodology before touching code

Decide and write down (in a scratch file or at the top of a notebook) the answers to these, because they dictate everything else:

- **Repetitions per cell:** 3 runs, report median. Non-negotiable.
- **Sampling strategy:** `df.sample(frac=..., random_state=42)` — same seed across fractions so subsets are nested and comparable. No `.iloc`.
- **What counts as a "phase":** feature engineering (train), feature engineering (test), single-fit training, inference. That's it. CV searches are a separate "context" measurement, not part of the sweep.
- **What hyperparameters are used in single fits:** the best params found by the full CV search (run once per model, on the full dataset, on CPU). Both CPU and GPU single fits use these same params.
- **Timing instrument:** `time.perf_counter()` only. No `%%timeit`.
- **Correctness tolerance:** AUC within 0.005, feature-matrix max relative diff under 1e-4 (except avg_word_length: under 2%).

This document becomes the "Methodology" section of your report. Write it now, not later.

## Step 2 — Redesign the notebook split

Three artifacts, strict separation of concerns:

- **CPU_Baseline.ipynb** — runs on Colab CPU runtime only. Loads data, feature engineers, runs CV once to find best hyperparams (timed, used as a context number), runs the sweep. Writes `results/timings_cpu.csv` and `results/best_params.json`.
- **GPU_Solution.ipynb** — runs on Colab GPU runtime only. Loads data, loads best params from the CPU notebook's JSON, GPU-feature-engineers, runs GPU CV once for context, runs the GPU sweep. Writes `results/timings_gpu.csv`.
- **Analysis.ipynb** (small, ~10 cells) — loads both CSVs, computes medians and speedups, produces the tables and plots that go in the report.

The reason for three notebooks: each one runs in a single runtime, the comparison happens offline, and if you need to re-run just the GPU benchmarks you don't disturb the CPU numbers.

## Step 3 — Standardize the timing/logging pattern

Before writing any experiment code, build one small utility that every timing call uses. Something like a helper that takes a label, a callable, and a run index, times it, and appends a row to the CSV. Use it everywhere. Consistency matters more than cleverness here — when the analysis notebook joins the two CSVs on `(device, fraction, phase, run_idx)`, everything needs to line up.

Columns: `device, fraction, n_rows, phase, run_idx, seconds, notes`.

## Step 4 — Strip the CPU notebook

From the current CPU notebook, remove:

- Everything that isn't load, FE, train, predict, sweep.
- The narrative rationale for why LR vs XGB, the 40-row feature-description table, etc. Those belonged in the original course submission — they don't belong here. A grader reading this notebook wants to see what's being benchmarked, not an essay.
- Threshold-sweep-for-F1 logic. You don't need optimal thresholds to measure speedup. Just time `fit` and `predict_proba`.

Keep:

- Load, train/test split (fixed seed), `feature_engineering()` function, one LR CV (once, for best params), one XGB CV (once, for best params), the sweep loop (3 repetitions × 5 fractions), write-to-CSV.

## Step 5 — Rebuild the sweep loop (CPU first)

Single sweep, clean logic:

```
for run_idx in range(3):
    for frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
        sample data with frac, random_state=42
        time FE_train, log
        time FE_test (using fitted encoders), log
        time LR fit (best params), log
        time LR predict_proba, log
        time XGB fit (best params), log
        time XGB predict_proba, log
```

Run it. Save CSV. Verify the file has 3 × 5 × 6 = 90 rows. This is your CPU baseline, locked.

## Step 6 — Strip the GPU notebook

Remove the entire CPU-side sweep that currently lives inside it. Remove the big 2.5-page discussion — it will be rewritten after experiments run.

Keep:

- RAPIDS install + restart guard + imports + GPU warm-up.
- `feature_engineering_gpu()` function.
- Correctness verification block (this is already good — keep it).
- Load `best_params.json` from the CPU run.
- GPU CV runs (once each, for context).
- GPU-only sweep loop (mirror of CPU's structure).
- Write `timings_gpu.csv`.

No CPU operations in this notebook. That's the one rule.

## Step 7 — Run the CPU sweep, then the GPU sweep

Colab sessions die. Save CSVs after every run iteration, not at the end. Commit them to git as you go.

Expected wall time: CPU sweep is probably 30–60 min (depends on XGB CV); GPU sweep probably 10–20 min. Budget accordingly.

## Step 8 — Build the analysis notebook

Load both CSVs. Group by `(device, fraction, phase)`, take median. Pivot so you have CPU-time and GPU-time side-by-side per phase per fraction. Compute `speedup = cpu_median / gpu_median`.

Produce:

- One main speedup table (5 fractions × 4 phases).
- One speedup-vs-fraction plot (one line per phase, dashed horizontal line at 1.0 for break-even).
- One stacked bar showing total pipeline time CPU vs GPU at each fraction.
- The full-CV context number (once, at 100%) mentioned as a separate bullet.

This notebook is short and mechanical. It exists so the numbers that go in the report come from a single, auditable source.

## Step 9 — Write the report from the numbers

Only now. Four pages, written in this order:

1. Problem + dataset + what you ported (half a page).
2. Methodology — the document from Step 1 (half a page).
3. Results — one table, one plot, maybe a second plot (one page).
4. Discussion — what actually happened, with honest explanations of overheads where speedup is small/negative, and where it's large (one page).
5. AI tools reflection (one page).

The discussion section writes itself from the data. Do not pre-write it.

## Step 10 — Sanity checks before submission

- Both notebooks execute top-to-bottom on a fresh Colab runtime without errors.
- `results/` folder is present in the zip with the CSVs.
- `best_params.json` is present so the GPU notebook can actually find it.
- Report is exactly 4 pages, not 5.
- Zip file is actually a zip, not a rar or a folder.
