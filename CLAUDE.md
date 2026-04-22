# Assignment 1 — GPU-Accelerated Data Analysis

Course: Engineering of Data Analysis (2779), T4 2025–26, NOVA SBE.
Deadline: 27 April 2026, 23:59 Lisbon time.
Target grade: 20/20.

## What the assignment asks for

Take an existing data analysis / ML project, port the compute-heavy parts to a GPU-accelerated framework (RAPIDS cuDF / cuML, CuPy, or equivalent), and produce a systematic performance comparison between the CPU baseline and the GPU version.

The solution must run on Google Colab (GPU runtime).

## Deliverables

A single `.zip` containing:
- `report.pdf` — max 4 pages. Pages 1–3 are the performance analysis (problem, solution design, results, discussion). Page 4 is a reflection on how AI tools were used.
- All `.ipynb` notebooks needed to reproduce the experiments (CPU baseline + GPU solution).

## Required performance analysis

- **Dataset-size sweep:** measure wall-clock time at several dataset sizes (e.g. 10%, 25%, 50%, 75%, 100%) and report the speedup at each scale.
- **Training vs inference:** if the project has both phases, report speedups separately.
- Present results with tables and/or plots.
- Discuss *why* the observed speedups happen (or don't), including overheads like host↔device transfer, memory allocation, and kernel launch latency.

## Grading weights (optimize effort accordingly)

- GPU solution — correctness and completeness: **25%**
- Critical evaluation — performance analysis and discussion: **55%**
- Report quality and AI tools reflection: **20%**

The middle bucket is where this assignment is won or lost.

## AI tools reflection (page 4 of report)

Page 4 must cover:
- Which tools and model versions were used.
- Agentic vs chat-based usage.
- Development strategy: iterative / design-first / interview-driven / debugging-led / benchmark-guided.
- Honest reflection on what worked, what didn't, how AI shaped the final solution.

Keep a lightweight log as we work so this page is grounded in what actually happened.

## How we're working

- Source lives in a GitHub repo (this repo).
- Execution happens on Google Colab with GPU runtime enabled.
- Claude Code is used from the terminal for development.
- Before non-trivial changes to notebooks or code, propose the change in plain English and wait for approval.
- Commit to git at meaningful checkpoints with clear messages.
