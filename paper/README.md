# LaTeX working paper

`main.tex`, `references.bib` and `figures/*.png` form a self-contained manuscript source directory. The figures are copies of `results/experiment1/figures/*.png`, `results/experiment2/figures/*.png` and `results/experiment2b/figures/*.png` (written by `uv run macro-analyze`) plus the original single-seed `learning_curves.png`.

To compile with a standard TeX installation:

```bash
cd paper
latexmk -pdf main.tex
```

No TeX distribution was available in the environment where the current revision was written, so the source has been checked for structure (balanced environments, defined citation keys, existing figure files) but not compiled. Numbers in the tables are copied from `results/experiment*/summary/tables.md`; regenerate those files before editing the tables.

**Status: research note with controlled multi-seed evidence; not submission-ready.** Before submission: replace the placeholder authorship, run the locality-aware raw-state baselines described in `docs/next-experiments.md`, and have the limitations section reviewed. The Markdown twin of this manuscript is `docs/manuscript-draft.md`.
