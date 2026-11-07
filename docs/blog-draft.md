# Blog post

The essay now lives in [`index.md`](index.md), which GitHub Pages serves as the site's front page when Pages is configured to deploy from the `docs/` folder. Its figures are in [`figures/`](figures/) and are regenerated from the experiment summaries with:

```bash
uv run macro-analyze --experiment1 results/experiment1 --experiment2 results/experiment2 --blog-figures docs/figures
```
