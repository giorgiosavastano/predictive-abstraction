# Contributing

Scientific questions, reproducibility reports, bug fixes and carefully designed experiments are welcome. This is a small independent research project; a clear explanation is more useful than a large patch.

## Getting started

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/giorgiosavastano/predictive-abstraction.git
cd predictive-abstraction
uv sync --locked
uv run pytest
uv run macro-experiment1 --quick
```

The project requests Python 3.13; uv can install it if needed. The quick experiment uses two seeds and writes to the ignored `results/quick/experiment1/` directory. It checks the workflow, rather than reproducing the full published estimates.

The CLI options define experimental configurations: run `uv run macro-experiment1 --help`, `uv run macro-experiment2 --help` or `uv run macro-experiment2b --help`. [Methods](docs/methods.md) documents the simulator, seeds, splits and features. [Results](results/README.md) maps the saved data; each experiment's `metadata.json` records its invocation and configuration. Use `results/runs/<your-run>/` for other scratch runs and pass it with `--output`.

## Reporting an issue

Please include the problem or research question, steps to reproduce, the random seed and configuration, expected and observed behaviour, and relevant software versions. A small example or exact command helps. Negative results and unsuccessful reproductions are useful too.

## Proposing an experiment

Explain the hypothesis, the prediction that could falsify it, the baselines and controls, and how the configuration and analysis can be reproduced. Account for data, tuning and computational budgets. Report negative findings and distinguish supplied features from learned abstractions. For substantial work, opening an issue first can avoid duplicated effort.

## Pull requests

Work on a feature branch, explain what changed and why, and run `uv run pytest`. Include relevant commands or checks in the PR description. Keep changes focused and preserve the author's voice when editing the essay.

Do not overwrite published raw records, summaries or figures as a side effect of a test run. Put new results in a separate directory and describe their provenance. If a published result needs correction, explain the discrepancy and preserve the evidence needed to understand it.
