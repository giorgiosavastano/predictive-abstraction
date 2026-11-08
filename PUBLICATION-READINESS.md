# Publication readiness — 8 November 2026

Publication-readiness work is on `chore/publication-readiness`. The essay is ready to share after merging and checking the deployment; the manuscript remains a working draft. This report records the local audit before the subsequently authorized feature-branch commit and PR. GitHub settings were read without modification; no release or website submission was made.

## Branch protection: active, with two gaps

Inspected the ruleset list, [ruleset 26795309](https://github.com/giorgiosavastano/predictive-abstraction/rules/26795309), effective `main` rules, classic protection, Pages settings, recent check runs and releases through authenticated, read-only GitHub API calls.

| Setting | Observed configuration |
| :-- | :-- |
| Enforcement and target | Active; targets `~DEFAULT_BRANCH`, currently `main`; no exclusions |
| Force pushes and deletion | Blocked by `non_fast_forward` and `deletion` rules, subject to bypass |
| Pull requests | Required, with zero approving reviews; code-owner and last-push approval requirements disabled |
| Required CI checks | None |
| Bypass | Repository administrators (`RepositoryRole` 5) can bypass **always**; the current maintainer can do so |
| Classic branch protection | Endpoint returns 404; the active ruleset still protects the branch |
| Pages | HTTPS enforced; branch deployment from `main:/docs`; latest inspected deployment succeeded |

**Recommended manual changes:** keep the rule active and the PR requirement at **zero approvals**; remove the administrator entry from the bypass list; enable **Require status checks to pass before merging**, adding **`test`**, from **GitHub Actions** (app ID 15368). The workflow is called `Tests`, but its actual check name is `test`, confirmed from completed runs. Requiring an up-to-date branch is optional for this single-maintainer project. No extra reviewer is needed to merge your own passing PR.

The Copilot extra-approval flag adds no requirement when the approval count is zero ([API reference](https://docs.github.com/en/rest/repos/rules)). A “For pull requests only” bypass would prevent direct pushes but still allow bypassing checks; removing bypass is stronger ([GitHub documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)).

Do **not** require the observed Pages checks `build`, `deploy` or `report-build-status` for PR merging: the branch-based Pages workflow deploys after merging. The current branch rules do not restrict release tags. No workflow conflict was found.

## Scientific corrections

The contradiction remained in the README and both manuscript versions: some passages denied measuring closure, while others inferred approximately autonomous, Markov macroscopic dynamics. These now consistently describe Experiment 2b as an **observational conditional-mean closure diagnostic**.

| Concept | What this project establishes |
| :-- | :-- |
| Useful engineered representation | Lower global prediction error for the tested learner and resource settings; reversed ranking for the local learning control |
| Observational predictive sufficiency | Full future-distribution sufficiency is not measured |
| Approximate closure | Evidence for agreement in conditional means of the measured global observables, over sampled configurations and horizons |
| Markov dynamics | Not established: one previous-summary refinement and conditional means cannot test independence from the entire past |
| Causal emergence | Not tested: no intervention-based comparison |

The [saved summary](results/experiment2b/summary/exp2b_summary.csv) confirms **0.918 ± 0.004**, versus **0.266** for random-3 and **0.572** for initial disorder: 10 seeds, 10,000 configurations per seed sampled after two recovery sweeps, sixteen futures each, twelve further sweeps. Raw and one-step-history refinements add **0.0103** and **0.0015** over random refinement. The small nonzero history effect and finite matching do not establish a Markov law; full distributions were not compared. See also the [macrostate literature](https://arxiv.org/abs/cond-mat/0303625).

Removed claims of measured local-target closure, necessary reconstruction of these exact features, and rigorous matching bounds. Corrected “within 10%” to about **11% above the best medium model** (0.4052195 versus 0.3657011). Qualified universal random-projection claims and fixed-time-training descriptions. Numerical records and tables are unchanged.

## Changes and preservation

- Updated `README.md`, `docs/manuscript-draft.md`, `paper/main.tex`, `docs/methods.md`, `docs/research-vision.md`, `docs/next-experiments.md` and the results README files. Updated the experiment map and the LaTeX figure-provenance note. The module docstring in `experiment2b.py` now states the same scope; its executable AST is unchanged.
- Added [CONTRIBUTING.md](CONTRIBUTING.md) with verified setup, test and CLI commands, reproducibility reports, research controls and PR guidance. CI now installs with `uv sync --locked`; the `test` job name is unchanged.
- Kept the essay's opening, first four sections and existing explanations. Added a short results overview, a limited closure qualification and the simulator diagram before the results. The original `docs/index_draft.md` was preserved during the audit and later removed from the current tree at the author's request; it remains in Git history.
- Added [sorting-experiment.svg](docs/figures/sorting-experiment.svg), about 25 KB, and [its generator](scripts/render_simulator.py). It uses `make_episodes` and `_sweep`, data seed 314159 and future seed 20261108. The close-up is replayed against the actual first sweep and RNG state; permutation, feature-count and disorder checks pass. The SVG has a title, description, high-contrast labels and a hatched fault interval. It scales with the page and opens separately for zooming.
- Changed the closure plot's “lower-bound estimate” label to “matching-limited estimate” and regenerated only its blog PNG/PDF. All plotted curves and confidence-band coordinates match the previous plotting function on the correctly filtered saved closure rows. All other existing figures are unchanged.
- Added explicit Pages URL/base URL, author/language, a sharing image, large-image card and supported `jekyll-seo-tag`/`jekyll-sitemap` configuration. The existing theme already invokes the SEO tag; no duplicate metadata or new framework was added. The local build produces a canonical URL, Open Graph image and image alternative text, sitemap and robots file. Corrected eleven Markdown manuscript image links to the same figures on the public repository, because the old relative paths fell outside the Pages source folder.
- Added the essay URL, software type and scope to `CITATION.cff`; kept version 0.1.0 and left the release date unset pending the release. No DOI, arXiv ID or publication citation was invented. `pyproject.toml`, `uv.lock`, the MIT license and bibliography are unchanged.

The original personal draft was subsequently removed at the author's request. Its earlier versions remain in Git history. The editorial report and obsolete blog pointer remain tracked and excluded from **Pages**. Exclusion does not make them private or remove them from GitHub/history; `.gitignore` would not untrack them either. No Git history was rewritten. This audit supersedes the older report’s outstanding manuscript suggestion.

Scratch runs, environments, caches and root plot copies are ignored. Added local Jekyll/Sass and `.qodo` ignore rules. Published data, metadata, summaries and paper figures remain versioned and unchanged; no original work was deleted.

## Validation

- **23 tests pass on Python 3.13.12**, with the locked dependencies; the same suite also passed on 3.12.12 while resolving the original download failure. The final environment uses 3.13; the project pin is unchanged. Two warnings concern core-count detection and near-identical values in a statistics test fixture.
- A separate small CLI run completed under ignored `results/runs/publication-smoke/`. No published experiment campaign was rerun.
- Local production Jekyll **3.10.0**, theme **0.2.0**, SEO **2.8.0**, sitemap **1.4.0** and GitHub Pages Markdown versions build successfully. Build tools stay under `/tmp`; no Ruby dependencies were added to the repository. Local GitHub metadata values were supplied from the audit because the unauthenticated metadata request returned a gateway error. The actual Pages service supplies these values during deployment. Versions follow the [supported Pages dependencies](https://pages.github.com/versions/).
- Chrome checked a 390-pixel page viewport: no horizontal overflow; all five essay images load, have alternative text and fit. The SVG and updated closure figure were visually inspected. Built-page local links/assets resolve; all eleven corrected manuscript image URLs return HTTP 200 with image content.
- `CITATION.cff` validates against the official CFF 1.2.0 schema using temporary uv-managed validators. No project dependencies were changed.
- LaTeX environments, citation keys, cross-references and all eleven figure paths pass structural checks. **No TeX compiler is installed, so PDF compilation remains unverified.**
- Verified the saved numerical values, exact preservation of the essay opening and original draft, unchanged raw/summary data, unchanged experimental logic and clean whitespace checks. A credential-pattern scan found no matches in tracked or proposed files; no private profile document or authentication material was added.

## Manual publication steps and initial release

1. Review the feature-branch PR and adjust the two protection gaps above. Confirm the new `test` run passes before merging.
2. After merging, verify the real Pages deployment, the new diagram, sharing image, sitemap and research-note links. This audit did not deploy the local changes. Then the essay is ready to share, including on Hacker News if you choose.
3. For an initial **v0.1.0** research-software release, set `date-released` to the actual release date in a reviewed change, tag the reviewed commit and publish release notes manually. No existing GitHub releases were returned by the API. Include the source, lockfile, MIT license, citation file, saved numerical results, figure generators, essay and clearly labelled draft research note. A DOI/archive is optional; add an identifier only after one is actually issued.

Suggested release notes:

> Initial research-software release: reproducible stochastic local-sorting experiments comparing supplied task-relevant summaries, full states and random projections across ten seeds. Includes the local-task control, granularity sweep, observational conditional-mean closure diagnostic, saved results, tests and an accompanying essay. These experiments use engineered features and one learner family; they do not demonstrate autonomous abstraction discovery, Markov macroscopic laws or causal emergence. The manuscript is a working draft.

Optional GitHub enhancement: set the repository homepage to the essay URL. No repository setting, release or external submission was changed automatically.

## Before an academic preprint

Resolve the authorship placeholder, compile and inspect the LaTeX PDF, reconcile the final Markdown/LaTeX presentation and check the full bibliography and numerical tables. The manuscript's own research gate still calls for locality-aware raw-state baselines, tuning of the largest boosting models and horizon controls; these were not added in this cleanup. Obtain an independent review of the estimator and its assumptions before stronger sufficiency or autonomy claims. Discovery, transfer and adaptation remain future work, rather than prerequisites for an honestly scoped release of the present supplied-feature results.
