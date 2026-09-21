# Bayesian IPO Study

This project studies whether the first trading days of an IPO contain useful information about its subsequent performance through day 60. It builds an event-time dataset from IPO prices, trading volume and benchmark returns, then uses Bayesian models to produce full predictive distributions rather than single return estimates.

The analysis separates the initial offer-to-first-close move from aftermarket performance and compares models that progressively add IPO size, early price behaviour, trading intensity and realised uncertainty. Model quality is assessed chronologically on later IPO cohorts using out-of-sample scoring and calibration tests.

## Quick start

Clone the **whole repository**. The notebook is only the presentation layer; it deliberately depends on the versioned helpers, frozen data and selected model traces beside it.

```bash
git clone <repository-url>
cd bayesian_ipo_study

python3 -m venv .venv
source .venv/bin/activate            
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m jupyter lab bayesian_ipo_study.ipynb
```

Python 3.13 is recommended because it is the version used to create the frozen research snapshot.

For a quick completeness check before opening Jupyter:

```bash
python scripts/validate_release.py
```

To verify the entire notebook non-interactively:

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute bayesian_ipo_study.ipynb \
  --output bayesian_ipo_study.executed.ipynb \
  --ExecutePreprocessor.timeout=300
```

## Portable path handling

There are no hard-coded paths to the folder in which this project was originally developed. The first notebook cell finds the repository from the current directory and its parents, so the repository can be moved or cloned anywhere.

Normally, launch Jupyter from the repository root as shown above. If the Jupyter server must be started from an unrelated directory, set the repository explicitly before launching it:

```bash
export BAYESIAN_IPO_ROOT=/absolute/path/to/bayesian_ipo_study
python -m jupyter lab /absolute/path/to/bayesian_ipo_study/bayesian_ipo_study.ipynb
```

Moving only `bayesian_ipo_study.ipynb` will not work because the notebook needs the rest of the repository. If required files are absent, the bootstrap cell now reports which files are missing and how to fix the launch context.

## Repository layout

```text
bayesian_ipo_study.ipynb  Main narrative and executable analysis
data/processed/           Frozen derived tables and event panel
models/                   Selected posterior traces, registry and draws
src/                      Data, model, prediction and plotting helpers
scripts/validate_release.py  Fast completeness and loadability check
requirements.txt          Reproducible Python environment
```

The repository is an offline replay of the frozen 7 September 2026 research snapshot. It does not download market data or rerun the original model-selection pipeline. Generated figures are written to `reports/tape_figures/` and are ignored by Git.

GitHub displays notebooks as static documents. Clone the repository and follow the quick-start commands for live execution. All included artifacts fit in normal Git storage; Git LFS is not required.

## Scope

This is a research and educational artifact, not investment advice. The notebook documents material limitations, including final-cohort interval undercoverage, selected-sample effects and the small number of IPOs above $1 billion.
