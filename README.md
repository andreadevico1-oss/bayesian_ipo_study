# Bayesian IPO Study

This project asks a simple question:

> Can the first trading days of an IPO help predict its market-adjusted return through trading day 60?

The project studies IPO prices, trading volume, volatility, offer size and market returns. It uses Bayesian models, so its output is a range of possible future returns with probabilities—not one exact price target.

The main result is presented in [`bayesian_ipo_study.ipynb`](bayesian_ipo_study.ipynb). The repository also contains the code, frozen data and saved model results needed to run the notebook on another computer.

## What this repository contains

This is a self-contained, offline copy of the research as at **7 September 2026**. It contains:

- the full research notebook;
- a frozen sample of 903 usable IPOs;
- price, volume and market observations for the event-time analysis;
- saved Bayesian model draws;
- model predictions, scores and robustness checks; and
- Python code for loading data, making predictions and drawing charts.

It does **not** contain the original raw-data download pipeline. Running the notebook replays the published analysis from the frozen files in this repository. It does not download fresh market data or repeat the full model search.

## How the project works

The project follows four main steps:

1. **Build the IPO sample.** Check which IPOs have usable offer data, prices and trading volume.
2. **Create early-trading measures.** Measure early market-adjusted price performance, trading intensity and realised volatility.
3. **Compare Bayesian models.** Test models using only information that would have been available at each prediction date.
4. **Evaluate later IPOs.** Compare the models on later IPO cohorts using predictive accuracy and calibration.

The predicted outcome is the IPO's remaining market-adjusted log return from trading day 1, 5, 10 or 20 through day 60.

The model sequence is:

| Model | What it adds |
|---|---|
| M0 | A simple historical reference model |
| M1 | Early price performance |
| M2 | Trading intensity relative to IPO proceeds |
| M3 | Realised volatility as a guide to forecast uncertainty |

IPO size is also included in the selected specifications. The saved selection uses the growth-market benchmark and a Student-t likelihood, which allows more probability in the tails than a normal distribution.

## Quick start

### 1. Download the whole repository

Do not download or move only the notebook. The notebook needs the `src`, `data` and `models` folders.

```bash
git clone <repository-url>
cd bayesian_ipo_study
```

### 2. Create a Python environment

Python 3.13 is recommended because it was used to create this research snapshot.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Check that all files are present

```bash
python scripts/validate_release.py
```

A successful check reports 903 usable IPOs, 82,215 event-panel rows and 86 model-registry records.

### 4. Open the notebook

```bash
python -m jupyter lab bayesian_ipo_study.ipynb
```

Using `python -m jupyter` helps ensure that Jupyter uses the same Python environment in which the requirements were installed.

## Repository map

```text
bayesian_ipo_study/
├── bayesian_ipo_study.ipynb   Main analysis and written explanation
├── README.md                  Guide to the repository
├── requirements.txt           Exact Python package versions
├── data/processed/            Frozen data and analysis results
├── models/                    Saved model draws and model settings
├── src/                       Reusable Python code
└── scripts/                   Repository checks
```

The next sections explain every published file.

## Main files

| File | Purpose |
|---|---|
| [`bayesian_ipo_study.ipynb`](bayesian_ipo_study.ipynb) | The main research document. It explains the question, builds the analysis tables, compares the models, creates charts and discusses the results and limitations. Start here if you want to understand the study. |
| [`README.md`](README.md) | The guide you are reading. It explains how to install, run and navigate the project. |
| [`requirements.txt`](requirements.txt) | The exact package versions used by the notebook, including NumPy, pandas, PyMC, ArviZ, Matplotlib and Jupyter. |
| [`.gitignore`](.gitignore) | Keeps local environments, caches, generated figures and executed notebook copies out of Git. |

## Python code in `src`

| File | Purpose |
|---|---|
| [`src/tape_data.py`](src/tape_data.py) | Defines the project paths, loads the frozen IPO sample and event panel, and calculates model features and day-60 outcomes. It also contains the original data-building functions, although their upstream raw inputs are not part of this offline release. |
| [`src/tape_models.py`](src/tape_models.py) | Defines the Bayesian models, prepares model matrices, loads saved posterior draws, makes predictions and calculates model diagnostics and forecast scores. |
| [`src/tape_api.py`](src/tape_api.py) | Provides the higher-level prediction interface. It loads the frozen historical population and builds or updates a probability distribution for a new IPO. |
| [`src/tape_plots.py`](src/tape_plots.py) | Creates and saves the charts used in the notebook. Generated charts are written to `reports/tape_figures/`. |
| [`src/__init__.py`](src/__init__.py) | Marks `src` as a Python package and stores its version number. |

## Data files in `data/processed`

These are frozen research inputs and outputs. CSV files can be opened in a spreadsheet. The Parquet file is best read with pandas or another data-analysis tool.

| File | Purpose |
|---|---|
| [`tape_sample_audit.csv`](data/processed/tape_sample_audit.csv) | One row per IPO candidate. It records dates, offer size, data coverage, whether the IPO is usable and the reason for any exclusion. This is the main sample-audit file. |
| [`tape_event_panel.parquet`](data/processed/tape_event_panel.parquet) | The main event-time dataset. It contains daily prices, volume, benchmark returns, VIX and trading-intensity measures for each IPO. |
| [`tape_predictions.csv`](data/processed/tape_predictions.csv) | IPO-level model predictions and realised outcomes. It includes predictive intervals, probability of a positive return, loss probability and forecast-quality measures. |
| [`tape_scores.csv`](data/processed/tape_scores.csv) | Summary scores for each model, cutoff and validation phase. It includes log score, CRPS, error measures and interval coverage. |
| [`tape_coefficients.csv`](data/processed/tape_coefficients.csv) | Posterior summaries for model coefficients, including median estimates, uncertainty intervals and sign probabilities. |
| [`tape_pseudo_live.csv`](data/processed/tape_pseudo_live.csv) | Historical examples treated as if they were being observed live. It shows how predictions change as days 1, 5, 10 and 20 become available. |
| [`tape_benchmark_sensitivity.csv`](data/processed/tape_benchmark_sensitivity.csv) | Tests whether conclusions change when a different market benchmark is used. |
| [`tape_initial_return_validation.csv`](data/processed/tape_initial_return_validation.csv) | Checks offer prices and first-day closing prices against an independent source. |
| [`tape_offered_share_validation.csv`](data/processed/tape_offered_share_validation.csv) | Checks the relationship between gross IPO proceeds, offer price and shares offered using selected filing data. |
| [`tape_normalization_distributions.csv`](data/processed/tape_normalization_distributions.csv) | Day-by-day distributions used to study how trading intensity and volatility develop over an IPO's first trading days. |
| [`tape_normalization_models.csv`](data/processed/tape_normalization_models.csv) | Results from models that compare possible time-normalisation rules for early-trading measures. |
| [`tape_size_contrasts.csv`](data/processed/tape_size_contrasts.csv) | Compares predictive results across IPO-size thresholds. |
| [`tape_size_performance.csv`](data/processed/tape_size_performance.csv) | Reports forecast accuracy, probabilities and calibration separately for different IPO-size groups. |
| [`tape_survivorship_bounds.csv`](data/processed/tape_survivorship_bounds.csv) | Measures how missing outcomes could affect the conclusions by reporting identification bounds under different assumptions. |

## Model files in `models`

The `.npz` files contain compressed NumPy arrays of saved posterior draws. They allow the notebook to replay the selected Bayesian models without fitting every model again.

| File | Purpose |
|---|---|
| [`tape_selection.json`](models/tape_selection.json) | The final model-selection settings: benchmark, likelihood, cutoff dates, selected model at each cutoff and comparison results. |
| [`tape_registry.json`](models/tape_registry.json) | The full model catalogue. It records specifications, training periods, diagnostics and file references for all fitted candidates. |
| [`tape_runtime_versions.json`](models/tape_runtime_versions.json) | The Python, operating-system and package versions used when the release was created. |
| [`growth_2024_p1.0_M0_t10_student_bfb3915f5b7f05f0.npz`](models/growth_2024_p1.0_M0_t10_student_bfb3915f5b7f05f0.npz) | Saved development-period M0 draws for day 10. The notebook uses this model for the prior-predictive check. |
| [`growth_final_p1.0_M0_t1_student_569abdc3bc636ef9.npz`](models/growth_final_p1.0_M0_t1_student_569abdc3bc636ef9.npz) | Saved final M0 reference distribution for day 1. |
| [`growth_final_p1.0_M2_t1_student_ca2832a68966bcfc.npz`](models/growth_final_p1.0_M2_t1_student_ca2832a68966bcfc.npz) | Saved selected M2 posterior draws for day 1. |
| [`growth_final_p1.0_M3_t5_student_2e3a41139d5c7817.npz`](models/growth_final_p1.0_M3_t5_student_2e3a41139d5c7817.npz) | Saved selected M3 posterior draws for day 5. |
| [`growth_final_p1.0_M3_t10_student_c6e484f7c52f3059.npz`](models/growth_final_p1.0_M3_t10_student_c6e484f7c52f3059.npz) | Saved selected M3 posterior draws for day 10. |
| [`growth_final_p1.0_M3_t20_student_e3022542b90e8eda.npz`](models/growth_final_p1.0_M3_t20_student_e3022542b90e8eda.npz) | Saved selected M3 posterior draws for day 20. |
| [`tape_pseudo_live_draws.npz`](models/tape_pseudo_live_draws.npz) | Predictive draws behind the historical pseudo-live examples and density charts. |

The long model filenames are deliberate. They record the benchmark, sample phase, prior scale, model name, trading-day cutoff, likelihood and a short specification fingerprint.

## Scripts

| File | Purpose |
|---|---|
| [`scripts/validate_release.py`](scripts/validate_release.py) | Checks that the important data and model files exist, loads the frozen data and confirms that every selected model trace can be opened. This is the fastest way to test a fresh clone. |

## Generated files

Running the notebook creates `reports/tape_figures/` and saves the charts there. This folder is ignored by Git because the figures can be recreated from the notebook.

If you run the non-interactive notebook command below, it also creates `bayesian_ipo_study.executed.ipynb`. Executed copies are ignored by Git.

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute bayesian_ipo_study.ipynb \
  --output bayesian_ipo_study.executed.ipynb \
  --ExecutePreprocessor.timeout=300
```

## Using the project from another folder

The notebook uses paths relative to the repository. It can therefore be moved or cloned to a different location, provided the **whole repository** moves with it.

Normally, start Jupyter from the repository root. If Jupyter must start from an unrelated folder, set the project location first.

macOS or Linux:

```bash
export BAYESIAN_IPO_ROOT=/absolute/path/to/bayesian_ipo_study
python -m jupyter lab /absolute/path/to/bayesian_ipo_study/bayesian_ipo_study.ipynb
```

Windows PowerShell:

```powershell
$env:BAYESIAN_IPO_ROOT="C:\absolute\path\to\bayesian_ipo_study"
python -m jupyter lab C:\absolute\path\to\bayesian_ipo_study\bayesian_ipo_study.ipynb
```

## Common problems

### “Could not locate a complete Bayesian IPO Study repository”

The notebook cannot see one or more required folders. Make sure `bayesian_ipo_study.ipynb`, `src`, `data` and `models` are still inside the same repository folder. Run `python scripts/validate_release.py` to see which file is missing.

### `ModuleNotFoundError`

Activate the project environment and install the requirements again:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activation is `.venv\Scripts\Activate.ps1`.

### Jupyter uses the wrong Python environment

Close Jupyter, activate `.venv`, and launch it with:

```bash
python -m jupyter lab bayesian_ipo_study.ipynb
```

## Important limitations

- The analysis is predictive, not causal. An association does not prove that early trading behaviour causes later returns.
- The usable sample is selected by public-data availability.
- Validation for very large IPOs is based on a much smaller sample than validation for the full market.
- Some final-cohort prediction intervals undercovered their nominal levels.
- Aggregate price and volume cannot identify which investor groups caused a move.
- This frozen release cannot collect new raw data or reproduce the full original model-search pipeline.

## Project status and use

This repository is a research and educational artifact. It does not provide investment advice, and its forecasts should not be treated as trading recommendations.
