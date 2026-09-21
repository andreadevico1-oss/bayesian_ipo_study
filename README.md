# Bayesian IPO Study

This project studies whether the first trading days of an IPO contain useful information about its subsequent performance through day 60. It builds an event-time dataset from IPO prices, trading volume and benchmark returns, then uses Bayesian models to produce full predictive distributions rather than single return estimates.

The analysis separates the initial offer-to-first-close move from aftermarket performance and compares a sequence of models that progressively add IPO size, early price behaviour, trading intensity and realised uncertainty. Model quality is assessed chronologically on later IPO cohorts using out-of-sample scoring and calibration tests, with particular attention to uncertainty and the risk of overfitting a relatively small event sample.

## Current status

This is an ongoing research project. The current version covers data construction and validation, exploratory price-discovery evidence, the Bayesian modelling framework, the M0–M3 model sequence and chronological out-of-sample comparison.

Planned extensions include:

- sampling diagnostics and prior-sensitivity analysis
- sequential Bayesian updating as new trading information becomes available
- pseudo-live demonstrations on historical IPOs
- robustness tests for price-discovery normalization
- survivorship analysis and partial identification for missing outcomes
- separate interpretation and validation for large and mega-IPOs
- a practical workflow for applying the model to future IPOs
- a consolidated discussion of limitations and final research conclusions

## Repository contents

- `bayesian_ipo_study.ipynb` — the main research notebook containing the data work, methodology, model comparisons and results developed so far.

This repository is intended for research and educational purposes and does not provide investment advice.
