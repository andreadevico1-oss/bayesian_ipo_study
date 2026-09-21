"""Validate that the published offline notebook bundle is complete and loadable."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tape_api import load_frozen_population  # noqa: E402
from src.tape_data import load_prepared  # noqa: E402


REQUIRED_NOTEBOOK_FILES = (
    "bayesian_ipo_study.ipynb",
    "requirements.txt",
    "models/tape_selection.json",
    "models/tape_registry.json",
    "data/processed/tape_sample_audit.csv",
    "data/processed/tape_event_panel.parquet",
    "data/processed/tape_predictions.csv",
    "data/processed/tape_scores.csv",
    "data/processed/tape_coefficients.csv",
    "data/processed/tape_initial_return_validation.csv",
    "data/processed/tape_offered_share_validation.csv",
    "data/processed/tape_size_contrasts.csv",
)


def main() -> None:
    missing = [path for path in REQUIRED_NOTEBOOK_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete release; missing: {missing}")

    selection = json.loads((ROOT / "models/tape_selection.json").read_text())
    audit, panel = load_prepared()
    population = load_frozen_population()

    if audit.empty or panel.empty or not audit["ipo_id"].is_unique:
        raise ValueError("Frozen data are empty or the IPO identifiers are not unique")

    # Loading each published cutoff proves that its selected portable trace exists.
    for cutoff in selection["cutoffs"]:
        population.get(cutoff)

    print(
        "Release validation passed: "
        f"{int(audit['usable'].sum())} usable IPOs, "
        f"{len(panel):,} event-panel rows, "
        f"{len(population.registry)} registry records."
    )


if __name__ == "__main__":
    main()
