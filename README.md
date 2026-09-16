# FACs-TrapSignatures

[![DOI (code)](https://zenodo.org/badge/DOI/10.5281/zenodo.22791960.svg)](https://doi.org/10.5281/zenodo.22791960)
[![DOI (data)](https://zenodo.org/badge/DOI/10.5281/zenodo.22791430.svg)](https://doi.org/10.5281/zenodo.22791430)

Code and data for: Single Best Fits Can Be Misleading: Resolving Common1
Trapping Signatures across FA–Cs Perovskites.

This repo holds the analysis code and raw measurement data. The larger
intermediate/fit outputs (`FitResults/`, `MCMCResults/`) are archived
separately on Zenodo (DOI: https://doi.org/10.5281/zenodo.22791430) because of
their size, and are fetched on demand with `scripts/download_results.py`.

## Repository structure

```
FACs-TrapSignatures/
├── README.md
├── LICENSE
├── requirements.txt
├── ExperimentalNotes.txt
├── src/
│   ├── measurement/
│   │   └── 250121-trPL_PowerDeps-MeasProcedure.py   # hardware acquisition script (see note below)
│   ├── fitting/
│   │   └── run_all_opti_12Traps_BDF_CsSeries.py     # trap-model fitting (BoTorch/Ax) -> results/FitResults
│   └── mcmc/
│       └── run_mcmc_from_turbo_CsContent.py         # MCMC over the best fits -> results/MCMCResults
├── notebooks/
│   └── plots.ipynb                                   # figures
├── data/raw/
│   ├── CsSeries/     # raw time-resolved PL traces, Cs-content series (~35M)
│   └── TwoStep/      # raw traces, two-step reference measurements (~10M)
├── vendor/
│   └── optimpv_github/   # vendored copy of your own unpublished fitting code (see below)
├── results/           # NOT in git - populated by scripts/download_results.py
│   ├── FitResults/     (~557M on Zenodo)
│   └── MCMCResults/    (~2.1G on Zenodo, minus a superseded "OLD" run - see below)
└── scripts/
    └── download_results.py
```

## Pipeline

1. **Acquisition** (`src/measurement/`) - raw `.dat` traces are recorded on
   the lab PC and land in `data/raw/CsSeries/` and `data/raw/TwoStep/`.
   These raw files are already included in this repo.
2. **Fitting** (`src/fitting/run_all_opti_12Traps_BDF_CsSeries.py`) - fits
   the rate-equation trap models to `data/raw/CsSeries/*` using
   [optimPV](https://github.com/openPV-lab/optimPV)'s Bayesian/BoTorch
   optimizer, writing pickled optimizer/agent objects to
   `results/FitResults/...`.
3. **MCMC** (`src/mcmc/run_mcmc_from_turbo_CsContent.py`) - reads the best
   fits from `results/FitResults/CsSeries/Run5_12345Traps_L500nm/` and runs
   `emcee`-based MCMC sampling, writing corner plots, trace plots and
   sample CSVs to `results/MCMCResults/...`.


### About `optimpv` vs `optimpv_github`

The code imports two different things with confusingly similar names:

- **`optimpv`** is the public [openPV-lab/optimPV](https://github.com/openPV-lab/optimPV)
  package, installed normally via `pip install optimpv` (see `requirements.txt`).
- **`optimpv_github`** local, unpublished rate-equation fitting code
  (`RateEqAgent`, `RateEqModel`, `Pumps`, `axBOtorchOptimizer`, etc.) that also contains
  the appropriate functions for the log likelihood formulation based on NRMSE.

## Getting the archived results

```bash
python scripts/download_results.py --record-id <ZENODO_RECORD_ID>
```

This fetches `FitResults.zip` and `MCMCResults.zip` from the Zenodo record
and unpacks them into `results/FitResults/` and `results/MCMCResults/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`optimpv_github` doesn't need a separate install step - it's vendored in
`vendor/optimpv_github/` and picked up automatically (see "About `optimpv`
vs `optimpv_github`" above). See `requirements.txt` for filling in the
exact pinned `optimpv` (PyPI) version you used, for reproducibility.

## Citation

TODO - add paper citation once published, and the Zenodo DOI (Zenodo
mints one automatically for the data archive; you can also link a GitHub
release of this code repo to Zenodo to get a DOI for the code itself, via
https://zenodo.org/account/settings/github/).

## License

Code: MIT (see `LICENSE`). Data archived on Zenodo: CC-BY-4.0 (set this in
the Zenodo upload form).
