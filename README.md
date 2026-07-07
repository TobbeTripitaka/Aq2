# Aq2 / Kq2: Subglacial Geothermal Heat Flow for Antarctica and Greenland

**Status:** Active development  
**Configuration:** `config.py` v4.0  
**Model output version:** `0_5`

This repository contains a modelling pipeline for producing subglacial geothermal heat-flow datasets for Antarctica and Greenland. The outputs are intended as basal boundary conditions for ice-sheet, glaciological, and other geophysical modelling.

The pipeline combines machine-learning prediction, uncertainty quantification, ensemble fusion, and physics-based forward corrections to estimate geothermal heat flow beneath the Antarctic and Greenland ice sheets.

## Overview

This repository updates the earlier Aq1 / Kq1 geothermal heat-flow workflow by combining:

- a Quantile Regression Forest model;
- a Gradient Boosting quantile model;
- a similarity / analogue model derived from the Aq1 approach;
- conformal calibration and centre-bias correction;
- uncertainty diagnostics and clustering;
- ensemble fusion of model outputs;
- forward corrections for volcanic, hydrothermal, and topographic effects.

The models are trained using the IHFC 2024 Global Heat Flow Database together with gridded geophysical observables, including seismic tomography, crustal and lithospheric structure, gravity, magnetics, bed topography, and related derived quantities.

## Methods

The pipeline uses three empirical prediction methods.

### Quantile Regression Forest

The QRF model uses the `quantile_forest` package to estimate conditional quantiles of geothermal heat flow. This provides a probabilistic prediction distribution from a single fitted ensemble and is the main model used for uncertainty characterisation.

### Gradient Boosting Quantile Model

The GBM approach uses `sklearn.ensemble.HistGradientBoostingRegressor` with quantile loss. Separate models are trained for different target quantiles, giving an independent estimate of prediction intervals and median heat flow.

### Similarity model

The similarity model follows the Aq1-style analogue approach. Target cells are predicted from similar reference observations in normalised feature space. The method uses feature-wise kernel widths and an optimised neighbourhood size, with acceleration via `numba`.

### Calibration and correction

The model outputs include conformal calibration and a centre-bias correction spline. These corrections are applied to improve interval coverage and reduce systematic bias while preserving uncertainty-width structure.

### Ensemble and forward correction (draft version)

Notebook 7 combines the QRF, GBM, and similarity predictions into an ensemble field. Notebook 8 applies forward-model corrections, including oversampling, volcanic/hydrothermal heat injection, and topographic refraction corrections.

## Installation

Python 3.10 or later is recommended.

Clone the repository:

```bash
git clone https://github.com/TobbeTripitaka/Aq2.git
cd Aq2
```

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas xarray scipy scikit-learn statsmodels
pip install quantile_forest optuna numba joblib
pip install geopandas pyproj rasterio rioxarray netCDF4 stripy
pip install matplotlib seaborn cmcrameri pyyaml
```

A LaTeX installation with `pdflatex` and `bibtex` is also needed if you want to build the generated observable documentation using `compile_docs.sh`.

## Quick start

The pipeline is notebook-driven. Shared paths, constants, features, and model parameters are configured in:

- `config.py`
- `recipe.py`
- `forward_params.yaml`

Input data are expected under `../data/`, relative to the repository root.

Run the notebooks in order:

```text
1_IMPORT
2_OBSERVABLES
3a_QRF_SWEEP
3b_GBM_SWEEP
3c_SIM_SWEEP
3d_DEPTH_SWEEP
4a_QRF_MODEL
4b_GBM_MODEL
4c_SIM_MODEL
5a_QRF_TARGETS
5b_GBM_TARGETS
5c_SIM_TARGETS
6_CLUSTERING
7_ENSEMBLE (draft)
8_FORWARD (draft)
```

The `3x` notebooks perform hyperparameter sweeps and feature-selection experiments. These are the most computationally expensive stages. The `4x` notebooks train or load the fitted models, and the `5x` notebooks apply them to Antarctic and Greenland target grids.

## Notebook inventory

| Notebook | Stage | Description |
|---|---|---|
| `1_IMPORT.ipynb` | Data import | Loads and filters IHFC heat-flow data, imports observables, builds the reference dataset, computes sample weights, and prepares Antarctic and Greenland prediction grids. |
| `2_OBSERVABLES.ipynb` | Observable inspection | Visualises and documents the geophysical observables used by the pipeline. Can generate figures and LaTeX documentation. |
| `3a_QRF_SWEEP.ipynb` | QRF sweep | Runs hyperparameter optimisation and feature-selection experiments for the Quantile Regression Forest model. |
| `3b_GBM_SWEEP.ipynb` | GBM sweep | Runs Optuna-based hyperparameter optimisation and feature analysis for the Gradient Boosting quantile model. |
| `3c_SIM_SWEEP.ipynb` | Similarity sweep | Optimises feature-wise similarity kernel widths and neighbourhood size for the analogue model. |
| `3d_DEPTH_SWEEP.ipynb` | Depth-feature sweep | Tests seismic tomography depth slices and derived depth features for predictive skill. |
| `4a_QRF_MODEL.ipynb` | QRF model | Trains or loads the QRF model, applies calibration and bias correction, and exports model artefacts. |
| `4b_GBM_MODEL.ipynb` | GBM model | Trains or loads GBM quantile models, applies calibration and bias correction, and exports model artefacts. |
| `4c_SIM_MODEL.ipynb` | Similarity model | Configures and evaluates the similarity model using optimised parameters. |
| `5a_QRF_TARGETS.ipynb` | QRF targets | Applies the QRF model to Antarctic and Greenland target grids and writes gridded prediction outputs. |
| `5b_GBM_TARGETS.ipynb` | GBM targets | Applies the GBM quantile models to Antarctic and Greenland target grids. |
| `5c_SIM_TARGETS.ipynb` | Similarity targets | Applies the similarity model to Antarctic and Greenland target grids. |
| `6_CLUSTERING.ipynb` | Uncertainty analysis | Performs clustering-based QRF uncertainty analysis and regional uncertainty characterisation. |
| `7_ENSEMBLE.ipynb` | Ensemble | Combines QRF, GBM, and similarity outputs into a fused ensemble prediction. This notebook is not fully developed and contains a few bugs. |
| `8_FORWARD.ipynb` | Forward correction | Applies physics-based forward corrections to the ensemble outputs. This notebook is not fully developed and contains a few bugs. |
| `X_EXTRAS.ipynb` | Utilities | Generates auxiliary tables and documentation, including observable summaries. |

## Repository structure

```text
Aq2/
├── 1_IMPORT.ipynb
├── 2_OBSERVABLES.ipynb
├── 3a_QRF_SWEEP.ipynb
├── 3b_GBM_SWEEP.ipynb
├── 3c_SIM_SWEEP.ipynb
├── 3d_DEPTH_SWEEP.ipynb
├── 4a_QRF_MODEL.ipynb
├── 4b_GBM_MODEL.ipynb
├── 4c_SIM_MODEL.ipynb
├── 5a_QRF_TARGETS.ipynb
├── 5b_GBM_TARGETS.ipynb
├── 5c_SIM_TARGETS.ipynb
├── 6_CLUSTERING.ipynb
├── 7_ENSEMBLE.ipynb
├── 8_FORWARD.ipynb
├── X_EXTRAS.ipynb
├── config.py
├── recipe.py
├── forward_params.yaml
├── compile_docs.sh
├── lib/
├── scripts/
├── output/
├── data/
├── fig/
└── temp/
```

- `config.py`: central configuration for paths, model settings, feature lists, versions, and shared constants.
- `recipe.py`: observable definitions, import logic, and supporting data recipes.
- `forward_params.yaml`: parameters for the forward-correction stage.
- `compile_docs.sh`: helper script for compiling generated LaTeX documentation.
- `lib/`: project library code imported by the notebooks.
- `scripts/`: utility scripts used by later workflow stages.
- `output/`: model artefacts, sweep results, diagnostics, and generated products.
- `data/`: local data directory or symlink. Raw input data are not version-controlled.

## Dependencies

The current dependency set is derived from the imports used across the notebooks and supporting modules.

### Core scientific Python

- `numpy`
- `pandas`
- `xarray`
- `scipy`
- `scikit-learn`
- `statsmodels`

### Machine learning and optimisation

- `quantile_forest`
- `optuna`
- `numba`
- `joblib`

### Geospatial and gridded data

- `geopandas`
- `pyproj`
- `rasterio`
- `rioxarray`
- `netCDF4`
- `stripy`
- `cartopy` 

### Plotting and configuration

- `matplotlib`
- `seaborn`
- `cmcrameri`
- `pyyaml`

### External tools

- LaTeX toolchain for generated documentation
- `pdflatex`
- `bibtex`


## Data requirements

Input data are expected under `../data/`, with exact paths configured in `config.py`.

The pipeline uses datasets including (not complete):

| Dataset | Purpose |
|---|---|
| IHFC 2024 Global Heat Flow Database | Training observations for geothermal heat flow |
| REVEAL seismic tomography | Seismic velocity observables and depth-sweep features |
| BedMachine Antarctica | Antarctic bed geometry, ice thickness, and related gridded fields |
| BedMachine Greenland | Greenland bed geometry, ice thickness, and related gridded fields |
| GEMMA | Gravity-derived crustal information |
| Szwillus et al. Moho model | Seismic Moho information |
| LithoRef18 | Lithospheric structure and related observables |
| EIGEN-6C4 | Gravity and geoid observables |
| Smithsonian GVP volcano data | Volcanic forcing for forward correction |

The raw input datasets are not stored in this repository.

## Outputs and reproducibility

The workflow writes outputs to `output/`. These include:

- sweep parameter files;
- model artefacts;
- model diagnostics;
- uncertainty diagnostics;
- ensemble products;
- forward-model products;
- NetCDF prediction grids.

The model output version is currently set to `0_5`, and output filenames are versioned accordingly.

Model splits use a fixed random seed where configured, and the sweep notebooks write parameter files that are reused by later notebooks. This allows later stages to be rerun without repeating expensive hyperparameter searches.

Large data products and raw input data are intentionally not version-controlled.

## Development notes

- The canonical workflow is the numbered notebook sequence in the repository root.
- The `3x` notebooks are computationally expensive and should usually be rerun only when changing model structure, features, or optimisation settings.
- The `4x` notebooks train or load model artefacts.
- The `5x` notebooks apply trained models to target grids.
- Notebook execution is controlled by toggles near the top of several notebooks.
- `config.py` should be treated as the main source of truth for paths, feature selections, constants, and version identifiers.
- `recipe.py` should be treated as the main source of truth for observable definitions and import logic.

## Known issues

- Notebooks 7 and 8 are not fully developed and contain a few bugs.
- Outputs from Notebooks 7 and 8 may change as the methodology is refined.
- Adding a pinned environment file is recommended before publication.
- Adding a `LICENSE` file and `CITATION.cff` is recommended before external release.
- The Greenland hydrothermal-vent input is currently a placeholder.
- Some forward-model inputs appear to be placeholders.
- A formal citation file has not yet been added.
- A pinned software environment to be added.
- There is currently no pinned `requirements.txt` or `environment.yml`. Adding one before publication is recommended.


Specifically:
- `7_ENSEMBLE.ipynb` should be treated as a provisional ensemble-combination workflow.
- `8_FORWARD.ipynb` should be treated as a provisional forward-correction workflow.

## Academic context

This work builds on earlier geothermal heat-flow modelling for polar regions, including the Aq1 / Kq1 datasets and subsequent work on uncertainty quantification for subglacial geothermal heat-flow prediction.

If you use this repository or its outputs, please cite the relevant Aq1 / Kq1 publications, the IHFC 2024 Global Heat Flow Database, the geophysical datasets used as inputs, and this repository.

## Contact

Tobias Stål  
University of Tasmania / ACEAS
