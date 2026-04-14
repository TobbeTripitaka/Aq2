# Aq2 / Kq2 — Subglacial Geothermal Heat Flow for Antarctica and Greenland

**Status:** Active development — v2.0 (March 2026)

---

## Overview

This repository contains the full modelling pipeline to produce the Aq2 (Antarctica) and Kq2 (Greenland) subglacial geothermal heat flow datasets. These datasets are intended as boundary conditions for ice sheet models, improving on the earlier Aq1 / Kq1 products by combining three independent methods and applying physics-based forward corrections.

The work builds on the global IHFC heat flow database and a set of ~22 geophysical observables sampled onto 5 km polar stereographic grids, drawing on seismic tomography, crustal structure models, gravity, magnetics, and lithospheric density.

---

## Methods

Three complementary modelling approaches are used in parallel:

### 1. Gradient Boosting (HistGradientBoostingRegressor)
A gradient boosted tree model trained with a quantile loss function, providing the best-calibrated point estimates. Three separate models are trained for q₀₅, q₅₀, and q₉₅. Post-hoc isotonic regression calibration is applied on a held-out set.

### 2. Quantile Regression Forest (QRF)
A random forest trained via the `quantile_forest` library. Predicts the full conditional distribution at any quantile simultaneously, offering robust probabilistic uncertainty estimates. This is the primary model for uncertainty characterisation.

### 3. Similarity (Aq1 heritage)
A distance-weighted analogue method. Each target point receives a heat flow estimate from the K most-similar reference points in normalised observable space. Gaussian kernel widths (σ) per feature are optimised with Optuna. Provides backwards compatibility with Aq1 and distributional metrics that are useful for entropy-based uncertainty analysis.

### Forward model corrections (Aq2 / Kq2)
Applied after the empirical predictions:
- **Volcanic heat injection:** Mexican-hat (Ricker wavelet) Gaussian kernels centred on Holocene and Pleistocene volcanoes from the GVP catalogue, with age-dependent amplitude and spatial scale.
- **Topographic refraction correction:** Adjusts heat flow for sub-glacial topographic relief using BedMachine bed geometry and a rock/ice thermal conductivity model.

---

## Repository Structure

```
aq2/
├── 1_Import.ipynb          # Load IHFC DB, cluster, build 5 km grids, import observables
├── 2_Observables.ipynb     # Visualise and describe all observables
├── 3_Test_Sweep.ipynb      # Param sweep, REVEAL depth sweep, feature ablation
├── 4_Run_Models.ipynb      # Train QRF and GB; predict all target grids
├── 5_Sim.ipynb             # Train and apply Similarity model
├── 6_Target_Grids.ipynb    # Load 5 km products; multi-model comparison maps
├── 7_Uncertainty.ipynb     # Entropy, inter-model agreement, robustness figures
├── 8_Forward_Models.ipynb  # Volcanic and topographic forward corrections (Aq2/Kq2)
│
├── config.py               # All settings: paths, features, hyperparameters
├── recipe.py               # Observable definitions, import functions, volcano catalogue
│
├── lib/
│   ├── __init__.py
│   ├── agrid.py            # Grid import library (v0.4.9)
│   ├── utils.py            # Utility functions (latlon_to_epsg, sort_quantiles)
│   ├── plotting.py         # Plotting mixin
│   └── xarray_mixin.py     # xarray mixin
│
├── output/
│   ├── param_sweep/        # Sweep CSV results (tracked in git)
│   ├── depth_sweep/
│   ├── feature_sweep/
│   └── *.nc                # NetCDF products (NOT tracked in git)
│
├── fig/                    # Figures (NOT yet tracked in git)
├── data/                   # Symlink or local copy of data (NOT tracked in git)
└── temp/                   # Scratch files
```

---

## Data Requirements

All input datasets are expected under `../data/` (one level above the repo root). Key datasets include:

| Dataset | Reference | Variable |
|---|---|---|
| IHFC 2024 GHFDB | Fuchs et al. 2024 | Heat flow observations |
| REVEAL | Auer et al. 2024 | Seismic tomography (S & P wave) |
| BedMachine Antarctica v3 | Morlighem et al. 2020 | Bed topography, ice thickness |
| BedMachine Greenland v5 | Morlighem et al. 2017 | Bed topography, ice thickness |
| GEMMA | Reguzzoni et al. 2013 | Gravity-derived Moho |
| Szwillus et al. 2019 | Szwillus et al. 2019 | Seismic Moho |
| LithoRef18 | Afonso et al. 2019 | LAB depth, densities |
| EIGEN-6C4 | Förste et al. 2014 | Gravity, geoid |
| GVP Holocene/Pleistocene | Smithsonian GVP | Volcano catalogue |

---

## Setup

```bash
# Clone
git clone https://github.com/TobbeTripitaka/Aq2.git
cd Aq2

# Install dependencies
pip install numpy pandas xarray scipy scikit-learn quantile_forest optuna
pip install geopandas pyproj rasterio rioxarray stripy
pip install cartopy matplotlib cmcrameri

# Run notebooks in order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

---

## Key References

- Stål et al. (2021) — Aq1 original Antarctic heat flow, *Journal of Geophysical Research*
- Al-Aghbary et al. (2026) — Uncertainty characterisation of subglacial heat flow predictions
- Fuchs et al. (2024) — IHFC 2024 Global Heat Flow Database
- Morlighem et al. (2020) — BedMachine Antarctica v3

---

*Tobias Stål — University of Tasmania, IMAS — tobias.staal@utas.edu.au*
