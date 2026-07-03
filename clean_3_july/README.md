# Aq2 / Kq2 — Subglacial Geothermal Heat Flow for Antarctica and Greenland

**Status:** Active development — model version `0_4` (see `MODEL_VERSION` in `config.py`)

---

## Overview

This repository contains the modelling pipeline that produces the Aq2 (Antarctica)
and Kq2 (Greenland) subglacial geothermal heat-flow datasets. The products are
intended as basal boundary conditions for ice-sheet models. They improve on the
earlier Aq1 / Kq1 products by combining three independent statistical methods and
then applying physics-based forward corrections.

The work builds on the global IHFC heat-flow database and a set of ~22 geophysical
observables sampled onto 5 x 5 km polar-stereographic grids (EPSG:3031 for
Antarctica, EPSG:3413 for Greenland), drawing on seismic tomography, crustal
structure, gravity, magnetics, and lithospheric density.

---

## Methods

Three complementary methods are trained and applied in parallel. Each is developed
across a matched trio of notebooks (sweep, model, target).

### 1. Gradient Boosting (GBM)
`HistGradientBoostingRegressor` trained with the quantile loss at each of the five
prediction quantiles (0.05, 0.25, 0.50, 0.75, 0.95). Provides the best-calibrated
point estimate. One fitted model per quantile is stored (dict), so the artefact key
is `models` (plural) rather than `model`.

### 2. Quantile Regression Forest (QRF)
A random forest trained with the `quantile_forest` library. A single fitted forest
predicts any quantile simultaneously, which makes it the primary model for
uncertainty characterisation.

### 3. Similarity (Aq1 heritage)
A distance-weighted analogue method. Each target point is estimated from the K
most-similar reference points in z-scored observable space, using per-feature
Gaussian kernel widths (sigma) optimised with Optuna. It is non-parametric — there
is no fitted estimator object — so its artefact carries the reference database and
kernel parameters instead. It provides the full empirical distribution used for
entropy-based uncertainty analysis and backward compatibility with Aq1.

### Forward-model corrections (Aq2 / Kq2)
Applied to the empirical predictions in the forward step (`lib/forward.py`,
`7_FORWARD`):
- **Volcanic heat injection:** Gaussian kernels centred on Holocene ("active") and
  Pleistocene volcanoes, with age-dependent amplitude and spatial scale (see the
  `VOLC_KERNEL_*` constants in `config.py`).
- **Topographic refraction correction:** the Colgan-style correction adjusts heat
  flow for sub-glacial relief using BedMachine bed geometry and rock/ice thermal
  conductivities.

---

## Pipeline

```
1_IMPORT           Load IHFC DB, cluster reference points, build 5 km grids,
                   import all observables to parquet.
2_OBSERVABLES      Visualise and describe each observable.

3a_QRF_SWEEP  }
3b_GBM_SWEEP  }    Hyperparameter / feature sweeps. Each writes the chosen
3c_SIM_SWEEP  }    feature list and tuning constants to output/sweeps/<method>_params.json.
3d_DEPTH_SWEEP     REVEAL depth-slice sweep.

4a_QRF_MODEL  }    Train each method on the reference data using the sweep JSON,
4b_GBM_MODEL  }    fit the conformal / correction spline, and export
4c_SIM_MODEL  }    output/models/<method>_artefacts.pkl.

5a_QRF_TARGETS }   Load the artefact, predict on the Antarctica and Greenland
5b_GBM_TARGETS }   target grids, and write the output NetCDF products.
5c_SIM_TARGETS }

6_CLUSTERING_v2    Multi-model comparison / clustering.
7_FORWARD          Volcanic + topographic forward corrections (Aq2 / Kq2).
X_EXTRAS           Ancillary figures and checks.
```

---

## How configuration drives selection

- **`config.py`** is the single source for paths, grid definitions
  (`TARGET_GRIDS`), CRS, thresholds, model defaults, and plotting constants
  (`hf_cmap`, `hf_v_min`, `hf_v_max`, `unc_cmap`). Notebooks import it rather than
  hard-coding literals.
- **`recipe.py`** defines every observable (`dd`) and its import routing. The schema
  for a `dd` entry is documented at the top of the file.
- **Feature hierarchy:** `obs_model ⊆ obs_sweep ⊆ obs`.
  - `obs` — full parquet catalogue (everything `1_IMPORT` writes).
  - `obs_sweep` — candidate universe fed into the sweeps.
  - `obs_model` — the features actually used in prediction.
- **Sweep JSON = single source of truth for `obs_sel`.** Each sweep notebook writes
  the human-approved final feature list under the key `obs_sel` in
  `output/sweeps/<method>_params.json`. The model notebooks read it back with
  `obs_sel = PARAMS['obs_sel']` (no silent fallback), so the selection cannot drift
  between sweep and training.

---

## Artefact flow

```
output/sweeps/<method>_params.json   (obs_sel, PARAMS, tuning constants)
        │  read by stage 4
        ▼
output/models/<method>_artefacts.pkl (model|models, scaler, spline, obs_sel, PARAMS)
        │  read by stage 5
        ▼
output/*.nc                          (canonical output variables)
        │  read by lib/forward.py
        ▼
forward-corrected Aq2 / Kq2 products
```

Artefact bundle keys are standardised across methods:
`model` (QRF / Similarity) or `models` (GBM quantile dict), `scaler`, `spline`,
`obs_sel`, `PARAMS`. The Similarity bundle additionally carries its reference
database and kernel parameters. Model quality metrics are written separately to
`output/models/model_metrics.csv`.

## Canonical output variable naming

Stage-5 NetCDF files use a uniform, method-prefixed scheme:

| Variable | Meaning |
|---|---|
| `<method>_q05_corr` … `<method>_q95_corr` | bias-corrected quantiles |
| `<method>_std_corr` | Gaussian-equivalent standard deviation |
| `qrf_iqr50_corr`, `qrf_iqr90_corr` | QRF interquantile ranges |
| `sim_N`, `sim_hist` | Similarity neighbour count / distribution |

`lib/forward.py:load_empirical_grid()` maps any `qrf_`/`gbm_`/`sim_` prefixed
canonical variable to the internal `q_q05 … q_q95` / `q_std` names, so the forward
step is method-agnostic.

---

## Current status

- **Production:** import, observables, all three sweep/model/target trios, and the
  volcanic + topographic forward corrections.
- **Phase-2 placeholders (not yet production):**
  - `lib/forward.py` topographic correction uses **constant** rock/ice conductivity
    placeholders (`TOPO_K_ROCK_DEFAULT`, `TOPO_K_ICE_DEFAULT`); spatially varying
    conductivity is not yet wired in.
  - Submarine / marine heat-flow correction is not implemented.
  - `lib/uncertainty.py:ensemble_forward()` is a Monte-Carlo stub that raises
    `NotImplementedError`; only the parametric (analytical) uncertainty path is
    active.

---

## Data requirements

All input datasets are expected under `../data/` (one level above the repo root).
Key datasets:

| Dataset | Reference | Variable |
|---|---|---|
| IHFC 2024 GHFDB | Fuchs et al. 2024 | Heat-flow observations |
| REVEAL | Auer et al. 2024 | Seismic tomography (S & P wave) |
| BedMachine Antarctica v3 | Morlighem et al. 2020 | Bed topography, ice thickness |
| BedMachine Greenland v6 | Morlighem et al. 2017 | Bed topography, ice thickness |
| GEMMA | Reguzzoni et al. 2013 | Gravity-derived Moho |
| Szwillus et al. 2019 | Szwillus et al. 2019 | Seismic Moho |
| LithoRef18 | Afonso et al. 2019 | LAB depth, densities |
| EIGEN-6C4 | Förste et al. 2014 | Gravity, geoid |
| GVP Holocene/Pleistocene | Smithsonian GVP | Volcano catalogue |

---

## Running

The notebooks are developed and run in **VS Code** (Jupyter extension) against the
project virtual environment. Run them in pipeline order:

```
1_IMPORT → 2_OBSERVABLES →
3a/3b/3c(/3d) → 4a/4b/4c → 5a/5b/5c →
6_CLUSTERING_v2 → 7_FORWARD
```

Dependencies:

```bash
pip install numpy pandas xarray scipy scikit-learn quantile_forest optuna numba
pip install geopandas pyproj rasterio rioxarray
pip install cartopy matplotlib cmcrameri
```

`config.py` creates the required `output/`, `data/`, `fig/`, `temp/`, and `logs/`
directories on import.

---

## Key references

- Stål et al. (2021) — Aq1 original Antarctic heat flow, *JGR*
- Al-Aghbary et al. (2026) — Uncertainty characterisation of subglacial heat flow
- Fuchs et al. (2024) — IHFC 2024 Global Heat Flow Database
- Morlighem et al. (2020) — BedMachine Antarctica v3

---

*Tobias Stål — University of Tasmania, IMAS — tobias.staal@utas.edu.au*
