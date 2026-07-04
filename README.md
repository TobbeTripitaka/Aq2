# Aq2 / Kq2 — Notebook 8 Forward-Model Rework (v0_5)

**Status:** Active development — model version `0_5` (`MODEL_VERSION` in `config.py`).
This package updates the forward step (`8_FORWARD`) and its supporting library and
configuration. It layers on top of the main `Aq2_v0_5_delivery` package — drop these
files into the repository, replacing their namesakes.

---

## What changed

The forward model (`8_FORWARD.ipynb`) is reworked in four ways, per your brief:

1. **Input is now the fused ensemble** from `7_ENSEMBLE`, not the single QRF grid.
2. **Ice conductivity is temperature-dependent**, `k_ice = k_ice(T_b)`, driven by a
   basal-temperature input file.
3. **Basal temperature, upper-crust conductivity and hydrology are read from files**
   (Phase-1 placeholders provided by a generator script).
4. **A development bounding-box subset** lets you run the whole chain at full
   resolution over a small area (default: West Marie Byrd Land) and remove it with a
   single config flag.

The four notebook bugs found in review are also fixed (see *Review findings* below).

---

## Files in this package

```
8_FORWARD.ipynb                       reworked forward-model notebook
config.py                             + FORWARD-MODEL block (see below)
forward_params.yaml                   topo_correction + output notes updated to v0_5
lib/forward.py                        + k_ice_from_temperature(), crop_to_bbox(),
                                        fusion-aware load_empirical_grid()
lib/uncertainty.py                    unchanged (included for a self-contained lib/)
scripts/make_forward_placeholders.py  generates the Phase-1 input NetCDFs
README.md                             this file
```

> **Deliverable policy:** no media or output files are included. Run the notebook
> and the generator yourself to produce grids and figures.

---

## Run order

```
… 7_ENSEMBLE            writes output/ensemble/{ant,grl}_ensemble_v0_5.nc
      │
      ▼
python scripts/make_forward_placeholders.py     (one-off; rerun with --overwrite)
      │  writes output/forward_inputs/{ant,grl}_{basal_temperature,k_upper_crust,hydrology}.nc
      ▼
8_FORWARD               reads the fused grid + the three input files, applies
                        oversampling → volcanic kernels → topographic correction
```

The generator reads the notebook-7 ensemble grid for its coordinates, so run
`7_ENSEMBLE` first. If the ensemble file is missing it falls back to the stage-5 QRF
target grid (same 5 km definition) with a printed warning, so you can still smoke-test
the pipeline before the ensemble exists.

```bash
# both regions (default)
python scripts/make_forward_placeholders.py
# one region / overwrite existing
python scripts/make_forward_placeholders.py --region antarctica --overwrite
```

---

## 1 — Fusion input

`config.FORWARD_USE_FUSION = True` switches `8_FORWARD` from the QRF-only grid to the
fused ensemble `output/ensemble/{key}_ensemble_v{MODEL_VERSION}.nc`. Set it `False` to
fall back to the old QRF grid (`Aq1_5_QRF_v0_5.nc` / `Kq1_5_QRF_v0_5.nc`).

`lib/forward.py:load_empirical_grid()` now recognises the ensemble `ens_*` variables
and maps them onto the internal names the forward step expects:

| Ensemble variable (NB7) | Internal name | Notes |
|---|---|---|
| `ens_q05 … ens_q95` | `q_q05 … q_q95` | pooled quantiles |
| `ens_q50` | `q_mean` | pooled median = central estimate |
| `ens_band` | → `q_std` | `q_std = ens_band / 0.674489…` (semi-IQR → Gaussian σ, matching NB7's own convention) |

`ens_var_total`, `ens_struct_share`, `ens_robustness`, `ens_confidence` and
`ens_explainability` are carried through unchanged for traceability. The loader stamps
`ds.attrs["forward_input_kind"] = "fusion_ensemble"` (vs `"single_method"`) so the
provenance is recorded in the output NetCDF.

The `q_mean → ens_q50` mapping uses the pooled **median** as the central field. This
is the honest choice for a quantile-pooled ensemble (the mixture median, not a mean of
medians); it is documented in the loader docstring and in the NetCDF attributes.

---

## 2 & 3 — File-driven inputs and temperature-dependent k_ice

`8_FORWARD` cell "Step 3 input fields" now reads three per-region NetCDFs
(paths in `config.FORWARD_BASAL_TEMP_NC`, `FORWARD_K_CRUST_NC`, `FORWARD_HYDROLOGY_NC`)
and interpolates them onto the oversampled target grid:

- **Upper-crust conductivity** `k_upper_crust` → `k_rock` (used directly).
- **Basal temperature** `t_basal` → drives `k_ice` via
  `lib/forward.py:k_ice_from_temperature()`.

`k_ice(T)` follows **Cuffey & Paterson (2010)**, *The Physics of Glaciers* (4th ed.):

```
k_ice(T) = 9.828 · exp(−0.0057 · T)      T in kelvin,  k in W m⁻¹ K⁻¹
```

At the freezing point (T = 273.15 K = 0 °C) this gives **2.072 W m⁻¹ K⁻¹**, consistent
with the old constant `2.1` placeholder. Output is clamped to
`[KICE_CLIP_MIN, KICE_CLIP_MAX] = [1.5, 4.0] W m⁻¹ K⁻¹` as a physical guard.

**Documented approximation:** the basal temperature is used as a *single-value proxy
for the whole ice column*. The topographic ice term integrates over a column whose
temperature runs from the cold surface to the warm bed; using `T_b` alone is a Phase-2
first approximation that is most accurate near the bed interface, where the correction
concentrates. This is stated in the function docstring and the NetCDF attributes.

The library's `topographic_correction()` already accepted a 2-D `k_ice` array, so no
physics was altered — `k_ice(T_b)` is computed in the notebook and passed in.

### Placeholder values (Phase 1)

| Field | Variable | Value | Units |
|---|---|---|---|
| Basal temperature | `t_basal` | `0.0` | °C |
| Upper-crust conductivity | `k_upper_crust` | `2.5` | W m⁻¹ K⁻¹ |
| Hydrology | `hydrology` | `0.0` | — (no flux) |

Constants live in `config.py` (`PLACEHOLDER_BASAL_TEMP_C`, `PLACEHOLDER_K_UPPER_CRUST`,
`PLACEHOLDER_HYDROLOGY`). Files are written on the **5 km fusion grid** per region and
oversampled to the forward target spacing alongside the heat-flow grid.

> **Note:** hydrology is generated and documented but is **not yet consumed** by the
> forward correction — it is a Phase-1 stub for the eventual hydrological advection
> term.

---

## 4 — Development subset

`config.DEV_SUBSET = True` crops the pipeline to `config.DEV_SUBSET_BBOX`, a full-
resolution patch that exercises the whole chain quickly. Default box (EPSG:3031):

```
x: −1,240,000 … −650,000 m      y: −1,515,000 … −925,000 m
```

≈ 583 × 590 km over **West Marie Byrd Land**, across the West Antarctic Rift System —
a region where volcanic kernels and structural uncertainty are most active, so it is a
genuine end-to-end stress test rather than a quiet corner.

- Applied to both BedMachine and the 5 km input grid via
  `lib/forward.py:crop_to_bbox()` (robust to descending-`y` grids).
- Cache and output filenames are tagged with `config.DEV_SUBSET_TAG = "devMBL"` so
  subset runs never overwrite full-grid caches.
- The box is defined for Antarctica (`DEV_SUBSET_REGION = "antarctica"`); when
  `REGION` differs, the crop is skipped with a printed note.

**To process the whole grid:** set `DEV_SUBSET = False`. Nothing else changes.

### Forward target spacing

`config.FORWARD_TARGET_SPACING_M = {"antarctica": 500.0, "greenland": 150.0}` records
the intended oversampling targets (Antarctica BedMachine ≈ 500 m, Greenland ≈ 150 m).
The notebook still reads the true spacing from the BedMachine file at run time; these
values also drive the placeholder generator's documentation.

---

## Review findings (bugs fixed in this rework)

1. **Empty code cells removed** — two stray empty cells deleted.
2. **Colormap objects, not strings** — `hf_cmap` / `unc_cmap` are `"cmc.*"` strings in
   `config.py`. The notebook now resolves them once to real `Colormap` objects
   (`resolve_cmap()`), so plotting no longer depends on `cmcrameri`'s string
   registration being active in every cell. The `cmap == "RdBu_r"` comparison in the
   Step-3 panel was reworked to a `"diverging"` sentinel so it still works with objects.
3. **Input grid switched to the fusion product** — the old `INPUT_NC` pointed at the
   QRF-only grid; it now selects the ensemble grid (or QRF as a fallback).
4. **Version consistency** — `forward_params.yaml` output filenames were `v0_3` while
   `MODEL_VERSION = "0_5"`. Bumped to `v0_5` and annotated. The legacy
   `k_rock_default` / `k_ice_default` constants are retained but flagged **unused** in
   v0_5 (conductivity is now file-driven).

All variables in the edited notebook were checked to be defined (static namespace
scan across cells: no undefined names). The new library functions were unit-tested:
`k_ice(0 °C) = 2.072`, clip works, `crop_to_bbox` handles descending `y`, the
fusion loader derives `q_std` exactly as `ens_band / 0.674489…`, and the placeholder
generator round-trips through `_read_field` → `k_ice(T_b)` correctly.

---

## Phase-2 notes (not done here)

- Replace the constant placeholders with real fields: a modelled basal-temperature
  grid, a geology-derived upper-crust conductivity map, and a basal-hydrology field.
  Only `scripts/make_forward_placeholders.py` (or the file paths in `config.py`) needs
  to change — the notebook contract (file names, variable names) is stable.
- Upgrade `k_ice(T_b)` from a single basal-temperature proxy to a **column-integrated**
  ice conductivity once a temperature profile is available.
- Provide spatially varying conductivity uncertainties (`k_rock_std`, `k_ice_std`;
  currently `None`) to propagate through `topographic_correction()`.
- Wire the hydrology field into an advective correction term.
- Implement the submarine / marine topographic correction (still not implemented).

---

## References

- **Cuffey, K. M., & Paterson, W. S. B. (2010).** *The Physics of Glaciers* (4th ed.).
  Elsevier. — temperature-dependent ice conductivity `k_ice(T) = 9.828·exp(−0.0057·T)`.
- **Colgan, W., et al. (2021).** Topographic correction of geothermal heat flow beneath
  ice sheets — the relief-refraction approach the Step-3 correction follows.
- **Morlighem, M., et al. (2020).** BedMachine Antarctica v3; **(2017)** BedMachine
  Greenland v6 — bed and surface geometry.
- **Stål, T., et al. (2021),** *G-Cubed* — Aq1 heritage and the similarity method.
- **Al-Aghbary, Sobh, Stål, et al. (2026),** *Geophysical Journal International* — the
  fused-ensemble uncertainty framework produced by `7_ENSEMBLE`.

---

*Tobias Stål — University of Tasmania, IMAS — tobias.staal@utas.edu.au*
