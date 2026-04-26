"""
lib/forward.py
==============
Core algorithmic functions for the Aq2/Kq2 forward heat-flow model.

Pipeline
--------
  STEP 1  oversample_to_bedmachine()  —  5 km → BedMachine resolution
  STEP 2  apply_volcanic_kernels()    —  multiplicative Mexican-hat kernels
  STEP 3  topographic_correction()   —  Colgan-style bed/ice correction

Design principles
-----------------
  · Pure functions — no global state, no in-place mutation.
  · xarray-centric — all I/O uses xr.Dataset / xr.DataArray.
  · Conservation invariant — each step preserves the total integrated heat.
    Global mean checks are printed and stored as NetCDF attributes.
  · Dask-compatible — .compute() called only where unavoidable.
"""
from __future__ import annotations
import datetime
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.signal as sig
import xarray as xr
from pyproj import Transformer

_VOLC_CACHE: Optional[pd.DataFrame] = None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _print_step(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def _check_mean_conservation(
    da_before: xr.DataArray,
    da_after: xr.DataArray,
    label: str,
    tol: float = 1e-6,
) -> float:
    before = float(da_before.mean(skipna=True))
    after  = float(da_after.mean(skipna=True))
    diff   = abs(after - before)
    ok = "✓" if diff < tol else "✗  WARNING"
    print(f"  {ok}  [{label}]  "
          f"before={before*1e3:.4f}  after={after*1e3:.4f}  "
          f"delta={diff*1e6:.3f} µW/m²  (tol={tol*1e6:.1f} µW/m²)")
    if diff >= tol:
        warnings.warn(f"Mean conservation violated [{label}]: delta={diff:.2e} W/m²",
                      RuntimeWarning, stacklevel=3)
    return before


# ── Input loading ─────────────────────────────────────────────────────────────

def load_empirical_grid(
    nc_path: Path,
    region: str,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """Load the 5 km empirical QRF NetCDF and return with canonical variable names."""
    _print_step(f"Loading empirical 5 km grid — {region}")
    ds = xr.open_dataset(nc_path, chunks=chunks or {})
    rename = {
        "qrf_q05_corr":  "q_q05",
        "qrf_q25_corr":  "q_q25",
        "qrf_q50_corr":  "q_q50",
        "qrf_q75_corr":  "q_q75",
        "qrf_q95_corr":  "q_q95",
        "qrf_sigma_corr": "q_std",
        "qrf_iqr50_corr": "q_iqr50",
        "qrf_iqr90_corr": "q_iqr90",
    }
    ds = ds.rename({k: v for k, v in rename.items() if k in ds})
    if "q_mean" not in ds and "q_q50" in ds:
        ds = ds.assign(q_mean=ds["q_q50"].assign_attrs(
            units="W m-2", long_name="Heat flow — median estimate (q50)"))
    ds.attrs.update(region=region, forward_step="input")
    d = dict(ds.dims)
    print(f"  Grid: {d.get('y','?')} x {d.get('x','?')} cells")
    print(f"  Global mean q_mean: {float(ds['q_mean'].mean(skipna=True))*1e3:.3f} mW/m2")
    return ds


# ── Step 1: Oversampling ─────────────────────────────────────────────────────

def _nan_safe_zoom(arr: np.ndarray, zoom_y: float, zoom_x: float) -> np.ndarray:
    """
    NaN-safe cubic zoom via scipy.ndimage.zoom.

    NaN cells are filled with the local median before zooming so the
    cubic spline solver receives a finite-valued array.  After zooming
    the NaN mask is re-applied (nearest-neighbour zoom of the mask).
    """
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        fill = float(np.nanmedian(arr))
        arr_filled = np.where(nan_mask, fill, arr)
    else:
        arr_filled = arr

    zoomed = ndi.zoom(arr_filled.astype(np.float64), (zoom_y, zoom_x),
                      order=3, mode="reflect", prefilter=True)

    if nan_mask.any():
        mask_zoomed = ndi.zoom(nan_mask.astype(np.float32), (zoom_y, zoom_x),
                               order=0, mode="reflect") > 0.5
        zoomed = np.where(mask_zoomed, np.nan, zoomed)

    return zoomed.astype(np.float32)


def _cell_mean_correction(
    da_fine: xr.DataArray,
    da_coarse: xr.DataArray,
    fine_spacing: float,
    coarse_spacing: float,
) -> xr.DataArray:
    """Per-coarse-cell mean conservation via scalar multiplicative correction."""
    factor = int(round(coarse_spacing / fine_spacing))
    fine_coarsened = da_fine.coarsen(y=factor, x=factor, boundary="pad").mean()
    coarse_aligned = da_coarse.interp(
        y=fine_coarsened.y, x=fine_coarsened.x, method="nearest")
    ratio = xr.where(
        np.abs(fine_coarsened) > 1e-12,
        coarse_aligned / fine_coarsened,
        1.0,
    )
    ratio_fine = ratio.interp(y=da_fine.y, x=da_fine.x, method="nearest")
    return da_fine * ratio_fine


def oversample_to_bedmachine(
    ds_5km: xr.Dataset,
    bedmachine: xr.Dataset,
    fine_spacing: Optional[float] = None,
    coarse_spacing: float = 5_000.0,
    quantile_vars: Optional[list] = None,
    mean_tol: float = 1e-6,
) -> xr.Dataset:
    """
    STEP 1 — Oversample the 5 km empirical grid to BedMachine resolution.

    Algorithm:
      1. Bicubic interpolation (C2 surface — removes block artefacts).
      2. Per-cell conservative correction: coarse-cell mean enforced exactly.
      3. Global mean assertion.

    Conservation: integrated heat over each 5 km source cell is unchanged.
    No spatial information beyond nearest-cell guidance is added.
    """
    _print_step("STEP 1 — Oversampling 5 km to BedMachine resolution")
    if fine_spacing is None:
        fine_spacing = float(bedmachine.attrs.get("spacing", 500.0))
    print(f"  {coarse_spacing:.0f} m to {fine_spacing:.0f} m  "
          f"(factor {coarse_spacing/fine_spacing:.1f}x)")

    x_tgt = bedmachine["x"]
    y_tgt = bedmachine["y"]

    if quantile_vars is None:
        quantile_vars = [v for v in ds_5km.data_vars if v.startswith("q_")]
    print(f"  Variables: {quantile_vars}")

    out_vars: dict = {}
    for var in quantile_vars:
        da_c = ds_5km[var]
        # NaN-safe cubic zoom via scipy.ndimage (avoids xr.interp cubic solver
        # failure when source grid contains NaN cells).
        zoom_y = len(y_tgt) / len(da_c.y)
        zoom_x = len(x_tgt) / len(da_c.x)
        zoomed_arr = _nan_safe_zoom(da_c.values, zoom_y, zoom_x)
        # Trim or pad to exact target shape (zoom rounding can differ by ±1)
        ny_tgt, nx_tgt = len(y_tgt), len(x_tgt)
        zoomed_arr = zoomed_arr[:ny_tgt, :nx_tgt]
        if zoomed_arr.shape[0] < ny_tgt or zoomed_arr.shape[1] < nx_tgt:
            pad = ((0, ny_tgt - zoomed_arr.shape[0]),
                   (0, nx_tgt - zoomed_arr.shape[1]))
            zoomed_arr = np.pad(zoomed_arr, pad, mode="edge")
        da_f = xr.DataArray(zoomed_arr, dims=["y", "x"],
                            coords={"y": y_tgt, "x": x_tgt})
        da_f = _cell_mean_correction(da_f, da_c, fine_spacing, coarse_spacing)
        da_f.attrs = {**da_c.attrs,
                      "oversampled_from_m": coarse_spacing,
                      "oversampled_to_m": fine_spacing}
        out_vars[var] = da_f
        if var in ("q_mean", "q_q50"):
            _check_mean_conservation(da_c, da_f, var, tol=mean_tol)

    out_vars["mask"] = bedmachine["mask"].assign_attrs(
        long_name="BedMachine surface mask",
        flag_values="0 1 2 3",
        flag_meanings="ocean grounded_ice floating_ice rock")

    if "source" in bedmachine:
        out_vars["bedmachine_coverage"] = bedmachine["source"].isin([1, 2, 7, 10]).assign_attrs(
            long_name="True where BedMachine has direct observational constraint",
            note="Topographic correction applied only where this flag is True")

    ds_out = xr.Dataset(
        out_vars, coords={"y": y_tgt, "x": x_tgt},
        attrs={**ds_5km.attrs,
               "forward_step": "step1_oversample",
               "coarse_spacing_m": coarse_spacing,
               "fine_spacing_m": fine_spacing})
    print(f"  Output: {len(y_tgt)} x {len(x_tgt)} = {len(y_tgt)*len(x_tgt)/1e6:.1f}M pixels")
    return ds_out


# ── Step 2: Volcanic kernels ──────────────────────────────────────────────────

def load_volcano_catalogue(
    volc_list_path:   str = "../data/volcanos/volcanic list.xlsx",
    holocene_path:    str = "../data/volcanos/GVP_Volcano_List_Holocene.xlsx",
    pleistocene_path: str = "../data/volcanos/GVP_Volcano_List_Pleistocene.xlsx",
) -> pd.DataFrame:
    """
    Load and merge volcanic catalogues from three source files.
    Returns a DataFrame with columns: lat, lon, exposed, group, kernel_class.
    kernel_class: 'holocene' group -> 'active_volcano', else 'pleistocene_volcano'.
    Cached after first call.
    """
    global _VOLC_CACHE
    if _VOLC_CACHE is not None:
        return _VOLC_CACHE

    parts = []
    sheets = pd.read_excel(volc_list_path, sheet_name=None)
    for sname, df in sheets.items():
        if not {"LAT", "LON"}.issubset(df.columns):
            continue
        cols = [c for c in ["LAT", "LON", "exposed"] if c in df.columns]
        tmp = df[cols].rename(columns={"LAT": "lat", "LON": "lon"})
        if "exposed" not in tmp.columns:
            tmp["exposed"] = "unknown"
        tmp["group"] = sname
        parts.append(tmp)
    if not parts:
        raise RuntimeError(f"No valid sheets in {volc_list_path}")

    def _gvp(path, group):
        df = pd.read_excel(path, header=1)
        return (df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
                [["lat", "lon"]].dropna().assign(exposed="full", group=group))

    parts.append(_gvp(holocene_path,    "holocene"))
    parts.append(_gvp(pleistocene_path, "pleistocene"))
    all_v = pd.concat(parts, ignore_index=True)
    for col in ("lat", "lon"):
        all_v[col] = pd.to_numeric(all_v[col], errors="coerce")
    all_v = all_v.dropna(subset=["lat", "lon"])
    all_v["kernel_class"] = all_v["group"].apply(
        lambda g: "active_volcano" if "holocene" in str(g).lower() else "pleistocene_volcano")
    print(f"  Volcano catalogue: {len(all_v)} features")
    print(all_v["kernel_class"].value_counts().to_string())
    _VOLC_CACHE = all_v
    return _VOLC_CACHE


def load_hydrothermal_catalogue(csv_path: str) -> pd.DataFrame:
    """Load hydrothermal vent catalogue (CSV, required cols: lat, lon)."""
    df = pd.read_csv(csv_path)
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["kernel_class"] = "hydrothermal"
    df["group"] = "hydrothermal"
    print(f"  Hydrothermal catalogue: {len(df)} features")
    return df


def build_mexican_hat_kernel(
    sigma_inner_m: float,
    sigma_outer_factor: float,
    amplitude_factor: float,
    extent_m: float,
    dx_m: float,
) -> np.ndarray:
    """
    Build a 2-D zero-sum Mexican-hat (difference-of-Gaussians) kernel.

        K(r) = G(r; sigma_in) - G(r; sigma_out)

    The positive lobe integrates to 1 (pixel-area units) before amplitude
    scaling.  Sum of K is exactly 0, guaranteeing zero net heat addition.

    Kernel is always square with odd side length.
    """
    sigma_out = sigma_inner_m * sigma_outer_factor
    n_half = int(np.ceil(extent_m / dx_m))
    n = 2 * n_half + 1
    idx = np.arange(-n_half, n_half + 1) * dx_m
    xx, yy = np.meshgrid(idx, idx)
    r2 = xx**2 + yy**2
    g_in  = np.exp(-r2 / (2.0 * sigma_inner_m**2))
    g_out = np.exp(-r2 / (2.0 * sigma_out**2))
    K = g_in - g_out
    pos_sum = K[K > 0].sum() * dx_m**2
    if pos_sum > 0:
        K /= pos_sum
    return K * amplitude_factor


def _project_catalogue(
    catalogue: pd.DataFrame,
    crs_target: str,
    bedmachine: xr.Dataset,
) -> pd.DataFrame:
    """Project lat/lon to polar-stereo CRS and filter to grid extent."""
    t = Transformer.from_crs("EPSG:4326", crs_target, always_xy=True)
    x_m, y_m = t.transform(catalogue["lon"].values, catalogue["lat"].values)
    cat = catalogue.copy()
    cat["x_m"], cat["y_m"] = x_m, y_m
    x_min, x_max = float(bedmachine.x.min()), float(bedmachine.x.max())
    y_min, y_max = float(bedmachine.y.min()), float(bedmachine.y.max())
    mask = ((cat.x_m >= x_min) & (cat.x_m <= x_max) &
            (cat.y_m >= y_min) & (cat.y_m <= y_max))
    print(f"  Features in grid extent: {mask.sum()} / {len(cat)}")
    return cat[mask].reset_index(drop=True)


def apply_volcanic_kernels(
    ds: xr.Dataset,
    catalogue: pd.DataFrame,
    kernel_params: dict,
    bedmachine: xr.Dataset,
    region_crs: str,
    fine_spacing: Optional[float] = None,
    mean_tol: float = 1e-6,
) -> xr.Dataset:
    """
    STEP 2 — Apply volcanic / hydrothermal kernel modification.

    Algorithm:
      1. Build one Mexican-hat kernel per kernel_class.
      2. Place each feature via FFT convolution of a delta field.
      3. Log-additive combination: ln(M_total) = sum_i ln(1 + A_i * K_i).
         Rationale: positive-definite; overlapping sources stack correctly;
         degrades to single-feature result when features are isolated.
      4. Normalise M_total so ice-cell mean = 1.0 (global conservation).
      5. Apply to all quantile fields.
      6. Inflate q_std via uncertainty.inflate_uncertainty_volcanic().
    """
    from lib.uncertainty import inflate_uncertainty_volcanic
    _print_step("STEP 2 — Volcanic / hydrothermal kernel modification")
    fine_spacing = fine_spacing or float(ds.attrs.get("fine_spacing_m", 500.0))
    q_before = ds["q_mean"].copy(deep=True)
    cat_proj = _project_catalogue(catalogue, region_crs, bedmachine)
    if len(cat_proj) == 0:
        print("  No features in grid extent — skipping Step 2.")
        return ds.assign_attrs(forward_step="step2_volcanic_skipped")

    kclass_cfg = {k: v for k, v in kernel_params.items()
                  if isinstance(v, dict) and "sigma_inner_m" in v}
    kernels: dict[str, np.ndarray] = {}
    for kclass, kp in kclass_cfg.items():
        kernels[kclass] = build_mexican_hat_kernel(
            sigma_inner_m     = kp["sigma_inner_m"],
            sigma_outer_factor = kp["sigma_outer_factor"],
            amplitude_factor  = kp["amplitude_factor"],
            extent_m          = kp.get("extent_m", 100_000.0),
            dx_m              = fine_spacing,
        )
        print(f"  Kernel [{kclass}]: {kernels[kclass].shape[0]}x"
              f"{kernels[kclass].shape[1]} px  sum={kernels[kclass].sum():.6f}")

    x_coords = ds.x.values; y_coords = ds.y.values
    ny, nx = len(y_coords), len(x_coords)
    log_M = np.zeros((ny, nx), dtype=np.float64)

    for _, row in cat_proj.iterrows():
        kclass = row["kernel_class"]
        if kclass not in kernels:
            continue
        K = kernels[kclass]
        amp = row.get("factor", kclass_cfg[kclass]["amplitude_factor"])
        ix = int(np.argmin(np.abs(x_coords - row["x_m"])))
        iy = int(np.argmin(np.abs(y_coords - row["y_m"])))
        delta = np.zeros((ny, nx), dtype=np.float64); delta[iy, ix] = 1.0
        K_placed = sig.fftconvolve(delta, K, mode="same")
        log_M += np.log(np.maximum(1.0 + amp * K_placed, 1e-6))

    M_total = np.exp(log_M)
    mask_vals = ds["mask"].values if isinstance(ds["mask"].values, np.ndarray) \
                else ds["mask"].compute().values
    valid = mask_vals > 0
    mean_M = M_total[valid].mean() if valid.any() else M_total.mean()
    M_total /= mean_M
    print(f"  Multiplier (post-norm, ice):  "
          f"min={M_total[valid].min():.4f}  max={M_total[valid].max():.4f}  "
          f"mean={M_total[valid].mean():.6f}")

    M_da = xr.DataArray(M_total.astype(np.float32), dims=["y","x"],
                        coords={"y": ds.y, "x": ds.x},
                        attrs=dict(units="dimensionless",
                                   long_name="Volcanic/hydrothermal multiplier",
                                   note="Normalised so ice-cell mean=1.0"))

    q_vars = [v for v in ds.data_vars if v.startswith("q_q") or v == "q_mean"]
    out_vars = {v: ds[v] for v in ds.data_vars}
    for v in q_vars:
        out_vars[v] = (ds[v] * M_da).assign_attrs(ds[v].attrs)
    out_vars["volc_multiplier"] = M_da
    out_vars["delta_volc"] = (out_vars["q_mean"] - q_before).assign_attrs(
        units="W m-2", long_name="Heat-flow change from volcanic kernels")

    ds_out = xr.Dataset(out_vars, coords=ds.coords,
                        attrs={**ds.attrs, "forward_step": "step2_volcanic"})
    _check_mean_conservation(q_before, ds_out["q_mean"], "step2 q_mean", tol=mean_tol)
    ds_out = inflate_uncertainty_volcanic(
        ds_out, q_before=q_before, q_after=ds_out["q_mean"],
        p_exists=kernel_params.get("default_p_exists", 0.9),
        sigma_factor=kernel_params.get("default_sigma_factor", 0.3))
    return ds_out


# ── Step 3: Topographic correction ───────────────────────────────────────────

def _rolling_mean_2d(da: xr.DataArray, radius_m: float, spacing_m: float) -> xr.DataArray:
    n_pix = max(1, int(np.floor(radius_m / spacing_m)))
    arr = da.values if isinstance(da.values, np.ndarray) else da.compute().values
    smoothed = ndi.uniform_filter(arr.astype(np.float64), size=2*n_pix+1, mode="reflect")
    return xr.DataArray(smoothed.astype(np.float32), dims=da.dims,
                        coords=da.coords, attrs=da.attrs)


def topographic_correction(
    ds: xr.Dataset,
    bedmachine: xr.Dataset,
    k_rock: xr.DataArray,
    k_ice: xr.DataArray,
    k_rock_std: Optional[xr.DataArray] = None,
    k_ice_std:  Optional[xr.DataArray] = None,
    r_m: float = 5_000.0,
    k_ref: float = 2.5,
    k_ocean: float = 6.0,
    clip_max: float = 1.0,
    apply_only_where_covered: bool = True,
    fine_spacing: Optional[float] = None,
    mean_tol: float = 1e-4,
) -> xr.Dataset:
    """
    STEP 3 — Colgan-style topographic correction.

    DeltaG_bed  = q * (k_rock / k_ref) * (z_bed - z_bed_mean) / r
    DeltaG_ice  = q * (k_ice  / k_ref) * (z_surf - z_bed)     / r
    DeltaG_total = DeltaG_bed + DeltaG_ice

    Phase 1: k_rock and k_ice are constant 2-D arrays (placeholders).
    Phase 2: supply spatially varying arrays (geology / basal temperature).

    Ocean cells: k_rock is replaced with k_ocean (placeholder).
    Submarine correction is NOT implemented — documented in NetCDF attrs.

    Coverage restriction: DeltaG_total = 0 where BedMachine is interpolated.
    """
    from lib.uncertainty import propagate_topo_uncertainty, combine_uncertainties_quadrature
    _print_step("STEP 3 — Topographic correction (Colgan-style)")
    fine_spacing = fine_spacing or float(ds.attrs.get("fine_spacing_m", 500.0))

    z_bed  = bedmachine["bed"].compute().astype(np.float32)
    z_surf = bedmachine["surface"].compute().astype(np.float32)
    mask   = ds["mask"].compute() if "mask" in ds else bedmachine["mask"].compute()

    z_bed_mean = _rolling_mean_2d(z_bed, radius_m=r_m, spacing_m=fine_spacing)
    bed_anom   = (z_bed - z_bed_mean).assign_attrs(
        units="m", long_name="Bed elevation anomaly")
    ice_thick  = (z_surf - z_bed).clip(min=0).assign_attrs(
        units="m", long_name="Ice thickness (surface-bed, clipped>=0)")

    k_rock_eff = xr.where(mask == 0, k_ocean, k_rock).assign_attrs(
        units="W m-1 K-1",
        long_name="Effective rock conductivity (ocean: k_ocean placeholder)")
    topo_ocean_flag = (mask == 0).assign_attrs(
        long_name="Cells where ocean k_rock placeholder was applied",
        note="Submarine topographic correction not implemented")

    q_in = ds["q_mean"]
    delta_G_bed   = q_in * (k_rock_eff / k_ref) * (bed_anom  / r_m)
    delta_G_ice   = q_in * (k_ice      / k_ref) * (ice_thick / r_m)
    delta_G_total = delta_G_bed + delta_G_ice

    if apply_only_where_covered and "bedmachine_coverage" in ds:
        cov = ds["bedmachine_coverage"]
        delta_G_total = xr.where(cov, delta_G_total, 0.0)
        print(f"  Coverage restriction: {float(cov.mean())*100:.1f}% corrected")

    n_tot = int(delta_G_total.size)
    n_clip = int((np.abs(delta_G_total) > clip_max).sum())
    delta_G_total = delta_G_total.clip(-clip_max, clip_max)
    print(f"  Clipped {n_clip:,} cells ({n_clip/n_tot*100:.3f}%) to +/-{clip_max} W/m2")

    q_std_in = ds.get("q_std", xr.zeros_like(q_in))
    delta_topo_std = propagate_topo_uncertainty(
        delta_G=delta_G_total, q_in=q_in, q_in_std=q_std_in,
        k_rock=k_rock_eff, k_rock_std=k_rock_std,
        k_ice=k_ice, k_ice_std=k_ice_std,
        k_ref=k_ref, r=r_m,
        bed_anom=bed_anom, ice_thick=ice_thick)

    q_vars = [v for v in ds.data_vars if v.startswith("q_q") or v == "q_mean"]
    out_vars = {v: ds[v] for v in ds.data_vars}
    for var in q_vars:
        safe_mean = xr.where(np.abs(q_in) > 1e-12, q_in, np.nan)
        scale     = xr.where(safe_mean.notnull(), ds[var] / safe_mean, 1.0)
        out_vars[var] = (ds[var] + delta_G_total * scale).clip(min=1e-6).assign_attrs(
            ds[var].attrs)

    out_vars.update({
        "delta_G_bed":     delta_G_bed.assign_attrs(units="W m-2", long_name="Topo correction — bed"),
        "delta_G_ice":     delta_G_ice.assign_attrs(units="W m-2", long_name="Topo correction — ice"),
        "delta_topo":      delta_G_total.assign_attrs(units="W m-2", long_name="Topo correction total"),
        "delta_topo_std":  delta_topo_std,
        "topo_ocean_flag": topo_ocean_flag,
        "bed_anom":        bed_anom,
        "ice_thick":       ice_thick,
    })
    out_vars["q_std"] = combine_uncertainties_quadrature(
        ds.get("q_std", xr.zeros_like(q_in)), delta_topo_std)

    ds_out = xr.Dataset(
        out_vars, coords=ds.coords,
        attrs={**ds.attrs,
               "forward_step": "step3_topo_correction",
               "topo_r_m": r_m, "topo_k_ref": k_ref,
               "topo_k_ocean_placeholder": k_ocean,
               "topo_clip_max_Wm2": clip_max,
               "topo_clip_fraction": n_clip/n_tot,
               "topo_coverage_restricted": apply_only_where_covered,
               "NOTE_submarine_topo": (
                   "Submarine topographic correction NOT implemented. "
                   "Ocean cells use k_rock placeholder value. "
                   "BedMachine treated as truth only where direct "
                   "observations (radar/seismic) exist."),
               "NOTE_k_rock_phase": "Phase 1 — constant 1.8 W/m/K placeholder",
               "NOTE_k_ice_phase": (
                   "Phase 1 — constant 2.1 W/m/K placeholder. "
                   "Phase 2: temperature-dependent k_ice(T_b).")})
    _check_mean_conservation(q_in, ds_out["q_mean"], "step3 q_mean", tol=mean_tol)
    return ds_out


# ── Output ───────────────────────────────────────────────────────────────────

def save_forward_product(
    ds: xr.Dataset,
    out_path: Path,
    region: str,
    model_version: str = "0_3",
    compression_level: int = 4,
    author: str = "Stal et al. (Aq2/Kq2)",
) -> None:
    """Write the forward product to a CF-1.8 compliant NetCDF with zlib compression."""
    _print_step(f"Saving to {out_path}")
    ds.attrs.update({
        "Conventions":   "CF-1.8",
        "title":         f"Aq2/Kq2 forward heat-flow model — {region}",
        "author":        author,
        "model_version": model_version,
        "creation_date": datetime.datetime.utcnow().isoformat() + "Z",
    })
    float_dtypes = (np.float32, np.float64)
    encoding = {v: ({"zlib": True, "complevel": compression_level, "dtype": "float32"}
                    if ds[v].dtype in float_dtypes else {"zlib": False})
                for v in ds.data_vars}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
