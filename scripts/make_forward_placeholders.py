#!/usr/bin/env python
"""Generate Phase-1 placeholder input NetCDFs for the forward model (notebook 8).

The forward model reads three gridded input fields per region:

  1. Basal ice temperature   T_b   (drives k_ice(T_b), see lib.forward)
  2. Upper-crust conductivity k_crust
  3. Basal hydrology          hydro (a placeholder flux/index, unused in Phase 1)

In Phase 1 every field is a spatially *constant* placeholder so the pipeline can
run end-to-end before the real physical fields exist. The constants come from
config:

    PLACEHOLDER_BASAL_TEMP_C   = 0.0   # deg C  (freezing point)
    PLACEHOLDER_K_UPPER_CRUST  = 2.5   # W m-1 K-1
    PLACEHOLDER_HYDROLOGY      = 0.0   # dimensionless, no flux

Each field is written on the SAME 5 km grid as the notebook-7 fused ensemble
(``output/ensemble/{key}_ensemble_v{MODEL_VERSION}.nc``). Reading the grid from
the ensemble product guarantees the placeholders align cell-for-cell with the
forward-model input, so no regridding is needed downstream. The forward model
oversamples these to the BedMachine target spacing (500 m Antarctica /
150 m Greenland) alongside the heat-flow grid.

Antarctica and Greenland get *individual* files, named per config
(FORWARD_BASAL_TEMP_NC / FORWARD_K_CRUST_NC / FORWARD_HYDROLOGY_NC).

Usage
-----
    python scripts/make_forward_placeholders.py                # both regions
    python scripts/make_forward_placeholders.py --region antarctica
    python scripts/make_forward_placeholders.py --overwrite    # replace existing

Replace the constants (or this whole script) with real-field ingestion in
Phase 2; the file names and variable names are the stable contract the notebook
depends on.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ── Import project config ─────────────────────────────────────────────────────
# Run from the repository root (so ``import config`` resolves) or from anywhere
# by adding the repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402


# region-key ("antarctica"/"greenland") -> ensemble file key ("ant"/"grl")
_REGION_TO_ENS_KEY = {"antarctica": "ant", "greenland": "grl"}
_REGION_TO_EPSG = {"antarctica": 3031, "greenland": 3413}


def _ensemble_path(ens_key: str) -> Path:
    """Path to the notebook-7 fused ensemble grid for a region key (ant/grl)."""
    return config.ENSEMBLE_DIR / f"{ens_key}_ensemble_v{config.MODEL_VERSION}.nc"


def _load_grid_coords(region: str):
    """Return (y, x) coordinate arrays for the region's 5 km fusion grid.

    Reads the notebook-7 ensemble product. Falls back to the QRF target grid if
    the ensemble file is missing, so placeholders can still be built before
    notebook 7 has run (with a clear warning).
    """
    ens_key = _REGION_TO_ENS_KEY[region]
    ens_path = _ensemble_path(ens_key)
    if ens_path.exists():
        with xr.open_dataset(ens_path) as ds:
            return ds["y"].values.copy(), ds["x"].values.copy(), str(ens_path.name)

    # Fallback: the stage-5 QRF target grid shares the same 5 km definition.
    fallback = {"antarctica": config.ant_Aq2_qrf_nc,
                "greenland":  config.grl_Kq2_qrf_nc}[region]
    if fallback.exists():
        print(f"  ! ensemble grid not found ({ens_path.name}); "
              f"falling back to {fallback.name}")
        with xr.open_dataset(fallback) as ds:
            return ds["y"].values.copy(), ds["x"].values.copy(), str(fallback.name)

    raise FileNotFoundError(
        f"No grid found for {region}: neither {ens_path} nor {fallback} exist. "
        f"Run notebook 7 (or at least notebook 5) first.")


def _const_field(y, x, value: float, dtype=np.float32) -> xr.DataArray:
    """A constant-valued (y, x) DataArray on the given coordinates."""
    data = np.full((y.size, x.size), value, dtype=dtype)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


def _base_attrs(region: str, source_grid: str) -> dict:
    return {
        "region": region,
        "crs": f"EPSG:{_REGION_TO_EPSG[region]}",
        "epsg": _REGION_TO_EPSG[region],
        "model_version": str(config.MODEL_VERSION),
        "phase": "1-placeholder",
        "source_grid": source_grid,
        "author": config.NETCDF_AUTHOR,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "history": "created by scripts/make_forward_placeholders.py",
        "WARNING": ("Phase-1 PLACEHOLDER: spatially constant value, not a real "
                    "physical field. Replace before scientific use."),
    }


def _write(ds: xr.Dataset, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"  · exists, skipping (use --overwrite): {path.name}")
        return
    enc = {v: {"zlib": True, "complevel": config.netcdf_compression_level}
           for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"  ✓ wrote {path}")


def make_region(region: str, overwrite: bool) -> None:
    if region not in _REGION_TO_ENS_KEY:
        raise ValueError(f"region must be one of {list(_REGION_TO_ENS_KEY)}, "
                         f"got {region!r}")
    print(f"[{region}]")
    y, x, source_grid = _load_grid_coords(region)
    print(f"  grid: {y.size} x {x.size} cells (from {source_grid})")
    base = _base_attrs(region, source_grid)

    # 1. Basal ice temperature -------------------------------------------------
    t_b = _const_field(y, x, config.PLACEHOLDER_BASAL_TEMP_C)
    t_b.attrs.update(units="degC", long_name="Basal ice temperature (placeholder)",
                     standard_name="land_ice_basal_temperature",
                     note="single-value proxy for the whole ice column; drives k_ice(T_b)")
    ds_t = xr.Dataset({"t_basal": t_b}, coords={"y": y, "x": x})
    ds_t.attrs.update(base, title="Basal ice temperature (Phase-1 placeholder)",
                      placeholder_value_degC=config.PLACEHOLDER_BASAL_TEMP_C)
    _write(ds_t, config.FORWARD_BASAL_TEMP_NC[region], overwrite)

    # 2. Upper-crust thermal conductivity -------------------------------------
    k_c = _const_field(y, x, config.PLACEHOLDER_K_UPPER_CRUST)
    k_c.attrs.update(units="W m-1 K-1",
                     long_name="Upper-crust thermal conductivity (placeholder)",
                     note="typical crystalline upper-crust value")
    ds_k = xr.Dataset({"k_upper_crust": k_c}, coords={"y": y, "x": x})
    ds_k.attrs.update(base, title="Upper-crust conductivity (Phase-1 placeholder)",
                      placeholder_value_W_m_K=config.PLACEHOLDER_K_UPPER_CRUST)
    _write(ds_k, config.FORWARD_K_CRUST_NC[region], overwrite)

    # 3. Basal hydrology -------------------------------------------------------
    hyd = _const_field(y, x, config.PLACEHOLDER_HYDROLOGY)
    hyd.attrs.update(units="1", long_name="Basal hydrology index (placeholder)",
                     note="set to zero in Phase 1 (no hydrological advection)")
    ds_h = xr.Dataset({"hydrology": hyd}, coords={"y": y, "x": x})
    ds_h.attrs.update(base, title="Basal hydrology (Phase-1 placeholder)",
                      placeholder_value=config.PLACEHOLDER_HYDROLOGY)
    _write(ds_h, config.FORWARD_HYDROLOGY_NC[region], overwrite)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=["antarctica", "greenland", "both"],
                    default="both", help="region(s) to generate (default: both)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing placeholder files")
    args = ap.parse_args(argv)

    regions = (["antarctica", "greenland"] if args.region == "both"
               else [args.region])
    print(f"Writing Phase-1 placeholder inputs to {config.forward_input_dir}\n"
          f"  T_basal = {config.PLACEHOLDER_BASAL_TEMP_C} degC, "
          f"k_crust = {config.PLACEHOLDER_K_UPPER_CRUST} W/m/K, "
          f"hydrology = {config.PLACEHOLDER_HYDROLOGY}\n")
    for r in regions:
        make_region(r, args.overwrite)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
