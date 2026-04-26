"""
lib/uncertainty.py
==================
Uncertainty propagation for the Aq2/Kq2 forward model.

Phase 1: parametric (analytical) propagation — fast, Gaussian assumption.
Phase 2: ensemble / Monte Carlo — stub raises NotImplementedError.
"""
from __future__ import annotations
import numpy as np
import xarray as xr
from typing import Optional


def quantile_to_sigma(q05: xr.DataArray, q95: xr.DataArray) -> xr.DataArray:
    """Gaussian-equivalent sigma from the 90% credible interval."""
    return (q95 - q05) / (2.0 * 1.6449)


def compute_uncertainty_summary(ds: xr.Dataset) -> xr.Dataset:
    """
    Derive compact uncertainty fields from quantile variables in ds.

    Accepts canonical forward names (q_q05 … q_q95) or raw QRF names
    (qrf_q05_corr … qrf_q95_corr).

    Adds / updates:
        q_std          Gaussian-equivalent sigma from 90% CI
        q_iqr          q75 - q25
        q_cv           coefficient of variation = q_std / q_mean  (clipped 0-10)
        q_ci90_lower   q05
        q_ci90_upper   q95
    """
    def _get(ds, fwd, raw):
        return ds[fwd] if fwd in ds else ds[raw]

    q05 = _get(ds, "q_q05", "qrf_q05_corr")
    q25 = _get(ds, "q_q25", "qrf_q25_corr")
    q50 = _get(ds, "q_q50", "qrf_q50_corr")
    q75 = _get(ds, "q_q75", "qrf_q75_corr")
    q95 = _get(ds, "q_q95", "qrf_q95_corr")

    q_std = quantile_to_sigma(q05, q95)
    q_iqr = q75 - q25
    q_cv  = xr.where(q50 > 1e-9, q_std / q50, np.nan).clip(0, 10)

    return ds.assign({
        "q_std": q_std.assign_attrs(
            units="W m-2",
            long_name="Gaussian-equivalent sigma (90% CI / 2z)"),
        "q_iqr": q_iqr.assign_attrs(
            units="W m-2", long_name="Interquartile range q75-q25"),
        "q_cv": q_cv.assign_attrs(
            units="dimensionless", long_name="Coefficient of variation q_std/q_mean"),
        "q_ci90_lower": q05.assign_attrs(
            units="W m-2", long_name="Lower bound 90% CI (q05)"),
        "q_ci90_upper": q95.assign_attrs(
            units="W m-2", long_name="Upper bound 90% CI (q95)"),
    })


def inflate_uncertainty_volcanic(
    ds: xr.Dataset,
    q_before: xr.DataArray,
    q_after: xr.DataArray,
    p_exists: float = 0.9,
    sigma_factor: float = 0.3,
) -> xr.Dataset:
    """
    Add volcanic kernel uncertainty in quadrature.

    Variance contribution:
        var_inflate = (q_after - q_before)^2 * (sigma_factor^2 + (1 - p_exists))

    sigma_factor: fractional amplitude uncertainty.
    p_exists:     prior P(feature exists) — currently a global placeholder (0.9).
                  Per-feature variation deferred to Phase 2 via catalogue column.

    Stores:
        q_std              updated in quadrature
        volc_inflate_factor  diagnostic ratio new/old sigma
    """
    delta       = q_after - q_before
    var_inflate = delta**2 * (sigma_factor**2 + (1.0 - p_exists))
    old_std = ds.get("q_std", xr.zeros_like(q_before))
    new_std = np.sqrt(old_std**2 + var_inflate)
    inflate_factor = xr.where(old_std > 1e-12, new_std / old_std, 1.0)
    inflate_factor.attrs.update(
        units="dimensionless",
        long_name="Uncertainty inflation factor from volcanic/hydrothermal kernels")
    new_std.attrs.update(
        units="W m-2", long_name="Heat-flow std (cumulative, post-volcanic)")
    return ds.assign({"q_std": new_std, "volc_inflate_factor": inflate_factor})


def propagate_topo_uncertainty(
    delta_G: xr.DataArray,
    q_in: xr.DataArray,
    q_in_std: xr.DataArray,
    k_rock: xr.DataArray,
    k_rock_std: Optional[xr.DataArray],
    k_ice: xr.DataArray,
    k_ice_std: Optional[xr.DataArray],
    k_ref: float,
    r: float,
    bed_anom: xr.DataArray,
    ice_thick: xr.DataArray,
) -> xr.DataArray:
    """
    First-order error propagation for the Colgan topographic correction.

    Phase 1 (k_rock_std / k_ice_std = None):
        sigma_deltaG = |delta_G / q_in| * q_in_std

    Phase 2 (conductivity std arrays provided):
        Full propagation including conductivity uncertainty:
            d(DeltaG)/d(k_rock) = q_in * bed_anom  / (k_ref * r)
            d(DeltaG)/d(k_ice)  = q_in * ice_thick / (k_ref * r)
    """
    safe_q = xr.where(q_in > 1e-12, q_in, np.nan)
    dG_dq  = xr.where(safe_q.notnull(), delta_G / safe_q, 0.0)
    var_from_q = (dG_dq * q_in_std) ** 2

    if k_rock_std is not None and k_ice_std is not None:
        dG_dkrock = q_in * bed_anom  / (k_ref * r)
        dG_dkice  = q_in * ice_thick / (k_ref * r)
        var_from_k = (dG_dkrock * k_rock_std)**2 + (dG_dkice * k_ice_std)**2
    else:
        var_from_k = xr.zeros_like(var_from_q)

    return np.sqrt(var_from_q + var_from_k).assign_attrs(
        units="W m-2",
        long_name="Topographic correction uncertainty (1-sigma, Phase 1: from q_in only)")


def combine_uncertainties_quadrature(*sigma_fields: xr.DataArray) -> xr.DataArray:
    """Return sqrt(sum sigma_i^2) — quadrature combination of independent sources."""
    total_var = sum(s**2 for s in sigma_fields)
    return np.sqrt(total_var).assign_attrs(
        units="W m-2", long_name="Combined heat-flow uncertainty (quadrature)")


def ensemble_forward(forward_func, ds_5km, param_sampler, n_samples=200, seed=42):
    """[PHASE 2 NOT IMPLEMENTED] Monte Carlo ensemble propagation."""
    raise NotImplementedError(
        "Monte Carlo ensemble mode deferred to Phase 2. "
        "Set mode='parametric' in forward_params.yaml for now.")
