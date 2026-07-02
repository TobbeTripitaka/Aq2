# lib/utils.py — shared utility functions
import numpy as np
import pandas as pd
from pyproj import Transformer


def latlon_to_epsg(df, epsg_tgt, lat_col="lat", lon_col="lon",
                   x_col="x", y_col="y"):
    """
    Convert lat/lon columns (degrees, EPSG:4326) to projected x/y.

    Parameters
    ----------
    df       : DataFrame with lat_col and lon_col
    epsg_tgt : target EPSG code (int or str, e.g. 3031)
    Returns  : copy of df with new x_col and y_col columns
    """
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_tgt}", always_xy=True)
    x, y = t.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy())
    out = df.copy()
    out[x_col] = x
    out[y_col] = y
    return out


def sort_quantiles(*arrays):
    """
    Enforce monotonicity across matched quantile arrays element-wise.

    Accepts any number of 1-D arrays (e.g. q05, q50, q95) and returns them
    sorted so that arrays[0][i] <= arrays[1][i] <= ... for every index i.

    Applied to all three models (GB, QRF, Similarity) as a safety layer.
    QRF guarantees this internally; the sort costs nothing and makes the
    guarantee explicit even after calibration or log back-transform.

    Example
    -------
    q05, q50, q95 = sort_quantiles(q05, q50, q95)
    """
    stacked = np.stack(arrays, axis=0)
    sorted_ = np.sort(stacked, axis=0)
    return tuple(sorted_[i] for i in range(len(arrays)))


# ── Model / calibration helpers (shared by 4a/4b/4c) ─────────────────────────
# These were previously copy-pasted inline into 4a_QRF_MODEL, 4b_GBM_MODEL and
# 4c_SIM_MODEL. Moved here to remove drift risk. Behaviour is unchanged; the
# per-notebook constants (q_clip_min/max, bin width, percentile knots) are now
# passed in explicitly so callers keep full control.

def weighted_percentile(vals, weights, pcts):
    """Weighted quantile (ignores NaN/inf). pcts are in [0, 100]."""
    mask = np.isfinite(vals) & np.isfinite(weights)
    vs, ws = vals[mask], weights[mask]
    idx  = np.argsort(vs)
    cumw = np.cumsum(ws[idx]) / ws.sum()
    return np.interp(pcts / 100.0, cumw, vs[idx])


def build_correction_spline(y_pred_cal, y_true_cal, w_cal, n_pctls=200):
    """
    PCHIP spline mapping model predictions -> empirical reference distribution.
    Fitted on calibration slice only (no test leakage).
    Addresses regression-to-the-mean / centre-of-distribution bias.

    Returns (spline, pred_p, true_p).
    """
    from scipy.interpolate import PchipInterpolator
    pctls      = np.linspace(1, 99, n_pctls)
    pred_p     = weighted_percentile(y_pred_cal, w_cal, pctls)
    true_p     = weighted_percentile(y_true_cal, w_cal, pctls)
    _, keep    = np.unique(pred_p, return_index=True)
    spline     = PchipInterpolator(pred_p[keep], true_p[keep], extrapolate=True)
    return spline, pred_p, true_p


def empirical_entropy(vals, q_min, q_max, bin_width=0.010):
    """Shannon entropy (bits/nats per scipy default) of the value histogram."""
    from scipy.stats import entropy as scipy_entropy
    bins   = np.arange(q_min, q_max + bin_width, bin_width)
    counts, _ = np.histogram(vals[np.isfinite(vals)], bins=bins)
    probs  = counts / counts.sum()
    return float(scipy_entropy(probs[probs > 0]))


def scatter_residuals_fig(rows, suptitle, savepath, q_min, q_max, fig_dpi=150):
    """
    rows: list of (y_true, y_raw, y_corr, label, color)
    3-column figure: uncorrected | corrected | residual overlay
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    lim  = (q_min * 1e3, q_max * 1e3)
    bins = np.linspace(lim[0], lim[1], 80)
    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 5 * len(rows)))
    if len(rows) == 1: axes = axes[None, :]

    for i, (yt, yraw, ycorr, label, color) in enumerate(rows):
        yt_mW, yr_mW, yc_mW = yt*1e3, yraw*1e3, ycorr*1e3
        for j, (yp, cmap, alpha, lbl) in enumerate([
                (yr_mW, 'Oranges', 0.8, 'uncorr'),
                (yc_mW, 'Blues',   0.8, 'corr')]):
            ax = axes[i, j]
            h, xe, ye = np.histogram2d(yt_mW, yp, bins=bins)
            h = np.ma.masked_where(h == 0, h)
            ax.pcolormesh(xe, ye, h.T, cmap=cmap,
                          norm=plt.matplotlib.colors.LogNorm(),
                          rasterized=True)
            ax.plot(lim, lim, 'k--', lw=0.9, label='1:1')
            r2 = r2_score(yt_mW, yp)
            ax.set_title(f'{label} — {lbl}  R²={r2:.3f}', fontsize=9)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel('Observed [mW/m²]'); ax.set_ylabel('Predicted [mW/m²]')
            ax.legend(fontsize=7)

        ax = axes[i, 2]
        rr = yr_mW - yt_mW
        rc = yc_mW - yt_mW
        ax.hist(rr, bins=60, color='#FF9800', alpha=0.5, edgecolor='none',
                label=f'uncorr  bias={np.mean(rr):.1f} σ={np.std(rr):.1f}')
        ax.hist(rc, bins=60, color=color, alpha=0.65, edgecolor='none',
                label=f'corr    bias={np.mean(rc):.1f} σ={np.std(rc):.1f}')
        ax.axvline(0, color='k', lw=0.9)
        ax.set_xlabel('Residual [mW/m²]'); ax.set_ylabel('Count')
        ax.set_xlim(-200, 200); ax.set_title(f'{label} — residuals', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(savepath, dpi=fig_dpi, bbox_inches='tight')
    plt.show()
    print(f'Saved {savepath}')


import pyproj
import xarray as xr
from lib.agrid import Grid  # adjust import to your package


def grid_from_xr_dataset(ds: xr.Dataset, name: str = "Grid", **kwargs) -> Grid:
    """
    Build an agrid.Grid from an xarray Dataset with projected coordinates,
    copying all data variables onto the Grid as numpy arrays.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'x' and 'y' dimension coordinates and
        ds.attrs['epsg'] (int or str) defining the projected CRS.
    name : str
        Name passed to Grid.__init__.
    **kwargs
        Extra keyword arguments forwarded to Grid.__init__
        (e.g. verbose=True, garbage_collect=False).

    Returns
    -------
    Grid
        Fully initialised Grid with .x, .y, .lat, .lon,
        .reshape_tuple, .nn, and all dataset variables populated.
    """
    # --- 1. Pull projected coordinates ---
    x_1d = ds["x"].values.astype(float)
    y_1d = ds["y"].values.astype(float)

    ny, nx = len(y_1d), len(x_1d)

    xx, yy = np.meshgrid(x_1d, y_1d)     # both (ny, nx)
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    # --- 2. Reproject to geographic lon/lat ---
    epsg = int(ds.attrs["epsg"])
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )
    lon_flat, lat_flat = transformer.transform(x_flat, y_flat)

    # --- 3. Construct Grid ---
    g = Grid(
        lats=lat_flat,
        lons=lon_flat,
        x=x_flat,
        y=y_flat,
        name=name,
        regular_grid=(ny, nx),
        crs=epsg,
        **kwargs,
    )

    # --- 4. Copy all data variables ---
    for var in ds.data_vars:
        da = ds[var]
        arr = da.values

        # Flatten 2-D (y, x) arrays to match the Grid's flat point order.
        # Higher-dimensional vars (e.g. z, time, y, x) are left as-is so
        # the leading extra axes are preserved and only (y, x) are flattened.
        if arr.ndim == 2 and arr.shape == (ny, nx):
            arr = arr.ravel()
        elif arr.ndim > 2 and arr.shape[-2:] == (ny, nx):
            arr = arr.reshape(*arr.shape[:-2], ny * nx)

        setattr(g, var, arr)

        # Mirror into the internal DataFrame for 1-D scalar fields
        if arr.ndim == 1 and len(arr) == len(g.df):
            g.df[var] = arr

    return g
