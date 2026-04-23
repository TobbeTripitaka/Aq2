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


import numpy as np
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
