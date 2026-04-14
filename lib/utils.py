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
