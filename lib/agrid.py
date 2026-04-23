#!/usr/bin/env python3
# agrid.py v0.5.2b — Tobias Staal 2026
# tobias.staal@utas.edu.au
# MIT License

"""Grid class for Aq2/Kq2 observables management and plotting.

Changes in v0.5.2
-----------------
* Fixed two missing closing parentheses in organise_recipe() that caused
  the entire class to fail to parse (AttributeError on all methods).
* _compute_extent() restored to @staticmethod; call site in __init__
  updated to Grid._compute_extent(...).

Changes in v0.5.1
-----------------
* _grid_matches() static helper for grid-key filtering.
* organise_recipe() supports grid key as None/str/list, case-insensitive
  matching, whitespace stripping, and duplicate-label handling.

Changes in v0.5.0
-----------------
* PlottingMixin and XarrayMixin merged into Grid body (mixin removed).
* New method: Grid.quicklook(labels, dd, ncols=3, cell_size=3.0)
- Tiles multiple observables as small panels (no cartopy).
- Attempts imshow (regular grid) then falls back to scatter.
- Reads v_range and cmap from the recipe dict.
* Grid.look() unchanged (single-observable quick-check).
* Grid.map() unchanged (full cartopy production map).
"""

import os, gc, time, re, math, warnings, inspect, importlib
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy import ndimage
from pyproj import CRS, Geod, Transformer
geod = Geod(ellps="WGS84")

__version__ = "0.5.2"

import stripy
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import cartopy.feature as cfeature
from matplotlib import pyplot as plt

# =============================================================================
class Grid:
    """Multidimensional grid for import, storage, and visualisation of
    geophysical observables.

    Parameters
    ----------
    lats, lons : array-like
        Geographic coordinates (decimal degrees, WGS84).
    x, y : array-like, optional
        Projected coordinates (metres). Required for EPSG != 4326.
    name : str
        Grid name used to filter recipe entries with a ``grid`` key.
        Three canonical values: ``'IHFC'``, ``'Antarctica'``, ``'Greenland'``.
    crs : int
        EPSG code (default 4326).
    regular_grid : None | False | True | (ny, nx)
        If a 2-tuple, sets reshape_tuple to (ny, nx) directly.
        If True, infers shape from unique coordinate values.
    verbose : bool
    garbage_collect : bool
    log_file : str, optional
    """

    km = 1000

    # ------------------------------------------------------------------
    def __init__(self, lats, lons, x=None, y=None, m=None,
                 name="Grid", verbose=False, garbage_collect=False,
                 regular_grid=None, log_file=None, *args, **kwargs):

        self.verbose = verbose
        self.garbage_collect = garbage_collect
        self.log_file = log_file
        self.recipe = None
        self.name = name
        self.crs = kwargs.get("crs", 4326)

        self.lats = np.array(lats, dtype=float)
        self.lons = np.array(lons, dtype=float)
        self.rad_lats = np.radians(self.lats)
        self.rad_lons = np.radians(self.lons)

        assert len(self.lats) == len(self.lons), \
            f"lats/lons length mismatch ({len(self.lats)} vs {len(self.lons)})"

        self.df = pd.DataFrame({"lon": self.lons, "lat": self.lats})

        if x is not None and y is not None:
            x = np.array(x, dtype=float); y = np.array(y, dtype=float)
            self.df["x"] = x; self.df["y"] = y

        _pad = kwargs.get("extent_pad", 0.0)
        self.extent = kwargs.get("extent", None) or \
            Grid._compute_extent(self.df, self.crs, pad_frac=_pad)

        self.reshape_tuple = None
        if isinstance(regular_grid, tuple):
            self.nn = regular_grid
        if regular_grid is not None and regular_grid is not False:
            if isinstance(regular_grid, (tuple, list)) and len(regular_grid) == 2:
                ny, nx = regular_grid
                if ny * nx != len(self.df):
                    raise ValueError(
                        f"regular_grid ({ny}x{nx}={ny*nx}) != n_points ({len(self.df)})")
                self.reshape_tuple = (int(ny), int(nx))
            elif regular_grid is True:
                self._infer_regular_shape()
            else:
                raise TypeError(f"regular_grid must be None/False/True/(ny,nx)")

        if self.verbose:
            print(f'Grid "{self.name}" init: {len(self.lats)} nodes, '
                  f'reshape={self.reshape_tuple}')

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def dict_key(self, lst, key, value):
        return [d for d in lst if key in d and d[key] == value][0]

    def _log(self, msg, level="info", log_file=None):
        if level == "info" and not self.verbose:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        out = f"[{ts}] [{level.upper():5s}] {msg}"
        print(out)
        target = log_file or getattr(self, "log_file", None)
        if target:
            try:
                with open(target, "a") as f: f.write(out + "\n")
            except IOError:
                pass
    log = _log

    def _make_stats_string(self, data, label, t0):
        nv = np.sum(~np.isnan(data)); nt = len(data)
        return (f"{label} min={np.nanmin(data):.4g} mean={np.nanmean(data):.4g} "
                f"max={np.nanmax(data):.4g} valid={nv}/{nt} "
                f"[{time.time()-t0:.2f}s]")

    def _user_to_array(self, im_data):
        if isinstance(im_data, str):
            im_data = self.df[im_data].values if im_data else None
        if im_data is None:
            a = np.empty(len(self.df)); a[:] = np.nan; return a
        if isinstance(im_data, list): im_data = np.array(im_data)
        try: return np.copy(im_data).astype(float)
        except: return im_data.values.astype(float)

    def _epsg_to_cartopy(self, epsg, **kw):
        import cartopy.crs as ccrs
        if isinstance(epsg, str):
            m = re.search(r"\d+", epsg)
            if not m: raise ValueError(f"No EPSG code in: {epsg}")
            epsg = int(m.group())
        _map = {
            3031: ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71, **kw),
            3030: ccrs.Stereographic(central_latitude=-90, true_scale_latitude=71, **kw),
            4326: ccrs.PlateCarree(**kw),
            3413: ccrs.Stereographic(central_longitude=-45, central_latitude=90,
                                     true_scale_latitude=70, **kw),
        }
        if epsg in _map: return _map[epsg]
        try: return ccrs.epsg(epsg)
        except: raise KeyError(f"EPSG:{epsg} not supported")

    def _resolve_map_crs(self, map_crs=None, **kw):
        import cartopy.crs as ccrs
        if hasattr(map_crs, "__module__") and "cartopy" in map_crs.__module__:
            return map_crs
        if map_crs is not None:
            if isinstance(map_crs, str):
                m = re.search(r"\d+", map_crs)
                map_crs = int(m.group()) if m else map_crs
            return self._epsg_to_cartopy(map_crs, **kw)
        if hasattr(self, "crs") and self.crs:
            return self._epsg_to_cartopy(self.crs, **kw)
        return self._epsg_to_cartopy(4326, **kw)

    # ------------------------------------------------------------------
    # COORDINATE HELPERS
    # ------------------------------------------------------------------
    def _pre_processing(self, xs, ys, zs, d):
        for k, arr in [("x_preproc", xs), ("y_preproc", ys), ("z_preproc", zs)]:
            if k in d:
                try:
                    arr = d[k](arr)
                    if k == "x_preproc": xs = arr
                    elif k == "y_preproc": ys = arr
                    else: zs = arr
                except Exception as e:
                    self._log(f"Preprocessing {k} failed: {e}", level="warn")
        return xs, ys, zs

    def _remove_duplicate_lonlat(self, data, xcol=0, ycol=1):
        s = np.lexsort([data[:, xcol], data[:, ycol]])
        sd = data[s]
        mask = np.append([True], np.any(np.diff(sd[:, [xcol, ycol]], axis=0), 1))
        return sd[mask]

    def _mask_date_line(self, lats, lons, gates=(-89.99, 89.99, -179.99, 179.99)):
        return ((lats > gates[0]) & (lats < gates[1]) &
                (lons > gates[2]) & (lons < gates[3]))

    def _transform_and_mask(self, xx, yy, crs_src, crs):
        xx = np.array(xx, copy=True); yy = np.array(yy, copy=True)
        if crs_src != crs:
            t = Transformer.from_crs(CRS(crs_src), CRS(crs), always_xy=True)
            xx, yy = t.transform(xx, yy)
        return xx, yy, self._mask_date_line(yy, xx)

    def _infer_regular_shape(self, coord_x="lon", coord_y="lat"):
        x = self.df[coord_x].to_numpy(); y = self.df[coord_y].to_numpy()
        nx = np.unique(x).size; ny = np.unique(y).size
        if nx * ny != len(self.df):
            raise ValueError(f"Not regular: {nx}*{ny} != {len(self.df)}")
        self.reshape_tuple = (ny, nx)
        self._coord_x_name = coord_x; self._coord_y_name = coord_y

    def make_2D(self, label, coord_x='lon', coord_y='lat'):
        if self.reshape_tuple is None:
            self._infer_regular_shape(coord_x, coord_y)
        ny, nx = self.reshape_tuple
        arr = self.df[label].to_numpy(dtype=float)
        if arr.size != ny * nx:
            raise ValueError(f"make_2D {label}: size {arr.size} != {ny}x{nx}")
        return arr.reshape(ny, nx)

    # ------------------------------------------------------------------
    # DATA IMPORT
    # ------------------------------------------------------------------
    def _read_data(self, d, verbose=False):
        t = d.get("import_type")
        if t == "read_ascii":  return self.ascii_to_mesh(d)
        if t == "read_raster": return self.raster_to_mesh(d)
        if t == "read_grid":   return self.grid_to_mesh(d)
        if t == "read_shape":  return self.polygons_to_mesh(d)
        if t == "compute":     return d["func"](self, d)
        raise RuntimeError(f"Unknown import_type: {t}")

    def ascii_to_mesh(self, d, return_coords=False, **kw):
        t0 = time.time(); d = d.copy()
        d.setdefault("crs", "EPSG:4326"); d.setdefault("crs_src", "EPSG:4326")
        d.setdefault("x_col", 0); d.setdefault("y_col", 1); d.setdefault("z_col", 2)
        d.setdefault("interpol_method", "linear")
        kk = {k: d[k] for k in inspect.signature(pd.read_csv).parameters if k in d}
        tab = (pd.read_csv(**kk).dropna()
               .apply(pd.to_numeric, errors="coerce").values)
        tab = self._remove_duplicate_lonlat(tab, d["x_col"], d["y_col"])
        tab[:, d["x_col"]], tab[:, d["y_col"]], tab[:, d["z_col"]] = \
            self._pre_processing(tab[:, d["x_col"]], tab[:, d["y_col"]],
                                 tab[:, d["z_col"]], d)
        xx, yy, mask = self._transform_and_mask(
            tab[:, d["x_col"]], tab[:, d["y_col"]], d["crs_src"], d["crs"])
        zz = tab[:, d["z_col"]]
        st = stripy.sTriangulation(lons=np.radians(xx[mask]),
                                   lats=np.radians(yy[mask]), permute=True)
        _ord = {"nearest": 0, "nearest-neighbour": 0, "linear": 1, "cubic": 2}
        f, _ = st.interpolate(lats=self.rad_lats, lons=self.rad_lons,
                              zdata=zz[mask], order=_ord[d["interpol_method"]])
        del st
        f = np.asarray(f, dtype=float)
        if self.verbose: self._log(self._make_stats_string(f, d.get("label","ascii"), t0))
        if self.garbage_collect: gc.collect()
        return (xx, yy, zz, f) if return_coords else f

    def raster_to_mesh(self, d, **kw):
        t0 = time.time()
        pts = [(lon, lat) for lon, lat in zip(self.lons, self.lats)]
        with rasterio.open(d["filepath_or_buffer"]) as src:
            nodata = src.nodata
            data = np.array(list(src.sample(pts)), dtype=float).flatten()
        if nodata is not None:
            data[data == nodata] = np.nan
        if "z_preproc" in d:
            try:
                data = d["z_preproc"](data)
            except Exception as e:
                self._log(f"z_preproc failed for '{d.get('label')}': {e}", level="warn")
        if self.verbose: self._log(self._make_stats_string(data, d.get("label","raster"), t0))
        if self.garbage_collect: gc.collect()
        return data

    def grid_to_mesh(self, d, **kw):
        t0 = time.time(); d = d.copy()
        d.setdefault("crs", "EPSG:4326"); d.setdefault("crs_src", "EPSG:4326")
        d.setdefault("x_col", "longitude"); d.setdefault("y_col", "latitude")
        d.setdefault("depth", None); d.setdefault("data_col", "data")
        d.setdefault("interpol_method", "linear")
        ds = xr.open_dataset(d["filepath_or_buffer"])
        tab = ds.to_dataframe().reset_index().dropna().apply(pd.to_numeric, errors="coerce")
        if d["depth"] is not None:
            tab = tab.loc[tab["depth"] == d["depth"]]
        tab[d["x_col"]], tab[d["y_col"]], tab[d["data_col"]] = \
            self._pre_processing(tab[d["x_col"]], tab[d["y_col"]], tab[d["data_col"]], d)
        arr = tab[[d["x_col"], d["y_col"], d["data_col"]]].values
        arr = self._remove_duplicate_lonlat(arr, 0, 1)
        xx, yy, mask = self._transform_and_mask(arr[:, 0], arr[:, 1],
                                                d["crs_src"], d["crs"])
        st = stripy.sTriangulation(lons=np.radians(xx[mask]),
                                   lats=np.radians(yy[mask]), permute=False)
        _ord = {"nearest": 0, "nearest-neighbour": 0, "linear": 1, "cubic": 2}
        f, _ = st.interpolate(lats=self.rad_lats, lons=self.rad_lons,
                              zdata=arr[:, 2][mask], order=_ord[d["interpol_method"]])
        del st
        f = np.asarray(f, dtype=float)
        if self.verbose: self._log(self._make_stats_string(f, d.get("label","grid"), t0))
        if self.garbage_collect: gc.collect()
        return f

    def polygons_to_mesh(self, d, return_labels=False, **kw):
        t0 = time.time(); d = d.copy()
        d.setdefault("crs", "EPSG:4326"); d.setdefault("crs_src", "EPSG:4326")
        reg = gpd.read_file(d["filepath_or_buffer"]); reg.crs = d.get("crs_src","EPSG:4326")
        gmesh = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.lons, self.lats,
                                 crs=d["crs"])).to_crs(reg.crs)
        rm = gpd.sjoin_nearest(left_df=gmesh, right_df=reg, how="left", max_distance=1.)
        rm = rm[~rm.index.duplicated(keep="first")]
        labels, levels = pd.factorize(rm[d["attribute"]].values.flatten(),
                                      use_na_sentinel=False)
        labels = np.asarray(labels, dtype=float)
        if self.verbose: self._log(self._make_stats_string(labels, d.get("label","shape"), t0))
        if self.garbage_collect: gc.collect()
        return (labels, levels) if return_labels else labels

    # ------------------------------------------------------------------
    # RECIPE MANAGEMENT
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_matches(grid_key, grid_name, case_sensitive=False, strip=True):
        """Return True if the recipe's ``grid`` key matches this grid's name.

        Parameters
        ----------
        grid_key : None | str | list[str]
            Value of ``d["grid"]`` from the recipe entry.
            None  → shared (always matches).
            str   → matches that single grid name.
            list  → matches any grid name in the list.
        grid_name : str
            ``self.name`` of the Grid being built.
        case_sensitive : bool
            If False (default) comparison is case-insensitive.
        strip : bool
            If True (default) leading/trailing whitespace is stripped before
            comparing.
        Returns
        -------
        bool
        """
        if grid_key is None:
            return True  # shared entry

        def _norm(s):
            s = s.strip() if strip else s
            return s if case_sensitive else s.lower()

        gn = _norm(grid_name)

        if isinstance(grid_key, str):
            return _norm(grid_key) == gn

        if isinstance(grid_key, (list, tuple)):
            return any(_norm(k) == gn for k in grid_key)

        return False  # unknown type → skip

    def organise_recipe(self, dd=None, verbose=True,
                        case_sensitive=False, strip_names=True,
                        on_duplicate="warn"):
        """Filter recipe for this grid, resolve import types, and check files.

        Parameters
        ----------
        dd : list[dict], optional
            Full recipe list.  Falls back to ``self.recipe.dd`` if None.
        verbose : bool
            Log skipped / duplicate entries.
        case_sensitive : bool
            Grid-name comparison (default: False → case-insensitive).
        strip_names : bool
            Strip whitespace from grid names before comparing (default True).
        on_duplicate : str
            What to do when two kept entries share the same label:
            ``"warn"``  – keep both, print a warning, append ``_2``, ``_3`` …
            ``"error"`` – raise ``ValueError``.
            ``"ignore"`` – keep both silently (labels may collide in df).

        Returns
        -------
        list[dict]  dd_read + dd_compute (already filtered for this grid)
        """
        if dd is None:
            dd = getattr(getattr(self, "recipe", None), "dd", [])

        dd_read = []; dd_compute = []
        seen_labels = {}  # label → count of times seen so far

        for d in dd:
            d = d.copy()
            lb = d.get("label", "?")
            gk = d.get("grid", None)

            # ── grid-key filtering ────────────────────────────────────────
            if not self._grid_matches(gk, self.name,
                                      case_sensitive=case_sensitive,
                                      strip=strip_names):
                if verbose:
                    self._log(
                        f"organise_recipe: skipping '{lb}' "
                        f"(grid={gk!r} != '{self.name}')",
                        level="info"
                    )
                continue

            # ── set defaults ──────────────────────────────────────────────
            d.setdefault("import_type", self._define_data_type(d, verbose=verbose))
            d.setdefault("weight", 1.0)
            d.setdefault("sigma",  1.0)
            d.setdefault("cmap",   "viridis")

            # ── file-existence check for file-based imports ───────────────
            it = d.get("import_type")
            if it in ("read_ascii", "read_raster", "read_grid", "read_shape"):
                fp = d.get("filepath_or_buffer")
                if fp and isinstance(fp, str) and not os.path.exists(fp):
                    self._log(
                        f"File not found: {fp} (skipping '{lb}')",
                        level="warn"
                    )
                    continue

            # ── duplicate-label handling ──────────────────────────────────
            if lb in seen_labels:
                seen_labels[lb] += 1
                msg = (f"organise_recipe: label '{lb}' seen {seen_labels[lb]}x "
                    f"after grid filtering for '{self.name}' — keeping original label")
                if on_duplicate == "error":
                    raise ValueError(msg)
                if on_duplicate == "warn" or verbose:
                    self._log(msg, level="warn")
            else:
                seen_labels[lb] = 1

            (dd_compute if it == "compute" else dd_read).append(d)

        return dd_read + dd_compute

    def _define_data_type(self, d, verbose=True,
                          grid_ext=(".nc",".grd"), raster_ext=(".tiff",".asc",".tif"),
                          ascii_ext=(".xyz",".gdf",".csv",".tsv",".txt"),
                          vector_ext=(".shp",)):
        if "filepath_or_buffer" not in d: return None
        f = d["filepath_or_buffer"]
        if isinstance(f, np.ndarray): return "read_array"
        if "read_method" in d: return d["read_method"]
        _, ext = os.path.splitext(f)
        if ext in grid_ext:   return "read_grid"
        if ext in raster_ext: return "read_raster"
        if ext in vector_ext: return "read_shape"
        return "read_ascii"

    def _validate_observable(self, d):
        it = d.get("import_type"); lb = d.get("label","?")
        if it in ("read_ascii","read_raster","read_grid","read_shape"):
            if "filepath_or_buffer" not in d:
                raise KeyError(f"'{lb}': filepath_or_buffer required")
        if it == "compute" and "func" not in d:
            raise KeyError(f"'{lb}': func required")

    def read_recipe(self, dd, force_read=False, verbose=True,
                import_list=None, max_trials=None, on_file_error="warn"):
        """Import observables from a prepared recipe list into self.df.

        Parameters
        ----------
        dd : list[dict]
            Recipe list (already filtered by organise_recipe).
        force_read : bool | list[str]
            False  → skip labels already present in self.df.
            True   → re-read every label.
            list   → re-read only the named labels.
        import_list : list[str], optional
            If given, restrict import to these labels only.
        max_trials : int, optional
            Maximum retry attempts per entry (default: len(dd)).
        on_file_error : str
            'warn' or 'error' (currently unused, reserved).
        """
        if max_trials is None:
            max_trials = max(len(dd), 1)

        if not import_list:
            import_list = []

        force_labels = (None if force_read is True
                        else set(force_read)
                        if isinstance(force_read, (list, tuple, set))
                        else set())

        existing = set(self.df.columns)

        # ── build selected list, preserving duplicates by index ──────────
        selected = []
        for d in dd:
            lb = d["label"]
            if import_list and lb not in import_list:
                continue
            if lb in existing:
                if force_labels is None:
                    pass                          # force_read=True → always re-read
                elif lb in force_labels:
                    pass                          # named force → re-read
                else:
                    if verbose:
                        self._log(f"Skipping '{lb}' (already in df)")
                    continue
            selected.append(d)

        if verbose:
            self._log(f"Grid '{self.name}': {len(selected)} to import")

        # ── key tries/done/failed by index, not label ────────────────────
        #    two entries may share the same label (different grid sources)
        tries      = {i: 0 for i in range(len(selected))}
        failed_idx = set()
        done_idx   = set()
        failed     = set()   # label set for reporting
        done       = []      # label list for reporting (order preserved)

        while True:
            progress  = False
            remaining = [(i, d) for i, d in enumerate(selected)
                        if i not in failed_idx and i not in done_idx]
            if not remaining:
                break

            for i, d in remaining:
                lb = d["label"]
                if tries[i] >= max_trials:
                    failed_idx.add(i)
                    failed.add(lb)
                    continue
                try:
                    self._validate_observable(d)
                    if verbose:
                        self._log(f"  {lb}")
                    print(lb)
                    self.df[lb] = self._read_data(d, verbose=verbose)
                    done_idx.add(i)
                    done.append(lb)
                    progress = True
                except Exception as e:
                    tries[i] += 1
                    msg = str(e)
                    if tries[i] >= max_trials:
                        if verbose:
                            self._log(f"  {lb}: failed — {msg[:60]}", level="error")
                        failed_idx.add(i)
                        failed.add(lb)

            if not progress:
                # no forward progress this pass → mark all remaining as failed
                for i, d in remaining:
                    if i not in failed_idx and i not in done_idx:
                        failed_idx.add(i)
                        failed.add(d["label"])
                break

        if verbose:
            self._log(f"Done: {len(done)} ok, {len(failed)} failed")
        return done

    def import_recipe(self, module_name, verbose=False):
        try: self.recipe = importlib.import_module(module_name)
        except ImportError as e:
            self._log(f"Cannot import '{module_name}': {e}", level="error"); raise
        if not hasattr(self.recipe, "dd"):
            raise AttributeError(f"'{module_name}' must define dd list")
        return self.recipe

    # ------------------------------------------------------------------
    # XARRAY
    # ------------------------------------------------------------------
    class GridShapeError(ValueError): pass

    def df_to_ds(self):
        df = self.df
        if "x" not in df.columns or "y" not in df.columns:
            raise self.GridShapeError("df_to_ds requires x and y columns")
        xs = np.sort(df["x"].unique()); ys = np.sort(df["y"].unique())
        if xs.size * ys.size != len(df):
            raise self.GridShapeError(f"Grid not full: {xs.size}x{ys.size} != {len(df)}")
        df_idx = df.set_index(["y","x"]) \
            .reindex(pd.MultiIndex.from_product([ys,xs], names=["y","x"])) \
            .sort_index()
        ds = xr.Dataset.from_dataframe(df_idx).transpose("y","x")
        cn = [n for n in ("lat","lon") if n in ds]
        if cn: ds = ds.set_coords(cn)
        return ds

    # ------------------------------------------------------------------
    # PLOTTING HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _plotting_imports():
        import cartopy.crs as ccrs
        import cartopy.feature as cfeat
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpath
        import matplotlib.ticker as mticker
        try: import cmcrameri.cm as cmc
        except ImportError: cmc = None
        return ccrs, cfeat, plt, mpath, mticker, cmc

    @staticmethod
    def _compute_extent(df, crs, pad_frac=0.10):
        if crs == 4326:
            return None
        col_x = "x" if "x" in df.columns else "lon"
        col_y = "y" if "y" in df.columns else "lat"
        xmin, xmax = df[col_x].min(), df[col_x].max()
        ymin, ymax = df[col_y].min(), df[col_y].max()
        dx = (xmax - xmin) * pad_frac
        dy = (ymax - ymin) * pad_frac
        return [xmin - dx, xmax + dx, ymin - dy, ymax + dy]

    def _resolve_extent(self, extent, map_crs, map_crs_obj, use_xy, ccrs):
        ext = extent if extent is not None else self.extent
        if ext is not None:
            return ext, map_crs_obj if use_xy else ccrs.PlateCarree()
        return None, None

    def _apply_gridlines(self, ax, gl_dict, gl_kw, ccrs, use_xy, 
                         gridlabels = [False, False, False, False]):
        import matplotlib.ticker as mticker
        kw = {"linewidth": 0.4, "alpha": 0.5, "color": "gray", "linestyle": "-"}
        if gl_kw: kw.update(gl_kw)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), **kw)
        if "step_x" in gl_dict or "step_y" in gl_dict:
            gl.xlocator = mticker.FixedLocator(
                np.arange(-180, 181, gl_dict.get("step_x", 10)))
            gl.ylocator = mticker.FixedLocator(
                np.arange(-90, 91, gl_dict.get("step_y", 10)))
        elif "parallels" in gl_dict and "meridians" in gl_dict:
            gl.xlocator = mticker.FixedLocator(gl_dict["meridians"])
            gl.ylocator = mticker.FixedLocator(gl_dict["parallels"])
        elif "step" in gl_dict:
            s = gl_dict["step"]
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, s))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, s))
        gl.top_labels = gridlabels[0]
        gl.right_labels = gridlabels[1]
        gl.left_labels   = gridlabels[2]
        gl.right_labels  = gridlabels[3]

        

    # ------------------------------------------------------------------
    # look — single-observable, no cartopy
    # ------------------------------------------------------------------
    def look(self, data, cmap="viridis", vmin=None, vmax=None,
             title=None, cbar=True, cbar_label=None, cbar_units_label=None,
             figsize=None, show=True, return_fig=False, **kw):
        """Quick single-observable view — imshow (regular) or scatter (irregular)."""
        _, _, plt, _, _, _ = self._plotting_imports()
        t0 = time.time()
        arr = self._user_to_array(data)
        lbl = data if isinstance(data, str) else "Data"
        if title is None: title = lbl
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0: vmin = vmin or 0; vmax = vmax or 1
        else:
            if vmin is None: vmin = np.percentile(valid, 2)
            if vmax is None: vmax = np.percentile(valid, 98)

        is_reg = self.reshape_tuple is not None
        if not is_reg:
            try: self._infer_regular_shape(); is_reg = True
            except ValueError: pass

        if figsize is None:
            lr = self.lons.max()-self.lons.min()
            la = self.lats.max()-self.lats.min()
            asp = lr/la if la > 0 else 1.0
            figsize = (5*asp*0.7, 5*0.7) if asp >= 1 else (5*0.7, 5/asp*0.7)

        cbl = (cbar_label or "")
        if cbar_units_label: cbl = f"{cbl} {cbar_units_label}".strip()

        use_xy = (getattr(self, "crs", 4326) != 4326 and
                  "x" in self.df.columns and "y" in self.df.columns)
        if is_reg:
            ny, nx = self.reshape_tuple
            if isinstance(data, str):
                if use_xy:
                    d2 = self.make_2D(data, coord_x="x", coord_y="y")
                else:
                    d2 = self.make_2D(data)
            else:
                d2 = arr.reshape(ny, nx)
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(d2, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", **kw)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            if cbar: plt.colorbar(im, ax=ax).set_label(cbl)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            if use_xy:
                sc = ax.scatter(self.df["x"].values, self.df["y"].values,
                                c=arr, cmap=cmap, vmin=vmin, vmax=vmax,
                                s=10, edgecolors="none", **kw)
                ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
            else:
                sc = ax.scatter(self.lons, self.lats, c=arr, cmap=cmap,
                                vmin=vmin, vmax=vmax, s=10, edgecolors="none", **kw)
                ax.set(title=title, xlabel="Longitude (deg)", ylabel="Latitude (deg)")
            if cbar: plt.colorbar(sc, ax=ax).set_label(cbl)
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        if self.verbose: self._log(f"look() [{time.time()-t0:.2f}s]")
        return (fig, ax) if return_fig else None

    def quicklook(self, labels, dd,
                  ncols=3,
                  scatter_size=2,
                  cell_size=3.0,
                  show=True):
        """Tile multiple observables as small flat panels (no cartopy).

        For each label:
        - Attempts imshow using make_2D() (requires regular grid).
        - Falls back to scatter if grid is irregular or make_2D fails.
        Reads ``v_range`` and ``cmap`` from the recipe dict (``dd``).

        Parameters
        ----------
        labels : list[str]
            Column names to display.  Labels absent from self.df are skipped.
        dd : list[dict]
            Recipe list — provides ``v_range`` and ``cmap`` per label.
        ncols : int
            Tile columns (default 3).
        cell_size : float
            Each panel width/height in inches (default 3.0).
        show : bool
            plt.show() if True, plt.close() if False.

        Returns
        -------
        matplotlib.figure.Figure or None
        """

        available = list(dict.fromkeys(lb for lb in labels if lb in self.df.columns))
        if not available:
            print("quicklook: no requested labels found in df")
            return None

        recipe_map = {}
        for d in dd:
            recipe_map[d["label"]] = d

        n     = len(available)
        ncols = min(ncols, n)
        nrows = math.ceil(n / ncols)

        is_reg = self.reshape_tuple is not None
        if not is_reg:
            try: self._infer_regular_shape(); is_reg = True
            except ValueError: pass

        if is_reg:
            ny, nx = self.reshape_tuple
            aspect = nx / ny
        else:
            lr = self.lons.max()-self.lons.min()
            la = self.lats.max()-self.lats.min()
            aspect = (lr/la) if la > 0 else 1.0

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(cell_size*aspect*ncols, cell_size*nrows),
                                 squeeze=False)

        for idx, label in enumerate(available):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            d    = recipe_map.get(label, {})
            cmap = d.get("cmap", "viridis")
            vr   = d.get("v_range", None)
            vmin = vr[0] if vr is not None else None
            vmax = vr[1] if vr is not None else None

            arr   = self.df[label].to_numpy(dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                ax.set_title(label, fontsize=8); ax.axis("off"); continue

            if vmin is None: vmin = np.percentile(valid, 2)
            if vmax is None: vmax = np.percentile(valid, 98)

            if is_reg:
                try:
                    d2 = self.make_2D(label)
                    ax.imshow(d2,
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax,
                              origin="upper",
                              interpolation="nearest",
                              aspect="auto")
                except Exception:
                    ax.scatter(self.lons, self.lats, c=arr, cmap=cmap,
                               vmin=vmin, vmax=vmax, s=1, linewidths=0, rasterized=True)
            else:
                ax.scatter(self.lons, self.lats, c=arr, cmap=cmap,
                           vmin=vmin, vmax=vmax, s=scatter_size, linewidths=0, rasterized=True)

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=8, pad=2)

        for idx in range(n, nrows*ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        fig.suptitle(f"{self.name} — quicklook ({n} observables)",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # scatter (always scatter, no pcolormesh)
    # ------------------------------------------------------------------
    def scatter(self, data=None, map_crs=None, cmap="viridis",
                vmin=None, vmax=None, s=10, edgecolors="none",
                coastlines=False, continents=False, gridlines=None,
                gridlines_kwargs=None, vector_file=None, vector_kwargs=None,
                save_file=None, figsize=None, extent=None,
                show=True, return_fig=False, transparent=False,
                cbar=True, cbar_label=None, cbar_units_label=None, **kw):
        """Cartopy scatter plot (forces scatter regardless of regularity)."""
        ccrs, cfeat, plt, mpath, mticker, cmc = self._plotting_imports()
        t0 = time.time()
        map_crs_obj = ccrs.Mollweide() if (map_crs is None and
            getattr(self,"crs",4326)==4326) else self._resolve_map_crs(map_crs)
        use_xy = "x" in self.df.columns and "y" in self.df.columns
        arr = self._user_to_array(data) if data is not None else None
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            if len(valid):
                if vmin is None: vmin = np.nanpercentile(valid, 2)
                if vmax is None: vmax = np.nanpercentile(valid, 98)
        if figsize is None: figsize = (7, 7)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=map_crs_obj)
        if transparent:
            fig.patch.set_alpha(0); ax.patch.set_alpha(0)
        ext, ecrs = self._resolve_extent(extent, map_crs, map_crs_obj, use_xy, ccrs)
        if ext is not None: ax.set_extent(ext, crs=ecrs)
        if coastlines: ax.coastlines()
        if continents: ax.add_feature(cfeat.LAND, facecolor="lightgray")
        if gridlines: self._apply_gridlines(ax, gridlines, gridlines_kwargs, ccrs, use_xy)
        if arr is not None:
            if use_xy:
                sc = ax.scatter(self.df["x"].values, self.df["y"].values,
                                c=arr, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                                edgecolors=edgecolors, **kw)
            else:
                sc = ax.scatter(self.lons, self.lats, c=arr, s=s,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                edgecolors=edgecolors,
                                transform=ccrs.PlateCarree(), **kw)
            if cbar:
                lbl = (cbar_label or "")
                if cbar_units_label: lbl = f"{lbl} {cbar_units_label}".strip()
                plt.colorbar(sc, ax=ax, pad=0.02).set_label(lbl)
        if save_file:
            plt.savefig(save_file, bbox_inches="tight", transparent=transparent, dpi=150)
        if show: plt.show()
        else: plt.close(fig)
        return fig, ax

    # ------------------------------------------------------------------
    # map (production cartographic visualisation)
    # ------------------------------------------------------------------
    def map(self, data=None, ax=None, map_crs=None, cmap="viridis",
            vmin=None, vmax=None, title="", cbar=False,
            cbar_title="", cbar_units_label=None, cbar_title_rotation=0,
            cbar_orientation="horizontal", coastlines=True, continents=False,
            gridlines=None, gridlines_kwargs=None, vector_file=None,
            vector_kwargs=None, vector_crs=None, cbar_ratio=0.1,
            figsize=None, extent=None, dpi=300, no_frame=True,
            transparent=True, show=True, return_fig=False,
            save_fig=None, ext_cbar=False, save_cbar=None,
            gridlabels = [False, False, False, False],
            cbarsize=None, no_frame_cbar=True, histogram=True,
            hist_dict=None, scatter_kwargs=None, **kw):
        """Production cartographic visualisation.

        pcolormesh for regular grids, scatter for irregular.
        Accepts data=None (basemap only), str, 1D/2D array, or list of layers.
        """
        ccrs, cfeat, plt, mpath, mticker, cmc = self._plotting_imports()
        t0 = time.time()

        if not isinstance(data, list): data_list = [data]
        else: data_list = data
        n_layers = len(data_list)

        def _tol(v, n): return v if isinstance(v, list) else [v]*n
        cmap_l = _tol(cmap, n_layers)
        vmin_l = _tol(vmin, n_layers)
        vmax_l = _tol(vmax, n_layers)

        own_fig = ax is None
        if own_fig:
            map_crs_obj = (ccrs.Mollweide() if (map_crs is None and
                           getattr(self,"crs",4326)==4326)
                           else self._resolve_map_crs(map_crs))
        else:
            map_crs_obj = ax.projection; fig = ax.figure

        use_xy = "x" in self.df.columns and "y" in self.df.columns

        is_reg = self.reshape_tuple is not None
        if not is_reg:
            try: self._infer_regular_shape(); is_reg = True
            except ValueError: pass
        if not is_reg and hasattr(self,"nn") and isinstance(self.nn, tuple):
            self.reshape_tuple = self.nn; is_reg = True

        if own_fig:
            if figsize is None:
                ext_ = extent or self.extent
                if ext_: ar = (ext_[1]-ext_[0]) / max(ext_[3]-ext_[2], 1e-9)
                elif use_xy:
                    ar = (self.df["x"].max()-self.df["x"].min()) / \
                         max(self.df["y"].max()-self.df["y"].min(), 1e-9)
                else:
                    ar = (self.lons.max()-self.lons.min()) / \
                         max(self.lats.max()-self.lats.min(), 1e-9)
                figsize = (5*ar, 5) if ar >= 1 else (5, 5/ar)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=map_crs_obj)
            if transparent:
                fig.patch.set_alpha(0); fig.patch.set_facecolor("none")
                ax.patch.set_alpha(0)
            if no_frame and hasattr(ax,"spines") and "geo" in ax.spines:
                ax.spines["geo"].set_edgecolor("none")

        ext, ecrs = self._resolve_extent(extent, map_crs, map_crs_obj, use_xy, ccrs)
        if ext is not None: ax.set_extent(ext, crs=ecrs)
        if coastlines: ax.coastlines()
        if continents: ax.add_feature(cfeat.LAND, facecolor="lightgray")
        if gridlines: self._apply_gridlines(ax, 
                                            gridlines, 
                                            gridlines_kwargs, 
                                            ccrs, 
                                            use_xy, gridlabels=gridlabels)

        if vector_file:
            try:
                gdf = gpd.read_file(vector_file)
                if gdf.crs is None: gdf = gdf.set_crs(vector_crs or self.crs)
                vec_kw = {"edgecolor":"black","facecolor":"none","linewidth":0.5}
                if vector_kwargs: vec_kw.update(vector_kwargs)
                gdf.plot(ax=ax, **vec_kw)
            except Exception as e:
                self._log(f"Vector overlay failed: {e}", level="warn")

        im = None; plot_data = None; valid_data = np.array([])
        for i, layer in enumerate(data_list):
            _cm = cmap_l[i]; _vn = vmin_l[i]; _vx = vmax_l[i]
            if layer is None: continue
            plot_data = self._user_to_array(layer)
            valid_data = plot_data[~np.isnan(plot_data)]
            if len(valid_data) == 0: _vn = _vn or 0; _vx = _vx or 1
            else:
                if _vn is None: _vn = np.percentile(valid_data, 2)
                if _vx is None: _vx = np.percentile(valid_data, 98)
            alpha = kw.pop("alpha", 1.0) if n_layers == 1 else kw.get("alpha", 0.7)

            if isinstance(layer, np.ndarray) and layer.ndim == 2:
                xs = np.sort(self.df["x" if use_xy else "lon"].unique())
                ys = np.sort(self.df["y" if use_xy else "lat"].unique())
                X, Y = np.meshgrid(xs, ys)
                im = ax.pcolormesh(X, Y, layer, cmap=_cm, vmin=_vn, vmax=_vx,
                                   shading="auto", rasterized=True,
                                   alpha=alpha, zorder=1+i, **kw)
            elif is_reg:
                pd_ = plot_data.reshape(*self.reshape_tuple) \
                    if plot_data.ndim == 1 else plot_data
                if use_xy:
                    ny, nx = self.reshape_tuple
                    x2 = self.df["x"].values.reshape(ny, nx)
                    y2 = self.df["y"].values.reshape(ny, nx)
                    im = ax.pcolormesh(x2, y2, pd_, cmap=_cm,
                                       vmin=_vn, vmax=_vx, shading="auto",
                                       alpha=alpha, zorder=1+i, **kw)
                else:
                    ny, nx = self.reshape_tuple
                    x2 = self.lons.reshape(ny, nx); y2 = self.lats.reshape(ny, nx)
                    im = ax.pcolormesh(x2, y2, pd_, transform=ccrs.PlateCarree(),
                                       cmap=_cm, vmin=_vn, vmax=_vx, shading="auto",
                                       alpha=alpha, zorder=1+i, **kw)
            else:
                sc_kw = {"marker":"o","s":10,"edgecolors":"none"}
                if scatter_kwargs: sc_kw.update(scatter_kwargs)
                im = ax.scatter(self.lons, self.lats, c=plot_data,
                                cmap=_cm, vmin=_vn, vmax=_vx,
                                transform=ccrs.PlateCarree(),
                                alpha=alpha, zorder=1+i, **sc_kw)

        if title: ax.set_title(title, fontsize=14, fontweight="bold")
        if cbar and im is not None:
            lbl = cbar_title or ""
            if cbar_units_label: lbl = f"{lbl} {cbar_units_label}".strip()
            plt.colorbar(im, ax=ax, orientation=cbar_orientation,
                         pad=0.02).set_label(lbl, rotation=cbar_title_rotation)

        if own_fig:
            plt.tight_layout()
            if save_fig:
                os.makedirs(os.path.dirname(save_fig) or ".", exist_ok=True)
                fig.savefig(save_fig, dpi=dpi, transparent=transparent,
                            bbox_inches="tight", pad_inches=0)
            if ext_cbar:
                self._make_ext_cbar(
                    im=im, plot_data=plot_data,
                    valid_data=valid_data, all_nan=(data_list[-1] is None),
                    cmap=cmap_l[-1], vmin=vmin_l[-1], vmax=vmax_l[-1],
                    cbar_title=cbar_title, cbar_units_label=cbar_units_label,
                    cbar_title_rotation=cbar_title_rotation,
                    cbar_orientation=cbar_orientation, cbar_ratio=cbar_ratio,
                    figsize=figsize, cbarsize=cbarsize,
                    no_frame_cbar=no_frame_cbar, histogram=histogram,
                    hist_dict=hist_dict, transparent=transparent,
                    dpi=dpi, save_cbar=save_cbar, show=show)
            if show: plt.show()
            else: plt.close(fig)
        if self.verbose: self._log(f"map() [{time.time()-t0:.2f}s]")
        return fig, ax

    def _make_ext_cbar(self, *, im, plot_data, valid_data, all_nan,
                       cmap, vmin, vmax, cbar_title, cbar_units_label,
                       cbar_title_rotation, cbar_orientation, cbar_ratio,
                       figsize, cbarsize, no_frame_cbar, histogram, hist_dict,
                       transparent, dpi, save_cbar, show):
        _, _, plt, _, _, _ = self._plotting_imports()
        if im is None:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        if cbarsize is None:
            fw, fh = figsize
            cbarsize = (0.5*fw, cbar_ratio*fh) if cbar_orientation=="horizontal" \
                else (cbar_ratio*fw, 0.5*fh)
        if hist_dict is None: hist_dict = {}
        n_bins  = hist_dict.get("n_bins",  51 if not all_nan else 10)
        h_alpha = hist_dict.get("alpha",   0.6)
        h_color = hist_dict.get("color",   "#999999")
        h_dens  = hist_dict.get("density", True)
        if histogram:
            if cbar_orientation == "horizontal":
                cfig, (dax, cax) = plt.subplots(2, 1, figsize=cbarsize,
                    sharex=True, gridspec_kw={"height_ratios": [9, 4]})
            else:
                cfig, (dax, cax) = plt.subplots(1, 2, figsize=cbarsize,
                    sharey=True, gridspec_kw={"width_ratios": [4, 9]})
            if transparent: cfig.patch.set_alpha(0)
            if not all_nan:
                dax.hist(valid_data, bins=n_bins, range=(vmin, vmax),
                         density=h_dens, color=h_color, alpha=h_alpha)
                dax.set_xlim(vmin, vmax) if cbar_orientation=="horizontal" \
                    else dax.set_ylim(vmin, vmax)
            dax.axis("off")
            plt.subplots_adjust(hspace=0.02, wspace=0.02)
        else:
            cfig, cax = plt.subplots(figsize=cbarsize)
            if transparent: cfig.patch.set_alpha(0)
        cb = im.axes.figure.colorbar(im, cax=cax, orientation=cbar_orientation,
                          pad=0, fraction=1.0)
        cb.ax.tick_params(labelsize=10)
        lbl = cbar_title or ""
        if cbar_units_label: lbl = f"{lbl} {cbar_units_label}".strip()
        if cbar_orientation == "horizontal":
            cb.ax.set_xlabel(lbl, rotation=cbar_title_rotation, size=12)
        else:
            cb.ax.set_ylabel(lbl, rotation=cbar_title_rotation, size=12)
        if no_frame_cbar:
            for sp in ("top","right","left","bottom"):
                cax.spines[sp].set_visible(False)
        if save_cbar:
            os.makedirs(os.path.dirname(save_cbar) or ".", exist_ok=True)
            cfig.savefig(save_cbar, dpi=dpi, transparent=transparent,
                         bbox_inches="tight", pad_inches=0)
        if show: plt.show()
        else: plt.close(cfig)


    # ------------------------------------------------------------------
    # _smooth_data  (internal helper)
    # ------------------------------------------------------------------
    def _smooth_data(self, data_2d, kernel_size=3, kernel_type="gaussian"):
        """Apply 2D smoothing to a 2D array, NaN-aware.

        Parameters
        ----------
        data_2d     : 2D ndarray (may contain NaN)
        kernel_size : int or float
            For "gaussian": sigma value (std-dev in grid cells).
            For "uniform"  (boxcar): filter window width in grid cells.
        kernel_type : {"gaussian", "uniform"}

        Returns
        -------
        smoothed 2D ndarray (same shape, NaN mask preserved)
        """
        from scipy.ndimage import gaussian_filter, uniform_filter

        has_nan = np.any(np.isnan(data_2d))

        if has_nan:
            # NaN-aware normalised convolution: smooth(vals) / smooth(weights)
            nan_mask = np.isnan(data_2d)
            filled   = np.where(nan_mask, 0.0, data_2d)
            weights  = np.where(nan_mask, 0.0, 1.0)

            if kernel_type == "gaussian":
                smooth_fn = lambda a: gaussian_filter(
                    a, sigma=float(kernel_size), mode="nearest")
            else:
                smooth_fn = lambda a: uniform_filter(
                    a, size=int(kernel_size), mode="nearest")

            num            = smooth_fn(filled)
            den            = smooth_fn(weights)
            den[den == 0]  = np.nan
            result         = num / den
            result[nan_mask] = np.nan
        else:
            if kernel_type == "gaussian":
                result = gaussian_filter(data_2d, sigma=float(kernel_size),
                                         mode="nearest")
            else:
                result = uniform_filter(data_2d, size=int(kernel_size),
                                        mode="nearest")
        return result

    # ------------------------------------------------------------------
    # contour_map  (production contour cartographic visualisation)
    # ------------------------------------------------------------------
    def contour_map(self, data=None, ax=None, map_crs=None,
                    # ---- filled contours ----------------------------------------
                    cmap="viridis", vmin=None, vmax=None,
                    n_contour_fills=10,
                    # ---- line contours ------------------------------------------
                    n_contours=10,
                    contour_colors="k",
                    contour_linewidths=0.6,
                    contour_linestyles="solid",
                    contour_levels=None,
                    # ---- contour labels -----------------------------------------
                    contour_labels=False,
                    contour_label_fmt="%.4g",
                    contour_label_fontsize=8,
                    contour_label_inline=True,
                    contour_label_colors=None,
                    # ---- smoothing ----------------------------------------------
                    smooth=False,
                    smooth_kernel_size=2,
                    smooth_kernel_type="gaussian",
                    # ---- colorbar -----------------------------------------------
                    cbar=False,
                    cbar_title="",
                    cbar_units_label=None,
                    cbar_title_rotation=0,
                    cbar_orientation="horizontal",
                    ext_cbar=False,
                    save_cbar=None,
                    cbar_ratio=0.1,
                    cbarsize=None,
                    no_frame_cbar=True,
                    histogram=True,
                    hist_dict=None,
                    # ---- map features -------------------------------------------
                    title="",
                    coastlines=True,
                    continents=False,
                    gridlines=None,
                    gridlines_kwargs=None,
                    gridlabels=[False, False, False, False],
                    vector_file=None,
                    vector_kwargs=None,
                    vector_crs=None,
                    # ---- figure -------------------------------------------------
                    figsize=None,
                    extent=None,
                    dpi=300,
                    no_frame=True,
                    transparent=True,
                    show=True,
                    return_fig=False,
                    save_fig=None,
                    **kw):
        """Production cartographic contour visualisation.

        Draws filled contours (contourf) and/or line contours (contour)
        over a Cartopy basemap.  Mirrors the signature of ``map()`` and adds
        dedicated contour controls, optional smoothing, and contour labelling.

        Parameters
        ----------
        data : None | str | 1-D/2-D ndarray
            Data to visualise.  Passed through ``_user_to_array``.
            Pass ``None`` for a basemap-only plot.
        ax : GeoAxes, optional
            Existing Cartopy axes to draw into.  If None a new figure is
            created.
        map_crs : Cartopy CRS or int EPSG, optional
            Projection for the map axes.

        Filled contours
        ---------------
        cmap : str or Colormap
            Colormap for ``contourf``.
        vmin, vmax : float, optional
            Colour-scale limits.  Defaults to the 2nd / 98th percentile.
        n_contour_fills : int
            Number of filled contour bands (levels = n + 1 edges).
            Set to 0 to suppress filled contours.

        Line contours
        -------------
        n_contours : int
            Number of line contour levels.  Set to 0 to suppress.
        contour_colors : color spec
            Colour(s) for line contours (e.g. "k", ["k","grey"]).
        contour_linewidths : float or list of float
            Line width(s).
        contour_linestyles : str or list of str
            Line style(s), e.g. "solid", "dashed", "dotted".
        contour_levels : array-like, optional
            Explicit level values.  Overrides ``n_contours`` /
            ``n_contour_fills`` when given.

        Contour labels
        --------------
        contour_labels : bool
            Annotate line contours with their values.
        contour_label_fmt : str or callable
            Format for contour labels.  Accepts:
            - %-format string: ``"%.1f"``, ``"%.0f"``
            - new-style callable: ``"{:.2f}".format``
            - any callable ``f(float) -> str``
        contour_label_fontsize : int
            Font size for labels.
        contour_label_inline : bool
            Remove the line segment under each label when True.
        contour_label_colors : color spec, optional
            Label colour(s).  Defaults to matching contour line colours.

        Smoothing
        ---------
        smooth : bool
            Apply spatial smoothing to the data array before contouring.
        smooth_kernel_size : int or float
            Kernel size / spread:
            - "gaussian" → sigma (standard deviation in grid cells)
            - "uniform"  → window width in grid cells
        smooth_kernel_type : {"gaussian", "uniform"}
            Type of smoothing kernel.  Both are NaN-aware.

        Colorbar
        --------
        cbar : bool
            Draw an inline colorbar on the map axes.
        cbar_title : str
            Main colorbar label.
        cbar_units_label : str, optional
            Units string appended to ``cbar_title``.
        cbar_title_rotation : float
            Rotation of the colorbar label in degrees.
        cbar_orientation : {"horizontal", "vertical"}
        ext_cbar : bool
            Generate a separate colourbar figure (delegates to
            ``_make_ext_cbar``).
        save_cbar : str, optional
            File path to save the external colorbar figure.
        cbar_ratio, cbarsize, no_frame_cbar, histogram, hist_dict
            Passed directly to ``_make_ext_cbar``.

        Map features
        ------------
        title : str
            Axes title.
        coastlines : bool
            Draw Cartopy coastlines.
        continents : bool
            Fill land with light grey.
        gridlines : bool or dict, optional
            Draw map gridlines.
        gridlabels : list of 4 bool
            [left, right, top, bottom] gridline label sides.
        vector_file : str, optional
            Path to a vector file (Shapefile / GeoJSON) to overlay.
        vector_kwargs : dict, optional
            Style overrides for the vector overlay.
        vector_crs : Cartopy CRS, optional
            CRS of the vector file (used if the file has no embedded CRS).

        Figure
        ------
        figsize : tuple (w, h) in inches
        extent : [lon_min, lon_max, lat_min, lat_max]
        dpi : int
        no_frame : bool
            Hide the geo-spine frame.
        transparent : bool
            Transparent figure background.
        show : bool
        save_fig : str, optional
            File path to save the figure.

        Returns
        -------
        fig : Figure
        ax  : GeoAxes
        """
        ccrs, cfeat, plt, mpath, mticker, cmc = self._plotting_imports()
        t0 = time.time()

        # ── 1. Resolve figure / axes ──────────────────────────────────────────
        own_fig = ax is None
        if own_fig:
            map_crs_obj = (
                ccrs.Mollweide()
                if (map_crs is None and getattr(self, "crs", 4326) == 4326)
                else self._resolve_map_crs(map_crs)
            )
        else:
            map_crs_obj = ax.projection
            fig = ax.figure

        use_xy = "x" in self.df.columns and "y" in self.df.columns

        # ── 2. Determine regularity / reshape tuple ───────────────────────────
        is_reg = self.reshape_tuple is not None
        if not is_reg:
            try:
                self._infer_regular_shape()
                is_reg = True
            except ValueError:
                pass
        if not is_reg and hasattr(self, "nn") and isinstance(self.nn, tuple):
            self.reshape_tuple = self.nn
            is_reg = True

        # ── 3. Figure creation ────────────────────────────────────────────────
        if own_fig:
            if figsize is None:
                ext_ = extent or self.extent
                if ext_:
                    ar = (ext_[1] - ext_[0]) / max(ext_[3] - ext_[2], 1e-9)
                elif use_xy:
                    ar = (self.df["x"].max() - self.df["x"].min()) / max(
                        self.df["y"].max() - self.df["y"].min(), 1e-9)
                else:
                    ar = (self.lons.max() - self.lons.min()) / max(
                        self.lats.max() - self.lats.min(), 1e-9)
                figsize = (5 * ar, 5) if ar >= 1 else (5, 5 / ar)
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(1, 1, 1, projection=map_crs_obj)
            if transparent:
                fig.patch.set_alpha(0)
                fig.patch.set_facecolor("none")
                ax.patch.set_alpha(0)
            if no_frame and hasattr(ax, "spines") and "geo" in ax.spines:
                ax.spines["geo"].set_edgecolor("none")

        # ── 4. Extent + basemap features ─────────────────────────────────────
        ext, ecrs = self._resolve_extent(extent, map_crs, map_crs_obj,
                                         use_xy, ccrs)
        if ext is not None:
            ax.set_extent(ext, crs=ecrs)
        if coastlines:
            ax.coastlines(zorder = 10)
        if continents:
            ax.add_feature(cfeat.LAND, facecolor="lightgray")
        if gridlines:
            self._apply_gridlines(ax, gridlines, gridlines_kwargs,
                                  ccrs, use_xy, gridlabels=gridlabels)

        # ── 5. Optional vector overlay ────────────────────────────────────────
        if vector_file:
            try:
                gdf = gpd.read_file(vector_file)
                if gdf.crs is None:
                    gdf = gdf.set_crs(vector_crs or self.crs)
                vec_kw = {"edgecolor": "black", "facecolor": "none",
                          "linewidth": 0.5}
                if vector_kwargs:
                    vec_kw.update(vector_kwargs)
                gdf.plot(ax=ax, **vec_kw)
            except Exception as e:
                self._log(f"Vector overlay failed: {e}", level="warn")

        # ── 6. Early exit: basemap only ───────────────────────────────────────
        if data is None:
            if own_fig:
                plt.tight_layout()
                if save_fig:
                    import os
                    os.makedirs(os.path.dirname(save_fig) or ".", exist_ok=True)
                    fig.savefig(save_fig, dpi=dpi, transparent=transparent,
                                bbox_inches="tight", pad_inches=0)
                if show:
                    plt.show()
                else:
                    plt.close(fig)
            if self.verbose:
                self._log(f"contour_map() basemap [{time.time()-t0:.2f}s]")
            return fig, ax

        # ── 7. Prepare data array ─────────────────────────────────────────────
        plot_data  = self._user_to_array(data)
        valid_data = plot_data[~np.isnan(plot_data)]

        if len(valid_data) == 0:
            _vn = vmin if vmin is not None else 0
            _vx = vmax if vmax is not None else 1
        else:
            _vn = vmin if vmin is not None else float(np.percentile(valid_data, 2))
            _vx = vmax if vmax is not None else float(np.percentile(valid_data, 98))

        # ── 8. Build 2-D coordinate arrays and grid ───────────────────────────
        if isinstance(data, np.ndarray) and data.ndim == 2:
            pd_ = data
            xs  = np.sort(self.df["x" if use_xy else "lon"].unique())
            ys  = np.sort(self.df["y" if use_xy else "lat"].unique())
            X2, Y2 = np.meshgrid(xs, ys)
        elif is_reg:
            pd_ = (plot_data.reshape(*self.reshape_tuple)
                   if plot_data.ndim == 1 else plot_data)
            ny, nx = self.reshape_tuple
            col_x  = "x" if use_xy else "lon"
            col_y  = "y" if use_xy else "lat"
            X2 = self.df[col_x].values.reshape(ny, nx)
            Y2 = self.df[col_y].values.reshape(ny, nx)
        else:
            self._log(
                "contour_map() requires a regular grid; "
                "irregular scatter data is not supported for contouring.",
                level="warn",
            )
            return fig, ax

        # ── 9. Optional smoothing ─────────────────────────────────────────────
        if smooth:
            pd_ = self._smooth_data(pd_,
                                    kernel_size=smooth_kernel_size,
                                    kernel_type=smooth_kernel_type)

        # ── 10. Build level arrays ─────────────────────────────────────────────
        if contour_levels is not None:
            levels_fill = np.asarray(contour_levels)
            levels_line = levels_fill
        else:
            levels_fill = np.linspace(_vn, _vx, max(int(n_contour_fills) + 1, 2))
            levels_line = np.linspace(_vn, _vx, max(int(n_contours) + 1, 2))

        # ── 11. CRS transform for data ────────────────────────────────────────
        # lon/lat data always needs PlateCarree transform;
        # projected XY data should use the map CRS itself.
        transform = ccrs.PlateCarree() if not use_xy else map_crs_obj

        # ── 12. Filled contours ───────────────────────────────────────────────
        im_cf = None
        if n_contour_fills > 0 or contour_levels is not None:
            cf_kw = dict(
                cmap=cmap,
                vmin=_vn,
                vmax=_vx,
                levels=levels_fill,
                zorder=2,
            )
            # Allow caller to pass extend / alpha via **kw
            for _k in ("alpha", "extend"):
                if _k in kw:
                    cf_kw[_k] = kw[_k]
            im_cf = ax.contourf(X2, Y2, pd_, transform=transform, **cf_kw)

        # ── 13. Line contours ─────────────────────────────────────────────────
        cs = None
        if n_contours > 0 or contour_levels is not None:
            cs = ax.contour(
                X2, Y2, pd_,
                transform=transform,
                levels=levels_line,
                colors=contour_colors,
                linewidths=contour_linewidths,
                linestyles=contour_linestyles,
                zorder=3,
            )

        # ── 14. Contour labels ────────────────────────────────────────────────
        if contour_labels and cs is not None:
            clabel_kw = dict(
                fontsize=contour_label_fontsize,
                inline=contour_label_inline,
                fmt=contour_label_fmt,
            )
            if contour_label_colors is not None:
                clabel_kw["colors"] = contour_label_colors
            ax.clabel(cs, cs.levels, **clabel_kw)

        # ── 15. Title ─────────────────────────────────────────────────────────
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # ── 16. Inline colorbar ───────────────────────────────────────────────
        if cbar and im_cf is not None:
            lbl = cbar_title or ""
            if cbar_units_label:
                lbl = f"{lbl} {cbar_units_label}".strip()
            plt.colorbar(im_cf, ax=ax, orientation=cbar_orientation,
                         pad=0.02).set_label(lbl, rotation=cbar_title_rotation)

        # ── 17. Save / show / external colorbar ──────────────────────────────
        if own_fig:
            plt.tight_layout()
            if save_fig:
                import os
                os.makedirs(os.path.dirname(save_fig) or ".", exist_ok=True)
                fig.savefig(save_fig, dpi=dpi, transparent=transparent,
                            bbox_inches="tight", pad_inches=0)
            if ext_cbar and im_cf is not None:
                self._make_ext_cbar(
                    im=im_cf,
                    plot_data=plot_data,
                    valid_data=valid_data,
                    all_nan=(data is None),
                    cmap=cmap,
                    vmin=_vn,
                    vmax=_vx,
                    cbar_title=cbar_title,
                    cbar_units_label=cbar_units_label,
                    cbar_title_rotation=cbar_title_rotation,
                    cbar_orientation=cbar_orientation,
                    cbar_ratio=cbar_ratio,
                    figsize=figsize,
                    cbarsize=cbarsize,
                    no_frame_cbar=no_frame_cbar,
                    histogram=histogram,
                    hist_dict=hist_dict,
                    transparent=transparent,
                    dpi=dpi,
                    save_cbar=save_cbar,
                    show=show,
                )
            if show:
                plt.show()
            else:
                plt.close(fig)

        if self.verbose:
            self._log(f"contour_map() [{time.time()-t0:.2f}s]")
        return fig, ax

