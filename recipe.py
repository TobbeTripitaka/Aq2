#!/usr/bin/env python3
# recipe.py v0.4.3
# ============================================================================
# Observable definitions for Aq2/Kq2.
# v0.4.3: _nearest_volcano_distance uses tight 100 km chord window + hard 100 km NaN mask
# v0.4.2: _nearest_volcano_distance rewritten with 3D ECEF KD-tree (no lat distortion)
# v0.4.1: v_range updated from observables_ref.csv statistics (p01/p99 ≈5th/95th
#         percentile). DEM centred on 0; categorical ranges span all classes.

# Need to fix: 
# SEDIMENT Antarctica Li2022




# grid-key routing
# ----------------
# Entries with a ``grid`` key are imported ONLY by the Grid whose ``name``
# attribute matches that value. Entries without ``grid`` are shared across
# all three grids (IHFC, Antarctica, Greenland).

# Currently grid-keyed: DEM (3 variants) and SEDIMENT (3 variants).
# All other observables are shared.
# ============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from scipy.spatial import cKDTree

# ── Volcano catalogue ─────────────────────────────────────────────────────
_VOLC_CACHE = None

def build_volcano_catalog():
    global _VOLC_CACHE
    if _VOLC_CACHE is not None: return _VOLC_CACHE
    path_exp = "../data/volcanos/volcanic list.xlsx"
    sheets = pd.read_excel(path_exp, sheet_name=None)
    parts = []
    for sname, df in sheets.items():
        if not {"LAT","LON"}.issubset(df.columns): continue
        cols = [c for c in ["LAT","LON","exposed"] if c in df.columns]
        tmp = df[cols].rename(columns={"LAT":"lat","LON":"lon"})
        if "exposed" not in tmp.columns: tmp["exposed"] = "unknown"
        tmp["group"] = sname
        parts.append(tmp)
    if not parts: raise RuntimeError("No valid sheets in volcano list")
    exposed = pd.concat(parts, ignore_index=True)
    def _gvp(path, group):
        df = pd.read_excel(path, header=1)
        return (df.rename(columns={"Latitude":"lat","Longitude":"lon"})
                [["lat","lon"]].dropna().assign(exposed="full", group=group))
    all_v = pd.concat([exposed,
                       _gvp("../data/volcanos/GVP_Volcano_List_Holocene.xlsx", "holocene"),
                       _gvp("../data/volcanos/GVP_Volcano_List_Pleistocene.xlsx", "pleistocene"),
                       ], ignore_index=True)
    for col in ("lat","lon"):
        all_v[col] = pd.to_numeric(all_v[col], errors="coerce")
    _VOLC_CACHE = all_v.dropna(subset=["lat","lon"])
    return _VOLC_CACHE

def _haversine_m(la1, lo1, la2, lo2):
    la1,lo1,la2,lo2 = map(np.radians,[la1,lo1,la2,lo2])
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6_371_000.0 * 2.0 * np.arcsin(np.sqrt(a))

def _xyz(lat_deg, lon_deg):
    """Unit-sphere ECEF coords from lat/lon in degrees."""
    la = np.radians(lat_deg)
    lo = np.radians(lon_deg)
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])

# Maximum detection radius in metres; anything beyond this is set to NaN.
_VOLC_MAX_DIST_M = 100_000.0
# Chord length on the unit sphere for _VOLC_MAX_DIST_M, plus 10 % margin so
# the KD-tree window is tight but guaranteed to contain all candidates.
_VOLC_CHORD_LIMIT = 1.1 * 2.0 * np.sin(_VOLC_MAX_DIST_M / (2.0 * 6_371_000.0))

def _nearest_volcano_distance(lats, lons, volc_df, max_deg=2.0):
    """Return distance (m) to nearest volcano within _VOLC_MAX_DIST_M (100 km).

    max_deg is retained for API compatibility but no longer controls the search
    window or the masking threshold — use _VOLC_MAX_DIST_M for that.
    KD-tree is built in 3D ECEF (unit sphere) so there is no latitude-dependent
    distortion; the search window is the chord equivalent of 100 km + 10 % margin.
    """
    lats, lons = np.asarray(lats), np.asarray(lons)
    vl, vn = volc_df["lat"].to_numpy(), volc_df["lon"].to_numpy()

    tree = cKDTree(_xyz(vl, vn))
    da, idx = tree.query(_xyz(lats, lons), k=1,
                         distance_upper_bound=_VOLC_CHORD_LIMIT)

    out = np.full(lats.shape, np.nan)
    v = np.isfinite(da) & (idx < len(vl))
    if v.any():
        dist = _haversine_m(lats[v], lons[v], vl[idx[v]], vn[idx[v]])
        out[v] = np.where(dist <= _VOLC_MAX_DIST_M, dist, np.nan)
    return out

# ── Compute functions ─────────────────────────────────────────────────────

def compute_volcano_distance(self, d):
    return _nearest_volcano_distance(
        self.df["lat"].values, self.df["lon"].values,
        build_volcano_catalog(), max_deg=2.0)

def compute_corrected_dem(self, d):
    import xarray as xr
    import numpy as np
    from scipy.ndimage import uniform_filter, gaussian_filter

    bed = xr.open_dataset(d["bed_path"])[d["bed_var"]]
    iso = xr.open_dataset(d["iso_path"])[d["iso_var"]]

    x = xr.DataArray(self.df["x"].values, dims="points")
    y = xr.DataArray(self.df["y"].values, dims="points")

    bed_pts = bed.interp(x=x, y=y, method="linear").values
    iso_pts = iso.interp(x=x, y=y, method="linear").values
    corrected = bed_pts + iso_pts

    grid_spacing_m = d.get("grid_spacing_m", 5_000)
    kernel_km      = d.get("kernel_km", 5.0)
    kernel_cells   = max(1, round(kernel_km * 1000 / grid_spacing_m))

    if self.reshape_tuple is None:
        self._infer_regular_shape(coord_x="x", coord_y="y")
    ny, nx = self.reshape_tuple
    grid2d = corrected.reshape(ny, nx)

    nan_mask = np.isnan(grid2d)

    size = 2 * kernel_cells + 1
    filled  = np.where(nan_mask, 0.0, grid2d)
    weights = np.where(nan_mask, 0.0, 1.0)

    if d.get("kernel_type", "uniform") == "gaussian":
        smoothed = gaussian_filter(filled,   sigma=kernel_cells, mode="nearest")
        wsum     = gaussian_filter(weights,  sigma=kernel_cells, mode="nearest")
    else:
        smoothed = uniform_filter(filled,   size=size, mode="nearest")
        wsum     = uniform_filter(weights,  size=size, mode="nearest")

    smoothed = np.where(wsum > 0, smoothed / wsum, np.nan)

    smoothed[nan_mask] = np.nan

    return smoothed.ravel()

def compute_sediment_log_antarctica(self, d):
    """
    Compute log-transformed sediment thickness for the Antarctica grid.

    Steps:
    1. Read Li 2022 sediment likelihood raster (hardcoded path).
    2. Build a weight mask: 0 where likelihood < 0.5, else 1.
       (NaN likelihood → weight = 1, i.e. treat as "present".)
    3. Apply weight mask to SEDIMENT (zero out low-likelihood cells).
    4. Smooth the masked sediment field using a NaN-aware weighted kernel
       (identical pattern to compute_corrected_dem).
    5. Apply log1p to the smoothed field → 0 where sediment=0, log-scale above.
    6. Return flat array (no NaNs).
    """
    import numpy as np
    import rasterio
    from scipy.ndimage import uniform_filter, gaussian_filter

    # ── 1. Read likelihood raster ──────────────────────────────────────────
    likelihood_path = "../data/li_2022/SSB_Likelihood.tif"
    pts = list(zip(self.lons, self.lats))
    with rasterio.open(likelihood_path) as src:
        nodata = src.nodata
        lik = np.array(list(src.sample(pts)), dtype=float).flatten()
        if nodata is not None:
            lik[lik == nodata] = np.nan

    # ── 2. Weight mask: 0 where likelihood is defined AND < 0.5, else 1 ───
    weight_mask = np.where(np.isfinite(lik) & (lik < 0.5), 0.0, 1.0)

    # ── 3. Apply mask to SEDIMENT ──────────────────────────────────────────
    sed = self.df["SEDIMENT"].values.copy().astype(float)
    sed = sed * weight_mask   # zero out low-likelihood cells

    # ── 4. Smooth on the 2-D grid (NaN-aware weighted kernel) ─────────────
    grid_spacing_m = d.get("grid_spacing_m", 5_000)
    kernel_km      = d.get("kernel_km", 15.0)
    kernel_cells   = max(1, round(kernel_km * 1000 / grid_spacing_m))

    if self.reshape_tuple is None:
        self._infer_regular_shape(coord_x="x", coord_y="y")
    ny, nx = self.reshape_tuple

    sed_2d      = sed.reshape(ny, nx)
    nan_mask    = np.isnan(sed_2d)

    size   = 2 * kernel_cells + 1
    filled  = np.where(nan_mask, 0.0, sed_2d)
    weights = np.where(nan_mask, 0.0, 1.0)

    if d.get("kernel_type", "uniform") == "gaussian":
        smoothed = gaussian_filter(filled,  sigma=kernel_cells, mode="nearest")
        wsum     = gaussian_filter(weights, sigma=kernel_cells, mode="nearest")
    else:
        smoothed = uniform_filter(filled,  size=size, mode="nearest")
        wsum     = uniform_filter(weights, size=size, mode="nearest")

    smoothed = np.where(wsum > 0, smoothed / wsum, 0.0)
    smoothed[nan_mask] = 0.0   # outside-coverage cells → 0 (not NaN)

    # ── 5. log1p transform ─────────────────────────────────────────────────
    # log1p(0) = 0  (no -inf); log1p(x) ≈ log(x) for large x
    result = np.log1p(smoothed.ravel())

    return result

# def compute_corrected_dem(self, d):
#     bed = xr.open_dataset(d["bed_path"])[d["bed_var"]]
#     iso = xr.open_dataset(d["iso_path"])[d["iso_var"]]
#     x = self.df["x"].values; y = self.df["y"].values
#     return (bed.interp(x=("points",x), y=("points",y)).values +
#             iso.interp(x=("points",x), y=("points",y)).values)



# ── Observable dictionary ─────────────────────────────────────────────────







dd = [

# ── Moho (GEMMA) ──────────────────────────────────────────────────
{"label":"MOHO_GRAV",
 "filepath_or_buffer":"../data/GEMMA/moho/t6.asc",
 "import_type":"read_raster",
 "z_preproc": lambda d: d*1000.0, "interpol_method":"linear",
 "unit":"metre", "v_range":(-54_000,-9_200), "cmap":"cmc.bamako",
 "reference":"GEMMA",
 "description":"Moho depth (GEMMA gravity-derived)"},

{"label":"MOHO_GRAV_U",
 "filepath_or_buffer":"../data/GEMMA/moho/moho_err.asc",
 "import_type":"read_raster",
 "z_preproc": lambda d: d*1000.0, "interpol_method":"linear",
 "unit":"metre", "v_range":(1_600,8_600), "cmap":"cmc.bamako",
 "reference":"GEMMA",
 "description":"Moho depth uncertainty (GEMMA)"},

# ── Moho (Szwillus 2019) ──────────────────────────────────────────
{"label":"MOHO",
 "filepath_or_buffer":"../data/Szwillus_2019/jgrb53251-sup-0003-data_set_si-s03.txt",
 "import_type":"read_ascii",
 "delim_whitespace":True, "usecols":[0,1,2],
 "names":["lon","lat","depth"], "header":34, "encoding":"ISO-8859-1",
 "z_preproc": lambda d: -d*1000.0, "interpol_method":"linear",
 "unit":"metre", "v_range":(-52_000,-9_200), "cmap":"cmc.bamako",
 "reference":"Szwillus2019",
 "description":"Moho depth (Szwillus et al. 2019)"},

{"label":"MOHO_U",
 "filepath_or_buffer":"../data/Szwillus_2019/jgrb53251-sup-0003-data_set_si-s03.txt",
 "import_type":"read_ascii",
 "delim_whitespace":True, "usecols":[0,1,3],
 "names":["lon","lat","uncertainty"], "header":34, "encoding":"ISO-8859-1",
 "z_preproc": lambda d: d*1000.0, "interpol_method":"linear",
 "unit":"metre", "v_range":(1_400,7_200),
 "reference":"Szwillus2019",
 "description":"Moho depth uncertainty (Szwillus et al. 2019)"},

# ── DEM — three grid-keyed variants ──────────────────────────────
# v_range centred on 0; extent = ceil2(max(|p01|,|p99|)) = 4900 m
{"label":"DEM", "grid":"IHFC",
 "filepath_or_buffer":"../data/ETOPO1/ETOPO1_Bed_g_geotiff.tif",
 "import_type":"read_raster",
 "unit":"metre", "v_range":(-4_900,4_900), "cmap":"cmc.oleron",
 "reference":"ETOPO1",
 "description":"ETOPO1 DEM (reference grid)"},


{"label":"DEM", "grid":"Antarctica",
 "import_type":"compute", "func":compute_corrected_dem,
 "bed_path":"../data/bedmap3/bedmap3.nc", "bed_var":"bed_topography",
 "iso_path":"../data/paxman_relaxed_topo/Total_Isostatic_Adjustment_Antarctica_BedMachinev3.nc",
 "iso_var":"topography change",
 "unit":"metre", "v_range":(-4_900,4_900), "cmap":"cmc.oleron",
     "grid_spacing_m": 5_000,
    "kernel_km":      5.0,  
    "kernel_type":   "uniform",  
 "description":"Antarctic DEM: bed topo + Paxman isostatic adjustment"},

{"label":"DEM", "grid":"Greenland",
 "import_type":"compute", "func":compute_corrected_dem,
 "bed_path":"../data/bed_machine_v3/IDBMG4_6-20260115_001332/BedMachineGreenland-v6.nc",
 "bed_var":"bed",
 "iso_path":"../data/paxman_relaxed_topo/Total_Isostatic_Adjustment_Greenland_BedMachinev5.nc",
 "iso_var":"topography change",
 "unit":"metre", "v_range":(-4_900,4_900), "cmap":"cmc.oleron",
"grid_spacing_m": 5_000,
    "kernel_km":      5.0,  
    "kernel_type":   "uniform",  
 "description":"Greenland DEM: bed topo + Paxman isostatic adjustment"},

# ── LAB ────────────────────────────────────────────────────────────
{"label":"LAB",
 "filepath_or_buffer":"../data/Afonso2019/LithoRef18.xyz",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,4], "names":["lon","lat","depth"], "header":10,
 "x_preproc": lambda x: np.where(x>180, x-360, x), "interpol_method":"linear",
 "unit":"metre", "v_range":(-220_000,-41_000), "cmap":"cmc.bamako",
 "reference":"Afonso2019",
 "description":"LAB depth (LithoRef18)"},

{"label":"DYNAMIC",
 "reference":"10.1029/2025JB031837",
 "description":"Tomography-based thermo-chemical mantle convection (Cui et al 2026)"},

# ── CTD ────────────────────────────────────────────────────────────
{"label":"CTD",
 "filepath_or_buffer":"../data/gard_CTD/GardHasterok2021_wocean.txt",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","depth"], "header":1,
 "z_preproc": lambda d: -d*1000.0, "interpol_method":"linear",
 "unit":"metre", "v_range":(-52_000,-2_300), "cmap":"cmc.bamako",
 "description":"Curie temperature depth (Gard & Hasterok 2021)"},

# ── EMAG2 ──────────────────────────────────────────────────────────
{"label":"EMAG2_LOG",
 "filepath_or_buffer":"../data/emag2/EMAG2_V3_UpCont_DataTiff_m0.tif",
 "import_type":"read_raster",
 "z_preproc": lambda m: np.clip(np.sign(m)*np.log(1+np.abs(m)/300), -1, 1),
 "unit":"dimensionless", "v_range":(-1,1), "cmap":"cmc.bilbao_r",
 "description":"EMAG2 magnetic anomaly (log-scaled, zero-centred)"},

# ── Gravity & Geoid ────────────────────────────────────────────────
{"label":"FREE_AIR",
 "filepath_or_buffer":"../data/EIGEN-6C4/EIGEN-6C4_gravity_anomaly_cl.gdf",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","gal"], "header":34,
 "z_preproc": lambda g: g*0.001, "interpol_method":"linear",
 "unit":"mGal", "v_range":(-0.11,0.09), "cmap":"cmc.broc",
 "description":"Free-air gravity anomaly (EIGEN-6C4)"},

{"label":"BOUGUER",
 "filepath_or_buffer":"../data/EIGEN-6C4/EIGEN-6C4_gravity_anomaly_bg.gdf",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","gal"], "header":34,
 "z_preproc": lambda g: g*0.001, "interpol_method":"linear",
 "unit":"mGal", "v_range":(-0.25,0.34), "cmap":"cmc.oleron",
 "description":"Bouguer gravity anomaly (EIGEN-6C4)"},

{"label":"SI",
 "filepath_or_buffer":"../data/GOCE_Curvature/GOCE_Curvature_Topo_Iso_Corr.txt",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","SI"], "header":0,
 "interpol_method":"linear",
 "unit":"dimensionless", "v_range":(-1,1), "cmap":"cmc.broc",
 "description":"GOCE gravity shape index"},

{"label":"GEOID",
 "filepath_or_buffer":"../data/EIGEN-6C4/EIGEN-6C4_geoid.gdf",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","m"], "header":37,
 "interpol_method":"linear",
 "unit":"metre", "v_range":(-56,62), "cmap":"cmc.bamako",
 "description":"Geoid height (EIGEN-6C4)"},

# ── S-wave velocity slices (REVEAL) ───────────────────────────────
# v_range updated to actual absolute velocity range (km/s) from data statistics
*[{"label":f"REVEAL_S{depth}",
   "filepath_or_buffer":"../data/reveal/REVEAL.nc",
   "import_type":"read_grid",
   "x_col":"longitude", "y_col":"latitude", "data_col":"vsv",
   "depth":depth, "interpol_method":"linear",
   "unit":"km/s", "v_range":vr, "cmap":"cmc.roma",
   "description":f"Vsv at {depth} km (REVEAL)"}
  for depth,vr in [
      (60,  (3.8, 4.8)),
      (70,  (3.9, 4.7)),   # best S-wave depth by univariate CV R²
      (80,  (3.9, 4.8)), 
      (90,  (3.9, 4.7)),   # second best
      (100,  (4.0, 4.7)),
      (140,  (4.1, 4.8)),
      (200,  (4.1, 4.8)),
  ]],

# ── P-wave velocity slices (REVEAL) ───────────────────────────────
# v_range updated to actual absolute velocity range (km/s) from data statistics
*[{"label":f"REVEAL_P{depth}",
   "filepath_or_buffer":"../data/reveal/REVEAL.nc",
   "import_type":"read_grid",
   "x_col":"longitude", "y_col":"latitude", "data_col":"vpv",
   "depth":depth, "interpol_method":"linear",
   "unit":"km/s", "v_range":vr, "cmap":"cmc.roma",
   "description":f"Vpv at {depth} km (REVEAL)"}
  for depth,vr in [
      (50,  (7.8, 8.8)),
      (60,  (7.8, 8.8)),
      (90,  (7.8, 8.8)),
      (150, (7.8, 8.5)),   # least-negative P-wave R²   
  ]],

# ── Densities ──────────────────────────────────────────────────────
{"label":"LITH_RHO",
 "filepath_or_buffer":"../data/Afonso2019/LithoRef18.xyz",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,6], "names":["lon","lat","density"], "header":10,
 "x_preproc": lambda x: np.where(x>180, x-360, x), "interpol_method":"linear",
 "unit":"kg/m3", "v_range":(3_200,3_400), "cmap":"cmc.batlow_r",
 "sigma":40, "weight":1.0, "reference":"Afonso2019",
 "description":"Lithospheric mantle density (LithoRef18)"},

{"label":"CRUST_RHO",
 "filepath_or_buffer":"../data/Afonso2019/LithoRef18.xyz",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,5], "names":["lon","lat","density"], "header":10,
 "x_preproc": lambda x: np.where(x>180, x-360, x), "interpol_method":"linear",
 "unit":"kg/m3", "v_range":(2_600,3_000), "cmap":"cmc.batlow_r",
 "sigma":100, "weight":1.0, "reference":"Afonso2019",
 "description":"Crustal density (LithoRef18)"},

# ── Classification ─────────────────────────────────────────────────
# v_range spans all classes present in the data (data_min to data_max)
{"label":"GLiM",
 "filepath_or_buffer":"../data/hartmann-moosdorf_2012/glim_wgs84_0point5deg.txt.asc",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","class"], "header":None,
 "z_preproc": lambda a: np.where((a>0)&(a<16), a, np.nan),
 "interpol_method":"nearest",
 "unit":"class", "v_range":(3,10), "cmap":"Dark2",
 "description":"Global Lithological Map (GLiM)"},

{"label":"TC1",
 "filepath_or_buffer":"../data/artemieva/TC128ageT290606-1x1.csv",
 "import_type":"read_ascii",
 "usecols":[0,1,7], "names":["lon","lat","class"], "header":1,
 "z_preproc": lambda a: np.where(np.isnan(a), np.nan, a.astype(float)),
 "interpol_method":"nearest",
 "unit":"class", "v_range":(15,3_500), "cmap":"Dark2",
 "description":"TC1 thermal age classification (Artemieva)"},

{"label":"GPRV",
 "filepath_or_buffer":"../data/global_tectonics-main/plates&provinces/global_gprv_wage.shp",
 "import_type":"read_shape", "attribute":"prov_type",
 "unit":"type", "v_range":(0,14), "cmap":"Dark2",
 "description":"Geodynamic provinces (GPRV)"},

{"label":"REG",
 "filepath_or_buffer":"../data/SL2013sv_TectRegn_2d/SL2013sv_Cluster_2d",
 "import_type":"read_ascii", "delim_whitespace":True,
 "usecols":[0,1,2], "names":["lon","lat","class"], "header":None,
 "z_preproc": lambda a: np.where(np.isnan(a), np.nan, a.astype(float)),
 "interpol_method":"nearest",
 "unit":"class", "v_range":(1,6), "cmap":"Dark2",
 "description":"Seismic tectonic regions SL2013sv",
 "note":"NOT in obs_list — diagnostics only."},

# ── SEDIMENT_LOG — three grid-keyed variants ──────────────────────────

# IHFC and Greenland: plain log1p, no likelihood masking needed
{
    "label": "SEDIMENT_LOG", "grid": ["IHFC", "Greenland"],
    "import_type": "compute",
    "func": lambda s, d: np.log1p(s.df["SEDIMENT"].values.astype(float)),
    "depends_on": ["SEDIMENT"],
    "unit": "log(m+1)", "v_range": (0, 8.5), "cmap": "cmc.oslo_r",
    "description": "log1p sediment thickness (GST1)",
},

# Antarctica: likelihood-masked, smoothed, then log1p
{
    "label": "SEDIMENT_LOG", "grid": "Antarctica",
    "import_type": "compute",
    "func": compute_sediment_log_antarctica,
    "depends_on": ["SEDIMENT"],
    "grid_spacing_m": 5_000,
    "kernel_km": 15.0,
    "kernel_type": "uniform",
    "unit": "log(m+1)", "v_range": (0, 8.5), "cmap": "cmc.oslo_r",
    "description": "log1p sediment thickness, Li2022 likelihood-masked + smoothed (Antarctica)",
},


# ── Derived / computed ─────────────────────────────────────────────
{"label":"MAG_SEIS_MOHO", "import_type":"compute",
 "func": lambda s,d: s.df["MOHO"].values - s.df["CTD"].values,
 "depends_on":["MOHO","CTD"],
 "unit":"metre", "v_range":(-23_000,16_000), "cmap":"cmc.broc",
 "sigma":500, "weight":1.0,
 "description":"Seismic Moho minus Curie depth"},


{"label":"REVEAL_VP60VS70", "import_type":"compute",
 "func": lambda s,d: np.where(s.df["REVEAL_S70"].values==0, np.nan,
                               s.df["REVEAL_P60"].values/s.df["REVEAL_S70"].values),
 "depends_on":["REVEAL_P60","REVEAL_S70"],
 "unit":"dimensionless", "v_range":(1.6,2.1), "cmap":"cmc.batlow",
 "sigma":0.08, "weight":1.0, "description":"Vp/Vs at 40-50 km"},

{"label":"REVEAL_VP90VS60", "import_type":"compute",
 "func": lambda s,d: np.where(s.df["REVEAL_S60"].values==0, np.nan,
                               s.df["REVEAL_P90"].values/s.df["REVEAL_S60"].values),
 "depends_on":["REVEAL_P90","REVEAL_S60"],
 "unit":"dimensionless", "v_range":(1.7,2.0), "cmap":"cmc.batlow",
 "sigma":0.08, "weight":1.0, "description":"Vp/Vs at 120-140 km"},



{"label":"REVEAL_VP50VS80", "import_type":"compute",
 "func": lambda s,d: np.where(s.df["REVEAL_S80"].values==0, np.nan,
                               s.df["REVEAL_P50"].values/s.df["REVEAL_S80"].values),
 "depends_on":["REVEAL_P50","REVEAL_S80"],
 "unit":"dimensionless", "v_range":(1.7,2.0), "cmap":"cmc.batlow",
 "sigma":0.08, "weight":1.0, "description":"Vp/Vs at 120-140 km"},


{"label":"VOLC_DIST", "import_type":"compute",
 "func":compute_volcano_distance,
 "unit":"metre", "v_range":(4_900,220_000), "cmap":"cmc.batlow",
 "sigma":10_000, "weight":1.0,
 "description":"Distance to nearest volcano (NaN beyond 2 deg); diagnostics only."},

]

