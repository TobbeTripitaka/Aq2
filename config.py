# =============================================================================
# CONFIG.PY  —  Aq2 / Kq2  project configuration  (v4.1)
#
# Single source of truth for paths, parameters, thresholds, colour maps and
# plot ranges shared across notebooks 1–7. Nothing numeric should be hard-coded
# in a notebook: add it here and import it.
#
# v4.1 adds the clustering (6_CLUSTERING) and multi-method ensemble
# (7_ENSEMBLE) blocks: shared paths, the 25 km coarsened grids, cluster-count
# range, mixture-of-experts tree settings, ensemble quantiles/weighting, the
# PICP target, the structural-share threshold, and the to_mW plot helper.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import sys; sys.path.insert(0, str(Path('.').resolve()))
from recipe import dd


MODEL_VERSION = '0_4'

VERBOSE = True

# ── Directory layout ──────────────────────────────────────────────────────────
data_root    = Path("../data")
output_root  = Path("./output")
local_data   = Path("./data")
fig_root     = Path("./fig")
local_temp   = Path("./temp")
log_dir      = Path("./logs")
sweep_dir    = output_root / "sweeps"
model_dir    = output_root / "models"
targets_dir  = output_root / "targets"
fig_dir_obs  = Path("fig/observables")

# ── Clustering / UQ paths (6_CLUSTERING) and ensemble paths (7_ENSEMBLE) ─────
cluster_dir   = output_root / "clustering"
ensemble_dir  = output_root / "ensemble"
CLUSTER_DIR   = cluster_dir            # upper-case aliases used in the notebooks
ENSEMBLE_DIR  = ensemble_dir

fig_dir_cluster  = fig_root / "6_CLUSTERING"
fig_dir_ensemble = fig_root / "7_ENSEMBLE"

for _p in [output_root, local_data, fig_root, local_temp, log_dir,
           sweep_dir, model_dir, targets_dir, fig_dir_obs,
           cluster_dir, ensemble_dir, fig_dir_cluster, fig_dir_ensemble]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Source data files ─────────────────────────────────────────────────────────
GHFDB_file = data_root / "NHFC" / "IHFC_2024_GHFDB.xlsx"
bedmachine_antarctica_file = (
    data_root / "bed_machine_v3"
    / "NSIDC-0756_3-20260115_001005" / "BedMachineAntarctica-v3.nc")
bedmachine_greenland_file = (
    data_root / "bed_machine_v3"
    / "IDBMG4_6-20260115_001332" / "BedMachineGreenland-v6.nc")

# ── Parquet paths (written by 1_Import) ──────────────────────────────────────
parquet_ref = local_data / "IHFC_obs.parquet"
parquet_ant = local_data / "antarctica.parquet"
parquet_grl = local_data / "greenland.parquet"

# ── Coarsened 25 km parquet grids (used by 6_CLUSTERING / 7_ENSEMBLE) ─────────
# The clustering and ensemble work on a coarser grid than the 5 km prediction
# grids to keep the per-cell CDF/entropy work tractable.
coarsen_target_spacing_m = 25_000
parquet_ant_25km = local_data / "antarctica_25km.parquet"
parquet_grl_25km = local_data / "greenland_25km.parquet"
CRS_ANT25 = parquet_ant_25km          # aliases kept for back-compat with drafts
CRS_GRL25 = parquet_grl_25km

# ── Output NetCDF paths ───────────────────────────────────────────────────────
ant_Aq2_qrf_nc  = output_root / "ant_Aq2_qrf.nc"
ant_Aq2_gb_nc   = output_root / "ant_Aq2_gb.nc"
ant_Aq2_sim_nc  = output_root / "ant_Aq2_sim.nc"
grl_Kq2_qrf_nc  = output_root / "grl_Kq2_qrf.nc"
grl_Kq2_gb_nc   = output_root / "grl_Kq2_gb.nc"
grl_Kq2_sim_nc  = output_root / "grl_Kq2_sim.nc"

# ── Model artefact paths (written by 4x_) ─────────────────────────────────
model_paths = {
    'qrf'        : model_dir / 'qrf_artefacts.pkl',
    'gbm_q05'    : model_dir / 'gbm_q05_model.pkl',
    'gbm_q25'    : model_dir / 'gbm_q25_model.pkl',
    'gbm_q50'    : model_dir / 'gbm_q50_model.pkl',
    'gbm_q75'    : model_dir / 'gbm_q75_model.pkl',
    'gbm_q95'    : model_dir / 'gbm_q95_model.pkl',
    "sim_correction" : model_dir / "sim_correction_spline.pkl",
    "model_metrics"  : model_dir / "model_metrics.csv",
}

# ── Sweep / parameter JSON paths (written by 3_SWEEPS) ───────────────────────
param_paths = {
    "qrf" : sweep_dir / "qrf_params.json",
    "gbm" : sweep_dir / "gbm_params.json",
    "sim" : sweep_dir / "sim_best_params.json",
}

# ── CRS ───────────────────────────────────────────────────────────────────────
ant_crs = "EPSG:3031"
grl_crs = "EPSG:3413"
ref_crs = "EPSG:4326"

# ── Grid parameters ───────────────────────────────────────────────────────────
grid_spacing_m_ant = 5_000
grid_spacing_m_grl = 5_000
ant_grid_extent_m  = 2_810_000
grl_grid_extent_m  = 910_000

# ── Target grid definitions (used by 5_TARGETS) ───────────────────────────────
TARGET_GRIDS = [
    dict(label="ant", parquet=parquet_ant,
         crs=ant_crs, epsg=3031,
         x_col="x", y_col="y",
         out_nc_qrf=ant_Aq2_qrf_nc,
         out_nc_gb=ant_Aq2_gb_nc,
         out_nc_sim=ant_Aq2_sim_nc),
    dict(label="grl", parquet=parquet_grl,
         crs=grl_crs, epsg=3413,
         x_col="x", y_col="y",
         out_nc_qrf=grl_Kq2_qrf_nc,
         out_nc_gb=grl_Kq2_gb_nc,
         out_nc_sim=grl_Kq2_sim_nc),
]

# ── Heat-flow thresholds [W/m²] ───────────────────────────────────────────────
q_min     = 0.0      # 'normal' lower for statistics and plots
q_clip_min = 0.001   # hard minimum
q_max     = 0.140    # 'normal' upper for statistics and plots
q_clip_max = 0.350   # upper clip applied in 1_Import — used as model ceiling

deep_ocean_threshold     = -5000.0   # DEM threshold for ocean masking [m]
ref_clustering_radius_km = (grid_spacing_m_ant / 1000) / np.sqrt(np.pi)

# =============================================================================
# OBSERVABLES
# Three lists, each a strict subset of the next:
#   obs_model  ⊆  obs_sweep  ⊆  obs
#
# RULES:
#   - obs_model   : exactly the columns used in prediction — 22 features.
#                   Change only after a new sweep has been completed.
#   - obs_sweep   : all candidates fed into 3_SWEEPS (add tentative features here).
#   - obs         : full parquet catalogue — everything 1_Import writes.
#
# Column naming convention:
#   REVEAL_*      — REVEAL tomography-derived columns (raw or computed)
#   No REVEAL_    — non-seismic observables
# =============================================================================

# ── obs_model : 22 modelling features ─────────────────────────────────────────
obs_model = [
    "MOHO", "MOHO_GRAV", "DEM", "LAB",
    "FREE_AIR", "BOUGUER", "SI", "GEOID",
    "REVEAL_S80", "REVEAL_S90", "REVEAL_S70", "REVEAL_S100",
    "REVEAL_VP60VS70", "REVEAL_VP90VS60", "REVEAL_VP50VS80",
    "LITH_RHO", "CRUST_RHO", "MAG_SEIS_MOHO",
    "SEDIMENT", "CTD", "EMAG2_LOG",
]

# ── obs_sweep : all sweep candidates (obs_model + extras worth testing) ────────
obs_sweep = obs_model + [
    # Additional REVEAL depths not in obs_model but worth sweeping
    "REVEAL_P150",        # Vpv at 150 km  R²=−0.056
]
obs_sweep = list(set(obs_sweep))

# ── obs : full parquet catalogue ──────────────────────────────────────────────

obs = list(dict.fromkeys(
    d["label"] for d in dd
    if not d.get("exclude_from_obs", False)
))

assert all(f in obs for f in obs_sweep), "obs_sweep has features not in recipe!"

# ── Per-model feature resolution ────────────────────────────────────────
def resolve_features(method):
    """Return the feature subset a given model was swept and trained on.

    Reads obs_sel from the method's sweep JSON (param_paths[method]) so the
    model uses the same columns at fit and predict time. Falls back to
    obs_model if the JSON or the obs_sel key is missing.
    method : one of "qrf", "gbm", "sim".
    """
    key = str(method).lower()
    if key not in param_paths:
        raise ValueError(
            f"resolve_features: unknown method {method!r}; "
            f"expected one of {sorted(param_paths)}")
    try:
        with open(param_paths[key]) as fh:
            params = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(obs_model)
    sel = params.get("obs_sel") or params.get("features")
    return list(sel) if sel else list(obs_model)


# =============================================================================
# MODEL TRAINING CONSTANTS
# Hyperparameters live in output/sweeps/*.json (written by 3_SWEEPS).
# Defaults below are used only when no JSON is found.
# =============================================================================
random_state          = 42
calibration_fraction  = 0.10
conformal_alpha       = 0.10
N_CV_FOLDS            = 5
USE_LOG_TARGET        = False
PRED_QUANTILES        = [0.05, 0.25, 0.50, 0.75, 0.95]

# QRF defaults
QRF_N_ESTIMATORS     = 2_000
QRF_MAX_FEATURES     = 0.3
QRF_MIN_SAMPLES_LEAF = 2
QRF_MAX_DEPTH        = 25
QRF_N_JOBS           = -1

# GBM defaults
GB_MAX_ITER          = 500
GB_MAX_DEPTH         = 6
GB_LEARNING_RATE     = 0.05
GB_MIN_SAMPLES_LEAF  = 20
GB_MAX_FEATURES      = 0.8
GB_L2_REG            = 1.0

# Distribution / entropy
ENTROPY_BINS   = 50
HIST_BIN_WIDTH = 0.010   # [W/m²] bin width for entropy / histogram outputs
HIST_MAX_BINS  = 400     # safety cap
BATCH_SIM      = 512     # Similarity kernel batch size (memory control)

# =============================================================================
# CLUSTERING  /  MIXTURE-OF-EXPERTS  (6_CLUSTERING)
# Follows Al-Aghbary et al. (2026): k-means on standardised observables,
# k chosen by Elbow + Davies–Bouldin; per-cluster QRF experts with tree depth
# scaled to cluster size (Md = 0.9·log(n)).
# =============================================================================
NCLUSTERS_RANGE_MIN = 2       # smallest k searched
NCLUSTERS_RANGE_MAX = 5       # largest k searched (paper optimum was 3)
NCLUSTERS_OVERRIDE  = None    # set to an int (e.g. 3) to pin k and skip the search
MOE_TREE_DEPTH_COEF = 0.9     # expert tree depth = coef · log2(cluster size)
MOE_MIN_TREE_DEPTH  = 5       # floor so small clusters still get a usable tree

# =============================================================================
# MULTI-METHOD ENSEMBLE  (7_ENSEMBLE)
# Quantile-mixture pooling of the QRF / GBM / SIM regressions, a three-term
# variance decomposition (aleatoric + within-method epistemic + between-method
# structural) and fused robustness / confidence / explainability maps.
# =============================================================================
ENSEMBLE_METHODS     = ["qrf", "gbm", "sim"]   # drop a method here to exclude it
ENSEMBLE_QUANTILES   = [0.05, 0.25, 0.50, 0.75, 0.95]
ENSEMBLE_WEIGHT_MODE = "inv_rmse"              # "inv_rmse" | "equal"
ENSEMBLE_PICP_TARGET = 0.90                    # nominal coverage of the 5–95 band
# A cell is labelled "structurally-uncertain" (5th explainability class) when
# the between-method structural variance is at least this share of the total —
# i.e. the methods disagree about location, not just spread. 0.5 = structural
# term is the single largest contributor.
STRUCTURAL_SHARE_THRESHOLD = 0.5

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================
PARAM_N_RUNS     = 200
MC_N_RUNS        = 300
N_OPTUNA_TRIALS  = 200
CV_FOLDS_SIM     = 5
# Optuna search bounds for per-feature sigma after StandardScaler.
# After z-score normalisation features live in roughly [-3, +3].
# sigma=0.25  → neighbourhood half-width ≈ 0.25 std, very sharp
# sigma=4.0   → neighbourhood half-width ≈ 4 std, very wide / near-flat
# standardisation to avoid wasting trials on degenerate extremes.
# SIM_SIGMA_BOUNDS is the single source of truth for the similarity per-feature
# sigma search range; 3c reads it from here and records the bounds it used in
# sim_best_params.json. SIGMA_BOUNDS is retained as an alias for back-compat.
SIM_SIGMA_BOUNDS = (0.25, 4.0)
SIGMA_BOUNDS = SIM_SIGMA_BOUNDS
K_RANGE          = (2.0, 20.0)

PARAM_GRID = {
    "n_estimators"    : [500, 1000, 1500, 2000],
    "max_depth"       : [None, 15, 20, 25, 30, 40, 50],
    "min_samples_leaf": [1, 2, 3, 5, 7, 10],
    "max_features"    : [0.3, 0.4, 0.5, 0.6, 0.7, "sqrt", "log2"],
    "use_log_target"  : [True, False],
    "q_clip_max"      : [0.200, 0.225, 0.250, 0.300, 0.350],
    "cal_frac"        : [0.10, 0.15, 0.20, 0.25],
}

# =============================================================================
# GRID / FORWARD MODEL
# =============================================================================
VOLC_KERNEL_SIGMA_ACTIVE_M      = 10_000.0
VOLC_KERNEL_SIGMA_PLEISTOCENE_M = 20_000.0
VOLC_KERNEL_FACTOR_ACTIVE       = 30.0
VOLC_KERNEL_FACTOR_PLEISTOCENE  = 5.0
VOLC_KERNEL_EXTENT_M            = 100_000.0
VOLC_KERNEL_DX_M                = 500.0
TOPO_CORRECTION_R_M             = 5_000.0
TOPO_CORRECTION_MIN_RELIEF_M    = 50.0
TOPO_K_ROCK_DEFAULT             = 1.8
TOPO_K_ICE_DEFAULT              = 2.1
TOPO_K_REF                      = 2.5

# =============================================================================
# PLOTTING CONSTANTS
# =============================================================================
milli = 1 / 1_000
micro = 1 / 1_000_000
km    = 1_000


def to_mW(x):
    """Convert heat flow from W/m² to mW/m².

    Apply twice to convert a variance from (W/m²)² to (mW/m²)². NaN-safe;
    accepts scalars, lists or arrays and always returns an ndarray.
    """
    return np.asarray(x) * 1_000.0


hf_cmap   = "cmc.lajolla"
hf_v_min  = 15
hf_v_max  = 180
hf_unit   = "mW/m²"

unc_cmap  = "cmc.batlow"
std_cmap  = "cmc.batlow"
entropy_cmap = "cmc.batlow"
unc_unit  = std_unit = "mW/m²"
entropy_unit = "bits"

coastline_color     = "black"
coastline_linewidth = 0.5
glacier_color       = "darkblue"
glacier_linewidth   = 1.0
lake_color          = "cyan"

FIG_DPI      = 150
FIG_EXT      = ".png"
MAP_W_REF    = 6.69;  MAP_H_REF = 3.54
MAP_W_ANT    = 3.35;  MAP_H_ANT = 3.35
MAP_W_GRL    = 1.38;  MAP_H_GRL = 3.35
MAP_W_XP     = 2.0;   MAP_H_XP  = 2.0
CBAR_W       = 2.00;  CBAR_H    = 0.55
HEXBIN_GRIDSIZE = 55
LOESS_FRAC   = 0.25
LOESS_N_EVAL = 200

FIG_OBS_DIR      = Path("fig/observables")
TEMPLATE_PATH    = FIG_OBS_DIR / "obs_template.tikz"
MANIFEST_PATH    = FIG_OBS_DIR / "manifest.csv"
ALL_OBS_TEX      = FIG_OBS_DIR / "all_observables.tex"
FIG_WRAPPER      = Path("fig/fig_observables.tex")
FIG_OBS_TEX_PREFIX = "observables"

# NetCDF metadata
netcdf_compression_level = 4
NETCDF_AUTHOR      = "Stål et al. (Aq2/Kq2)"
NETCDF_CONVENTIONS = "CF-1.8"

# =============================================================================
# STARTUP REPORT
# =============================================================================
if VERBOSE:
    print(
        f"✓ config.py v4.1 | "
        f"obs_model: {len(obs_model)} | "
        f"obs_sweep: {len(obs_sweep)} | "
        f"obs: {len(obs)} | "
        f"ensemble: {'+'.join(ENSEMBLE_METHODS)}"
    )
