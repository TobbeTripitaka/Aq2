# ============================================================================
# CONFIG.PY — Aq2 / Kq2 project configuration (v3.0)
# ============================================================================
# Changes from v2.1:
#   - Fixed syntax error: obs_sel and obs are now separate, properly closed lists
#   - obs_sel renamed to obs_model (22 curated modelling features, post-depth-sweep)
#   - obs is the full catalogue (all parquet columns, for sweeps)
#   - obs_sweep = obs minus uncertainty, categorical, and volcanic-distance columns
#   - REVEAL depth layers updated to post-sweep selection
#   - QRF hyperparameters removed from runtime — now in output/sweeps/qrf_params.json
#   - SIM_SIGMAS / SIM_BEST_K removed — now in output/sweeps/sim_best_params.json
#   - All sweep outputs unified under output/sweeps/
# ============================================================================
from pathlib import Path
import numpy as np

VERBOSE = True

data_root   = Path("../data")
output_root = Path("./output")
local_data  = Path("./data")
fig_root    = Path("./fig")
local_temp  = Path("./temp")
log_dir     = Path("./logs")
sweep_dir   = output_root / "sweeps"

for _p in [output_root, local_data, fig_root, local_temp, log_dir, sweep_dir]:
    _p.mkdir(parents=True, exist_ok=True)

GHFDB_file = data_root / "NHFC" / "IHFC_2024_GHFDB.xlsx"
bedmachine_antarctica_file = (data_root / "bed_machine_v3"
    / "NSIDC-0756_3-20260115_001005" / "BedMachineAntarctica-v3.nc")
bedmachine_greenland_file  = (data_root / "bed_machine_v3"
    / "IDBMG4_6-20260115_001332"    / "BedMachineGreenland-v6.nc")

parquet_ref = local_data / "IHFC_obs.parquet"
parquet_ant = local_data / "antarctica.parquet"
parquet_grl = local_data / "greenland.parquet"

ant_Aq1_5_qrf_nc  = output_root / "ant_Aq1_5_qrf.nc"
ant_Aq1_5_gb_nc   = output_root / "ant_Aq1_5_gb.nc"
ant_Aq1_5_sim_nc  = output_root / "ant_Aq1_5_sim.nc"
ant_Aq2_qrf_nc    = output_root / "ant_Aq2_qrf.nc"
ant_Aq2_gb_nc     = output_root / "ant_Aq2_gb.nc"
ant_Aq2_sim_nc    = output_root / "ant_Aq2_sim.nc"
grl_Kq1_5_qrf_nc  = output_root / "grl_Kq1_5_qrf.nc"
grl_Kq1_5_gb_nc   = output_root / "grl_Kq1_5_gb.nc"
grl_Kq1_5_sim_nc  = output_root / "grl_Kq1_5_sim.nc"
grl_Kq2_qrf_nc    = output_root / "grl_Kq2_qrf.nc"
grl_Kq2_gb_nc     = output_root / "grl_Kq2_gb.nc"
grl_Kq2_sim_nc    = output_root / "grl_Kq2_sim.nc"

ant_crs = "EPSG:3031"
grl_crs = "EPSG:3413"
ref_crs = "EPSG:4326"

grid_spacing_m_ant = 5_000
grid_spacing_m_grl = 5_000
ant_grid_extent_m  = 2_810_000
grl_grid_extent_m  =   910_000

deep_ocean_threshold     = -5000.0
ref_clustering_radius_km = (grid_spacing_m_ant / 1000) / np.sqrt(np.pi)

q_min      = 0.0
q_clip_min = 0.001
q_max      = 0.250
q_clip_max = 0.350

# ── obs_model : 22 features for ALL prediction models ─────────────────────
obs_model = [
    # Crustal structure
    "MOHO_GRAV",        # Moho depth, gravity-derived (GEMMA)
    "MOHO",             # Moho depth, seismic (Szwillus 2019)
    "DEM",              # Corrected bed topography
    "LAB",              # Lithosphere–asthenosphere boundary (Afonso 2019)
    # Gravity & geoid
    "FREE_AIR",         # Free-air gravity anomaly
    "BOUGUER",          # Bouguer gravity anomaly
    "SI",               # GOCE scalar invariant
    "GEOID",            # Geoid height (EIGEN-6C4)
    # Seismic — S-wave (top 2 by univariate CV R²)
    "REVEAL_S80",       # Vsv at  80 km  R²=+0.038
    "REVEAL_S90",       # Vsv at  90 km  R²=+0.014
    # Seismic — P-wave (wave-type diversity)
    "REVEAL_P180",      # Vpv at 180 km  R²=−0.038 (least-negative)
    "REVEAL_P150",      # Vpv at 150 km  R²=−0.056
    # Seismic — Vp/Vs cross-depth ratios
    "REVEAL_VP60VS70",  # Vp@60 / Vs@70  R²=+0.019
    "REVEAL_VP90VS60",  # Vp@90 / Vs@60  R²=+0.012
    # Densities
    "LITH_RHO",
    "CRUST_RHO",
    # Derived
    "MAG_SEIS_MOHO",    # Seismic Moho − Curie depth
    "SEDIMENT",
    "REVEAL_S_DIFF_200_220",   # S-wave gradient 200–220 km
    "REVEAL_P_DIFF_40_60",     # P-wave gradient  40–60 km
    "CTD",              # Curie temperature depth
    "EMAG2_LOG",        # EMAG2 magnetic anomaly (log-scaled)
]

# ── obs : full parquet catalogue ──────────────────────────────────────────
obs = [
    "MOHO_GRAV", "MOHO_GRAV_U", "MOHO", "MOHO_U",
    "DEM", "LAB", "CTD", "EMAG2_LOG",
    "FREE_AIR", "BOUGUER", "SI", "GEOID",
    "REVEAL_S50",  "REVEAL_S80",  "REVEAL_S90",
    "REVEAL_S100", "REVEAL_S140", "REVEAL_S200", "REVEAL_S220",
    "REVEAL_P40",  "REVEAL_P60",  "REVEAL_P120",
    "REVEAL_P150", "REVEAL_P180", "REVEAL_P210",
    "REVEAL_VP60VS70", "REVEAL_VP90VS60",
    "REVEAL_P_DIFF_40_60", "REVEAL_S_DIFF_200_220", "REVEAL_VPVS_DIFF",
    "LITH_RHO", "CRUST_RHO", "MAG_SEIS_MOHO",
    "REVEAL_S_DIFF_200_220", "REVEAL_P_DIFF_40_60",
    "REVEAL_VP_40_VS_50", "REVEAL_VP_120_VS_140", "VPVS_DIFF",
    "SEDIMENT", "VOLC_DIST", "REG", "GLiM", "TC1",
]

# ── obs_sweep : sweep input (obs minus uncertainty, categorical, VOLC_DIST) ─
obs_sweep = [
    "MOHO_GRAV", "MOHO",
    "DEM", "LAB", "CTD", "EMAG2_LOG",
    "FREE_AIR", "BOUGUER", "SI", "GEOID",
    "REVEAL_S50",  "REVEAL_S80",  "REVEAL_S90",
    "REVEAL_S100", "REVEAL_S140", "REVEAL_S200", "REVEAL_S220",
    "REVEAL_P40",  "REVEAL_P60",  "REVEAL_P120",
    "REVEAL_P150", "REVEAL_P180", "REVEAL_P210",
    "REVEAL_VP60VS70", "REVEAL_VP90VS60",
    "LITH_RHO", "CRUST_RHO", "MAG_SEIS_MOHO",
    "REVEAL_S_DIFF_200_220", "REVEAL_P_DIFF_40_60",
    "REVEAL_VP_40_VS_50", "REVEAL_VP_120_VS_140", "VPVS_DIFF",
    "SEDIMENT",
]

# ── Target grid definitions ────────────────────────────────────────────────
target_grids = [
    dict(label="ant", parquet=parquet_ant, crs=ant_crs, epsg=3031,
         spacing_m=grid_spacing_m_ant, extent_m=ant_grid_extent_m,
         bedmachine=bedmachine_antarctica_file, bm_res_m=500,
         out_nc_qrf=ant_Aq1_5_qrf_nc, out_nc_gb=ant_Aq1_5_gb_nc,
         out_nc_sim=ant_Aq1_5_sim_nc, out_nc_qrf2=ant_Aq2_qrf_nc,
         out_nc_gb2=ant_Aq2_gb_nc, out_nc_sim2=ant_Aq2_sim_nc),
    dict(label="grl", parquet=parquet_grl, crs=grl_crs, epsg=3413,
         spacing_m=grid_spacing_m_grl, extent_m=grl_grid_extent_m,
         bedmachine=bedmachine_greenland_file, bm_res_m=150,
         out_nc_qrf=grl_Kq1_5_qrf_nc, out_nc_gb=grl_Kq1_5_gb_nc,
         out_nc_sim=grl_Kq1_5_sim_nc, out_nc_qrf2=grl_Kq2_qrf_nc,
         out_nc_gb2=grl_Kq2_gb_nc, out_nc_sim2=grl_Kq2_sim_nc),
]

# ── Training parameters ────────────────────────────────────────────────────
random_state         = 42
calibration_fraction = 0.1
conformal_alpha      = 0.10
N_CV_FOLDS           = 5
USE_LOG_TARGET       = False
USE_CONFORMAL        = False
PRED_QUANTILES       = [0.05, 0.25, 0.50, 0.75, 0.95]

# ── QRF defaults (override from output/sweeps/qrf_params.json post-sweep) ─
QRF_N_ESTIMATORS     = 2_000
QRF_MAX_FEATURES     = 0.3
QRF_MIN_SAMPLES_LEAF = 2
QRF_MAX_DEPTH        = 25
QRF_N_JOBS           = -1

# ── GB defaults ────────────────────────────────────────────────────────────
GB_MAX_ITER          = 500
GB_MAX_DEPTH         = 6
GB_LEARNING_RATE     = 0.05
GB_MIN_SAMPLES_LEAF  = 20
GB_MAX_FEATURES      = 0.8
GB_L2_REG            = 1.0
GB_N_JOBS            = -1

ENTROPY_BINS         = 50

# ── Sweep run-count parameters ─────────────────────────────────────────────
PARAM_N_RUNS     = 200
MC_N_RUNS        = 300
N_OPTUNA_TRIALS  = 200

# ── Param sweep grid ───────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators"    : [500, 1000, 1500, 2000],
    "max_depth"       : [None, 15, 20, 25, 30, 40, 50],
    "min_samples_leaf": [1, 2, 3, 5, 7, 10],
    "max_features"    : [0.3, 0.4, 0.5, 0.6, 0.7, "sqrt", "log2"],
    "use_log_target"  : [True, False],
    "q_clip_max"      : [0.200, 0.225, 0.250, 0.300, 0.350],
    "cal_frac"        : [0.10, 0.15, 0.20, 0.25],
}

# ── Similarity sweep parameters ────────────────────────────────────────────
# NOTE: SIM_BEST_K and SIM_SIGMAS are NOT defined here.
# They are written to output/sweeps/sim_best_params.json by 3b_MODEL_SWEEPS.ipynb
# and must be loaded from there by 5_Sim.ipynb.
OPTIMISE_SIGMA = False
OPTIMISE_K     = False
CV_FOLDS_SIM   = 5
SIGMA_BOUNDS   = (0.1, 5.0)
K_RANGE        = (2.0, 60.0)
HIST_BIN_WIDTH = 0.010
SIM_BATCH_SIZE = 500

# ── Forward model parameters ───────────────────────────────────────────────
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

milli = 1 / 1_000
micro = 1 / 1_000_000
km    = 1_000

hf_cmap             = "cmc.lajolla"
hf_v_min            = 15
hf_v_max            = 180
hf_unit             = "mW/m²"
unc_cmap            = "cmc.batlow"
std_cmap            = "cmc.batlow"
entropy_cmap        = "cmc.batlow"
unc_unit = std_unit = entropy_unit = "mW/m²"
entropy_unit        = "bits"
coastline_color     = "black"
coastline_linewidth = 0.5
glacier_color       = "darkblue"
glacier_linewidth   = 1.0
lake_color          = "cyan"
fig_dpi             = 150
fig_facecolor       = "white"
fig_edgecolor       = "none"
font_size           = 10
font_family         = "sans-serif"

netcdf_compression_level = 4
NETCDF_AUTHOR      = "Stål et al. (Aq2/Kq2)"
NETCDF_CONVENTIONS = "CF-1.8"

FIG_DPI             = 150
FIG_EXT             = ".png"
MAP_W_REF = 6.69; MAP_H_REF = 3.54
MAP_W_ANT = 3.35; MAP_H_ANT = 3.35
MAP_W_GRL = 1.38; MAP_H_GRL = 3.35
MAP_W_XP  = 2.0;  MAP_H_XP  = 2.0
CBAR_W = 2.00; CBAR_H = 0.55
HEXBIN_GRIDSIZE = 55
LOESS_FRAC      = 0.25
LOESS_N_EVAL    = 200

FIG_OBS_DIR        = Path("fig/observables")
FIG_DIR            = FIG_OBS_DIR
TEMPLATE_PATH      = FIG_OBS_DIR / "obs_template.tikz"
MANIFEST_PATH      = FIG_OBS_DIR / "manifest.csv"
ALL_OBS_TEX        = FIG_OBS_DIR / "all_observables.tex"
FIG_WRAPPER        = Path("fig/fig_observables.tex")
FIG_OBS_TEX_PREFIX = "observables"
FIG_OBS_DIR.mkdir(parents=True, exist_ok=True)

if VERBOSE:
    print(
        f"✓ config.py v3.0 | "
        f"obs_model: {len(obs_model)} | "
        f"obs_sweep: {len(obs_sweep)} | "
        f"obs: {len(obs)}"
    )

# ── Model file paths (populated by 4_MODEL.ipynb) ─────────────────────────
# Paths to trained model artefacts. 5_TARGETS loads these directly.
# Update MODEL_DIR if you move the output folder.
MODEL_DIR   = Path('output/models')
PARAM_DIR   = Path('output/sweeps')
TARGETS_DIR = Path('output/targets')

model_paths = {
    "qrf"             : MODEL_DIR / "qrf_model.pkl",
    "gbm_q05"         : MODEL_DIR / "gbm_q05_model.pkl",
    "gbm_q50"         : MODEL_DIR / "gbm_q50_model.pkl",
    "gbm_q95"         : MODEL_DIR / "gbm_q95_model.pkl",
    "sim_correction"  : MODEL_DIR / "sim_correction_spline.pkl",
    "model_metrics"   : MODEL_DIR / "model_metrics.csv",
}

param_paths = {
    "qrf" : PARAM_DIR / "qrf_params.json",
    "gbm" : PARAM_DIR / "gbm_params.json",
    "sim" : PARAM_DIR / "sim_best_params.json",
}
