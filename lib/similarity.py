#!/usr/bin/env python3
# agrid.py v0.1.1b — Tobias Staal 2026
# tobias.staal@utas.edu.au
# MIT License

"""
lib/similarity.py

Core similarity functions for Aq2/Kq2.

Implements:
- Gaussian similarity kernel between reference and target observables
- Leave-one-out sweep on reference database (similarity_sweep)
- Prediction on arbitrary target grids (similarity_predict)

All heavy loops are JIT-compiled with numba. Observables should be
standardised before calling these functions.
"""

from typing import Tuple, Optional

import numpy as np
from numba import jit


# ----------------------------------------------------------------------
# Helpers: sigma preparation and feature standardisation
# ----------------------------------------------------------------------


def prepare_sigma(S_r: np.ndarray) -> np.ndarray:
    """
    Prepare sigma array for similarity computations.

    Parameters
    ----------
    S_r : np.ndarray
        Sigma information for the reference database records.
        Typically shaped (n_ref, n_feat) or (n_feat,).

    Returns
    -------
    sigma_arr : np.ndarray
        1-D array of length n_feat with per-feature sigma values.
        If S_r is 2-D, takes the median over references and clips
        to a small positive minimum to avoid division by zero.
    """
    S_r = np.asarray(S_r, dtype=np.float32)

    if S_r.ndim == 1:
        sigma_arr = S_r.copy()
    elif S_r.ndim == 2:
        # (n_ref, n_feat) → per-feature robust sigma
        sigma_arr = np.nanmedian(S_r, axis=0)
    else:
        raise ValueError(f"S_r must have 1 or 2 dimensions, got {S_r.ndim}")

    # avoid zero or negative sigmas
    sigma_arr = np.where(np.isfinite(sigma_arr), sigma_arr, np.nan)
    finite = sigma_arr[np.isfinite(sigma_arr)]
    if finite.size == 0:
        raise ValueError("No finite values in S_r to derive sigma_arr.")
    min_pos = np.nanpercentile(finite, 1)
    eps = max(min_pos * 1e-3, 1e-6)
    sigma_arr = np.clip(sigma_arr, eps, None)

    return sigma_arr.astype(np.float32)


def standardise_features(X: np.ndarray,
                         mean: np.ndarray,
                         std: np.ndarray) -> np.ndarray:
    """
    Standardise features: (X - mean) / std.

    Parameters
    ----------
    X : (n_samples, n_feat)
    mean, std : (n_feat,)

    Returns
    -------
    X_std : (n_samples, n_feat)
    """
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0

    return (X - mean[None, :]) / std_safe[None, :]


# ----------------------------------------------------------------------
# Gaussian similarity kernels
# ----------------------------------------------------------------------


@jit(nopython=True, fastmath=True, cache=True)
def sim_gaussian_2d(X_ref: np.ndarray,
                    X_tgt: np.ndarray,
                    sigma_arr: np.ndarray) -> np.ndarray:
    """
    Multi-feature Gaussian similarity between reference and target vectors.

    This is the JIT equivalent of applying your scalar sim_normal
    across all observables, assuming features are already standardised
    and sigma_arr holds optimised per-feature sigmas.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
        Reference feature matrix.
    X_tgt : (n_tgt, n_feat)
        Target feature matrix (flattened if original is higher-dim).
    sigma_arr : (n_feat,)
        Per-feature sigma values.

    Returns
    -------
    S : (n_tgt, n_ref)
        Similarity matrix S[j, i] between target j and reference i.
    """
    n_ref, n_feat = X_ref.shape
    n_tgt = X_tgt.shape[0]

    S = np.empty((n_tgt, n_ref), dtype=np.float32)

    for j in range(n_tgt):
        for i in range(n_ref):
            acc = 0.0
            for k in range(n_feat):
                d = (X_tgt[j, k] - X_ref[i, k]) / sigma_arr[k]
                acc += d * d
            S[j, i] = np.exp(-0.5 * acc)

    return S


# Placeholder for future kernels (step, etc.). Must match signature.
@jit(nopython=True, fastmath=True, cache=True)
def _sim_identity_2d(X_ref: np.ndarray,
                     X_tgt: np.ndarray,
                     sigma_arr: np.ndarray) -> np.ndarray:
    """
    Identity 'kernel' used as a placeholder for future similarities.
    Currently unused; kept to illustrate the expected signature.
    """
    n_ref, _ = X_ref.shape
    n_tgt, _ = X_tgt.shape
    S = np.zeros((n_tgt, n_ref), dtype=np.float32)
    for j in range(n_tgt):
        for i in range(n_ref):
            S[j, i] = 1.0
    return S


# ----------------------------------------------------------------------
# Similarity sweep on reference database (LOO)
# ----------------------------------------------------------------------


@jit(nopython=True, fastmath=True, cache=True)
def _similarity_sweep_core(X_ref: np.ndarray,
                           H_ref: np.ndarray,
                           W_r: np.ndarray,
                           sigma_arr: np.ndarray,
                           K: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core JIT loop implementing leave-one-out similarity_test on the
    reference database.

    This mirrors your original similarity_test semantics:

        for each reference i:
            B = sim(V_r, V_r[:,i], S_r, S_r[:,i])
            B[:, i] = 0
            W = B * W_r
            N_sim = sum(W, axis=0)
            w_i = K ** N_sim
            q   = sum(w_i * H_r) / sum(w_i)
            sig = sqrt(sum(w_i * (H_r - q)^2) / sum(w_i))

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
        Standardised reference features.
    H_ref : (n_ref,)
        Reference heat-flow values.
    W_r   : (n_ref,)
        Reference weights.
    sigma_arr : (n_feat,)
        Per-feature sigmas.
    K : float
        Exponential base.

    Returns
    -------
    Q      : (n_ref,)
        Leave-one-out predicted q for each reference.
    sig_Q  : (n_ref,)
        Leave-one-out standard deviation.
    N_sim  : (n_ref,)
        N_sim per reference (sum over W columns).
    obs_sum: (n_ref, n_ref)
        Sum of similarities per ref (for diagnostics).
    """
    n_ref, n_feat = X_ref.shape

    Q = np.zeros(n_ref, dtype=np.float32)
    sig_Q = np.zeros(n_ref, dtype=np.float32)
    N_sim = np.zeros(n_ref, dtype=np.float32)
    obs_sum = np.zeros((n_ref, n_ref), dtype=np.float32)

    for idx in range(n_ref):
        # Build similarity of all refs to the "test" reference idx
        # B_i(j) = similarity between ref j and ref idx
        # We reuse the same Gaussian kernel per pair.
        # Here we only need a 1D column B(:, 0), but keep 2D for clarity.
        B_col = np.empty(n_ref, dtype=np.float32)

        for j in range(n_ref):
            acc = 0.0
            for k in range(n_feat):
                d = (X_ref[j, k] - X_ref[idx, k]) / sigma_arr[k]
                acc += d * d
            B_col[j] = np.exp(-0.5 * acc)

        # remove self
        B_col[idx] = 0.0

        # obs_sum[:, idx] = sum over "rows" (here identical to B_col)
        for j in range(n_ref):
            obs_sum[j, idx] = B_col[j]

        # weighting by W_r (reference weights)
        W = B_col * W_r

        # N_sim per reference: here a single scalar for this test index
        N_val = 0.0
        for j in range(n_ref):
            N_val += W[j]
        N_sim[idx] = N_val

        # final weights w_i = K**N_sim for all refs
        # (here same scalar K**N_val for all j)
        if N_val <= 0.0:
            Q[idx] = np.nan
            sig_Q[idx] = np.nan
            continue

        w_scalar = K ** N_val

        Sw_i = 0.0
        num = 0.0
        for j in range(n_ref):
            wi = w_scalar * B_col[j]
            Sw_i += wi
            num += wi * H_ref[j]

        if Sw_i <= 0.0:
            Q[idx] = np.nan
            sig_Q[idx] = np.nan
            continue

        q_i = num / Sw_i
        Q[idx] = q_i

        var = 0.0
        for j in range(n_ref):
            wi = w_scalar * B_col[j]
            diff = H_ref[j] - q_i
            var += wi * diff * diff
        sig_Q[idx] = np.sqrt(var / Sw_i)

    return Q, sig_Q, N_sim, obs_sum


def similarity_sweep(X_ref: np.ndarray,
                     H_ref: np.ndarray,
                     W_r: np.ndarray,
                     S_r: np.ndarray,
                     K: float,
                     use_sigma_helper: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Public wrapper for leave-one-out similarity on reference database.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
        Standardised reference features.
    H_ref : (n_ref,)
        Heat-flow values for reference database.
    W_r   : (n_ref,)
        Reference weights.
    S_r   : array-like
        Sigma information for reference database.
        If use_sigma_helper is True, this is passed to prepare_sigma().
    K     : float
        Exponential base for similarity weights.
    use_sigma_helper : bool, default True
        If True, call prepare_sigma(S_r) to obtain per-feature sigmas.

    Returns
    -------
    Q      : (n_ref,)
    sig_Q  : (n_ref,)
    N_sim  : (n_ref,)
    obs_sum: (n_ref, n_ref)
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)
    W_r = np.asarray(W_r, dtype=np.float32)
    S_r = np.asarray(S_r, dtype=np.float32)

    if use_sigma_helper:
        sigma_arr = prepare_sigma(S_r if S_r.ndim <= 2 else S_r.reshape(S_r.shape[0], -1))
    else:
        # assume S_r already holds per-feature sigmas
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("When use_sigma_helper=False, S_r must be 1-D (per-feature sigmas).")

    return _similarity_sweep_core(X_ref, H_ref, W_r, sigma_arr, float(K))


# ----------------------------------------------------------------------
# Prediction on arbitrary target grids
# ----------------------------------------------------------------------


@jit(nopython=True, fastmath=True, cache=True)
def _similarity_predict_core(X_ref: np.ndarray,
                             X_tgt_flat: np.ndarray,
                             H_ref: np.ndarray,
                             W_t_flat: np.ndarray,
                             sigma_arr: np.ndarray,
                             K: float,
                             hist: bool,
                             n_bins: int,
                             q_min: float,
                             q_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core JIT loop for similarity-based prediction on flattened targets.

    This is the batched, per-target analogue of your original similarity
    function, extended to many targets at once.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
    X_tgt_flat : (n_tgt, n_feat)
    H_ref : (n_ref,)
    W_t_flat : (n_tgt,)
        Per-target weights.
    sigma_arr : (n_feat,)
    K : float
    hist : bool
    n_bins : int
    q_min, q_max : float

    Returns
    -------
    q_flat   : (n_tgt,)
    sig_flat : (n_tgt,)
    N_flat   : (n_tgt,)
    hist_flat: (n_tgt, n_bins) or (0, 0) if hist=False
    """
    n_ref, n_feat = X_ref.shape
    n_tgt = X_tgt_flat.shape[0]

    q_flat = np.empty(n_tgt, dtype=np.float32)
    sig_flat = np.empty(n_tgt, dtype=np.float32)
    N_flat = np.empty(n_tgt, dtype=np.float32)

    if hist:
        hist_flat = np.zeros((n_tgt, n_bins), dtype=np.float32)
        dq = (q_max - q_min) / n_bins
    else:
        hist_flat = np.empty((0, 0), dtype=np.float32)
        dq = 0.0

    for j in range(n_tgt):
        # similarity to all references
        # B(j, i) = Gaussian sim over features
        W_col = np.empty(n_ref, dtype=np.float32)

        # compute B and apply target weight W_t
        wt = W_t_flat[j]
        N_sim_j = 0.0

        for i in range(n_ref):
            acc = 0.0
            for k in range(n_feat):
                d = (X_tgt_flat[j, k] - X_ref[i, k]) / sigma_arr[k]
                acc += d * d
            Bij = np.exp(-0.5 * acc)
            Wij = Bij * wt
            W_col[i] = Wij
            N_sim_j += Wij

        N_flat[j] = N_sim_j

        if N_sim_j <= 0.0:
            q_flat[j] = np.nan
            sig_flat[j] = np.nan
            continue

        # weights per ref: w_i = K**N_sim_j * B_ji
        w_scalar = K ** N_sim_j

        Sw = 0.0
        num = 0.0
        for i in range(n_ref):
            wi = w_scalar * (W_col[i] / wt if wt != 0.0 else 0.0)  # keep dependence on B_ji
            Sw += wi
            num += wi * H_ref[i]

        if Sw <= 0.0:
            q_flat[j] = np.nan
            sig_flat[j] = np.nan
            continue

        qj = num / Sw
        q_flat[j] = qj

        var = 0.0
        for i in range(n_ref):
            wi = w_scalar * (W_col[i] / wt if wt != 0.0 else 0.0)
            diff = H_ref[i] - qj
            var += wi * diff * diff
        sig_flat[j] = np.sqrt(var / Sw)

        if hist:
            # histogram over H_ref with weights wi
            for i in range(n_ref):
                wi = w_scalar * (W_col[i] / wt if wt != 0.0 else 0.0)
                v = H_ref[i]
                if v < q_min or v >= q_max:
                    continue
                b = int((v - q_min) / dq)
                if 0 <= b < n_bins:
                    hist_flat[j, b] += wi

            # normalise to density
            total = 0.0
            for b in range(n_bins):
                total += hist_flat[j, b]
            if total > 0.0:
                for b in range(n_bins):
                    hist_flat[j, b] /= (total * dq)

    return q_flat, sig_flat, N_flat, hist_flat


def similarity_predict(X_ref: np.ndarray,
                       H_ref: np.ndarray,
                       X_tgt: np.ndarray,
                       W_t: Optional[np.ndarray],
                       S_r: np.ndarray,
                       K: float,
                       hist: bool = False,
                       n_bins: int = 150,
                       q_min: float = 0.0,
                       q_max: float = 0.15,
                       use_sigma_helper: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Public wrapper for similarity-based prediction on arbitrary target grids.

    Handles arbitrary target shapes by flattening over the first axes and
    preserving the last dimension as features, then reshaping outputs
    back to the original target shape.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
        Standardised reference features.
    H_ref : (n_ref,)
        Reference heat-flow values.
    X_tgt : (..., n_feat)
        Standardised target features. The shape before the last dimension
        is arbitrary and will be preserved in the outputs.
    W_t   : (...,) or None
        Per-target weights. If None, uses ones.
    S_r   : array-like
        Sigma information for the reference database. Used to derive
        per-feature sigmas via prepare_sigma if use_sigma_helper=True.
    K     : float
        Exponential base for similarity weights.
    hist  : bool, default False
        If True, compute histograms for each target.
    n_bins : int, default 150
    q_min, q_max : float, default (0.0, 0.15)
        Histogram range in W/m².
    use_sigma_helper : bool, default True
        If True, call prepare_sigma(S_r).

    Returns
    -------
    q     : same shape as X_tgt[..., 0]
    std   : same shape as q
    N     : same shape as q
    hist  : (X_tgt_shape, n_bins) if hist=True, else (0, 0)
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)
    S_r = np.asarray(S_r, dtype=np.float32)

    # Prepare sigmas
    if use_sigma_helper:
        sigma_arr = prepare_sigma(S_r if S_r.ndim <= 2 else S_r.reshape(S_r.shape[0], -1))
    else:
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("When use_sigma_helper=False, S_r must be 1-D (per-feature sigmas).")

    # Flatten targets to (n_tgt, n_feat)
    X_tgt = np.asarray(X_tgt, dtype=np.float32)
    if X_tgt.ndim < 2:
        raise ValueError("X_tgt must have at least 2 dimensions (..., n_feat).")

    tgt_shape = X_tgt.shape[:-1]
    n_tgt = int(np.prod(tgt_shape))
    n_feat = X_tgt.shape[-1]

    X_tgt_flat = X_tgt.reshape(n_tgt, n_feat)

    # Per-target weights
    if W_t is None:
        W_t_flat = np.ones(n_tgt, dtype=np.float32)
    else:
        W_t = np.asarray(W_t, dtype=np.float32)
        if W_t.shape != tgt_shape:
            raise ValueError(f"W_t shape {W_t.shape} incompatible with X_tgt spatial shape {tgt_shape}.")
        W_t_flat = W_t.reshape(n_tgt)

    # Core JIT computation
    q_flat, sig_flat, N_flat, hist_flat = _similarity_predict_core(
        X_ref, X_tgt_flat, H_ref, W_t_flat, sigma_arr, float(K),
        hist, int(n_bins), float(q_min), float(q_max)
    )

    # Reshape back to original target shape
    q = q_flat.reshape(tgt_shape)
    std = sig_flat.reshape(tgt_shape)
    N = N_flat.reshape(tgt_shape)

    if hist:
        hist = hist_flat.reshape(*tgt_shape, n_bins)
    else:
        hist = hist_flat

    return q, std, N, hist