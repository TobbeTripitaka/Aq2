#!/usr/bin/env python3
# agrid.py v0.1.1b — Tobias Staal 2026
# tobias.staal@utas.edu.au
# MIT License

"""
lib/similarity.py

Core similarity functions for Aq2/Kq2.

Faithful port of similarity_test() to numba + vectorised numpy.

Original semantics (V_r is n_feat × n_ref):
    B      = sim(V_r, V_r[:,i], S_r, S_r[:,i])   # (n_feat, n_ref)
    B[:,i] = 0                                     # remove self
    W      = B * W_r                               # (n_feat, n_ref)
    N_sim  = nansum(W, axis=0)                     # (n_ref,)  ← sum over FEATURES
    w_i    = K ** N_sim                            # (n_ref,)  ← per-reference
    Q[i]   = sum(w_i * H_r) / sum(w_i)

The key invariant: N_sim is a PER-REFERENCE vector (one value per ref),
so w_i varies across references and K genuinely sharpens the neighbourhood.
"""

from typing import Tuple, Optional

import numpy as np
from numba import jit

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def prepare_sigma(S_r: np.ndarray) -> np.ndarray:
    S_r = np.asarray(S_r, dtype=np.float32)
    if S_r.ndim == 1:
        sigma_arr = S_r.copy()
    elif S_r.ndim == 2:
        sigma_arr = np.nanmedian(S_r, axis=0)
    else:
        raise ValueError(f"S_r must have 1 or 2 dimensions, got {S_r.ndim}")
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
    X    = np.asarray(X,    dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std,  dtype=np.float32)
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0
    return (X - mean[None, :]) / std_safe[None, :]


# ----------------------------------------------------------------------
# Gaussian similarity kernel  (vectorised, no numba needed at this level)
# ----------------------------------------------------------------------

def sim_gaussian_2d(X_ref: np.ndarray,
                    X_tgt: np.ndarray,
                    sigma_arr: np.ndarray) -> np.ndarray:
    """
    Gaussian similarity matrix.

    Parameters
    ----------
    X_ref     : (n_ref, n_feat)
    X_tgt     : (n_tgt, n_feat)
    sigma_arr : (n_feat,)

    Returns
    -------
    S : (n_tgt, n_ref)
        S[j, i] = exp(-0.5 * sum_k((tgt[j,k]-ref[i,k])^2 / sigma[k]^2))
    """
    # (n_tgt, 1, n_feat) - (1, n_ref, n_feat)  → (n_tgt, n_ref, n_feat)
    diff = (X_tgt[:, None, :] - X_ref[None, :, :]) / sigma_arr[None, None, :]
    return np.exp(-0.5 * np.sum(diff ** 2, axis=-1)).astype(np.float32)


# ----------------------------------------------------------------------
# LOO sweep — faithful port of similarity_test()
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _similarity_sweep_core(X_ref: np.ndarray,
                            H_ref: np.ndarray,
                            W_r: np.ndarray,
                            sigma_arr: np.ndarray,
                            K: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-out sweep. Faithful port of similarity_test().

    Original: V_r is (n_feat, n_ref); W_r is (n_feat, n_ref).
    Here:     X_ref is (n_ref, n_feat); W_r is (n_ref,) — per-ref scalar weights.

    N_sim[j] = sum_over_features( B[j, :] * W_r[:] )
             = per-reference similarity-weighted feature sum
             → w_i[j] = K ** N_sim[j]   varies per reference ← KEY

    Parameters
    ----------
    X_ref     : (n_ref, n_feat)  standardised
    H_ref     : (n_ref,)
    W_r       : (n_ref,)         per-reference scalar weights
    sigma_arr : (n_feat,)
    K         : float

    Returns
    -------
    Q, sig_Q  : (n_ref,)
    N_sim_out : (n_ref,)  last computed N_sim vector (diagnostics)
    obs_sum   : (n_ref, n_ref)  raw B matrix (diagnostics)
    """
    n_ref, n_feat = X_ref.shape

    Q         = np.zeros(n_ref, dtype=np.float32)
    sig_Q     = np.zeros(n_ref, dtype=np.float32)
    N_sim_out = np.zeros(n_ref, dtype=np.float32)
    obs_sum   = np.zeros((n_ref, n_ref), dtype=np.float32)

    for idx in range(n_ref):
        # B[j] = similarity of each reference j to test point idx
        # Original: sim(V_r, V_r[:,idx], S_r, S_r[:,idx]) → (n_feat, n_ref)
        # then sum over features gives N_sim per reference.
        # Here we compute per-feature per-reference contributions directly.

        # B_feat_ref[k, j] = exp(-0.5 * ((X_ref[j,k] - X_ref[idx,k]) / sigma[k])^2)
        # N_sim[j] = sum_k( B_feat_ref[k,j] * W_r[j] )   ← original axis=0 sum

        N_sim = np.zeros(n_ref, dtype=np.float32)
        B_sum = np.zeros(n_ref, dtype=np.float32)  # sum of B over features

        for j in range(n_ref):
            feat_sim_sum = 0.0
            for k in range(n_feat):
                d = (X_ref[j, k] - X_ref[idx, k]) / sigma_arr[k]
                feat_sim_sum += np.exp(-0.5 * d * d)  # B_feat_ref[k,j]
            B_sum[j] = feat_sim_sum
            obs_sum[j, idx] = feat_sim_sum

        # Remove self (LOO)
        B_sum[idx] = 0.0

        # N_sim[j] = B_sum[j] * W_r[j]  (scalar weight per reference)
        for j in range(n_ref):
            N_sim[j] = B_sum[j] * W_r[j]

        N_sim_out = N_sim  # save last for diagnostics

        # w_i[j] = K ** N_sim[j]  — per-reference, varies with K
        Sw  = 0.0
        num = 0.0
        for j in range(n_ref):
            wi   = K ** N_sim[j]
            Sw  += wi
            num += wi * H_ref[j]

        if Sw <= 0.0:
            Q[idx]     = np.nan
            sig_Q[idx] = np.nan
            continue

        q_i        = num / Sw
        Q[idx]     = q_i

        var = 0.0
        for j in range(n_ref):
            wi   = K ** N_sim[j]
            diff = H_ref[j] - q_i
            var += wi * diff * diff
        sig_Q[idx] = np.sqrt(var / Sw)

    return Q, sig_Q, N_sim_out, obs_sum


def similarity_sweep(X_ref: np.ndarray,
                     H_ref: np.ndarray,
                     W_r: np.ndarray,
                     S_r: np.ndarray,
                     K: float,
                     use_sigma_helper: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)
    W_r   = np.asarray(W_r,   dtype=np.float32)
    S_r   = np.asarray(S_r,   dtype=np.float32)

    if use_sigma_helper:
        sigma_arr = prepare_sigma(S_r if S_r.ndim <= 2 else S_r.reshape(S_r.shape[0], -1))
    else:
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("When use_sigma_helper=False, S_r must be 1-D.")

    return _similarity_sweep_core(X_ref, H_ref, W_r, sigma_arr, float(K))


# ----------------------------------------------------------------------
# Prediction — faithful port of similarity_test(), vectorised over targets
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _similarity_predict_core(X_ref: np.ndarray,
                              X_tgt_flat: np.ndarray,
                              H_ref: np.ndarray,
                              W_r: np.ndarray,
                              sigma_arr: np.ndarray,
                              K: float,
                              hist: bool,
                              n_bins: int,
                              q_min: float,
                              q_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Faithful port of similarity_test() for prediction on arbitrary targets.

    For each target point j:
        B[j, i]  = Gaussian sim between target j and reference i
                   (this is the per-feature version: sum of per-feature sims)
        N_sim[i] = B[j, i] * W_r[i]     ← per-REFERENCE, varies across i
        w_i[i]   = K ** N_sim[i]         ← per-reference, K is meaningful
        q[j]     = sum(w_i * H_ref) / sum(w_i)

    NOTE: W_r is per-reference weights (not per-target). Pass the reference
    sample weights here, same as the original W_r argument in similarity_test.
    The old W_t (per-target) argument is removed — it was not in the original.

    Parameters
    ----------
    X_ref      : (n_ref, n_feat)  standardised reference features
    X_tgt_flat : (n_tgt, n_feat)  standardised target features (flattened)
    H_ref      : (n_ref,)         reference heat-flow values
    W_r        : (n_ref,)         per-reference weights (same as similarity_test W_r)
    sigma_arr  : (n_feat,)        per-feature bandwidths
    K          : float            sharpening exponent
    hist       : bool
    n_bins     : int
    q_min, q_max : float

    Returns
    -------
    q_flat, sig_flat, N_flat, hist_flat
    """
    n_ref, n_feat = X_ref.shape
    n_tgt = X_tgt_flat.shape[0]

    q_flat   = np.empty(n_tgt, dtype=np.float32)
    sig_flat = np.empty(n_tgt, dtype=np.float32)
    N_flat   = np.empty(n_tgt, dtype=np.float32)  # stores mean N_sim per target

    if hist:
        hist_flat = np.zeros((n_tgt, n_bins), dtype=np.float32)
        dq = (q_max - q_min) / n_bins
    else:
        hist_flat = np.empty((0, 0), dtype=np.float32)
        dq = 0.0

    for j in range(n_tgt):

        # B_feat_sum[i] = sum_k( exp(-0.5*((tgt[j,k]-ref[i,k])/sigma[k])^2) )
        # This is the per-feature similarity summed over features,
        # matching the original nansum(B, axis=0) after sim(V_r, V_r[:,i]...)
        B_feat_sum = np.empty(n_ref, dtype=np.float32)
        for i in range(n_ref):
            acc = 0.0
            for k in range(n_feat):
                d = (X_tgt_flat[j, k] - X_ref[i, k]) / sigma_arr[k]
                acc += np.exp(-0.5 * d * d)   # sum of per-feature similarities
            B_feat_sum[i] = acc

        # N_sim[i] = B_feat_sum[i] * W_r[i]  — per-reference
        # w_i[i]   = K ** N_sim[i]            — per-reference, K is meaningful
        Sw      = 0.0
        num     = 0.0
        N_total = 0.0

        for i in range(n_ref):
            N_sim_i = B_feat_sum[i] * W_r[i]
            wi       = K ** N_sim_i
            Sw      += wi
            num     += wi * H_ref[i]
            N_total += N_sim_i

        N_flat[j] = N_total / n_ref  # mean N_sim for diagnostics

        if Sw <= 0.0:
            q_flat[j]   = np.nan
            sig_flat[j] = np.nan
            continue

        qj          = num / Sw
        q_flat[j]   = qj

        var = 0.0
        for i in range(n_ref):
            N_sim_i = B_feat_sum[i] * W_r[i]
            wi       = K ** N_sim_i
            diff     = H_ref[i] - qj
            var     += wi * diff * diff
        sig_flat[j] = np.sqrt(var / Sw)

        if hist:
            for i in range(n_ref):
                N_sim_i = B_feat_sum[i] * W_r[i]
                wi = K ** N_sim_i
                v  = H_ref[i]
                if v < q_min or v >= q_max:
                    continue
                b = int((v - q_min) / dq)
                if 0 <= b < n_bins:
                    hist_flat[j, b] += wi
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
                       W_r: np.ndarray,
                       S_r: np.ndarray,
                       K: float,
                       hist: bool = False,
                       n_bins: int = 150,
                       q_min: float = 0.0,
                       q_max: float = 0.15,
                       use_sigma_helper: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Similarity-based prediction on arbitrary target grids.

    Faithful port of similarity_test(V_r, S_r, W_r, H_r, K).

    Parameters
    ----------
    X_ref : (n_ref, n_feat)   standardised reference features
    H_ref : (n_ref,)          reference heat-flow values
    X_tgt : (..., n_feat)     standardised target features
    W_r   : (n_ref,)          per-REFERENCE weights  ← same as original W_r
    S_r   : array-like        sigma info (see use_sigma_helper)
    K     : float             sharpening exponent
    hist  : bool
    n_bins, q_min, q_max : histogram params
    use_sigma_helper : bool

    Returns
    -------
    q, std, N, hist_out  — q/std/N shaped as X_tgt[..., 0]
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)
    W_r   = np.asarray(W_r,   dtype=np.float32)
    S_r   = np.asarray(S_r,   dtype=np.float32)

    if use_sigma_helper:
        sigma_arr = prepare_sigma(S_r if S_r.ndim <= 2 else S_r.reshape(S_r.shape[0], -1))
    else:
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("When use_sigma_helper=False, S_r must be 1-D.")

    X_tgt = np.asarray(X_tgt, dtype=np.float32)
    if X_tgt.ndim < 2:
        raise ValueError("X_tgt must have at least 2 dimensions (..., n_feat).")

    tgt_shape  = X_tgt.shape[:-1]
    n_tgt      = int(np.prod(tgt_shape))
    n_feat     = X_tgt.shape[-1]
    X_tgt_flat = X_tgt.reshape(n_tgt, n_feat)

    if W_r.shape != (X_ref.shape[0],):
        raise ValueError(f"W_r shape {W_r.shape} must match (n_ref,) = ({X_ref.shape[0]},).")

    q_flat, sig_flat, N_flat, hist_flat = _similarity_predict_core(
        X_ref, X_tgt_flat, H_ref, W_r, sigma_arr, float(K),
        hist, int(n_bins), float(q_min), float(q_max)
    )

    q        = q_flat.reshape(tgt_shape)
    std      = sig_flat.reshape(tgt_shape)
    N        = N_flat.reshape(tgt_shape)
    hist_out = hist_flat.reshape(*tgt_shape, n_bins) if hist else hist_flat

    return q, std, N, hist_out
