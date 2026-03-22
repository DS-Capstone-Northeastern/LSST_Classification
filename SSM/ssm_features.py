"""
SSM feature extraction -> fixed-length feature vectors per object.

We compute:
  - basic per-passband features from RTS-smoothed trajectories
  - global features from concatenated trajectories

This is a classical feature extractor (Kalman-based), not a neural SSM.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ssm.ssm_kalman import kalman_filter_rts


PASSBANDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Basic per passband (7 features) + has_obs (1 feature) = 8 features/passband
BASIC_NAMES = (
    "has_obs",
    "peak_flux",
    "time_to_peak",
    "total_duration",
    "rise_rate",
    "decline_rate",
    "asymmetry",
    "amplitude",
)


def _safe_div(n: float, d: float, eps: float = 1e-12) -> float:
    d = float(d)
    if abs(d) <= eps:
        return 0.0
    return float(n / d)


def _central_moments_stats(x: np.ndarray) -> Tuple[float, float]:
    if x.size < 2:
        return 0.0, 0.0
    mu = float(np.mean(x))
    c = x - mu
    v = float(np.mean(c**2))
    if v <= 1e-12:
        return 0.0, 0.0
    skew = float(np.mean(c**3) / (v ** 1.5))
    kurt_excess = float(np.mean(c**4) / (v**2) - 3.0)
    return skew, kurt_excess


def _basic_passband_features(
    *,
    times_mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    dt_scale: float,
    q: float,
    r_min: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      - basic feature vector (8,)
      - passband state dict with times, f_smooth, v_actual, obs_flux, obs_err
    """
    if times_mjd.size == 0:
        feat = np.zeros((len(BASIC_NAMES),), dtype=np.float64)
        return feat, {
            "times": np.zeros((0,), dtype=np.float64),
            "f_smooth": np.zeros((0,), dtype=np.float64),
            "v_actual": np.zeros((0,), dtype=np.float64),
            "obs_flux": np.zeros((0,), dtype=np.float64),
            "obs_err": np.zeros((0,), dtype=np.float64),
        }

    # Kalman/RTS
    res = kalman_filter_rts(
        times=times_mjd,
        observations=flux,
        flux_errs=flux_err,
        dt_scale=dt_scale,
        q=q,
        r_min=r_min,
    )
    f = res.x_smooth_mean[:, 0]
    v_scaled = res.x_smooth_mean[:, 1]
    v_actual = v_scaled / float(dt_scale)

    times = res.times
    idx_peak = int(np.argmax(f))
    t_first = float(times[0])
    t_last = float(times[-1])
    t_peak = float(times[idx_peak])
    peak_flux = float(f[idx_peak])
    baseline_flux = float(f[0])
    final_flux = float(f[-1])

    has_obs = 1.0
    time_to_peak = max(t_peak - t_first, 0.0)
    total_duration = max(t_last - t_first, 0.0)
    rise_rate = _safe_div(peak_flux - baseline_flux, time_to_peak)
    decline_duration = max(t_last - t_peak, 0.0)
    decline_rate = _safe_div(peak_flux - final_flux, decline_duration)
    asymmetry = _safe_div(time_to_peak, decline_duration)
    amplitude = peak_flux - baseline_flux

    feat = np.array(
        [
            has_obs,
            peak_flux,
            time_to_peak,
            total_duration,
            rise_rate,
            decline_rate,
            asymmetry,
            amplitude,
        ],
        dtype=np.float64,
    )

    return feat, {
        "times": times,
        "f_smooth": f,
        "v_actual": v_actual,
        "obs_flux": flux[np.argsort(times_mjd)],
        "obs_err": flux_err[np.argsort(times_mjd)],
    }


def _global_features_from_states(states: Dict[int, Dict[str, np.ndarray]], *, dt_scale: float) -> np.ndarray:
    """
    Compute a compact global feature set from concatenated passband trajectories.
    """
    # Concatenate all passband points
    times_all: List[np.ndarray] = []
    f_all: List[np.ndarray] = []
    v_all: List[np.ndarray] = []
    obs_all: List[np.ndarray] = []
    err_all: List[np.ndarray] = []

    for pb in PASSBANDS:
        st = states.get(pb)
        if st is None or st["times"].size == 0:
            continue
        times_all.append(st["times"])
        f_all.append(st["f_smooth"])
        v_all.append(st["v_actual"])
        obs_all.append(st["obs_flux"])
        err_all.append(st["obs_err"])

    if not times_all:
        return np.zeros((25,), dtype=np.float64)

    t = np.concatenate(times_all)
    f = np.concatenate(f_all)
    v = np.concatenate(v_all)
    obs = np.concatenate(obs_all)
    err = np.concatenate(err_all)

    order = np.argsort(t)
    t = t[order]
    f = f[order]
    v = v[order]
    obs = obs[order]
    err = err[order]

    # Peak-based velocity stats
    idx_peak = int(np.argmax(f))
    peak_flux = float(f[idx_peak])
    max_velocity = float(np.max(v))
    min_velocity = float(np.min(v))
    velocity_at_peak = float(v[idx_peak])
    velocity_range = float(max_velocity - min_velocity)

    # Acceleration via finite differences on irregular grid
    if t.size >= 3:
        dt = np.diff(t)
        dv = np.diff(v)
        valid = dt > 1e-12
        a = (dv[valid] / dt[valid]) if np.any(valid) else np.zeros((0,), dtype=np.float64)
    else:
        a = np.zeros((0,), dtype=np.float64)

    mean_acceleration = float(np.mean(a)) if a.size else 0.0
    max_acceleration = float(np.max(np.abs(a))) if a.size else 0.0
    acceleration_at_peak = float(a[0]) if a.size else 0.0
    jerk = float(np.mean(np.diff(a))) if a.size > 1 else 0.0

    # Statistical
    flux_variance = float(np.var(f))
    flux_skewness, flux_kurtosis = _central_moments_stats(f)
    smoothness = float(np.mean((obs - f) ** 2))
    mean_err = float(np.mean(np.maximum(err, 1e-8)))
    signal_to_noise = float(peak_flux / mean_err) if mean_err > 0 else 0.0
    chi_squared = float(np.sum(((obs - f) / np.maximum(err, 1e-8)) ** 2))

    # Model-fit proxies on tail (very lightweight)
    t_peak = float(t[idx_peak])
    tail = t >= t_peak
    if np.sum(tail) >= 3:
        tt = t[tail] - t_peak
        yy = f[tail]
        yy_shift = yy - np.min(yy) + 1e-6
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, _ = np.linalg.lstsq(A, np.log(yy_shift), rcond=None)[0]
        exponential_decay_rate = float(-slope)
    else:
        exponential_decay_rate = 0.0

    half_peak = 0.5 * float(np.max(f))
    stretch_factor = _safe_div(float(np.sum(f >= half_peak)), float(len(f)))

    model_residual_variance = float(np.var(obs - f))

    # Color proxies using g-r (passbands 1 and 2)
    g = states.get(1)
    r = states.get(2)
    if g is not None and r is not None and g["times"].size > 1 and r["times"].size > 1:
        grid = np.union1d(g["times"], r["times"])
        g_interp = np.interp(grid, g["times"], g["f_smooth"])
        r_interp = np.interp(grid, r["times"], r["f_smooth"])
        color = g_interp - r_interp
        idx_color_peak = int(np.argmax(f))
        color_at_peak = float(np.interp(t_peak, grid, color))
        dc = np.diff(color)
        dtc = np.diff(grid)
        valid = dtc > 1e-12
        color_evolution_rate = float(np.mean((dc[valid] / dtc[valid]))) if np.any(valid) else 0.0
    else:
        color_at_peak = 0.0
        color_evolution_rate = 0.0

    # Multi-band correlation (g vs r)
    if g is not None and r is not None and g["times"].size > 1 and r["times"].size > 1:
        grid = np.union1d(g["times"], r["times"])
        g_interp = np.interp(grid, g["times"], g["f_smooth"])
        r_interp = np.interp(grid, r["times"], r["f_smooth"])
        if np.std(g_interp) > 1e-12 and np.std(r_interp) > 1e-12:
            multi_band_correlation = float(np.corrcoef(g_interp, r_interp)[0, 1])
        else:
            multi_band_correlation = 0.0
    else:
        multi_band_correlation = 0.0

    # Remaining physical-ish proxies (heuristic)
    ejecta_mass_estimate = float(np.mean(np.abs(v))) if v.size else 0.0
    explosion_energy_estimate = float(np.max(np.abs(v)) * max(peak_flux, 0.0)) if v.size else 0.0
    nickel_mass_estimate = float(max(peak_flux, 0.0))
    opacity_estimate = _safe_div(1.0, 1.0 + max(signal_to_noise, 0.0))
    peak_luminosity = float(peak_flux)
    effective_temperature = float(np.tanh(color_at_peak) + 1.0)

    blue_to_red_ratio = 0.0  # placeholder for speed; extend later if needed
    rise_time_difference = 0.0
    peak_time_lag = 0.0
    g_minus_r_color = float(color_at_peak)
    color_change_rate = float(color_evolution_rate)
    band_asymmetry = 0.0
    spectral_evolution = float(color_evolution_rate)
    reddening = 0.0

    feats = np.array(
        [
            max_velocity,
            min_velocity,
            velocity_at_peak,
            velocity_range,
            mean_acceleration,
            max_acceleration,
            acceleration_at_peak,
            jerk,
            flux_variance,
            flux_skewness,
            flux_kurtosis,
            smoothness,
            signal_to_noise,
            chi_squared,
            exponential_decay_rate,
            stretch_factor,
            color_at_peak,
            color_evolution_rate,
            model_residual_variance,
            ejecta_mass_estimate,
            explosion_energy_estimate,
            nickel_mass_estimate,
            opacity_estimate,
            peak_luminosity,
            effective_temperature,
        ],
        dtype=np.float64,
    )

    # Sanity: keep fixed length 25
    if feats.shape[0] != 25:
        # Should never happen; guard to avoid silent misalignment.
        out = np.zeros((25,), dtype=np.float64)
        out[: min(25, feats.shape[0])] = feats[: min(25, feats.shape[0])]
        return out

    return feats


def extract_object_features(
    df_obj: pd.DataFrame,
    *,
    dt_scale: float,
    q: float,
    r_min: float,
    passbands: Sequence[int] = PASSBANDS,
) -> np.ndarray:
    """
    Extract fixed feature vector for one object.
    """
    passband_states: Dict[int, Dict[str, np.ndarray]] = {}
    basic_feats: List[np.ndarray] = []

    for pb in passbands:
        df_pb = df_obj[df_obj["passband"] == pb]
        times = df_pb["mjd"].to_numpy(dtype=np.float64)
        flux = df_pb["flux"].to_numpy(dtype=np.float64)
        flux_err = df_pb["flux_err"].to_numpy(dtype=np.float64)
        basic, st = _basic_passband_features(
            times_mjd=times,
            flux=flux,
            flux_err=flux_err,
            dt_scale=dt_scale,
            q=q,
            r_min=r_min,
        )
        passband_states[int(pb)] = st
        basic_feats.append(basic)

    basic_vec = np.concatenate(basic_feats, axis=0)  # 6 * 8 = 48
    global_vec = _global_features_from_states(passband_states, dt_scale=dt_scale)  # 25
    feats = np.concatenate([basic_vec, global_vec], axis=0)
    return feats.astype(np.float64, copy=False)


def extract_ssm_features_matrix(
    df_lc: pd.DataFrame,
    *,
    object_ids: Sequence[int],
    dt_scale: float,
    q: float,
    r_min: float,
    object_id_col: str = "object_id",
    passbands: Sequence[int] = PASSBANDS,
) -> np.ndarray:
    """
    Extract features for all object_ids in object_ids. df_lc must contain those IDs.
    """
    if df_lc.empty:
        return np.zeros((0, 73), dtype=np.float64)

    out: List[np.ndarray] = []
    grouped = df_lc.groupby(object_id_col, sort=False)
    id_to_df = {oid: g for oid, g in grouped}

    for oid in object_ids:
        df_obj = id_to_df.get(int(oid))
        if df_obj is None:
            out.append(np.zeros((73,), dtype=np.float64))
            continue
        out.append(
            extract_object_features(
                df_obj,
                dt_scale=dt_scale,
                q=q,
                r_min=r_min,
                passbands=passbands,
            )
        )

    return np.stack(out, axis=0).astype(np.float64, copy=False)

