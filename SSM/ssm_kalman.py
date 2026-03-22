"""
Irregularly-sampled classical linear state-space model:
Kalman filter + RTS smoother for a 2D constant-velocity model.

State (per passband):
  x_t = [f_t, v_t]
where:
  f_t is latent flux,
  v_t is d(f)/d(t_scaled)

Transition with scaled time increment dt_s:
  F(dt_s) = [[1, dt_s],
             [0, 1]]
  Q(dt_s) = q * [[dt_s^3/3, dt_s^2/2],
                  [dt_s^2/2, dt_s]]

Observation:
  y_t = H x_t + eps, H = [1, 0]
  R_t = flux_err_t^2 + r_min
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KalmanRTSResult:
    times: np.ndarray  # (T,)
    dt_scaled: np.ndarray  # (T-1,)
    x_filt_mean: np.ndarray  # (T, 2)
    P_filt: np.ndarray  # (T, 2, 2)
    x_smooth_mean: np.ndarray  # (T, 2)
    P_smooth: np.ndarray  # (T, 2, 2)


def _transition_matrices(dt_scaled: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = float(dt_scaled)
    if dt < 0:
        raise ValueError("dt_scaled must be non-negative.")
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    q11 = (dt**3) / 3.0
    q12 = (dt**2) / 2.0
    q22 = dt
    Q = q * np.array([[q11, q12], [q12, q22]], dtype=np.float64)
    return F, Q


def kalman_filter_rts(
    *,
    times: np.ndarray,
    observations: np.ndarray,
    flux_errs: np.ndarray,
    dt_scale: float,
    q: float,
    r_min: float,
    x0_mean: Optional[np.ndarray] = None,
    x0_cov: Optional[np.ndarray] = None,
) -> KalmanRTSResult:
    times = np.asarray(times, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    flux_errs = np.asarray(flux_errs, dtype=np.float64)

    if times.ndim != 1:
        raise ValueError("times must be 1D.")
    if observations.shape != times.shape or flux_errs.shape != times.shape:
        raise ValueError("observations and flux_errs must have the same shape as times.")
    if len(times) < 1:
        raise ValueError("Need at least one observation.")
    if dt_scale <= 0:
        raise ValueError("dt_scale must be > 0.")

    order = np.argsort(times)
    times = times[order]
    observations = observations[order]
    flux_errs = flux_errs[order]

    T = len(times)
    if T == 1:
        y0 = float(observations[0])
        r0 = float(flux_errs[0] ** 2 + r_min)
        if x0_mean is None:
            x0_mean = np.array([y0, 0.0], dtype=np.float64)
        if x0_cov is None:
            x0_cov = np.diag([r0, r0]).astype(np.float64)

        x0_mean = np.asarray(x0_mean, dtype=np.float64).reshape(2)
        x0_cov = np.asarray(x0_cov, dtype=np.float64).reshape(2, 2)

        x_filt_mean = x0_mean[None, :]
        P_filt = x0_cov[None, :, :]
        x_smooth_mean = x_filt_mean.copy()
        P_smooth = P_filt.copy()

        return KalmanRTSResult(
            times=times,
            dt_scaled=np.zeros((0,), dtype=np.float64),
            x_filt_mean=x_filt_mean,
            P_filt=P_filt,
            x_smooth_mean=x_smooth_mean,
            P_smooth=P_smooth,
        )

    dt_actual = times[1:] - times[:-1]
    dt_scaled = dt_actual / float(dt_scale)
    dt_scaled = np.maximum(dt_scaled, 0.0)

    H = np.array([1.0, 0.0], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)

    x_filt_mean = np.zeros((T, 2), dtype=np.float64)
    P_filt = np.zeros((T, 2, 2), dtype=np.float64)
    x_pred_mean = np.zeros((T, 2), dtype=np.float64)
    P_pred = np.zeros((T, 2, 2), dtype=np.float64)

    y0 = float(observations[0])
    r0 = float(flux_errs[0] ** 2 + r_min)
    if x0_mean is None:
        x0_mean = np.array([y0, 0.0], dtype=np.float64)
    if x0_cov is None:
        x0_cov = np.diag([r0, r0]).astype(np.float64)

    x = np.asarray(x0_mean, dtype=np.float64).reshape(2)
    P = np.asarray(x0_cov, dtype=np.float64).reshape(2, 2)

    for t in range(T):
        if t > 0:
            F, Q = _transition_matrices(dt_scaled[t - 1], q=q)
            x = F @ x
            P = F @ P @ F.T + Q

        R_t = float(flux_errs[t] ** 2 + r_min)
        y_pred = float(H @ x)
        resid = float(observations[t] - y_pred)

        S = float(H @ P @ H.T + R_t)  # scalar
        if not np.isfinite(S) or S <= 0:
            raise FloatingPointError(f"Invalid innovation covariance S={S} at t={t}.")

        K = (P @ H.reshape(2, 1)).reshape(2) / S  # (2,)
        x = x + K * resid
        P = (I - np.outer(K, H)) @ P

        x_filt_mean[t] = x
        P_filt[t] = P

        if t < T - 1:
            F, Q = _transition_matrices(dt_scaled[t], q=q)
            x_pred_mean[t + 1] = F @ x
            P_pred[t + 1] = F @ P @ F.T + Q

    x_smooth_mean = x_filt_mean.copy()
    P_smooth = P_filt.copy()

    for t in range(T - 2, -1, -1):
        P_pred_next = P_pred[t + 1]
        F_t, _ = _transition_matrices(dt_scaled[t], q=q)

        C = P_filt[t] @ F_t.T
        C = np.linalg.solve(P_pred_next, C.T).T

        dx = x_smooth_mean[t + 1] - x_pred_mean[t + 1]
        x_smooth_mean[t] = x_filt_mean[t] + C @ dx
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return KalmanRTSResult(
        times=times,
        dt_scaled=dt_scaled,
        x_filt_mean=x_filt_mean,
        P_filt=P_filt,
        x_smooth_mean=x_smooth_mean,
        P_smooth=P_smooth,
    )

