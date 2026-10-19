"""Dead reckoning: integrate the IMU alone, with no GPS and no bias estimation.

This is the "what happens without GNSS" baseline. The 2D navigation state is

    x = [px, py, vx, vy, yaw]      (ENU position [m], ENU velocity [m/s], heading [rad])

and the inputs are the level-frame IMU channels u = [af, al, wu] (forward / left acceleration and
yaw rate). Over one step dt = t[k+1] - t[k], with the inputs held constant (zero-order hold):

    yaw_{k+1} = yaw_k + wu dt
    a_enu     = R(yaw_k + wu dt / 2) [af, al]^T            R(.) = 2D rotation body -> ENU
    v_{k+1}   = v_k + a_enu dt
    p_{k+1}   = p_k + v_k dt + a_enu dt^2 / 2

Rotating by the **mid-step heading** rather than yaw_k matters. While turning, the body-frame
acceleration rotates with the car during the step. Using the start-of-step heading lags it by
wu dt / 2, which tilts the centripetal acceleration backwards and shortens every curve. For constant
wu, the mid-step rotation is exact up to a factor sin(wu dt / 2) / (wu dt / 2) ~ 1 - (wu dt)^2 / 24.
On a KITTI drive with clean inputs this cuts the integration error alone by ~7x (14 m instead of
94 m over 290 s); the remaining errors are then due to the sensor, not the integrator.

Why it diverges with a MEMS IMU: an accelerometer bias b integrates to a position error of
b t^2 / 2 (0.1 m/s^2 -> 5 m after 10 s, 180 m after 60 s), and a gyro bias b_g turns into a heading
error b_g t that rotates the whole velocity vector, adding a cross-track error ~ v b_g t^2 / 2.

The same kinematics are the prediction step of the EKF, which in addition subtracts its bias
estimates from u and propagates the covariance.
"""

from __future__ import annotations

import numpy as np

from navfusion.data.coordinate_transforms import wrap_angle

NAV_STATE = ("px", "py", "vx", "vy", "yaw")


def propagate(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One integration step of the 5-state navigation model (see module docstring).

    x: (5,) [px, py, vx, vy, yaw];  u: (3,) [af, al, wu];  returns the (5,) state at t + dt.
    """
    af, al, wu = u
    yaw_mid = x[4] + 0.5 * wu * dt
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    a = np.array([c * af - s * al, s * af + c * al])
    out = np.empty(5)
    out[0:2] = x[0:2] + x[2:4] * dt + 0.5 * a * dt**2
    out[2:4] = x[2:4] + a * dt
    out[4] = wrap_angle(x[4] + wu * dt)
    return out


def dead_reckon(t: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Integrate from state ``x0`` at ``t[0]`` through all of ``t``. Returns (N, 5) states.

    Row k is the state at t[k], so row 0 equals x0. The last input u[-1] is never used (it would
    propagate beyond t[-1]). To restart dead reckoning at sample s from some state (e.g. the
    filter's estimate when a GPS outage begins), pass the slices ``t[s:e], u[s:e]``.

    The recursion in ``propagate`` is linear once the heading sequence is known, so it is
    evaluated here with cumulative sums instead of a Python loop. The result is identical to
    calling ``propagate`` step by step, and fast enough to run thousands of times while
    calibrating training augmentation.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    dt = np.diff(t)
    af, al, wu = u[:-1, 0], u[:-1, 1], u[:-1, 2]

    yaw = x0[4] + np.concatenate([[0.0], np.cumsum(wu * dt)])  # unwrapped heading
    yaw_mid = yaw[:-1] + 0.5 * wu * dt
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    a = np.column_stack([c * af - s * al, s * af + c * al])  # (N-1, 2) ENU acceleration

    dv = a * dt[:, None]
    v = x0[2:4] + np.vstack([np.zeros(2), np.cumsum(dv, axis=0)])
    dp = v[:-1] * dt[:, None] + 0.5 * a * dt[:, None] ** 2
    p = x0[0:2] + np.vstack([np.zeros(2), np.cumsum(dp, axis=0)])
    return np.column_stack([p, v, wrap_angle(yaw)])
