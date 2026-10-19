"""8-state GPS/IMU Extended Kalman Filter with a non-holonomic constraint, in plain NumPy.

State (ENU frame, origin fixed per drive; yaw 0 = east, counter-clockwise positive):

    x = [px, py, vx, vy, yaw, b_ax, b_ay, b_gz]

Inputs are the level-frame IMU channels u = [af, al, wu] (forward / left acceleration, yaw rate).
The IMU biases are part of the state: a MEMS IMU's bias is the dominant error source (see
``dead_reckoning.py``), and while GPS is available it is observable, because the GPS track tells the
filter how far the integrated IMU has wandered off. Biases are modelled as slow random walks.

**Prediction** is the same mid-step-heading integration as dead reckoning, but with the estimated
biases removed from the inputs:

    w     = wu - b_gz
    a_b   = [af - b_ax, al - b_ay]
    a     = R(yaw + w dt / 2) a_b                  (body -> ENU)
    p'    = p + v dt + a dt^2 / 2,   v' = v + a dt,   yaw' = yaw + w dt,   b' = b

The covariance is propagated as P' = F P F^T + Q, with F = df/dx evaluated at the prior. The only
nonlinearity is the rotation R(.), whose derivative w.r.t. its angle is d/dpsi R(psi) a_b =
[-a_y, a_x]. Because the mid-step angle depends on yaw and on b_gz (through w), both pick up that
term. Q maps the IMU white noise through the input Jacobian G = df/du (Q_u = G diag(sa^2, sa^2,
sg^2) G^T) and adds the bias random walks.

**Updates** (all with the Joseph form, then P is symmetrized):

* GPS: z = [px, py] + noise, H = [I_2, 0], R = sigma_gps^2 I_2. 1 Hz, when a fix is available.
* Body-frame velocity: h(x) = R(yaw)^T [vx, vy] = [v_fwd, v_lat]. Its Jacobian is
      d v_fwd / d(vx, vy, yaw) = [ cos, sin,  v_lat]
      d v_lat / d(vx, vy, yaw) = [-sin, cos, -v_fwd]
  Any subset of the two components can be used. This is the update that a learned velocity
  measurement enters through during GPS outages.
* Non-holonomic constraint (NHC): a car's wheels do not slide sideways, so v_lat ~ 0. It is the
  body-velocity update with only the lateral component, z = 0 and R = (0.2 m/s)^2, applied at every
  IMU step. It couples heading to the velocity direction and makes b_ay observable, and during a
  GPS outage it stops the IMU from building up a sideways velocity.

**Why Joseph form.** The textbook update P+ = (I - K H) P is only correct for the optimal gain and
loses symmetry and positive-definiteness under rounding, especially with the tight NHC noise
applied 10 times a second. (I - K H) P (I - K H)^T + K R K^T is a sum of two PSD terms for any K.

**Initialisation** is from the noisy GPS only, never from the ground truth: position from a fix,
heading and velocity from the GPS displacement over the preceding ~2 s, biases 0 with a variance
matching the MEMS error model. The filter starts at the first fix whose 2 s baseline is long enough
for that heading to be meaningful (see ``initialise_from_gps``); earlier samples have no estimate
(NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navfusion.data.coordinate_transforms import wrap_angle
from navfusion.data.simulators import GpsMeasurements

STATE = ("px", "py", "vx", "vy", "yaw", "b_ax", "b_ay", "b_gz")
N_STATE = len(STATE)
YAW = 4


@dataclass(frozen=True)
class EkfConfig:
    """Noise model and initialisation.

    The process noise was tuned by hand on the validation drive only (5 noise seeds, full GPS),
    aiming for low position error *and* a consistent filter: the position NEES (squared error
    weighted by the filter's own covariance) averages ~1.8 against the ideal 2, and the GPS
    innovation NIS ~2.2 against 2.

    The input noise is far larger than the simulator's white noise (0.05 m/s^2, 0.01 rad/s per
    sample) because Q has to cover model error as well as sensor noise: the 2D model ignores road
    grade and body roll, and holds the inputs constant over each 0.1 s step. With the bare sensor
    noise the filter was overconfident (NEES ~22): its yaw variance collapsed to ~1 deg while the
    actual heading error was ~25 deg, and it then ignored the GPS evidence for over a minute.
    The bias random walks barely mattered in the range tried (10x either way).
    """

    # Process noise: IMU white noise per sample, bias random walks [unit / sqrt(s)]
    accel_noise_std: float = 0.5  # [m/s^2]
    gyro_noise_std: float = 0.02  # [rad/s]
    accel_bias_rw: float = 1e-3  # [m/s^2 / sqrt(s)]
    gyro_bias_rw: float = 1e-5  # [rad/s / sqrt(s)]

    # Measurement noise
    gps_sigma_m: float = 2.5  # matches the simulated GPS
    nhc_sigma: float = 0.2  # [m/s] allowed lateral velocity (tyre slip, IMU lever arm)

    # Initialisation
    init_window_s: float = 2.0  # GPS baseline used for the initial heading and velocity
    init_min_baseline_m: float = 10.0  # don't initialise before the car has moved this far
    init_accel_bias_std: float = 0.1  # [m/s^2], the MEMS bias range
    init_gyro_bias_std: float = float(np.deg2rad(0.5))  # [rad/s]

    use_nhc: bool = True


def _rot(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def motion_model(
    x: np.ndarray, u: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete motion model f(x, u) and its Jacobians F = df/dx (8x8) and G = df/du (8x3)."""
    w = u[2] - x[7]
    a_b = np.array([u[0] - x[5], u[1] - x[6]])
    R = _rot(x[YAW] + 0.5 * w * dt)
    a = R @ a_b
    da_dpsi = np.array([-a[1], a[0]])  # d/dpsi of R(psi) a_b
    h2 = 0.5 * dt**2

    x_new = x.copy()
    x_new[0:2] = x[0:2] + x[2:4] * dt + h2 * a
    x_new[2:4] = x[2:4] + dt * a
    x_new[YAW] = wrap_angle(x[YAW] + w * dt)

    F = np.eye(N_STATE)
    F[0:2, 2:4] = dt * np.eye(2)
    F[0:2, YAW] = h2 * da_dpsi
    F[2:4, YAW] = dt * da_dpsi
    F[0:2, 5:7] = -h2 * R
    F[2:4, 5:7] = -dt * R
    F[0:2, 7] = h2 * da_dpsi * (-0.5 * dt)  # b_gz moves the mid-step angle by -dt/2
    F[2:4, 7] = dt * da_dpsi * (-0.5 * dt)
    F[YAW, 7] = -dt

    G = np.zeros((N_STATE, 3))
    G[0:2, 0:2] = h2 * R
    G[2:4, 0:2] = dt * R
    G[0:2, 2] = h2 * da_dpsi * (0.5 * dt)  # wu moves the mid-step angle by +dt/2
    G[2:4, 2] = dt * da_dpsi * (0.5 * dt)
    G[YAW, 2] = dt
    return x_new, F, G


def body_velocity_model(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h(x) = [v_fwd, v_lat] = R(yaw)^T [vx, vy] and its Jacobian H (2x8)."""
    c, s = np.cos(x[YAW]), np.sin(x[YAW])
    v_fwd = c * x[2] + s * x[3]
    v_lat = -s * x[2] + c * x[3]
    H = np.zeros((2, N_STATE))
    H[0, 2:5] = [c, s, v_lat]
    H[1, 2:5] = [-s, c, -v_fwd]
    return np.array([v_fwd, v_lat]), H


class Ekf:
    """Stateful filter: call ``predict`` once per IMU step and the ``update_*`` methods as
    measurements arrive. ``x`` is (8,), ``P`` is (8, 8)."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, config: EkfConfig | None = None):
        self.config = config or EkfConfig()
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        if self.x.shape != (N_STATE,) or self.P.shape != (N_STATE, N_STATE):
            raise ValueError(f"expected x (8,) and P (8, 8), got {self.x.shape}, {self.P.shape}")
        self.x[YAW] = wrap_angle(self.x[YAW])

    # ----------------------------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------------------------

    def predict(self, u: np.ndarray, dt: float) -> None:
        """Propagate state and covariance over ``dt`` with IMU input u = [af, al, wu]."""
        cfg = self.config
        self.x, F, G = motion_model(self.x, u, dt)
        # IMU white noise enters through G; the biases drift as random walks.
        q_u = np.diag([cfg.accel_noise_std**2, cfg.accel_noise_std**2, cfg.gyro_noise_std**2])
        Q = G @ q_u @ G.T
        Q[5, 5] += cfg.accel_bias_rw**2 * dt
        Q[6, 6] += cfg.accel_bias_rw**2 * dt
        Q[7, 7] += cfg.gyro_bias_rw**2 * dt
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # ----------------------------------------------------------------------------------------
    # Updates
    # ----------------------------------------------------------------------------------------

    def _update(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        """Joseph-form Kalman update. Returns the normalised innovation squared (NIS).

        For a consistent filter NIS ~ chi^2 with dim(z) degrees of freedom, so its average over a
        run is a tuning diagnostic: well above dim(z) means the filter is overconfident.
        """
        PHt = self.P @ H.T
        S = H @ PHt + R
        K = np.linalg.solve(S, PHt.T).T  # K = P H^T S^-1 without forming the inverse
        self.x = self.x + K @ innovation
        self.x[YAW] = wrap_angle(self.x[YAW])
        IKH = np.eye(N_STATE) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(innovation @ np.linalg.solve(S, innovation))

    def update_gps(self, z: np.ndarray, sigma_m: float | None = None) -> float:
        """2D position fix z = [east, north]."""
        sigma = self.config.gps_sigma_m if sigma_m is None else sigma_m
        H = np.zeros((2, N_STATE))
        H[0, 0] = H[1, 1] = 1.0
        return self._update(np.asarray(z, float) - self.x[0:2], H, sigma**2 * np.eye(2))

    def body_velocity(self) -> np.ndarray:
        """Predicted measurement h(x) = [v_fwd, v_lat]: ENU velocity rotated into the body frame."""
        return body_velocity_model(self.x)[0]

    def update_body_velocity(
        self, z: np.ndarray, R: np.ndarray, components: tuple[int, ...] = (0, 1)
    ) -> float:
        """Body-frame velocity measurement. ``components`` selects v_fwd (0) and/or v_lat (1);
        ``z`` and ``R`` have one entry / row per selected component."""
        h, H = body_velocity_model(self.x)
        idx = list(components)
        innovation = np.atleast_1d(np.asarray(z, float)) - h[idx]
        return self._update(innovation, H[idx], np.atleast_2d(np.asarray(R, float)))

    def update_nhc(self) -> float:
        """Non-holonomic constraint: lateral body velocity = 0."""
        return self.update_body_velocity(
            np.zeros(1), np.array([[self.config.nhc_sigma**2]]), components=(1,)
        )


# --------------------------------------------------------------------------------------------
# Initialisation and a full run
# --------------------------------------------------------------------------------------------


def initialise_from_gps(
    gps: GpsMeasurements, config: EkfConfig | None = None, usable: np.ndarray | None = None
) -> tuple[int, np.ndarray, np.ndarray]:
    """Initial (fix index, x0, P0) from the noisy GPS fixes alone.

    The displacement d between the current fix and the one ~``init_window_s`` (dt) earlier gives
    both the heading and the velocity. With independent fixes of sigma per axis, d has noise
    sqrt(2) sigma per axis, so

        sigma_yaw ~ sqrt(2) sigma / |d|          sigma_v = sqrt(2) sigma / dt  (per axis)

    At 2.5 m noise and a 10 m baseline the heading is only good to ~20 deg, so the filter waits
    for the first fix with a baseline of at least ``init_min_baseline_m``. (Both are averages over
    the window rather than end-point values, an error well inside these sigmas at road speeds.)

    Starting with the GPS velocity instead of v = 0 matters for the heading. While v = 0, the NHC
    says nothing about yaw (d v_lat / d yaw = -v_fwd = 0), so a bad initial heading can only be
    corrected slowly through the position track. On the validation drive (30 seeds) the heading
    took 20 s on average and up to 63 s to settle within 5 deg starting from v = 0, against 4 s
    and at most 8 s with the GPS velocity.

    ``usable`` (bool per fix) lets the caller exclude fixes, e.g. during a simulated outage.
    """
    cfg = config or EkfConfig()
    usable = np.ones(len(gps), bool) if usable is None else np.asarray(usable, bool)
    for j in range(len(gps)):
        if not usable[j]:
            continue
        back = np.flatnonzero(usable[:j] & (gps.t[:j] <= gps.t[j] - cfg.init_window_s + 1e-6))
        if len(back) == 0:
            continue
        d = gps.pos[j] - gps.pos[back[-1]]
        baseline = float(np.hypot(*d))
        if baseline < cfg.init_min_baseline_m:
            continue
        x0 = np.zeros(N_STATE)
        x0[0:2] = gps.pos[j]
        dt = gps.t[j] - gps.t[back[-1]]
        x0[2:4] = d / dt
        x0[YAW] = np.arctan2(d[1], d[0])
        yaw_std = np.sqrt(2.0) * gps.sigma_m / baseline
        vel_std = np.sqrt(2.0) * gps.sigma_m / dt
        P0 = np.diag([
            gps.sigma_m**2, gps.sigma_m**2,
            vel_std**2, vel_std**2,
            yaw_std**2,
            cfg.init_accel_bias_std**2, cfg.init_accel_bias_std**2,
            cfg.init_gyro_bias_std**2,
        ])  # fmt: skip
        return j, x0, P0
    raise ValueError("no GPS baseline long enough to initialise the heading")


@dataclass(frozen=True)
class EkfResult:
    """Filter output on the IMU timeline. Rows before ``k0`` (initialisation) are NaN."""

    t: np.ndarray  # (N,)
    x: np.ndarray  # (N, 8) posterior state after all updates at that sample
    P: np.ndarray  # (N, 8, 8) posterior covariance
    k0: int  # first sample with an estimate
    gps_nis: np.ndarray  # (M,) NIS of each GPS update, NaN where the fix was not used

    @property
    def std(self) -> np.ndarray:
        """(N, 8) 1-sigma uncertainty of each state."""
        return np.sqrt(np.diagonal(self.P, axis1=1, axis2=2))


def run_ekf(
    t: np.ndarray,
    imu_planar: np.ndarray,
    gps: GpsMeasurements,
    config: EkfConfig | None = None,
    gps_usable: np.ndarray | None = None,
) -> EkfResult:
    """Run the EKF over a whole drive.

    t, imu_planar: (N,), (N, 3) IMU timeline and corrupted [af, al, wu].
    gps: noisy fixes; ``gps_usable`` (bool per fix) marks the fixes the filter may use (False
    during a simulated GPS outage). The IMU and NHC are never interrupted.

    At each sample k the filter predicts from k-1 to k with u[k-1] (the input is held over the
    step, as in dead reckoning), then applies a GPS update if a usable fix arrives at k, then the
    NHC.
    """
    cfg = config or EkfConfig()
    n = len(t)
    usable = np.ones(len(gps), bool) if gps_usable is None else np.asarray(gps_usable, bool)
    j0, x0, P0 = initialise_from_gps(gps, cfg, usable)
    k0 = int(gps.idx[j0])
    fix_at = np.full(n, -1)
    fix_at[gps.idx] = np.arange(len(gps))

    xs = np.full((n, N_STATE), np.nan)
    Ps = np.full((n, N_STATE, N_STATE), np.nan)
    nis = np.full(len(gps), np.nan)
    ekf = Ekf(x0, P0, cfg)
    for k in range(k0, n):
        if k > k0:
            ekf.predict(imu_planar[k - 1], t[k] - t[k - 1])
            j = fix_at[k]
            if j >= 0 and usable[j]:
                nis[j] = ekf.update_gps(gps.pos[j], gps.sigma_m)
        if cfg.use_nhc:
            ekf.update_nhc()
        xs[k] = ekf.x
        Ps[k] = ekf.P
    return EkfResult(t=np.asarray(t, float).copy(), x=xs, P=Ps, k0=k0, gps_nis=nis)
