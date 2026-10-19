"""8-state EKF: Jacobians, update algebra, covariance health, heading wrap, convergence, bias
observability. Everything runs on the synthetic circle drive, so no KITTI data is needed."""

import numpy as np
import pytest

from navfusion.data.coordinate_transforms import wrap_angle
from navfusion.data.kitti_loader import ground_truth
from navfusion.data.simulators import GpsMeasurements, simulate_sensors
from navfusion.filters.ekf import (
    N_STATE,
    Ekf,
    EkfConfig,
    body_velocity_model,
    initialise_from_gps,
    motion_model,
    run_ekf,
)

from .conftest import circle_drive


def numerical_jacobian(f, x, eps=1e-6):
    fx = f(x)
    J = np.empty((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        diff = f(x + dx) - f(x - dx)
        if len(fx) == N_STATE:
            diff[4] = wrap_angle(diff[4])
        J[:, i] = diff / (2 * eps)
    return J


STATE = np.array([3.0, -1.0, 7.0, 2.0, 2.9, 0.08, -0.06, 0.004])
INPUT = np.array([0.7, 1.9, 0.25])


def test_motion_model_jacobians_match_finite_differences():
    dt = 0.1
    _, F, G = motion_model(STATE, INPUT, dt)
    np.testing.assert_allclose(F, numerical_jacobian(lambda x: motion_model(x, INPUT, dt)[0],
                                                     STATE), atol=1e-8)
    np.testing.assert_allclose(G, numerical_jacobian(lambda u: motion_model(STATE, u, dt)[0],
                                                     INPUT), atol=1e-8)


def test_body_velocity_jacobian_matches_finite_differences():
    h, H = body_velocity_model(STATE)
    np.testing.assert_allclose(np.hypot(*h), np.hypot(*STATE[2:4]))
    np.testing.assert_allclose(H, numerical_jacobian(lambda x: body_velocity_model(x)[0], STATE),
                               atol=1e-8)


def test_shapes_and_validation():
    ekf = Ekf(np.zeros(N_STATE), np.eye(N_STATE))
    ekf.predict(INPUT, 0.1)
    assert ekf.x.shape == (8,) and ekf.P.shape == (8, 8)
    nis = ekf.update_body_velocity(np.array([1.0, 0.0]), np.eye(2) * 0.1)
    assert np.isscalar(nis) and nis >= 0
    with pytest.raises(ValueError):
        Ekf(np.zeros(5), np.eye(5))


def test_joseph_update_equals_standard_form_for_optimal_gain():
    """With the optimal gain, (I-KH) P (I-KH)^T + K R K^T reduces algebraically to (I-KH) P."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(N_STATE, N_STATE))
    P0 = A @ A.T + np.eye(N_STATE)
    ekf = Ekf(STATE, P0)
    z = STATE[:2] + [1.0, -2.0]
    ekf.update_gps(z, sigma_m=2.0)
    H = np.zeros((2, N_STATE))
    H[0, 0] = H[1, 1] = 1.0
    K = P0 @ H.T @ np.linalg.inv(H @ P0 @ H.T + 4.0 * np.eye(2))
    np.testing.assert_allclose(ekf.P, (np.eye(N_STATE) - K @ H) @ P0, atol=1e-9)
    np.testing.assert_allclose(ekf.x, STATE + K @ (z - STATE[:2]), atol=1e-12)


def test_nhc_removes_lateral_velocity():
    x0 = np.array([0, 0, 0.0, 5.0, 0.0, 0, 0, 0])  # heading east but moving north: v_lat = 5
    P0 = np.diag([1, 1, 4, 4, 1e-6, 1e-4, 1e-4, 1e-6])
    ekf = Ekf(x0, P0)
    for _ in range(10):
        ekf.update_nhc()
    assert abs(ekf.body_velocity()[1]) < 0.1


@pytest.fixture(scope="module")
def circle_run():
    """5 min on a 50 m circle at 10 m/s (yaw crosses +-pi ~10 times) with MEMS IMU + 1 Hz GPS."""
    seq = circle_drive(duration=300.0)
    sim = simulate_sensors(seq, seed=7)
    return ground_truth(seq), sim, run_ekf(sim.t, sim.imu.planar, sim.gps)


def test_covariance_stays_symmetric_positive_definite(circle_run):
    _, _, res = circle_run
    P = res.P[res.k0:]
    np.testing.assert_array_equal(P, np.swapaxes(P, 1, 2))
    assert np.linalg.eigvalsh(P).min() > 0


def test_yaw_stays_wrapped_and_tracks_across_pi(circle_run):
    gt, _, res = circle_run
    yaw = res.x[res.k0:, 4]
    assert np.all((yaw > -np.pi) & (yaw <= np.pi))
    assert np.sum(np.abs(np.diff(yaw)) > np.pi) >= 8  # it really crossed the +-pi seam
    err = wrap_angle(yaw - gt.yaw[res.k0:])
    assert np.max(np.abs(err[300:])) < np.deg2rad(3.0)  # no 2 pi glitches at the seam


def test_circle_converges_and_beats_raw_gps(circle_run):
    gt, sim, res = circle_run
    k = slice(res.k0 + 300, None)
    ekf_rmse = np.sqrt(np.mean(np.sum((res.x[k, :2] - gt.pos[k]) ** 2, axis=1)))
    gps_rmse = np.sqrt(np.mean(np.sum((sim.gps.pos - gt.pos[sim.gps.idx]) ** 2, axis=1)))
    assert ekf_rmse < 0.6 * gps_rmse
    speed_err = np.hypot(*res.x[k, 2:4].T) - gt.speed[k]
    assert np.sqrt(np.mean(speed_err**2)) < 0.2
    # Consistency: the error stays inside the filter's own 3-sigma bound almost always.
    pos_err = np.abs(res.x[k, :2] - gt.pos[k])
    assert np.mean(pos_err < 3 * res.std[k, :2]) > 0.97


def test_biases_are_observable(circle_run):
    _, sim, res = circle_run
    est, true = res.x[-1, 5:8], sim.imu.planar_bias
    np.testing.assert_allclose(est[:2], true[:2], atol=0.02)  # injected |b| 0.05-0.1 m/s^2
    np.testing.assert_allclose(est[2], true[2], atol=np.deg2rad(0.05))  # injected 0.1-0.5 deg/s
    # ... and the filter knows it: bias uncertainty shrank well below its initial value.
    cfg = EkfConfig()
    assert np.all(res.std[-1, 5:7] < 0.3 * cfg.init_accel_bias_std)
    assert res.std[-1, 7] < 0.3 * cfg.init_gyro_bias_std


def test_gps_mask_is_respected(circle_run):
    """Without GPS after initialisation, the position uncertainty must grow, and no fix is used."""
    gt, sim, _ = circle_run
    usable = np.zeros(len(sim.gps), bool)
    usable[:5] = True
    res = run_ekf(sim.t, sim.imu.planar, sim.gps, gps_usable=usable)
    assert np.all(np.isnan(res.gps_nis[5:]))
    k_last_fix = sim.gps.idx[4]
    assert res.std[-1, :2].min() > 4 * res.std[k_last_fix, :2].max()


def test_initialisation_waits_for_baseline_and_uses_gps_velocity():
    """Standing still for 5 s, then driving east at 10 m/s: no heading before the car moves."""
    t = np.arange(0.0, 20.0, 1.0)
    pos = np.column_stack([np.where(t < 5, 0.0, 10.0 * (t - 5)), np.zeros_like(t)])
    gps = GpsMeasurements(idx=np.arange(len(t)) * 10, t=t, pos=pos, sigma_m=2.5)
    j, x0, P0 = initialise_from_gps(gps)
    assert t[j] >= 6.0  # the 2 s baseline first reaches 10 m at t = 6 s
    assert x0[4] == pytest.approx(0.0)
    np.testing.assert_allclose(x0[:2], pos[j])
    # The velocity is the mean over the 2 s window: 1 s standing + 1 s at 10 m/s.
    np.testing.assert_allclose(x0[2:4], [5.0, 0.0])
    assert np.all(np.linalg.eigvalsh(P0) > 0)

    gps_still = GpsMeasurements(idx=gps.idx, t=t, pos=np.zeros_like(pos), sigma_m=2.5)
    with pytest.raises(ValueError):
        initialise_from_gps(gps_still)


def test_states_before_initialisation_are_nan(circle_run):
    _, _, res = circle_run
    assert res.k0 > 0
    assert np.all(np.isnan(res.x[: res.k0]))
    assert np.all(np.isfinite(res.x[res.k0 :]))
