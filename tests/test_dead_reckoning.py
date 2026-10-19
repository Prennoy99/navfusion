"""Dead-reckoning integrator: exactness on clean synthetic motion, analytic bias-drift laws."""

import numpy as np

from navfusion.data.kitti_loader import ground_truth
from navfusion.filters.dead_reckoning import dead_reckon, propagate

from .conftest import circle_drive


def true_initial_state(gt):
    return np.r_[gt.pos[0], gt.vel[0], gt.yaw[0]]


def analytic_circle(seq, radius=50.0):
    """Exact positions for circle_drive(): centre (0, radius), start at the origin heading east."""
    yaw = np.unwrap(seq["yaw"])
    return np.column_stack([radius * np.sin(yaw), radius * (1 - np.cos(yaw))])


def test_vectorised_matches_stepwise():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.choice([0.10, 0.11], size=300))
    u = rng.normal(0, [1.0, 1.0, 0.3], size=(300, 3))
    x0 = np.array([1.0, -2.0, 3.0, 0.5, 3.0])
    batch = dead_reckon(t, u, x0)
    x = x0.copy()
    np.testing.assert_allclose(batch[0], x0)
    for k in range(len(t) - 1):
        x = propagate(x, u[k], t[k + 1] - t[k])
        np.testing.assert_allclose(batch[k + 1], x, atol=1e-9)


def test_clean_circle_is_tracked():
    """Clean inputs on a 50 m circle: the mid-step heading keeps the error tiny after 3 laps."""
    seq = circle_drive(duration=95.0)  # ~3 laps at 10 m/s
    est = dead_reckon(seq.t, seq.imu_planar, true_initial_state(ground_truth(seq)))
    err = np.linalg.norm(est[:, :2] - analytic_circle(seq), axis=1)
    assert err.max() < 0.05
    np.testing.assert_allclose(np.linalg.norm(est[:, 2:4], axis=1), 10.0, atol=1e-3)


def test_start_of_step_heading_would_fail_the_same_circle():
    """Guards the design choice: integrating with yaw_k instead of the mid-step yaw drifts off."""
    seq = circle_drive(duration=95.0)
    u = seq.imu_planar.copy()
    dt = np.diff(seq.t)[0]
    lagged = u.copy()
    # Emulate start-of-step rotation: rotate the body acceleration back by wu dt / 2.
    phi = -0.5 * u[:, 2] * dt
    lagged[:, 0] = np.cos(phi) * u[:, 0] - np.sin(phi) * u[:, 1]
    lagged[:, 1] = np.sin(phi) * u[:, 0] + np.cos(phi) * u[:, 1]
    est = dead_reckon(seq.t, lagged, true_initial_state(ground_truth(seq)))
    assert np.linalg.norm(est[-1, :2] - analytic_circle(seq)[-1]) > 5.0


def test_accel_bias_gives_quadratic_drift():
    """Standing still with a forward bias b: position error = b t^2 / 2 along the heading."""
    t = np.arange(0.0, 30.0 + 1e-9, 0.1)
    b = 0.08
    u = np.zeros((len(t), 3))
    u[:, 0] = b
    yaw0 = np.deg2rad(30.0)
    est = dead_reckon(t, u, np.array([0, 0, 0, 0, yaw0]))
    expected = 0.5 * b * t**2
    np.testing.assert_allclose(est[:, 0], expected * np.cos(yaw0), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(est[:, 1], expected * np.sin(yaw0), rtol=1e-9, atol=1e-12)


def test_gyro_bias_gives_linear_heading_drift_and_wraps():
    t = np.arange(0.0, 100.0 + 1e-9, 0.1)
    bg = np.deg2rad(0.5)
    u = np.zeros((len(t), 3))
    u[:, 2] = bg
    est = dead_reckon(t, u, np.array([0, 0, 0, 0, np.pi - 0.1]))
    unwrapped = np.pi - 0.1 + bg * t
    np.testing.assert_allclose(np.cos(est[:, 4]), np.cos(unwrapped), atol=1e-9)
    assert np.all(est[:, 4] > -np.pi) and np.all(est[:, 4] <= np.pi)
    assert est[-1, 4] < 0  # crossed +pi and wrapped


def test_restart_from_given_state():
    seq = circle_drive()
    gt = ground_truth(seq)
    s = 200
    x_s = np.r_[gt.pos[s], gt.vel[s], gt.yaw[s]]
    est = dead_reckon(seq.t[s:], seq.imu_planar[s:], x_s)
    np.testing.assert_allclose(est[0], x_s, atol=1e-12)
    assert est.shape == (len(seq) - s, 5)
    assert np.linalg.norm(est[-1, :2] - analytic_circle(seq)[-1]) < 0.05
