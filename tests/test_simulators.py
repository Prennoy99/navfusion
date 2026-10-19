"""Seeded GPS and MEMS IMU simulators: reproducibility, rates and noise statistics."""

import numpy as np

from navfusion.data.kitti_loader import ground_truth
from navfusion.data.simulators import (
    GpsConfig,
    ImuNoiseConfig,
    corrupt_imu,
    make_rng,
    simulate_gps,
    simulate_sensors,
)

from .conftest import circle_drive


def test_make_rng_streams():
    a = make_rng(0, "gps", "drive").normal(size=5)
    np.testing.assert_array_equal(a, make_rng(0, "gps", "drive").normal(size=5))
    assert not np.allclose(a, make_rng(1, "gps", "drive").normal(size=5))
    assert not np.allclose(a, make_rng(0, "imu", "drive").normal(size=5))
    assert not np.allclose(a, make_rng(0, "gps", "other").normal(size=5))


def test_simulate_sensors_is_reproducible(circle):
    s1, s2 = simulate_sensors(circle, seed=3), simulate_sensors(circle, seed=3)
    np.testing.assert_array_equal(s1.gps.pos, s2.gps.pos)
    np.testing.assert_array_equal(s1.imu.body, s2.imu.body)
    s3 = simulate_sensors(circle, seed=4)
    assert not np.allclose(s1.gps.pos, s3.gps.pos)


def test_gps_noise_does_not_depend_on_imu_config(circle):
    a = simulate_sensors(circle, seed=0)
    b = simulate_sensors(circle, seed=0, imu_config=ImuNoiseConfig(accel_noise_std=1.0))
    np.testing.assert_array_equal(a.gps.pos, b.gps.pos)


def test_gps_rate_with_jittered_timestamps():
    rng = np.random.default_rng(0)
    t = np.concatenate([[0.0], np.cumsum(rng.choice([0.10, 0.11], size=3000))])
    pos = np.zeros((len(t), 2))
    gps = simulate_gps(t, pos, GpsConfig(), make_rng(0, "gps"))
    assert len(gps) == int(t[-1]) + 1
    # each fix lands on the first sample at/after its whole-second epoch
    assert np.all(gps.t - np.arange(len(gps)) >= -1e-9)
    assert np.all(gps.t - np.arange(len(gps)) < 0.11 + 1e-9)
    mask = gps.available(len(t))
    assert mask.sum() == len(gps) and mask[0]


def test_gps_noise_statistics():
    t = np.arange(0.0, 20000.0, 0.1)
    pos = np.column_stack([t, -t])
    gps = simulate_gps(t, pos, GpsConfig(sigma_m=2.5), make_rng(1, "gps"))
    err = gps.pos - pos[gps.idx]
    np.testing.assert_allclose(err.mean(axis=0), 0.0, atol=0.1)
    np.testing.assert_allclose(err.std(axis=0), 2.5, rtol=0.03)


def test_imu_bias_ranges_and_noise():
    n = 50000
    cfg = ImuNoiseConfig()
    for seed in range(20):
        imu = corrupt_imu(np.zeros((n, 6)), np.zeros((n, 3)), cfg, make_rng(seed, "imu"))
        assert np.all((np.abs(imu.accel_bias) >= 0.05) & (np.abs(imu.accel_bias) <= 0.10))
        lo, hi = cfg.gyro_bias_range
        assert np.all((np.abs(imu.gyro_bias) >= lo) & (np.abs(imu.gyro_bias) <= hi))
    # on zero input, the output is exactly bias + noise
    np.testing.assert_allclose(imu.body[:, :3].mean(axis=0), imu.accel_bias, atol=2e-3)
    np.testing.assert_allclose(imu.body[:, 3:].std(axis=0), cfg.gyro_noise_std, rtol=0.03)
    np.testing.assert_allclose(imu.body[:, :3].std(axis=0), cfg.accel_noise_std, rtol=0.03)


def test_planar_channels_share_the_body_error():
    seq = circle_drive()
    imu = simulate_sensors(seq, seed=0).imu
    body_err = imu.body - seq.imu_body
    planar_err = imu.planar - seq.imu_planar
    np.testing.assert_allclose(planar_err, body_err[:, [0, 1, 5]], atol=1e-12)
    np.testing.assert_allclose(imu.planar_bias,
                               [imu.accel_bias[0], imu.accel_bias[1], imu.gyro_bias[2]])


def test_gps_fixes_follow_truth(circle):
    gt = ground_truth(circle)
    sim = simulate_sensors(circle, seed=0)
    err = np.linalg.norm(sim.gps.pos - gt.pos[sim.gps.idx], axis=1)
    assert err.max() < 6 * 2.5 * np.sqrt(2)
