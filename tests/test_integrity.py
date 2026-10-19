"""Timestamp gap detection / splitting and the gravity-leak checks."""

import numpy as np

from navfusion.data.integrity import (
    check_gravity_leak,
    find_timestamp_gaps,
    split_at_gaps,
    standstill_mask,
)

from .conftest import circle_drive, make_sequence


def jittered_time(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.choice([0.10, 0.11], size=n))  # KITTI-like 0.10 / 0.11 s steps


def test_no_gaps_in_normal_jitter():
    assert find_timestamp_gaps(jittered_time()).size == 0


def test_detects_dropped_samples_and_backwards_steps():
    t = jittered_time()
    t[100:] += 0.5  # 5 missing samples after index 99
    t[150:] -= t[150] - t[149]  # sample 150 repeats the timestamp of sample 149
    np.testing.assert_array_equal(find_timestamp_gaps(t), [99, 149])


def test_split_at_gaps():
    t = jittered_time()
    t[60:] += 1.0
    seq = make_sequence(t)
    parts = split_at_gaps(seq)
    assert [len(p) for p in parts] == [60, 140]
    np.testing.assert_array_equal(parts[1].t, t[60:])
    assert split_at_gaps(make_sequence(jittered_time()))[0].data.shape == (200, 30)


def test_standstill_excludes_transients():
    t = np.arange(0, 10, 0.1)
    v = np.where((t > 3) & (t < 7), 0.0, 5.0)  # stopped for 4 s
    still = standstill_mask(make_sequence(t, vf=v))
    # stopped samples within 1 s of moving are excluded: 4 s stop -> ~2 s of standstill
    assert 18 <= still.sum() <= 22
    assert not still[t < 4].any() and not still[t > 6].any()


def test_clean_planar_accel_passes_and_body_with_tilt_is_flagged():
    seq = circle_drive()
    # a 2 deg pitch leaks g*sin(2 deg) = 0.34 m/s^2 into the body-frame ax only
    seq.data[:, 11] = 9.81 * np.sin(np.deg2rad(2.0))
    report = check_gravity_leak(seq)
    assert report.ok
    np.testing.assert_allclose(report.kinematic_bias, 0.0, atol=1e-3)
    assert abs(report.kinematic_bias_body[0]) > 0.3
    assert report.standstill_mean is None  # never stops


def test_leak_in_planar_accel_fails():
    seq = circle_drive()
    seq.data[:, 14] += 0.2  # af carries 0.2 m/s^2 of gravity
    assert not check_gravity_leak(seq).ok


def test_standstill_check_used_when_car_stops():
    t = np.arange(0, 10, 0.1)
    seq = make_sequence(t, af=np.full_like(t, 0.1))  # parked, but af reads 0.1 m/s^2
    report = check_gravity_leak(seq)
    assert report.n_standstill > 0
    np.testing.assert_allclose(report.standstill_mean, [0.1, 0.0])
    assert not report.ok
