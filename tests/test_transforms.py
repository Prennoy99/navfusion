"""WGS84 / ECEF / ENU transforms, checked against pymap3d as an independent oracle."""

import numpy as np
import pymap3d
import pytest

from navfusion.data.coordinate_transforms import (
    ecef_to_enu_rotation,
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    wrap_angle,
)

from .conftest import ORIGIN

rng = np.random.default_rng(0)
# Points scattered over +-5 km around Karlsruhe, a bit larger than any KITTI drive
LAT = ORIGIN[0] + rng.uniform(-0.045, 0.045, 200)
LON = ORIGIN[1] + rng.uniform(-0.07, 0.07, 200)
ALT = ORIGIN[2] + rng.uniform(-50, 50, 200)


def test_geodetic_to_ecef_matches_pymap3d():
    ours = geodetic_to_ecef(LAT, LON, ALT)
    ref = np.stack(pymap3d.geodetic2ecef(LAT, LON, ALT), axis=-1)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_ecef_to_geodetic_matches_pymap3d():
    ecef = geodetic_to_ecef(LAT, LON, ALT)
    ours = ecef_to_geodetic(ecef)
    ref = np.stack(pymap3d.ecef2geodetic(*ecef.T), axis=-1)
    np.testing.assert_allclose(ours[:, :2], ref[:, :2], atol=1e-10)  # deg, ~10 um
    np.testing.assert_allclose(ours[:, 2], ref[:, 2], atol=1e-6)  # m


def test_geodetic_to_enu_matches_pymap3d():
    ours = geodetic_to_enu(LAT, LON, ALT, ORIGIN)
    ref = np.stack(pymap3d.geodetic2enu(LAT, LON, ALT, *ORIGIN), axis=-1)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_enu_round_trip():
    enu = geodetic_to_enu(LAT, LON, ALT, ORIGIN)
    back = enu_to_geodetic(enu, ORIGIN)
    np.testing.assert_allclose(back[:, 0], LAT, atol=1e-10)
    np.testing.assert_allclose(back[:, 1], LON, atol=1e-10)
    np.testing.assert_allclose(back[:, 2], ALT, atol=1e-6)


def test_origin_maps_to_zero_and_axes_point_east_north():
    np.testing.assert_allclose(geodetic_to_enu(*ORIGIN, ORIGIN), 0.0, atol=1e-9)
    east = geodetic_to_enu(ORIGIN[0], ORIGIN[1] + 0.001, ORIGIN[2], ORIGIN)
    north = geodetic_to_enu(ORIGIN[0] + 0.001, ORIGIN[1], ORIGIN[2], ORIGIN)
    assert east[0] > 70 and abs(east[1]) < 0.01  # 0.001 deg lon ~ 73 m at 49 deg N
    assert north[1] > 110 and abs(north[0]) < 1e-6  # 0.001 deg lat ~ 111 m


def test_rotation_is_orthonormal():
    r = ecef_to_enu_rotation(*ORIGIN[:2])
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (np.pi, np.pi), (-np.pi, np.pi), (3 * np.pi, np.pi), (1.5 * np.pi, -0.5 * np.pi),
     (-1.5 * np.pi, 0.5 * np.pi), (2 * np.pi + 0.1, 0.1)],
)
def test_wrap_angle(angle, expected):
    assert np.isclose(wrap_angle(angle), expected)


def test_wrap_angle_range_and_invariance():
    a = np.linspace(-20, 20, 10001)
    w = wrap_angle(a)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)
