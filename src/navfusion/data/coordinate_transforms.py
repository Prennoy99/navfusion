"""WGS84 geodetic <-> ECEF <-> local ENU, written from scratch.

The filters work in a flat local East-North-Up (ENU) frame in metres, with its origin at the start
of the drive. GPS gives latitude/longitude/altitude on the WGS84 ellipsoid, so we go through
Earth-Centred Earth-Fixed (ECEF) Cartesian coordinates:

    geodetic (lat, lon, h)  ->  ECEF (X, Y, Z)  ->  ENU (e, n, u) relative to a reference point

Over a KITTI drive (a few km) the ENU tangent plane is an excellent approximation: the Earth's
curvature drops the surface ~0.08 m below the plane at 1 km and ~0.3 m at 2 km, which is well
below the 2.5 m GPS noise. Only the horizontal (e, n) part is used by the 2D filters.

All angles in this module are in degrees at the API boundary (that is how OXTS reports lat/lon)
and radians internally.
"""

from __future__ import annotations

import numpy as np

# WGS84 ellipsoid (NIMA TR8350.2)
WGS84_A = 6378137.0  # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the half-open interval (-pi, pi].

    ``pi - mod(pi - a, 2 pi)`` maps pi to pi and -pi to pi, so every heading has exactly one
    representation. That matters for the EKF: a yaw innovation of 359 deg must be treated as -1 deg.
    """
    return np.pi - np.mod(np.pi - a, 2.0 * np.pi)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m) -> np.ndarray:
    """Geodetic WGS84 coordinates -> ECEF [m]. Returns shape (..., 3).

    N(lat) = a / sqrt(1 - e^2 sin^2 lat) is the prime-vertical radius of curvature, i.e. the
    distance from the surface to the polar axis along the ellipsoid normal.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    h = np.asarray(alt_m, dtype=float)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(ecef: np.ndarray, iterations: int = 5) -> np.ndarray:
    """ECEF [m] -> geodetic (lat_deg, lon_deg, alt_m). Returns shape (..., 3).

    Longitude is closed-form. Latitude has no closed form on an ellipsoid, so it is found by
    fixed-point iteration on tan(lat) = Z / (p (1 - e^2 N / (N + h))), with p the distance from
    the polar axis. Starting from the spherical guess, it converges to well below a millimetre
    in 3-4 iterations for points near the Earth's surface (not valid at the poles, where
    cos(lat) -> 0, which is irrelevant for road vehicles).
    """
    ecef = np.asarray(ecef, dtype=float)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(iterations):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.stack([np.rad2deg(lat), np.rad2deg(lon), h], axis=-1)


def ecef_to_enu_rotation(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Rotation matrix R (3x3) with enu = R @ (ecef - ecef0).

    Rows are the local East, North and Up unit vectors expressed in ECEF:
      east  = (-sin lon,          cos lon,           0      )
      north = (-sin lat cos lon, -sin lat sin lon,   cos lat)
      up    = ( cos lat cos lon,  cos lat sin lon,   sin lat)
    R is orthonormal, so the inverse transform is simply R.T.
    """
    lat, lon = np.deg2rad(lat0_deg), np.deg2rad(lon0_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )


def ecef_to_enu(ecef: np.ndarray, origin_geodetic: tuple[float, float, float]) -> np.ndarray:
    """ECEF points (..., 3) -> ENU (..., 3) relative to ``origin_geodetic = (lat, lon, alt)``."""
    lat0, lon0, alt0 = origin_geodetic
    r = ecef_to_enu_rotation(lat0, lon0)
    d = np.asarray(ecef, dtype=float) - geodetic_to_ecef(lat0, lon0, alt0)
    return d @ r.T


def enu_to_ecef(enu: np.ndarray, origin_geodetic: tuple[float, float, float]) -> np.ndarray:
    """ENU points (..., 3) relative to ``origin_geodetic`` -> ECEF (..., 3)."""
    lat0, lon0, alt0 = origin_geodetic
    r = ecef_to_enu_rotation(lat0, lon0)
    return np.asarray(enu, dtype=float) @ r + geodetic_to_ecef(lat0, lon0, alt0)


def geodetic_to_enu(lat_deg, lon_deg, alt_m, origin_geodetic) -> np.ndarray:
    """Geodetic WGS84 -> local ENU [m] relative to ``origin_geodetic``. Returns (..., 3)."""
    return ecef_to_enu(geodetic_to_ecef(lat_deg, lon_deg, alt_m), origin_geodetic)


def enu_to_geodetic(enu: np.ndarray, origin_geodetic) -> np.ndarray:
    """Local ENU [m] -> geodetic (lat_deg, lon_deg, alt_m). Returns (..., 3)."""
    return ecef_to_geodetic(enu_to_ecef(enu, origin_geodetic))
