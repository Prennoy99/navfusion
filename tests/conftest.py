"""Synthetic OXTS sequences, so the tests never need the KITTI download."""

from __future__ import annotations

import numpy as np
import pytest

from navfusion.data.kitti_loader import OXTS_FIELDS, OxtsSequence

ORIGIN = (49.0495, 8.3966, 113.0)  # Karlsruhe, where KITTI was recorded


def make_sequence(t: np.ndarray, name: str = "synthetic", **fields: np.ndarray) -> OxtsSequence:
    """OxtsSequence with every field zero except the ones given (lat/lon/alt default to ORIGIN)."""
    data = np.zeros((len(t), len(OXTS_FIELDS)))
    defaults = {"lat": ORIGIN[0], "lon": ORIGIN[1], "alt": ORIGIN[2]}
    for key, value in {**defaults, **fields}.items():
        data[:, OXTS_FIELDS.index(key)] = value
    return OxtsSequence(name=name, t=np.asarray(t, float), t0=np.datetime64("2011-09-30T12:00"),
                        data=data)


def circle_drive(duration: float = 60.0, dt: float = 0.1, speed: float = 10.0,
                 radius: float = 50.0) -> OxtsSequence:
    """Constant-speed counter-clockwise circle with physically consistent OXTS fields."""
    t = np.arange(0.0, duration, dt)
    w = speed / radius
    yaw = w * t
    return make_sequence(
        t,
        yaw=yaw,
        ve=speed * np.cos(yaw),
        vn=speed * np.sin(yaw),
        vf=np.full_like(t, speed),
        al=np.full_like(t, speed * w),  # centripetal acceleration, pointing left
        ay=np.full_like(t, speed * w),
        wu=np.full_like(t, w),
        wz=np.full_like(t, w),
        az=np.full_like(t, 9.81),
    )


@pytest.fixture
def circle() -> OxtsSequence:
    return circle_drive()
