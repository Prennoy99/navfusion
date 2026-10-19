"""Data integrity checks run before any filtering.

1. **Timestamp gaps.** The EKF integrates the IMU over dt = t[k+1] - t[k]. A dropped OXTS packet
   would silently become one long integration step, so we find gaps and split the drive there.
2. **Gravity leak in af / al.** The 2D filters use the level-frame accelerations ``af``/``al``,
   which are supposed to have gravity removed. If they still carried g*sin(pitch) or g*sin(roll),
   the filter would see a slope-dependent "bias" that is not a sensor property at all. We check
   two ways:
     - at standstill the true acceleration is zero, so mean(af), mean(al) must be ~0;
     - over the whole drive, af/al must match the derivative of the OXTS velocity. In a frame
       rotating with the car at yaw rate w:  dvf/dt = af + w*vl  and  dvl/dt = al - w*vf.
       The second check works on drives where the car never stops.
   Body-frame ``ax``/``ay`` are run through the same checks as a positive control: they *do*
   contain gravity, and the checks should flag them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navfusion.data.kitti_loader import OxtsSequence

# Any leak smaller than the smallest accelerometer bias we inject in the MEMS simulator
# (0.05 m/s^2) is irrelevant next to the errors the filter has to deal with anyway.
GRAVITY_LEAK_TOL = 0.05  # m/s^2


def find_timestamp_gaps(t: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Indices ``i`` where the step t[i] -> t[i+1] is a gap.

    A gap is a step longer than ``factor`` times the median step (one or more missing samples
    at 10 Hz) or a non-positive step (duplicated / out-of-order timestamps). Normal OXTS jitter
    in KITTI is about +-30 ms around 100 ms, well inside 1.5x.
    """
    dt = np.diff(t)
    return np.flatnonzero((dt > factor * np.median(dt)) | (dt <= 0.0))


def split_at_gaps(seq: OxtsSequence, factor: float = 1.5) -> list[OxtsSequence]:
    """Split a sequence into gap-free segments (a gap-free drive returns ``[seq]``)."""
    cuts = find_timestamp_gaps(seq.t, factor) + 1
    bounds = [0, *cuts.tolist(), len(seq)]
    return [seq.segment(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def standstill_mask(seq: OxtsSequence, speed_tol: float = 0.05, hold_s: float = 1.0) -> np.ndarray:
    """Samples where the car has been and stays stationary for at least ``hold_s`` either side.

    A single-sample speed threshold also catches the moments just before moving off or just after
    stopping, where the true acceleration is clearly non-zero; requiring a stationary window
    around each sample excludes those transients.
    """
    still = (np.hypot(seq["vf"], seq["vl"]) < speed_tol).astype(int)
    k = max(1, round(hold_s / np.median(np.diff(seq.t))))
    window = np.convolve(still, np.ones(2 * k + 1, dtype=int), mode="same")
    return window == 2 * k + 1


@dataclass(frozen=True)
class GravityLeakReport:
    n_standstill: int
    standstill_mean: np.ndarray | None  # (2,) mean [af, al] at standstill, None if no standstill
    standstill_mean_body: np.ndarray | None  # (2,) the same for body-frame [ax, ay]
    kinematic_bias: np.ndarray  # (2,) mean of af/al minus the velocity-derived acceleration
    kinematic_bias_body: np.ndarray  # (2,) the same for body-frame ax/ay
    tol: float = GRAVITY_LEAK_TOL

    @property
    def ok(self) -> bool:
        """True when af/al show no gravity leak above ``tol`` in any available check."""
        checks = [self.kinematic_bias]
        if self.standstill_mean is not None:
            checks.append(self.standstill_mean)
        return all(bool(np.all(np.abs(c) < self.tol)) for c in checks)


def _kinematic_residual(seq: OxtsSequence, a_fwd: np.ndarray, a_left: np.ndarray) -> np.ndarray:
    """Mean of measured minus velocity-implied acceleration, in the yaw-rotating level frame."""
    t, vf, vl, w = seq.t, seq["vf"], seq["vl"], seq["wu"]
    implied_f = np.gradient(vf, t) - w * vl
    implied_l = np.gradient(vl, t) + w * vf
    return np.array([np.mean(a_fwd - implied_f), np.mean(a_left - implied_l)])


def check_gravity_leak(seq: OxtsSequence, tol: float = GRAVITY_LEAK_TOL) -> GravityLeakReport:
    still = standstill_mask(seq)
    has_still = bool(still.any())
    return GravityLeakReport(
        n_standstill=int(still.sum()),
        standstill_mean=seq.fields("af", "al")[still].mean(axis=0) if has_still else None,
        standstill_mean_body=seq.fields("ax", "ay")[still].mean(axis=0) if has_still else None,
        kinematic_bias=_kinematic_residual(seq, seq["af"], seq["al"]),
        kinematic_bias_body=_kinematic_residual(seq, seq["ax"], seq["ay"]),
        tol=tol,
    )
