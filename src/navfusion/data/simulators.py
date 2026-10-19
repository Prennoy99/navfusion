"""Seeded sensor simulators: a consumer 1 Hz GPS and a MEMS-grade IMU.

KITTI's OXTS unit is an RTK-GPS/INS with centimetre-level position and a tactical-grade IMU.
Feeding it to the filters would make every method look excellent: the EKF would be corrected
toward the ground truth itself, and dead reckoning on such an IMU barely drifts. So the clean OXTS
signals are kept as ground truth, and the filters receive degraded copies that resemble
the sensors in a production vehicle:

* **GPS:** horizontal position + white Gaussian noise (sigma = 2.5 m per axis), at 1 Hz.
* **IMU:** every channel gets a constant, randomly drawn bias plus white noise, with magnitudes
  typical of an automotive MEMS IMU (accelerometer bias 0.05-0.1 m/s^2, gyro bias 0.1-0.5 deg/s).

**Reproducibility.** Every random draw comes from ``make_rng(seed, *keys)``. Each sensor uses its
own named stream, e.g. ``make_rng(seed, "gps", drive)``, so for a given seed all methods see
identical noise, and changing the IMU model never reshuffles the GPS noise (or vice versa).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from navfusion.data.kitti_loader import OxtsSequence, ground_truth


def make_rng(seed: int, *keys: str) -> np.random.Generator:
    """Independent, reproducible random stream for ``(seed, *keys)``.

    The string keys are hashed with CRC32 (stable across runs and platforms, unlike ``hash()``)
    and fed to NumPy's SeedSequence together with the seed, which yields statistically
    independent streams for different keys.
    """
    return np.random.default_rng([seed, *(zlib.crc32(k.encode()) for k in keys)])


# --------------------------------------------------------------------------------------------
# GPS
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GpsConfig:
    rate_hz: float = 1.0
    sigma_m: float = 2.5  # per horizontal axis; the EKF's measurement noise R uses the same value


@dataclass(frozen=True)
class GpsMeasurements:
    """Noisy GPS fixes, aligned to the ~10 Hz OXTS/IMU timeline."""

    idx: np.ndarray  # (M,) index into the IMU timeline at which each fix arrives
    t: np.ndarray  # (M,) [s]
    pos: np.ndarray  # (M, 2) noisy east, north [m]
    sigma_m: float

    def __len__(self) -> int:
        return len(self.idx)

    def available(self, n: int) -> np.ndarray:
        """(n,) bool mask over the IMU timeline: True where a GPS fix arrives."""
        mask = np.zeros(n, dtype=bool)
        mask[self.idx] = True
        return mask


def simulate_gps(
    t: np.ndarray, pos_true: np.ndarray, config: GpsConfig, rng: np.random.Generator
) -> GpsMeasurements:
    """Downsample the true ENU track to ``config.rate_hz`` and add Gaussian noise.

    A fix is placed on the first IMU sample at or after each GPS epoch t0 + k / rate. The OXTS
    timestamps jitter by a few tens of ms, so a fixed every-10th-sample stride would let the GPS
    rate drift. Anchoring to wall-clock epochs keeps it at exactly 1 Hz on average.

    The noise is white (independent per fix). Real GNSS errors are time-correlated (multipath,
    atmosphere), so this is the optimistic textbook case and matches the EKF's white-noise
    assumption.
    """
    period = 1.0 / config.rate_hz
    epochs = t[0] + np.arange(0.0, t[-1] - t[0] + 1e-9, period)
    idx = np.unique(np.searchsorted(t, epochs - 1e-9, side="left"))
    idx = idx[idx < len(t)]
    noise = rng.normal(0.0, config.sigma_m, size=(len(idx), 2))
    return GpsMeasurements(idx=idx, t=t[idx], pos=pos_true[idx, :2] + noise, sigma_m=config.sigma_m)


# --------------------------------------------------------------------------------------------
# IMU
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImuNoiseConfig:
    """MEMS error model: constant bias (random magnitude and sign per axis) + white noise.

    Noise standard deviations are per sample at the ~10 Hz OXTS rate. A gyro sigma of 0.01 rad/s
    at 10 Hz corresponds to an angle random walk of 0.01 * sqrt(0.1 s) = 0.0032 rad/sqrt(s),
    about 11 deg/sqrt(h), which is consumer/automotive grade.
    """

    accel_bias_range: tuple[float, float] = (0.05, 0.10)  # |bias| [m/s^2]
    accel_noise_std: float = 0.05  # [m/s^2]
    gyro_bias_range: tuple[float, float] = (float(np.deg2rad(0.1)), float(np.deg2rad(0.5)))
    gyro_noise_std: float = 0.01  # [rad/s]


@dataclass(frozen=True)
class ImuMeasurements:
    """Corrupted IMU stream plus the injected biases (kept to check the EKF's bias estimates)."""

    body: np.ndarray  # (N, 6) [ax, ay, az, wx, wy, wz]: LSTM input
    planar: np.ndarray  # (N, 3) [af, al, wu]: EKF / dead-reckoning input
    accel_bias: np.ndarray  # (3,) injected [x, y, z] accelerometer bias [m/s^2]
    gyro_bias: np.ndarray  # (3,) injected [x, y, z] gyro bias [rad/s]

    @property
    def planar_bias(self) -> np.ndarray:
        """(3,) biases on [af, al, wu]: what the EKF states b_ax, b_ay, b_gz should converge to."""
        return np.array([self.accel_bias[0], self.accel_bias[1], self.gyro_bias[2]])


def _random_bias(lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, size=3) * rng.choice([-1.0, 1.0], size=3)


def corrupt_imu(
    imu_body: np.ndarray,
    imu_planar: np.ndarray,
    config: ImuNoiseConfig,
    rng: np.random.Generator,
) -> ImuMeasurements:
    """Apply one MEMS error realisation to the clean body-frame and level-frame IMU channels.

    A real MEMS IMU measures only the body frame (ax..wz). The 2D filters use the level-frame
    ``af, al, wu``, which a vehicle would obtain by rotating the body measurements with its
    roll/pitch estimate. With roll/pitch at most a few degrees, that rotation scales a sensor error
    by cos(tilt) > 0.98, so the *same* error realisation is applied to (ax, ay, wz) and
    (af, al, wu). The EKF and the LSTM therefore see one consistent sensor, and
    ``planar_bias`` is the ground truth for the EKF's bias states.
    """
    n = len(imu_body)
    accel_bias = _random_bias(*config.accel_bias_range, rng)
    gyro_bias = _random_bias(*config.gyro_bias_range, rng)
    err = np.empty((n, 6))
    err[:, :3] = accel_bias + rng.normal(0.0, config.accel_noise_std, size=(n, 3))
    err[:, 3:] = gyro_bias + rng.normal(0.0, config.gyro_noise_std, size=(n, 3))
    return ImuMeasurements(
        body=imu_body + err,
        planar=imu_planar + err[:, [0, 1, 5]],
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )


# --------------------------------------------------------------------------------------------
# One-call convenience for a whole drive
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedSensors:
    """Everything a filter is allowed to see for one drive and seed."""

    t: np.ndarray  # (N,) IMU timeline [s]
    imu: ImuMeasurements
    gps: GpsMeasurements
    seed: int
    gps_config: GpsConfig = field(default_factory=GpsConfig)
    imu_config: ImuNoiseConfig = field(default_factory=ImuNoiseConfig)


def simulate_sensors(
    seq: OxtsSequence,
    seed: int,
    gps_config: GpsConfig | None = None,
    imu_config: ImuNoiseConfig | None = None,
) -> SimulatedSensors:
    """Simulated GPS + MEMS IMU for one drive. Same (drive, seed) -> identical output."""
    gps_config = gps_config or GpsConfig()
    imu_config = imu_config or ImuNoiseConfig()
    gt = ground_truth(seq)
    gps = simulate_gps(gt.t, gt.pos, gps_config, make_rng(seed, "gps", seq.name))
    imu = corrupt_imu(seq.imu_body, seq.imu_planar, imu_config, make_rng(seed, "imu", seq.name))
    return SimulatedSensors(seq.t.copy(), imu, gps, seed, gps_config, imu_config)
