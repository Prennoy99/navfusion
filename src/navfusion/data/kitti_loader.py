"""Load KITTI Raw OXTS (GPS/IMU) sequences with plain NumPy.

Each synced drive has ``oxts/timestamps.txt`` (one timestamp per line) and ``oxts/data/*.txt``
(one line of 30 space-separated values per timestamp, ~10 Hz). The 30 fields are documented in
``oxts/dataformat.txt``; ``OXTS_FIELDS`` below lists them in order. (``dataformat.txt`` labels the
14th field "ay" a second time; it is actually the z acceleration, ``az``.)

Parsing this directly takes a few lines and avoids pykitti, whose package ``__init__`` imports
OpenCV even when only the OXTS text is needed.

The OXTS unit on the KITTI car is an RT3003 RTK-GPS/INS, far more accurate than a production
ADAS sensor set. Everything returned here is therefore *ground truth*: it is used for evaluation,
for training labels, and as the clean input to the sensor simulators in ``simulators.py``. The
filters never see it directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from navfusion.data.coordinate_transforms import geodetic_to_enu

OXTS_FIELDS: tuple[str, ...] = (
    "lat", "lon", "alt",            # WGS84 position [deg, deg, m]
    "roll", "pitch", "yaw",         # [rad]; yaw: 0 = east, counter-clockwise positive
    "vn", "ve",                     # velocity north / east [m/s]
    "vf", "vl", "vu",               # velocity forward / left (parallel to ground) / up [m/s]
    "ax", "ay", "az",               # body-frame specific force [m/s^2], includes gravity tilt
    "af", "al", "au",               # forward / left / up acceleration in the level frame [m/s^2]
    "wx", "wy", "wz",               # body-frame angular rate [rad/s]
    "wf", "wl", "wu",               # angular rate about forward / left / up axes [rad/s]
    "pos_accuracy", "vel_accuracy",
    "navstat", "numsats", "posmode", "velmode", "orimode",
)  # fmt: skip
_FIELD_INDEX = {name: i for i, name in enumerate(OXTS_FIELDS)}

# Drive split. Hyperparameters (filter Q/R, LSTM settings) are tuned on the validation drive only;
# the test drives are touched once, for the final evaluation.
TRAIN_DRIVES: tuple[str, ...] = (
    "2011_09_30_drive_0028",
    "2011_10_03_drive_0027",
    "2011_10_03_drive_0034",
    "2011_09_30_drive_0033",
    "2011_09_30_drive_0034",
)
VAL_DRIVE = "2011_09_30_drive_0018"
TEST_DRIVES: tuple[str, ...] = ("2011_09_30_drive_0020", "2011_10_03_drive_0042")
ALL_DRIVES: tuple[str, ...] = (*TRAIN_DRIVES, VAL_DRIVE, *TEST_DRIVES)


def default_data_root() -> Path:
    """``$NAVFUSION_DATA`` if set, else ``<repo>/data/kitti_raw``, where download_data.sh writes."""
    env = os.environ.get("NAVFUSION_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "kitti_raw"


@dataclass(frozen=True)
class OxtsSequence:
    """One OXTS stream: N timestamps and the 30 raw fields per timestamp.

    Access a field by name, e.g. ``seq["vf"]`` -> (N,) array.
    """

    name: str
    t: np.ndarray  # (N,) seconds since the first sample of the drive
    t0: np.datetime64  # absolute time of the first sample of the drive
    data: np.ndarray  # (N, 30) raw OXTS fields in OXTS_FIELDS order

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, field: str) -> np.ndarray:
        return self.data[:, _FIELD_INDEX[field]]

    def fields(self, *names: str) -> np.ndarray:
        """Stack several fields into an (N, len(names)) array."""
        return self.data[:, [_FIELD_INDEX[n] for n in names]]

    @property
    def imu_body(self) -> np.ndarray:
        """(N, 6) clean body-frame ``[ax, ay, az, wx, wy, wz]``, what a strapdown IMU measures."""
        return self.fields("ax", "ay", "az", "wx", "wy", "wz")

    @property
    def imu_planar(self) -> np.ndarray:
        """(N, 3) clean level-frame ``[af, al, wu]``: the 2D vehicle-dynamics inputs.

        ``af``/``al`` are accelerations parallel to the road surface with gravity removed, and
        ``wu`` is the yaw rate about the local vertical. The body-frame ``ax``/``ay`` instead
        contain g*sin(pitch) / g*sin(roll), which on a 2-3 deg slope is 0.3-0.5 m/s^2, larger
        than the accelerometer bias we want the filter to estimate.
        """
        return self.fields("af", "al", "wu")

    @property
    def origin(self) -> tuple[float, float, float]:
        """Geodetic (lat, lon, alt) of the first sample: the ENU origin for this drive."""
        lat, lon, alt = self.fields("lat", "lon", "alt")[0]
        return float(lat), float(lon), float(alt)

    def segment(self, start: int, stop: int) -> OxtsSequence:
        """Samples [start, stop). ``t`` keeps the drive's time base (it is not re-zeroed)."""
        return OxtsSequence(self.name, self.t[start:stop], self.t0, self.data[start:stop])


@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth 2D vehicle state in the local ENU frame, for evaluation and labels only."""

    t: np.ndarray  # (N,) [s]
    pos: np.ndarray  # (N, 2) east, north [m]
    vel: np.ndarray  # (N, 2) east, north velocity [m/s]
    yaw: np.ndarray  # (N,) heading [rad], 0 = east, counter-clockwise positive
    vf: np.ndarray  # (N,) forward body velocity [m/s]
    vl: np.ndarray  # (N,) leftward body velocity [m/s]

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.vel[:, 0], self.vel[:, 1])


def ground_truth(
    seq: OxtsSequence, origin: tuple[float, float, float] | None = None
) -> GroundTruth:
    """Convert an OXTS sequence to a 2D ENU ground truth.

    ``origin`` defaults to the first OXTS sample. The origin is only a fixed datum for the local
    frame; it carries no information the filter could exploit. The filter itself is initialised
    from the first *noisy* GPS fix.

    OXTS yaw is measured from local east at the vehicle, while our ENU axes are fixed at the
    origin. The two differ by the meridian convergence, ~sin(lat) * delta_lon, which over a few km
    is ~1e-4 rad and negligible.
    """
    origin = seq.origin if origin is None else origin
    enu = geodetic_to_enu(seq["lat"], seq["lon"], seq["alt"], origin)
    return GroundTruth(
        t=seq.t.copy(),
        pos=enu[:, :2],
        vel=seq.fields("ve", "vn"),
        yaw=seq["yaw"].copy(),
        vf=seq["vf"].copy(),
        vl=seq["vl"].copy(),
    )


def _parse_timestamps(path: Path) -> np.ndarray:
    """``2011-09-30 11:54:39.239671138`` lines -> datetime64[ns] array (nanosecond precision)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return np.array([ln.replace(" ", "T") for ln in lines], dtype="datetime64[ns]")


def load_oxts(oxts_dir: str | Path, name: str | None = None) -> OxtsSequence:
    """Parse one ``oxts/`` directory (``timestamps.txt`` + ``data/*.txt``)."""
    oxts_dir = Path(oxts_dir)
    stamps = _parse_timestamps(oxts_dir / "timestamps.txt")
    files = sorted((oxts_dir / "data").glob("*.txt"))
    if len(files) != len(stamps):
        raise ValueError(
            f"{oxts_dir}: {len(stamps)} timestamps but {len(files)} data files"
        )
    # One line per file; loadtxt on the list of lines parses them in a single vectorised call.
    data = np.loadtxt([f.read_text() for f in files], ndmin=2)
    if data.shape[1] != len(OXTS_FIELDS):
        raise ValueError(f"{oxts_dir}: expected {len(OXTS_FIELDS)} fields, got {data.shape[1]}")
    t = (stamps - stamps[0]).astype("timedelta64[ns]").astype(np.int64) * 1e-9
    return OxtsSequence(name=name or oxts_dir.parent.name, t=t, t0=stamps[0], data=data)


def load_drive(drive: str, root: str | Path | None = None) -> OxtsSequence:
    """Load a drive by name, e.g. ``load_drive("2011_09_30_drive_0018")``."""
    root = Path(root) if root is not None else default_data_root()
    oxts_dir = root / drive[:10] / f"{drive}_sync" / "oxts"
    if not oxts_dir.is_dir():
        raise FileNotFoundError(f"{oxts_dir} not found; run `bash data/download_data.sh` first")
    return load_oxts(oxts_dir, name=drive)
