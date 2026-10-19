"""OXTS parsing on a synthetic oxts/ directory with the real KITTI file layout."""

import numpy as np
import pytest

from navfusion.data.coordinate_transforms import enu_to_geodetic
from navfusion.data.kitti_loader import (
    OXTS_FIELDS,
    ground_truth,
    load_drive,
    load_oxts,
)

from .conftest import ORIGIN


def write_oxts(root, rows, stamps):
    oxts = root / "2011_09_30" / "2011_09_30_drive_9999_sync" / "oxts"
    (oxts / "data").mkdir(parents=True)
    (oxts / "timestamps.txt").write_text("\n".join(stamps) + "\n")
    for i, row in enumerate(rows):
        (oxts / "data" / f"{i:010d}.txt").write_text(" ".join(f"{v:.10g}" for v in row) + "\n")
    return oxts


STAMPS = ["2011-09-30 11:54:39.239671138", "2011-09-30 11:54:39.339674871",
          "2011-09-30 11:54:39.449844116"]


def rows(n=3):
    r = np.zeros((n, len(OXTS_FIELDS)))
    r[:, 0], r[:, 1], r[:, 2] = ORIGIN
    r[:, OXTS_FIELDS.index("vf")] = [1.0, 2.0, 3.0][:n]
    r[:, OXTS_FIELDS.index("posmode")] = 4
    return r


def test_load_oxts_parses_fields_and_time(tmp_path):
    seq = load_oxts(write_oxts(tmp_path, rows(), STAMPS))
    assert seq.data.shape == (3, 30)
    assert seq.name == "2011_09_30_drive_9999_sync"
    np.testing.assert_allclose(seq["vf"], [1.0, 2.0, 3.0])
    # nanosecond timestamps survive: 0.100003733 s and 0.210172978 s after the first sample
    np.testing.assert_allclose(seq.t, [0.0, 0.100003733, 0.210172978], atol=1e-9)
    assert seq.imu_body.shape == (3, 6) and seq.imu_planar.shape == (3, 3)


def test_load_drive_by_name(tmp_path):
    write_oxts(tmp_path, rows(), STAMPS)
    seq = load_drive("2011_09_30_drive_9999", root=tmp_path)
    assert seq.name == "2011_09_30_drive_9999" and len(seq) == 3


def test_load_drive_missing_gives_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        load_drive("2011_09_30_drive_0001", root=tmp_path)


def test_timestamp_count_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="timestamps"):
        load_oxts(write_oxts(tmp_path, rows(2), STAMPS))


def test_ground_truth_on_circle(circle):
    """Integrate the circle's ENU velocity into lat/lon, then check the ENU track comes back."""
    gt0 = ground_truth(circle)
    # build lat/lon from the analytic circle (radius 50 m, centre at (0, 50) m)
    r = 50.0
    e, n = r * np.sin(gt0.yaw), r * (1 - np.cos(gt0.yaw))
    geo = enu_to_geodetic(np.stack([e, n, np.zeros_like(e)], axis=-1), ORIGIN)
    data = circle.data.copy()
    data[:, :3] = geo
    seq = type(circle)(circle.name, circle.t, circle.t0, data)
    gt = ground_truth(seq)
    np.testing.assert_allclose(gt.pos, np.stack([e, n], axis=-1), atol=1e-6)
    np.testing.assert_allclose(gt.speed, 10.0)
    # heading is the direction of the ENU velocity
    np.testing.assert_allclose(np.arctan2(gt.vel[:, 1], gt.vel[:, 0]),
                               np.arctan2(np.sin(gt.yaw), np.cos(gt.yaw)), atol=1e-9)
