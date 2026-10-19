#!/usr/bin/env bash
# Download the KITTI Raw drives NavFusion uses: OXTS (GPS/IMU) only.
#
# KITTI has no OXTS-only download, so fetch_kitti_oxts.py reads the remote sync zips with
# HTTP range requests and pulls out just the oxts/ folders (~40 MB total instead of ~91 GB).
# Result: data/kitti_raw/<date>/<date>_drive_XXXX_sync/oxts/... (parsed by our own NumPy loader).
# Safe to re-run: drives that are already present are skipped.
#
# KITTI Raw is licensed CC BY-NC-SA 3.0 (Geiger et al., IJRR 2013).
set -euo pipefail

cd "$(dirname "$0")/.."

DRIVES=(
    # train
    2011_09_30_drive_0028
    2011_10_03_drive_0027
    2011_10_03_drive_0034
    2011_09_30_drive_0033
    2011_09_30_drive_0034
    # validation: the only drive Q/R and LSTM hyperparameters are tuned on
    2011_09_30_drive_0018
    # test
    2011_09_30_drive_0020
    2011_10_03_drive_0042
)

python3 data/fetch_kitti_oxts.py --out data/kitti_raw "${DRIVES[@]}"
