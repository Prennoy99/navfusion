"""Sanity report for the KITTI OXTS data and the sensor simulators.

    python -m navfusion.data.inspect_drive                  # report all drives, plot the val drive
    python -m navfusion.data.inspect_drive 2011_10_03_drive_0042 --seed 3

For every drive: sample count, duration, rate, timestamp gaps and the gravity-leak checks. For the
selected drive it also simulates the 1 Hz GPS and MEMS IMU, prints array shapes and the empirical
noise levels, and saves the ground-truth track with the noisy GPS fixes in ENU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from navfusion.data.integrity import check_gravity_leak, find_timestamp_gaps  # noqa: E402
from navfusion.data.kitti_loader import (  # noqa: E402
    ALL_DRIVES,
    TEST_DRIVES,
    TRAIN_DRIVES,
    VAL_DRIVE,
    ground_truth,
    load_drive,
)
from navfusion.data.simulators import simulate_sensors  # noqa: E402
from navfusion.visualization import style  # noqa: E402


def _split_of(drive: str) -> str:
    if drive in TRAIN_DRIVES:
        return "train"
    return "val" if drive == VAL_DRIVE else "test" if drive in TEST_DRIVES else "?"


def _fmt(v: np.ndarray | None) -> str:
    return "      n/a      " if v is None else f"{v[0]:+.3f} {v[1]:+.3f}"


def report_all(drives: tuple[str, ...]) -> None:
    print("Integrity checks (accelerations in m/s^2; af/al must be ~0, ax/ay contain gravity)")
    print(
        f"{'drive':<23}{'split':<6}{'N':>6}{'dur[s]':>8}{'Hz':>6}{'gaps':>5}"
        f"{'still':>6}  {'standstill af al':>16}  {'kinem. af al':>15}  {'kinem. ax ay':>15}  ok"
    )
    for drive in drives:
        seq = load_drive(drive)
        gaps = find_timestamp_gaps(seq.t)
        g = check_gravity_leak(seq)
        rate = (len(seq) - 1) / seq.t[-1]  # mean rate: KITTI steps alternate 0.10 / 0.11 s
        print(
            f"{drive:<23}{_split_of(drive):<6}{len(seq):>6}{seq.t[-1]:>8.1f}{rate:>6.2f}"
            f"{len(gaps):>5}{g.n_standstill:>6}  {_fmt(g.standstill_mean):>16}  "
            f"{_fmt(g.kinematic_bias):>15}  {_fmt(g.kinematic_bias_body):>15}  "
            f"{'yes' if g.ok else 'NO'}"
        )


def report_drive(drive: str, seed: int, out: Path) -> None:
    seq = load_drive(drive)
    gt = ground_truth(seq)
    sim = simulate_sensors(seq, seed)

    print(f"\n{drive}, seed {seed}")
    print(f"  oxts data       {seq.data.shape}   t {seq.t.shape}   start {seq.t0}")
    print(f"  truth pos/vel   {gt.pos.shape} {gt.vel.shape}   path length "
          f"{np.sum(np.linalg.norm(np.diff(gt.pos, axis=0), axis=1)):.0f} m")
    print(f"  imu body/planar {sim.imu.body.shape} {sim.imu.planar.shape}")
    print(f"  gps fixes       {sim.gps.pos.shape}   "
          f"mean interval {np.diff(sim.gps.t).mean():.3f} s")
    gps_err = sim.gps.pos - gt.pos[sim.gps.idx]
    print(f"  gps error std   east {gps_err[:, 0].std():.2f} m, north {gps_err[:, 1].std():.2f} m "
          f"(configured {sim.gps.sigma_m} m)")
    b = sim.imu.planar_bias
    print(f"  injected bias   b_ax {b[0]:+.3f} m/s^2, b_ay {b[1]:+.3f} m/s^2, "
          f"b_gz {np.rad2deg(b[2]):+.3f} deg/s")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.4), width_ratios=(1.35, 1))
    zoom_center = gt.pos[len(gt.pos) // 3]
    half = 40.0
    for ax in (ax_full, ax_zoom):
        ax.plot(gt.pos[:, 0], gt.pos[:, 1], color=style.TRUTH, lw=1.5,
                label="Ground truth (RTK OXTS)")
        ax.scatter(sim.gps.pos[:, 0], sim.gps.pos[:, 1], s=16, color=style.GPS,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=f"Simulated GPS (1 Hz, σ = {sim.gps.sigma_m} m)")
        style.style_axes(ax, "East [m]", "North [m]")
    ax_full.set_aspect("equal", adjustable="datalim")
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_full.plot(*gt.pos[0], "o", color=style.INK, ms=6)
    ax_full.annotate("start", gt.pos[0], xytext=(6, 6), textcoords="offset points",
                     color=style.INK)
    ax_full.add_patch(plt.Rectangle(zoom_center - half, 2 * half, 2 * half, fill=False,
                                    ec=style.INK_MUTED, lw=1, ls="--"))
    ax_full.set_title(f"{drive}: full drive in ENU", color=style.INK, loc="left", fontsize=11)
    ax_full.legend(loc="best", frameon=False, fontsize=9)
    ax_zoom.set_xlim(zoom_center[0] - half, zoom_center[0] + half)
    ax_zoom.set_ylim(zoom_center[1] - half, zoom_center[1] + half)
    ax_zoom.set_title("Zoom: 80 m × 80 m", color=style.INK, loc="left", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  plot saved to   {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("drive", nargs="?", default=VAL_DRIVE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="figure path (default: assets/gps_track_<drive>.png)")
    args = parser.parse_args()
    report_all(ALL_DRIVES)
    out = args.out or Path("assets") / f"gps_track_{args.drive}.png"
    report_drive(args.drive, args.seed, out)


if __name__ == "__main__":
    main()
