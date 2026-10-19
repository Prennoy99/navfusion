"""Act 1 figure: how fast a MEMS IMU alone drifts away from the truth.

    python -m navfusion.visualization.plot_dead_reckoning                 # validation drive
    python -m navfusion.visualization.plot_dead_reckoning 2011_09_30_drive_0028 --seeds 10

Dead reckoning runs over the whole drive without any GPS, for several noise seeds. It starts
from the *true* initial state (position, velocity, heading), which is the most favourable start
possible, so all of the drift comes from integrating the corrupted IMU. (In the outage evaluation,
dead reckoning instead restarts from the EKF's estimate at the start of each outage, so that all
methods start every outage from the same state.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from navfusion.data.kitti_loader import VAL_DRIVE, ground_truth, load_drive  # noqa: E402
from navfusion.data.simulators import simulate_sensors  # noqa: E402
from navfusion.filters.dead_reckoning import dead_reckon  # noqa: E402
from navfusion.visualization import style  # noqa: E402

REPORT_TIMES = (10.0, 30.0, 60.0, 120.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("drive", nargs="?", default=VAL_DRIVE)
    parser.add_argument("--seeds", type=int, default=5, help="number of noise seeds (0..n-1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure path (default: assets/dead_reckoning_<drive>.png)")
    args = parser.parse_args()

    seq = load_drive(args.drive)
    gt = ground_truth(seq)
    x0 = np.r_[gt.pos[0], gt.vel[0], gt.yaw[0]]
    runs = [dead_reckon(gt.t, simulate_sensors(seq, seed).imu.planar, x0)
            for seed in range(args.seeds)]
    errors = np.array([np.linalg.norm(r[:, :2] - gt.pos, axis=1) for r in runs])

    times = [T for T in REPORT_TIMES if T < gt.t[-1]] + [gt.t[-1]]
    idx = [int(np.searchsorted(gt.t, T)) for T in times]
    print(f"Dead reckoning on {args.drive}, {args.seeds} seeds, perfect initial state")
    print("  position error [m], mean ± std over seeds")
    for T, k in zip(times, idx, strict=True):
        print(f"  t = {T:6.1f} s   {errors[:, k].mean():8.1f} ± {errors[:, k].std():6.1f}")

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(12, 5.4), width_ratios=(1.15, 1))
    for seed, r in reversed(list(enumerate(runs))):
        first = seed == 0
        ax_traj.plot(r[:, 0], r[:, 1], color=style.DEAD_RECKONING, lw=2 if first else 1,
                     alpha=1.0 if first else 0.35,
                     label="Dead reckoning, seed 0" if first else
                     ("Dead reckoning, other seeds" if seed == 1 else None))
        ax_err.plot(gt.t, errors[seed], color=style.DEAD_RECKONING, lw=2 if first else 1,
                    alpha=1.0 if first else 0.35)
    ax_traj.plot(gt.pos[:, 0], gt.pos[:, 1], color=style.TRUTH, lw=1.5, label="Ground truth")
    ax_traj.plot(*gt.pos[0], "o", color=style.INK, ms=6)
    ax_traj.annotate("start", gt.pos[0], xytext=(6, -14), textcoords="offset points",
                     color=style.INK)
    ax_traj.set_aspect("equal", adjustable="datalim")
    style.style_axes(ax_traj, "East [m]", "North [m]",
                     f"{args.drive}: IMU only, no GPS")
    handles, labels = ax_traj.get_legend_handles_labels()
    order = [labels.index(lbl) for lbl in
             ("Ground truth", "Dead reckoning, seed 0", "Dead reckoning, other seeds")
             if lbl in labels]
    ax_traj.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="best", frameon=False, fontsize=9)

    k10 = int(np.searchsorted(gt.t, 10.0))
    ax_err.annotate(f"{errors[0, k10]:.0f} m after 10 s", (gt.t[k10], errors[0, k10]),
                    xytext=(40, 90), textcoords="offset points", color=style.INK, fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": style.INK_MUTED, "lw": 0.8})
    ax_err.annotate(f"{errors[0, -1]:.0f} m", (gt.t[-1], errors[0, -1]), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=style.INK, fontsize=9)
    ax_err.set_xlim(0, gt.t[-1])
    ax_err.set_ylim(0, None)
    style.style_axes(ax_err, "Time [s]", "Horizontal position error [m]",
                     f"Position error, {args.seeds} noise seeds")
    fig.tight_layout()

    out = args.out or Path("assets") / f"dead_reckoning_{args.drive}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"  plot saved to {out}")


if __name__ == "__main__":
    main()
