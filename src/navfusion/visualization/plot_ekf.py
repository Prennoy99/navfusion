"""Act 2 figure: the 8-state EKF with full GPS, against ground truth.

    python -m navfusion.visualization.plot_ekf                       # validation drive
    python -m navfusion.visualization.plot_ekf 2011_09_30_drive_0028 --seeds 10

The filter sees only the simulated 1 Hz GPS (sigma = 2.5 m) and the MEMS-corrupted IMU. The
script prints, per noise seed and on average:

* position RMSE (ATE) of the EKF and of the raw GPS fixes, and the yaw RMSE;
* consistency: the average position NEES (squared error weighted by the filter's own position
  covariance, ideal = 2 for a 2D position) and GPS-innovation NIS (ideal = 2);
* the final bias estimates against the biases injected by the IMU simulator.

The first 30 s after initialisation are excluded from the statistics (convergence transient).
The figure shows seed 0: trajectory, position error with the filter's own 2-sigma bound, and the
three bias estimates converging toward the injected values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from navfusion.data.coordinate_transforms import wrap_angle  # noqa: E402
from navfusion.data.kitti_loader import VAL_DRIVE, ground_truth, load_drive  # noqa: E402
from navfusion.data.simulators import simulate_sensors  # noqa: E402
from navfusion.filters.ekf import run_ekf  # noqa: E402
from navfusion.visualization import style  # noqa: E402

SETTLE_S = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("drive", nargs="?", default=VAL_DRIVE)
    parser.add_argument("--seeds", type=int, default=5, help="number of noise seeds (0..n-1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure path (default: assets/ekf_<drive>.png)")
    args = parser.parse_args()

    seq = load_drive(args.drive)
    gt = ground_truth(seq)

    print(f"EKF with full 1 Hz GPS on {args.drive}, {args.seeds} seeds "
          f"(statistics exclude the first {SETTLE_S:.0f} s after initialisation)")
    print(f"  {'seed':>4} {'ATE[m]':>7} {'GPS[m]':>7} {'yaw[deg]':>8} {'NEES':>5} {'NIS':>5}"
          f"   {'b_ax est/true':>15}  {'b_ay est/true':>15}  {'b_gz est/true [deg/s]':>21}")
    rows, runs = [], []
    for seed in range(args.seeds):
        sim = simulate_sensors(seq, seed)
        res = run_ekf(sim.t, sim.imu.planar, sim.gps)
        runs.append((sim, res))
        k = slice(int(np.searchsorted(gt.t, gt.t[res.k0] + SETTLE_S)), None)
        err = res.x[k, :2] - gt.pos[k]
        ate = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        late = sim.gps.t >= gt.t[k.start]
        gps_err = sim.gps.pos[late] - gt.pos[sim.gps.idx[late]]
        gps_rmse = np.sqrt(np.mean(np.sum(gps_err**2, axis=1)))
        yaw = np.rad2deg(np.sqrt(np.mean(wrap_angle(res.x[k, 4] - gt.yaw[k]) ** 2)))
        nees = np.mean(np.einsum("ni,nij,nj->n", err, np.linalg.inv(res.P[k, :2, :2]), err))
        nis = np.nanmean(res.gps_nis[late])
        b, bt = res.x[-1, 5:8], sim.imu.planar_bias
        rows.append([ate, gps_rmse, yaw, nees, nis])
        print(f"  {seed:>4} {ate:7.2f} {gps_rmse:7.2f} {yaw:8.2f} {nees:5.2f} {nis:5.2f}"
              f"   {b[0]:+.3f}/{bt[0]:+.3f}  {b[1]:+.3f}/{bt[1]:+.3f}  "
              f"{np.rad2deg(b[2]):+.3f}/{np.rad2deg(bt[2]):+.3f}")
    m, s = np.mean(rows, axis=0), np.std(rows, axis=0)
    print(f"  mean ATE {m[0]:.2f} ± {s[0]:.2f} m (raw GPS {m[1]:.2f} m), yaw {m[2]:.2f} ± "
          f"{s[2]:.2f} deg, NEES {m[3]:.2f} (ideal 2), NIS {m[4]:.2f} (ideal 2)")

    plot(args.drive, gt, *runs[0], args.out or Path("assets") / f"ekf_{args.drive}.png")


def plot(drive, gt, sim, res, out: Path) -> None:
    fig = plt.figure(figsize=(13, 8.6))
    grid = fig.add_gridspec(2, 6, height_ratios=(1.5, 1), hspace=0.38, wspace=0.9)
    ax_traj = fig.add_subplot(grid[0, :3])
    ax_err = fig.add_subplot(grid[0, 3:])
    ax_bias = [fig.add_subplot(grid[1, 2 * i:2 * i + 2]) for i in range(3)]
    valid = slice(res.k0, None)
    t = gt.t[valid]

    # Trajectory
    ax_traj.plot(*sim.gps.pos.T, ".", color=style.GPS, ms=3.5, label="Noisy GPS fixes (1 Hz)")
    ax_traj.plot(*gt.pos.T, color=style.TRUTH, lw=1.5, label="Ground truth")
    ax_traj.plot(*res.x[valid, :2].T, color=style.EKF, lw=1.6, label="EKF")
    ax_traj.set_aspect("equal", adjustable="datalim")
    style.style_axes(ax_traj, "East [m]", "North [m]", f"{drive}: EKF with full GPS (seed 0)")
    ax_traj.legend(loc="best", frameon=False, fontsize=9)

    # Position error with the filter's own uncertainty
    err = np.linalg.norm(res.x[valid, :2] - gt.pos[valid], axis=1)
    sigma_r = np.sqrt(res.P[valid, 0, 0] + res.P[valid, 1, 1])  # RMS radial 1-sigma
    gps_err = np.linalg.norm(sim.gps.pos - gt.pos[sim.gps.idx], axis=1)
    ax_err.plot(sim.gps.t, gps_err, ".", color=style.GPS, ms=3.5, label="Raw GPS fix error")
    ax_err.fill_between(t, 0, 2 * sigma_r, color=style.EKF, alpha=0.15, lw=0,
                        label="EKF 2σ (own covariance)")
    ax_err.plot(t, err, color=style.EKF, lw=1.4, label="EKF error")
    ax_err.set_xlim(0, gt.t[-1])
    ax_err.set_ylim(0, None)
    style.style_axes(ax_err, "Time [s]", "Horizontal position error [m]",
                     "Position error: EKF vs raw GPS")
    ax_err.legend(loc="upper right", frameon=False, fontsize=9)

    # Bias convergence
    scale = (1.0, 1.0, np.rad2deg(1.0))
    names = ("Forward accel bias b_ax", "Lateral accel bias b_ay", "Gyro bias b_gz")
    units = ("m/s²", "m/s²", "deg/s")
    for i, ax in enumerate(ax_bias):
        est = res.x[valid, 5 + i] * scale[i]
        sd = res.std[valid, 5 + i] * scale[i]
        true = sim.imu.planar_bias[i] * scale[i]
        ax.fill_between(t, est - 2 * sd, est + 2 * sd, color=style.EKF, alpha=0.15, lw=0)
        ax.plot(t, est, color=style.EKF, lw=1.4)
        ax.axhline(true, color=style.INK, lw=1.0, ls="--")
        ax.annotate("injected", (t[-1], true), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", color=style.INK_MUTED, fontsize=8)
        ax.set_xlim(0, gt.t[-1])
        lim = 3 * abs(true) + 1e-9
        ax.set_ylim(true - lim, true + lim)
        style.style_axes(ax, "Time [s]", units[i], names[i])
    ax_bias[0].text(0.02, 0.04, "estimate ± 2σ", transform=ax_bias[0].transAxes,
                    color=style.EKF, fontsize=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"  plot saved to {out}")


if __name__ == "__main__":
    main()
