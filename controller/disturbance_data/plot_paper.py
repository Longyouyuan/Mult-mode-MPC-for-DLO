from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


THIS_FILE = Path(__file__).resolve()
DATA_DIR = THIS_FILE.parent

REPO_ROOT = None
for candidate in THIS_FILE.parents:
    if (candidate / "common").exists() and (candidate / "controller").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not find repo root containing common/ and controller/.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import plot_tip_vs_goal_and_error


DPI = 600
FIG_SCALE = 1.3
FIGSIZE_HALF_COLUMN = (1.72 * FIG_SCALE, 1.52 * FIG_SCALE)
# Increase all font sizes for better readability
FONT_SIZE = 7
TICK_SIZE = 9
LEGEND_SIZE = 8
LINE_WIDTH = 1.2


def load_disturbance_data():
    data_path = DATA_DIR / "robustness_disturbance_data.npz"

    if data_path.exists():
        data = np.load(data_path)
        return (
            data["rope_trajectory"],
            data["goal_trajectory"],
            float(data["dt"]),
            int(data["ctr_period"]),
            data["disturbance_points"],
        )

    return (
        np.load(DATA_DIR / "rope_trajectory.npy"),
        np.load(DATA_DIR / "goal_trajectory.npy"),
        0.001,
        25,
        np.load(DATA_DIR / "disturbance_points.npy"),
    )


def normalize_data(pos_history, goal_traj, disturbance_points, tip_idx=-1):
    ph = np.asarray(pos_history)
    gt = np.asarray(goal_traj)
    dist = np.asarray(disturbance_points)

    if ph.ndim == 4:
        ph = ph[0]
    if gt.ndim == 3:
        gt = gt[0]

    if ph.ndim != 3 or ph.shape[-1] != 3:
        raise ValueError(f"pos_history should have shape (T, P, 3), got {ph.shape}.")
    if gt.ndim != 2 or gt.shape[-1] != 3:
        raise ValueError(f"goal_traj should have shape (T, 3), got {gt.shape}.")
    if dist.ndim != 2 or dist.shape[-1] != 3:
        raise ValueError(f"disturbance_points should have shape (N, 3), got {dist.shape}.")

    t_len = min(ph.shape[0], gt.shape[0])
    ph = ph[:t_len]
    gt = gt[:t_len]
    tip = ph[:, tip_idx, :]
    err_norm = np.linalg.norm(tip - gt, axis=1)

    return tip, gt, err_norm, dist


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": LINE_WIDTH,
        # Make title font larger if used
        "axes.titlesize": FONT_SIZE + 2,
    })


def set_equal_2d_axes(ax, *point_sets):
    all_pts = np.concatenate(point_sets, axis=0)
    xy_min = all_pts[:, :2].min(axis=0)
    xy_max = all_pts[:, :2].max(axis=0)
    center = 0.5 * (xy_min + xy_max)
    half = 0.5 * np.max(xy_max - xy_min)
    half = max(float(half), 1e-6)

    # Match the top-down orientation used in the reference: horizontal=y, vertical=x.
    ax.set_xlim(center[1] - half - 0.1, center[1] + half + 0.1)
    ax.set_ylim(center[0] - half - 0.05, 0.8-0.05)
    ax.set_aspect("equal", adjustable="box")


def save_figure(fig, stem):
    fig.savefig(DATA_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(DATA_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)


def plot_trajectory_3d(tip, goal, disturbance_points):
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF_COLUMN)

    ax.plot(
        goal[:, 1], goal[:, 0],
        "--", color="0.25", linewidth=LINE_WIDTH, label="Goal", zorder=1
    )
    ax.plot(
        tip[:, 1], tip[:, 0],
        "-", color="#d62728", linewidth=LINE_WIDTH, label="Tip", zorder=2
    )
    ax.scatter(
        goal[0, 1], goal[0, 0],
        s=16, color="#2ca02c", linewidths=0, label="Start", zorder=3
    )
    ax.scatter(
        disturbance_points[:, 1], disturbance_points[:, 0],
        marker="o",  # 👈 改成圆点（更容易看）
        s=48,  # 👈 放大
        facecolor="#0072B2",  # 填充色
        edgecolor="white",  # 👈 白边（关键！）
        linewidths=1.2,
        label="Disturbance",
        zorder=5  # 👈 压在最上面
    )

    ax.set_xlabel("$y$ (m)", labelpad=3)
    ax.set_ylabel("$x$ (m)", labelpad=0)
    ax.tick_params(axis="both", which="major", pad=0)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.grid(True, linewidth=0.3, alpha=0.45)
    set_equal_2d_axes(ax, tip, goal, disturbance_points)
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.00, 1.02), frameon=False, ncol=2,
        handlelength=2.0, handletextpad=0.3, borderpad=0.0, labelspacing=0.2
    )

    fig.tight_layout(pad=0.25)
    save_figure(fig, "disturbance_trajectory_3d")
    return fig


def plot_error(time_axis, err_norm):
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF_COLUMN)

    ax.plot(time_axis, err_norm * 100.0, color="#d62728", linewidth=LINE_WIDTH)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (cm)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, linewidth=0.3, alpha=0.45)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.25)
    save_figure(fig, "disturbance_tracking_error")
    return fig


def main():
    pos_history, Goal_run, dt, ctr_period, disturbance_points = load_disturbance_data()
    tip, goal, err_norm, disturbance_points = normalize_data(
        pos_history, Goal_run, disturbance_points, tip_idx=-1
    )
    time_axis = np.arange(err_norm.shape[0]) * dt * ctr_period

    set_paper_style()
    plot_trajectory_3d(tip, goal, disturbance_points)
    plot_error(time_axis, err_norm)
    print(f"Saved paper figures to: {DATA_DIR}")
    plot_tip_vs_goal_and_error(pos_history, Goal_run, dt, ctr_period,
                               dist=disturbance_points)


if __name__ == "__main__":
    main()
