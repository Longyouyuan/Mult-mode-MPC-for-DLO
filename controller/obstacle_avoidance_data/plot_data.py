from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

# ===================== Paths =====================
THIS_FILE = Path(__file__).resolve()
DATA_DIR = THIS_FILE.parent

# Expected files in the current directory:
GOAL_FILE = DATA_DIR / "goal_traj.npy"
M2PC_FILE = DATA_DIR / "m2pc_rope_traj.npy"
SVMPC_FILE = DATA_DIR / "svmpc_rope_traj.npy"
DBSCAN_FILE = DATA_DIR / "dbscan_mpc_rope_traj.npy"

# ===================== Paper style =====================
DPI = 600
FIG_SCALE = 1.45
FIGSIZE_TRAJ = (2.25 * FIG_SCALE, 1.85 * FIG_SCALE)
FIGSIZE_ERR = (2.35 * FIG_SCALE, 1.65 * FIG_SCALE)
FONT_SIZE = 8
TICK_SIZE = 8
LEGEND_SIZE = 7
LINE_WIDTH = 1.35

DT = 0.001
CTR_PERIOD = 25
TIP_IDX = -1  # rope tip index. Change to 0 if your saved rope tip is at index 0.

# Obstacle cylinder (world x, y coordinates)
CYL_X, CYL_Y = 0.0, 0.9
CYL_RADIUS = 0.06


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
        "axes.titlesize": FONT_SIZE + 1,
    })


def load_npy(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def normalize_goal(goal):
    goal = np.asarray(goal)
    if goal.ndim == 3:
        goal = goal[0]
    if goal.ndim != 2 or goal.shape[-1] != 3:
        raise ValueError(f"goal_traj should have shape (T, 3), got {goal.shape}")
    return goal.astype(np.float64)


def normalize_rope_traj(traj, tip_idx=TIP_IDX):
    traj = np.asarray(traj)
    if traj.ndim == 4:
        traj = traj[:, 0]
    if traj.ndim == 3 and traj.shape[-1] == 3:
        return traj[:, tip_idx, :].astype(np.float64)
    if traj.ndim == 2 and traj.shape[-1] == 3:
        return traj.astype(np.float64)
    raise ValueError(f"rope trajectory should have shape (T, P, 3) or (T, 3), got {traj.shape}")


def align_to_goal(goal, *tips):
    t_len = min([goal.shape[0], *[tip.shape[0] for tip in tips]])
    return (goal[:t_len],) + tuple(tip[:t_len] for tip in tips)


def set_data_bounds_2d(ax, *point_sets, margin=0.10):
    """Fit axis limits to actual data extent + margin. horizontal=y, vertical=x."""
    pts = np.concatenate([p[:, :2] for p in point_sets], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    ax.set_xlim(y_min - margin, y_max + margin)
    ax.set_ylim(x_min - margin - 0.1, x_max + margin)
    ax.set_aspect("equal", adjustable="box")


def save_figure(fig, stem):
    fig.savefig(DATA_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(DATA_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)


def plot_xy_trajectory(goal, m2pc, svmpc, dbscan):
    fig, ax = plt.subplots(figsize=FIGSIZE_TRAJ)

    # zorder: grid(0) < goal(1) < obstacle(2) < method curves(3-5) < start(6)
    ax.plot(goal[:, 1], goal[:, 0], "--", color="0.25", linewidth=LINE_WIDTH, label="Goal", zorder=1)

    # Cylinder: brown, above goal line, below method curves
    cyl = Circle((CYL_Y, CYL_X), CYL_RADIUS,
                 facecolor=(0.48, 0.26, 0.12, 1.0), edgecolor=(0.30, 0.15, 0.06),
                 linewidth=0.6, zorder=2, label="Obstacle")
    ax.add_patch(cyl)

    ax.plot(m2pc[:, 1], m2pc[:, 0], "-", color="#d62728", linewidth=LINE_WIDTH + 0.25, label="M2PC", zorder=5)
    ax.plot(svmpc[:, 1], svmpc[:, 0], "-", color="#1f77b4", linewidth=LINE_WIDTH, label="SVMPC", zorder=4)
    ax.plot(dbscan[:, 1], dbscan[:, 0], "-", color="#2ca02c", linewidth=LINE_WIDTH, label="C-MPPI", zorder=3)
    ax.scatter(goal[0, 1], goal[0, 0], s=18, color="black", linewidths=0, label="Start", zorder=6)

    ax.set_xlabel("$y$ (m)", labelpad=3)
    ax.set_ylabel("$x$ (m)", labelpad=0)
    ax.tick_params(axis="both", which="major", pad=0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, linewidth=0.3, alpha=0.45)
    set_data_bounds_2d(ax, goal, m2pc, svmpc, dbscan, margin=0.06)
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)
    # Legend: two separate legend objects to guarantee exact row layout
    obstacle_proxy = Line2D([0], [0], marker="o", linestyle="None",
                            markerfacecolor=(0.48, 0.26, 0.12, 1.0),
                            markeredgecolor=(0.30, 0.15, 0.06),
                            markeredgewidth=0.6, markersize=6)
    hl = {lbl: h for h, lbl in zip(*ax.get_legend_handles_labels())}

    # Row 1: Goal, M2PC, SVMPC, C-MPPI
    row1_handles = [hl["Goal"], hl["M2PC"], hl["SVMPC"], hl["C-MPPI"], obstacle_proxy]
    row1_labels  = ["Goal", "M2PC", "SVMPC", "C-MPPI", "Obstacle"]
    # Row 2: Start, Obstacle
    row2_handles = [hl["Start"]]
    row2_labels  = ["Start"]

    leg1 = ax.legend(row1_handles, row1_labels,
                     loc="upper left", bbox_to_anchor=(0.0, 1.01),
                     frameon=False, ncol=5,
                     handlelength=1.4, handletextpad=0.3,
                     borderpad=0.0, labelspacing=0.2, columnspacing=0.7)
    ax.add_artist(leg1)
    ax.legend(row2_handles, row2_labels,
              loc="upper left", bbox_to_anchor=(0.0, 0.94),
              frameon=False, ncol=2,
              handlelength=1.4, handletextpad=0.3,
              borderpad=0.0, labelspacing=0.2, columnspacing=0.7)

    fig.tight_layout(pad=0.25)
    save_figure(fig, "obstacle_avoidance_xy_trajectory_compare")
    return fig


def plot_tracking_error(goal, m2pc, svmpc, dbscan):
    err_m2pc = np.linalg.norm(m2pc - goal, axis=1) * 100.0
    err_svmpc = np.linalg.norm(svmpc - goal, axis=1) * 100.0
    err_dbscan = np.linalg.norm(dbscan - goal, axis=1) * 100.0
    time_axis = np.arange(goal.shape[0]) * DT * CTR_PERIOD

    fig, ax = plt.subplots(figsize=FIGSIZE_ERR)
    ax.plot(time_axis, err_m2pc, color="#d62728", linewidth=LINE_WIDTH + 0.25, label="M2PC")
    ax.plot(time_axis, err_svmpc, color="#1f77b4", linewidth=LINE_WIDTH, label="SV-MPC")
    ax.plot(time_axis, err_dbscan, color="#2ca02c", linewidth=LINE_WIDTH, label="DBSCAN-MPC")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (cm)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, linewidth=0.3, alpha=0.45)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), handlelength=1.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.25)
    save_figure(fig, "obstacle_avoidance_tracking_error_compare")
    return fig


def main():
    set_paper_style()

    goal = normalize_goal(load_npy(GOAL_FILE))
    m2pc = normalize_rope_traj(load_npy(M2PC_FILE))
    svmpc = normalize_rope_traj(load_npy(SVMPC_FILE))
    dbscan = normalize_rope_traj(load_npy(DBSCAN_FILE))

    goal, m2pc, svmpc, dbscan = align_to_goal(goal, m2pc, svmpc, dbscan)

    plot_xy_trajectory(goal, m2pc, svmpc, dbscan)
    plot_tracking_error(goal, m2pc, svmpc, dbscan)

    print(f"Saved figures to: {DATA_DIR}")
    print("- obstacle_avoidance_xy_trajectory_compare.pdf/png")
    print("- obstacle_avoidance_tracking_error_compare.pdf/png")


if __name__ == "__main__":
    main()
