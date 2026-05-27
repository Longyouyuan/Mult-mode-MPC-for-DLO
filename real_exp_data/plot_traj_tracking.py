from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


THIS_FILE = Path(__file__).resolve()
DATA_ROOT = THIS_FILE.parent / "data"
SAVE_STEM = THIS_FILE.parent / "real_traj_tracking_spid_m2pc_compare"

DPI = 600
FIGSIZE = (2.5, 1.55)  # IEEE single-column width, three panels in one horizontal row.
TIP_INDEX = 8
GOAL_STATS_REFERENCE = "SPiD"

GOAL_COLOR = "0.50"
M2PC_COLOR = "#FF2F2F"
SPID_COLOR = "#2F8CC8"  #'#2ca02c'
START_COLOR = '#2ca02c'

LINE_WIDTH_GOAL = 1.20
LINE_WIDTH_METHOD = 1.0

M2PC_ALPHA = 0.80
SPID_ALPHA = 0.80

DATASETS = [
    {
        "key": "sin",
        "title": "Sinusoid",
        "spid": DATA_ROOT / "spid" / "sin_2.5s" / "paper_data_spid_sin_1.csv",
        "m2pc": DATA_ROOT / "m2pc" / "sin_2.5s" / "paper_data_sin_2.csv",
    },
    {
        "key": "eight",
        "title": "Lemniscate (1 cycle)",
        "spid": DATA_ROOT / "spid" / "eight_4s" / "paper_data_spid_eight_1.csv",
        "m2pc": DATA_ROOT / "m2pc" / "eight_4s" / "paper_data_eight_2.csv",
    },
    {
        "key": "continuous_eight",
        "title": "Lemniscate (3 cycles)",
        "spid": DATA_ROOT / "spid" / "continuous_eight" / "paper_data_spid_continuous_eight_1.csv",
        "m2pc": DATA_ROOT / "m2pc" / "continuous_eight" / "paper_data_continuous_eight_1.csv",
    },
]


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.60,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.1,
        "ytick.major.size": 2.1,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    })


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")


def load_csv(path: Path):
    require_file(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    data = np.atleast_1d(data)

    tip = np.column_stack([
        data[f"rope_p{TIP_INDEX}_x"],
        data[f"rope_p{TIP_INDEX}_y"],
        data[f"rope_p{TIP_INDEX}_z"],
    ])
    goal = np.column_stack([
        data["goal_traj_x"],
        data["goal_traj_y"],
        data["goal_traj_z"],
    ])
    time = np.asarray(data["time"], dtype=np.float64)
    time = time - time[0]

    return {"time": time, "tip": tip, "goal": goal, "path": path}


def rmse_cm(traj: np.ndarray, goal: np.ndarray, dims=slice(0, 2)) -> float:
    n = min(traj.shape[0], goal.shape[0])
    err = traj[:n, dims] - goal[:n, dims]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))) * 100.0)


def mean_error_cm(traj: np.ndarray, goal: np.ndarray, dims=slice(0, 2)) -> float:
    n = min(traj.shape[0], goal.shape[0])
    err = traj[:n, dims] - goal[:n, dims]
    return float(np.mean(np.linalg.norm(err, axis=1)) * 100.0)


def max_error_cm(traj: np.ndarray, goal: np.ndarray, dims=slice(0, 2)) -> float:
    n = min(traj.shape[0], goal.shape[0])
    err = traj[:n, dims] - goal[:n, dims]
    return float(np.max(np.linalg.norm(err, axis=1)) * 100.0)


def trajectory_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    segments = np.diff(points, axis=0)
    return float(np.linalg.norm(segments, axis=1).sum())


def max_speed(points: np.ndarray, time: np.ndarray) -> float:
    if len(points) < 2 or len(time) < 2:
        return 0.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dt = np.diff(time)
    valid = dt > 0
    if not np.any(valid):
        return 0.0

    return float(np.max(segment_lengths[valid] / dt[valid]))


def load_all():
    loaded = []
    for spec in DATASETS:
        spid = load_csv(spec["spid"])
        m2pc = load_csv(spec["m2pc"])
        loaded.append({**spec, "SPiD": spid, "M2PC": m2pc})
    return loaded


def all_xy_points(item):
    return np.concatenate([
        item["SPiD"]["goal"][:, :2],
        item["SPiD"]["tip"][:, :2],
        item["M2PC"]["goal"][:, :2],
        item["M2PC"]["tip"][:, :2],
    ], axis=0)


def set_top_view_bounds(ax, item, margin_ratio=0.08):
    pts = all_xy_points(item)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span = max(x_max - x_min, y_max - y_min)
    margin = max(margin_ratio * span, 0.015)

    # Top-down view: horizontal axis is world y; vertical axis is world x.
    ax.set_xlim(y_min - margin, y_max + margin)
    ax.set_ylim(x_min - margin, x_max + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()


def plot_top_view(ax, item):
    goal = item["M2PC"]["goal"]
    spid_tip = item["SPiD"]["tip"]
    m2pc_tip = item["M2PC"]["tip"]

    ax.plot(
        goal[:, 0],
        goal[:, 1],
        # linestyle="--",
        color=GOAL_COLOR,
        linewidth=LINE_WIDTH_GOAL,
        label="Goal",
        zorder=2,
    )
    ax.plot(
        m2pc_tip[:, 0],
        m2pc_tip[:, 1],
        color=M2PC_COLOR,
        alpha=M2PC_ALPHA,
        linewidth=LINE_WIDTH_METHOD,
        label="M2PC",
        zorder=4,
    )
    ax.plot(
        spid_tip[:, 0],
        spid_tip[:, 1],
        color=SPID_COLOR,
        linewidth=LINE_WIDTH_METHOD,
        # linestyle="-.",
        alpha=SPID_ALPHA,
        label="SPiD",
        zorder=3,
    )
    ax.scatter(
        goal[0, 0],
        goal[0, 1],
        s=8,
        color=START_COLOR,
        linewidths=0,
        label="Start",
        zorder=5,
    )

    set_top_view_bounds(ax, item)
    ax.set_xlim(0.27, 0.70)
    ax.set_ylim(-0.35, 0.32)
    ax.set_xlabel("$x$ (m)", labelpad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, linewidth=0.28, alpha=0.42)
    ax.tick_params(axis="both", which="major", pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.60)


def compute_metrics(loaded):
    rows = []
    for item in loaded:
        m2pc = item["M2PC"]
        spid = item["SPiD"]
        m2pc_rmse = rmse_cm(m2pc["tip"], m2pc["goal"])
        spid_rmse = rmse_cm(spid["tip"], spid["goal"])
        rows.append({
            "name": item["key"],
            "m2pc_rmse": m2pc_rmse,
            "spid_rmse": spid_rmse,
            "m2pc_mean": mean_error_cm(m2pc["tip"], m2pc["goal"]),
            "spid_mean": mean_error_cm(spid["tip"], spid["goal"]),
            "m2pc_max": max_error_cm(m2pc["tip"], m2pc["goal"]),
            "spid_max": max_error_cm(spid["tip"], spid["goal"]),
        })
    return rows


def compute_goal_stats(loaded):
    rows = []
    for item in loaded:
        ref = item[GOAL_STATS_REFERENCE]
        rows.append({
            "name": item["key"],
            "duration_s": float(ref["time"][-1]) if len(ref["time"]) else 0.0,
            "length_m": trajectory_length(ref["goal"]),
            "max_vel_mps": max_speed(ref["goal"], ref["time"]),
        })
    return rows


def plot_figure(loaded):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)

    for ax, item in zip(axes, loaded):
        plot_top_view(ax, item)
        ax.set_title(item["title"], pad=2.0, fontsize=5.5, linespacing=0.9)

    axes[0].set_ylabel("$y$ (m)", labelpad=1.0)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=True, labelleft=False)
    for ax, anchor in zip(axes, ("E", "C", "W")):
        ax.set_anchor(anchor)

    handles, labels = axes[0].get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    legend_labels = ["Goal", "M2PC", "SPiD", "Start"]
    legend_handles = [handles_by_label[label] for label in legend_labels]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, 0.97),
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.165, top=0.820, wspace=0.00)
    fig.savefig(SAVE_STEM.with_suffix(".pdf"))
    fig.savefig(SAVE_STEM.with_suffix(".png"), dpi=DPI)
    return fig


def print_metrics(rows):
    print("RMSE in xy plane, rope_p8 vs goal_traj (cm)")
    print("-" * 86)
    print(f"{'Trajectory':<20} {'SPiD_RMSE':>11} {'M2PC_RMSE':>11} {'Winner':>8} {'SPiD_mean':>11} {'M2PC_mean':>11} {'SPiD_max':>10} {'M2PC_max':>10}")
    print("-" * 86)
    for row in rows:
        winner = "M2PC" if row["m2pc_rmse"] < row["spid_rmse"] else "SPiD"
        print(
            f"{row['name']:<20} "
            f"{row['spid_rmse']:11.3f} "
            f"{row['m2pc_rmse']:11.3f} "
            f"{winner:>8} "
            f"{row['spid_mean']:11.3f} "
            f"{row['m2pc_mean']:11.3f} "
            f"{row['spid_max']:10.3f} "
            f"{row['m2pc_max']:10.3f}"
        )
    print("-" * 86)
    print(f"{'Mean over 3':<20} {np.mean([r['spid_rmse'] for r in rows]):11.3f} {np.mean([r['m2pc_rmse'] for r in rows]):11.3f}")


def print_goal_stats(rows):
    print(f"Goal trajectory stats ({GOAL_STATS_REFERENCE} goal/time reference)")
    print("-" * 66)
    print(f"{'Trajectory':<20} {'Duration (s)':>12} {'Length (m)':>12} {'Max vel (m/s)':>16}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['name']:<20} "
            f"{row['duration_s']:12.3f} "
            f"{row['length_m']:12.3f} "
            f"{row['max_vel_mps']:16.3f}"
        )
    print("-" * 66)


def main():
    set_paper_style()
    loaded = load_all()
    rows = compute_metrics(loaded)
    goal_rows = compute_goal_stats(loaded)
    plot_figure(loaded)

    print(f"Saved figures:")
    print(f"- {SAVE_STEM.with_suffix('.pdf')}")
    print(f"- {SAVE_STEM.with_suffix('.png')}")
    print_goal_stats(goal_rows)
    print_metrics(rows)


if __name__ == "__main__":
    main()
