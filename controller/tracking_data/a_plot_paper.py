from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Put this script in controller/SPiD_tracking or another repo subfolder.
# It will try to locate the repo root and ../tracking_data automatically.
THIS_FILE = Path(__file__).resolve()

# Try common repo-root layouts.
CANDIDATE_ROOTS = [THIS_FILE.parent, *THIS_FILE.parents]
REPO_ROOT = None
for p in CANDIDATE_ROOTS:
    if (p / "common").exists() and (p / "my_trajs").exists():
        REPO_ROOT = p
        break
if REPO_ROOT is None:
    # If this file is inside controller/SPiD_tracking, parents[2] is repo root.
    REPO_ROOT = THIS_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import half_dense_then_uniform, sin_traj, egg_traj, eight_traj, build_goal_traj_from_drawn

# =========================
# User settings
# =========================
SPID_CTR_PERIOD = 10
M2PC_CTR_PERIOD = 25
PLOT_CTR_PERIOD = SPID_CTR_PERIOD
DEVICE = torch.device("cpu")

# tracking_data is one level above the script folder in your setup.
# If not found, the script also tries REPO_ROOT/tracking_data.
TRACKING_DATA_DIR = THIS_FILE.parent.parent / "tracking_data"
if not TRACKING_DATA_DIR.exists():
    alt = REPO_ROOT / "tracking_data"
    if alt.exists():
        TRACKING_DATA_DIR = alt
TRACKING_DATA_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PDF = TRACKING_DATA_DIR / "spid_m2pc_18traj_ral_xy_spid_switch.pdf"
SAVE_PNG = TRACKING_DATA_DIR / "spid_m2pc_18traj_ral_xy_spid_switch.png"

# RAL/IEEE double-column friendly size.
# IEEE double column is about 7.16 inch wide. Height chosen for 3x6 compact layout.
FIGSIZE = (7.16, 4.15)
DPI = 600

COL_ORDER = ["sin", "egg", "eight", "SpongeBob", "PatrickStar", "flower"]
COL_TITLES = ["Sin", "Egg", "Eight", "SpongeBob", "PatrickStar", "Flower"]
ROW_SPECS = [
    {"sin": 4.0, "egg": 4.0, "eight": 4.0, "SpongeBob": 12.0, "PatrickStar": 12.0, "flower": 12.0},
    {"sin": 5.0, "egg": 5.0, "eight": 5.0, "SpongeBob": 15.0, "PatrickStar": 15.0, "flower": 15.0},
    {"sin": 6.0, "egg": 6.0, "eight": 6.0, "SpongeBob": 18.0, "PatrickStar": 18.0, "flower": 18.0},
]

# =========================
# Plot switches
# =========================
# Set whether to draw the SPiD curve for each subplot.
# The key is (shape_name, total_time). Time is float, e.g. 15.0.
# Your requested setting: do not draw SPiD for all T=15s and T=18s subplots.
PLOT_SPID_BY_TRAJ = {
    ("SpongeBob", 15.0): False,
    ("PatrickStar", 15.0): False,
    ("flower", 15.0): False,
    ("SpongeBob", 18.0): False,
    ("PatrickStar", 18.0): False,
    ("flower", 18.0): False,
}

# Default for unspecified subplots.
PLOT_SPID_DEFAULT = True


def should_plot_spid(shape_name: str, total_time: float) -> bool:
    return PLOT_SPID_BY_TRAJ.get((shape_name, float(total_time)), PLOT_SPID_DEFAULT)

# Colors requested by user.
# GOAL_COLOR = "0.55"      # gray
# M2PC_COLOR = "#1f77b4"   # blue
# SPID_COLOR = "#ff7f0e"   # orange
# START_COLOR = "#2ca02c"  # green

GOAL_COLOR = "k"           # 黑（ground truth）
M2PC_COLOR = "#d62728"     # 红（主方法）
SPID_COLOR = "#1f77b4"     # 蓝（对比方法）
START_COLOR = "#2ca02c"  # green

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 9,
    "lines.linewidth": 1.10,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def build_goal(shape_name: str, total_time: float, ctr_period: int) -> np.ndarray:
    total_horizon_t = int(1000 / ctr_period * total_time)
    points = half_dense_then_uniform(
        N=total_horizon_t + 1,
        ratio=0.3,
        sharpness=2.0,
        mode="exp",
        interval=(0.0, 2.0),
        plot=False,
    )

    if shape_name == "sin":
        goal = sin_traj(points, width=0.45 * 2.0, plot=False, device=DEVICE)
    elif shape_name == "egg":
        goal = egg_traj(points, scale_x=0.38 * 2.0, scale_y=0.52 * 2.0, plot=False, device=DEVICE)
    elif shape_name == "eight":
        goal = eight_traj(points, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2, loops=1, plot=False, device=DEVICE)
    else:
        goal = build_goal_traj_from_drawn(
            drawn_path=str(REPO_ROOT / "my_trajs" / f"{shape_name}.npy"),
            total_horizon=total_horizon_t,
            device=DEVICE,
            z0=0.2,
            scale_x=3.0,
            scale_y=3.0,
            keep_aspect=False,
            sigma=0.0,
            uniform_M=1000,
            ratio=0.3,
            sharpness=2.0,
            interval=(0.0, 2.0),
        )
    return goal.detach().cpu().numpy()


def file_stem(method: str, shape_name: str, total_time: float) -> str:
    # Existing saved files use names like spid_egg_5s.npy and m2pc_PatrickStar_18s.npy.
    t = int(total_time) if float(total_time).is_integer() else total_time
    return f"{method}_{shape_name}_{t}s"


def load_endpoint(method: str, shape_name: str, total_time: float) -> np.ndarray:
    direct = TRACKING_DATA_DIR / f"{file_stem(method, shape_name, total_time)}.npy"
    if direct.exists():
        return np.load(direct)

    # Case-insensitive fallback for small naming differences.
    wanted = direct.name.lower()
    matches = [p for p in TRACKING_DATA_DIR.glob("*.npy") if p.name.lower() == wanted]
    if matches:
        return np.load(matches[0])

    # PatrickStar typo fallback.
    aliases = []
    if shape_name == "PatrickStar":
        aliases = ["patrikstar", "patrickstar", "Patrickstar"]
    elif shape_name == "SpongeBob":
        aliases = ["spongebob", "Spongebob"]
    for alias in aliases:
        candidate = TRACKING_DATA_DIR / f"{method}_{alias}_{int(total_time)}s.npy"
        if candidate.exists():
            return np.load(candidate)

    raise FileNotFoundError(f"Missing {method} endpoint file: {direct}")


def align_goal_to_endpoint(goal: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    # Most scripts saved endpoint length equal to goal[1:] length; use goal[1:] first.
    if len(goal) >= len(endpoint) + 1:
        return goal[1:1 + len(endpoint)]
    return goal[:len(endpoint)]


def rmse_cm(endpoint: np.ndarray, goal_aligned: np.ndarray) -> float:
    n = min(len(endpoint), len(goal_aligned))
    err = endpoint[:n, :2] - goal_aligned[:n, :2]
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 100.0)


def mean_cm(endpoint: np.ndarray, goal_aligned: np.ndarray) -> float:
    n = min(len(endpoint), len(goal_aligned))
    err = endpoint[:n, :2] - goal_aligned[:n, :2]
    return float(np.mean(np.linalg.norm(err, axis=1)) * 100.0)


def max_cm(endpoint: np.ndarray, goal_aligned: np.ndarray) -> float:
    n = min(len(endpoint), len(goal_aligned))
    err = endpoint[:n, :2] - goal_aligned[:n, :2]
    return float(np.max(np.linalg.norm(err, axis=1)) * 100.0)


def compute_column_limits(all_data):
    limits = {}
    for col_idx, shape_name in enumerate(COL_ORDER):
        xs, ys = [], []
        for row_idx in range(3):
            item = all_data[(row_idx, col_idx)]
            keys = ["goal", "m2pc"]
            if item.get("plot_spid", True):
                keys.append("spid")
            for key in keys:
                arr = item[key]
                xs.append(arr[:, 0])
                ys.append(arr[:, 1])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin)
        margin = max(0.04 * span, 0.02)
        half = 0.5 * span + margin
        limits[col_idx] = (cx - half, cx + half, cy - half, cy + half)
    return limits


def main():
    print(f"[Info] tracking_data = {TRACKING_DATA_DIR}")

    all_data = {}
    rows_for_print = []
    goal_cache = {}

    def get_goal(shape_name: str, total_time: float, ctr_period: int) -> np.ndarray:
        key = (shape_name, float(total_time), int(ctr_period))
        if key not in goal_cache:
            goal_cache[key] = build_goal(shape_name, total_time, ctr_period)
        return goal_cache[key]

    for row_idx, row_spec in enumerate(ROW_SPECS):
        for col_idx, shape_name in enumerate(COL_ORDER):
            total_time = row_spec[shape_name]
            goal = get_goal(shape_name, total_time, PLOT_CTR_PERIOD)
            # Labels follow filename prefixes: m2pc_*.npy -> M2PC, spid_*.npy -> SPiD.
            m2pc = load_endpoint("m2pc", shape_name, total_time)
            spid = load_endpoint("spid", shape_name, total_time)

            goal_m2pc = align_goal_to_endpoint(get_goal(shape_name, total_time, M2PC_CTR_PERIOD), m2pc)
            goal_spid = align_goal_to_endpoint(get_goal(shape_name, total_time, SPID_CTR_PERIOD), spid)

            m2pc_rmse = rmse_cm(m2pc, goal_m2pc)
            spid_rmse = rmse_cm(spid, goal_spid)
            m2pc_mean = mean_cm(m2pc, goal_m2pc)
            spid_mean = mean_cm(spid, goal_spid)
            m2pc_max = max_cm(m2pc, goal_m2pc)
            spid_max = max_cm(spid, goal_spid)

            all_data[(row_idx, col_idx)] = {
                "shape": shape_name,
                "time": total_time,
                "goal": goal,
                "m2pc": m2pc,
                "spid": spid,
                "plot_spid": should_plot_spid(shape_name, total_time),
                "m2pc_rmse": m2pc_rmse,
                "spid_rmse": spid_rmse,
            }
            rows_for_print.append((shape_name, total_time, m2pc_rmse, spid_rmse, m2pc_mean, spid_mean, m2pc_max, spid_max))

    col_limits = compute_column_limits(all_data)

    fig, axes = plt.subplots(3, 6, figsize=FIGSIZE, dpi=DPI)

    for row_idx in range(3):
        for col_idx, shape_name in enumerate(COL_ORDER):
            ax = axes[row_idx, col_idx]
            item = all_data[(row_idx, col_idx)]
            goal = item["goal"]
            m2pc = item["m2pc"]
            spid = item["spid"]

            if item.get("plot_spid", True):
                ax.plot(spid[:, 0], spid[:, 1], color=SPID_COLOR, linewidth=0.75, linestyle="--", zorder=1)
            ax.plot(goal[:, 0], goal[:, 1], color=GOAL_COLOR, linestyle="-", linewidth=1.0, zorder=2)
            ax.plot(m2pc[:, 0], m2pc[:, 1], color=M2PC_COLOR, linewidth=0.75, zorder=3)
            ax.scatter(goal[0, 0], goal[0, 1], s=12, color=START_COLOR, zorder=4, linewidths=0)

            xmin, xmax, ymin, ymax = col_limits[col_idx]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")

            # Keep the dashed grid at a fixed 1 m spacing on both axes.
            from matplotlib.ticker import MultipleLocator, NullLocator
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())

            ax.grid(False)
            ax.grid(True, which="major", linestyle="--", linewidth=0.25, alpha=0.5)

            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], pad=3.0)

            # Show the true duration inside every subplot.
            ax.text(
                0.75, 0.90,
                f"T={int(item['time'])}s",
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.8),
            )

            # Keep numeric tick labels visible, but avoid crowding.
            ax.tick_params(axis="both", which="major", pad=1.0)
            if row_idx < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("(m)", labelpad=1.0)
            ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=GOAL_COLOR, linestyle="-", linewidth=1.25, label="Goal"),
        Line2D([0], [0], color=M2PC_COLOR, linewidth=1.0, label="M2PC"),
        Line2D([0], [0], color=SPID_COLOR, linewidth=1.0, linestyle="--",  label="SPiD"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=START_COLOR, markeredgecolor=START_COLOR, markersize=4, label="Start"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.950),
        handlelength=2.0,
        columnspacing=3.0,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.070, top=0.855, wspace=0.035, hspace=0.08)
    fig.savefig(SAVE_PDF)
    fig.savefig(SAVE_PNG)
    print(f"[Save] {SAVE_PDF}")
    print(f"[Save] {SAVE_PNG}")

    print("=" * 118)
    print(f"{'Trajectory':<18} {'SPiD_RMSE(cm)':>14} {'M2PC_RMSE(cm)':>14} {'Winner':>10} {'SPiD_mean':>12} {'M2PC_mean':>12} {'SPiD_max':>12} {'M2PC_max':>12}")
    print("-" * 118)
    m2pc_rmse_all, spid_rmse_all = [], []
    for shape_name, total_time, m2pc_r, spid_r, m2pc_m, spid_m, m2pc_x, spid_x in rows_for_print:
        traj = f"{shape_name}_{int(total_time)}s"
        winner = "M2PC" if m2pc_r < spid_r else "SPiD"
        print(f"{traj:<18} {spid_r:14.3f} {m2pc_r:14.3f} {winner:>10} {spid_m:12.3f} {m2pc_m:12.3f} {spid_x:12.3f} {m2pc_x:12.3f}")
        m2pc_rmse_all.append(m2pc_r)
        spid_rmse_all.append(spid_r)
    print("-" * 118)
    print(f"{'Mean over 18':<18} {np.mean(spid_rmse_all):14.3f} {np.mean(m2pc_rmse_all):14.3f} {'M2PC' if np.mean(m2pc_rmse_all) < np.mean(spid_rmse_all) else 'SPiD':>10}")
    print(f"{'Median over 18':<18} {np.median(spid_rmse_all):14.3f} {np.median(m2pc_rmse_all):14.3f} {'M2PC' if np.median(m2pc_rmse_all) < np.median(spid_rmse_all) else 'SPiD':>10}")
    print("=" * 118)

    plt.show()


if __name__ == "__main__":
    main()
