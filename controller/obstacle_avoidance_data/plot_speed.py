from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ===================== Paths =====================
THIS_FILE = Path(__file__).resolve()
DATA_DIR = THIS_FILE.parent

PROFILE_FILES = (
    ("M2PC", DATA_DIR / "m2pc_profile_records.npz", "#d62728", ""),
    ("C-MPPI", DATA_DIR / "dbscan_profile_records.npz", "#2ca02c", "////"),
    ("SVMPC", DATA_DIR / "svmpc_profile_records.npz", "#1f77b4", "\\\\\\\\"),
)


# ===================== Plot setup =====================
DPI = 600
FIGSIZE_SPEED = (1.0, 1.22)  # Compact IEEE RAL single-column inset size.
FONT_SIZE = 7
TICK_SIZE = 6.5
LINE_WIDTH = 0.65

TIME_KEY = "improve"
WARMUP_STEPS = 1
N_EXAMPLE_POINTS = 12
SCATTER_JITTER = 0.22
SCATTER_SEED = 7


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "hatch.linewidth": 0.35,
    })


def load_timing_ms(path, key=TIME_KEY, warmup_steps=WARMUP_STEPS):
    if not path.exists():
        raise FileNotFoundError(f"Missing timing file: {path}")

    records = np.load(path)
    if key not in records.files:
        raise KeyError(f"{path.name} has keys {records.files}, but '{key}' was requested.")

    timing = np.asarray(records[key], dtype=np.float64)
    if warmup_steps > 0:
        timing = timing[warmup_steps:]
    timing = timing[np.isfinite(timing)]
    if timing.size == 0:
        raise ValueError(f"No finite timing samples in {path.name}.")

    return timing * 1000.0


def summarize(values_ms):
    return {
        "n": values_ms.size,
        "mean": float(np.mean(values_ms)),
        "std": float(np.std(values_ms, ddof=1)) if values_ms.size > 1 else 0.0,
        "var": float(np.var(values_ms, ddof=1)) if values_ms.size > 1 else 0.0,
        "min": float(np.min(values_ms)),
        "max": float(np.max(values_ms)),
    }


def select_example_points(values_ms, rng, count=N_EXAMPLE_POINTS):
    """Pick a reproducible random subset of recorded samples for scatter display."""
    if values_ms.size <= count:
        return values_ms.copy()
    sample_idx = rng.choice(values_ms.size, size=count, replace=False)
    return values_ms[sample_idx]


def save_figure(fig, stem):
    fig.savefig(DATA_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(DATA_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.015)


def plot_speed(timing_by_method):
    labels = [item["label"] for item in timing_by_method]
    colors = [item["color"] for item in timing_by_method]
    hatches = [item["hatch"] for item in timing_by_method]
    values = [item["values_ms"] for item in timing_by_method]
    stats = [summarize(v) for v in values]

    x = np.arange(len(labels), dtype=np.float64)
    means = np.asarray([s["mean"] for s in stats])
    stds = np.asarray([s["std"] for s in stats])
    rng = np.random.default_rng(SCATTER_SEED)

    fig, ax = plt.subplots(figsize=FIGSIZE_SPEED)

    bars = ax.bar(
        x,
        means,
        width=0.56,
        color=colors,
        edgecolor="0.12",
        linewidth=0.45,
        yerr=stds,
        error_kw={
            "ecolor": "0.12",
            "elinewidth": 0.6,
            "capsize": 2.0,
            "capthick": 0.6,
            "zorder": 4,
        },
        zorder=2,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    example_sets = []
    for i, values_ms in enumerate(values):
        examples = select_example_points(values_ms, rng)
        example_sets.append(examples)
        jitter = rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=examples.size)
        ax.scatter(
            x[i] + jitter,
            examples,
            s=4.0,
            facecolor="white",
            edgecolor="0.12",
            linewidth=0.35,
            alpha=0.95,
            zorder=3,
        )

    label_offset = max(0.45, 0.025 * float(np.max(means + stds)))
    label_positions = []
    for i, mean_ms in enumerate(means):
        label_y = max(mean_ms + stds[i], float(np.max(example_sets[i]))) + label_offset
        label_positions.append(label_y)
        ax.text(
            x[i],
            label_y,
            f"{mean_ms:.1f}",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
            color="0.12",
            zorder=4,
        )

    ax.set_ylabel("Time (ms)", labelpad=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, va="top")
    ax.tick_params(axis="x", pad=1.0)
    ax.tick_params(axis="y", pad=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", linewidth=0.25, alpha=0.42)
    ax.set_axisbelow(True)

    y_max = max(
        float(np.max(means + stds)),
        max(float(np.max(v)) for v in values),
        max(label_positions),
    )
    y_pad = max(0.6, 0.04 * y_max)
    ax.set_ylim(0.0, y_max + y_pad)
    ax.margins(x=0.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)

    fig.tight_layout(pad=0.08)
    save_figure(fig, "obstacle_avoidance_planning_time_compare")
    return fig, stats


def main():
    set_paper_style()

    timing_by_method = []
    for label, path, color, hatch in PROFILE_FILES:
        timing_by_method.append({
            "label": label,
            "values_ms": load_timing_ms(path),
            "color": color,
            "hatch": hatch,
        })

    _, stats = plot_speed(timing_by_method)

    print(f"Timing key: '{TIME_KEY}'")
    print(f"Discarded warm-up steps: {WARMUP_STEPS}")
    print("Method        n    mean(ms)   std(ms)   var(ms^2)   min(ms)   max(ms)")
    for item, stat in zip(timing_by_method, stats):
        label = item["label"].replace("\n", "-")
        print(
            f"{label:<11s} {stat['n']:4d}"
            f" {stat['mean']:10.3f} {stat['std']:9.3f} {stat['var']:11.3f}"
            f" {stat['min']:9.3f} {stat['max']:9.3f}"
        )
    print(f"Saved figures to: {DATA_DIR}")
    print("- obstacle_avoidance_planning_time_compare.pdf/png")


if __name__ == "__main__":
    main()
