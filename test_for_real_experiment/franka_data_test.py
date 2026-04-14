import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_franka_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "t_ns",
        "ee_pos_x",
        "ee_pos_y",
        "ee_pos_z",
        "ee_vel_x",
        "ee_vel_y",
        "ee_vel_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def compute_time_seconds(t_ns: np.ndarray) -> np.ndarray:
    t_ref = t_ns[0]
    return (t_ns - t_ref).astype(np.float64) * 1e-9


def integrate_velocity(t_s: np.ndarray, vel: np.ndarray) -> np.ndarray:
    if t_s.ndim != 1:
        raise ValueError("t_s must be 1D")
    if vel.ndim != 2 or vel.shape[0] != t_s.shape[0]:
        raise ValueError("vel must be shape (N, 3) matching t_s")

    dt = np.diff(t_s, prepend=t_s[0])
    # simple integration: delta = dt * vel
    increments = vel * dt[:, None]
    delta = np.concatenate([[0, 0, 0]], axis=0)[None, :] if vel.shape[0] > 0 else np.zeros_like(vel)
    delta = np.cumsum(increments, axis=0)
    return delta


def plot_position_comparison(t_s: np.ndarray, pos_act: np.ndarray, pos_int: np.ndarray, out_path: str = None):
    axis_labels = ["x", "y", "z"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    for i, label in enumerate(axis_labels):
        axs[i].plot(t_s, pos_act[:, i], label=f"pos_ee_{label} actual", linewidth=1.5)
        axs[i].plot(t_s, pos_int[:, i], label=f"pos_ee_{label} integrated", linestyle="--", linewidth=1.5)
        axs[i].set_ylabel(f"{label} [m]")
        axs[i].legend(loc="best", fontsize=9)
        axs[i].grid(True)

    error = pos_int - pos_act
    error_norm = np.linalg.norm(error, axis=1)
    axs[3].plot(t_s, error_norm, color="tab:red", label="position error magnitude")
    axs[3].set_xlabel("time [s]")
    axs[3].set_ylabel("error [m]")
    axs[3].legend(loc="best", fontsize=9)
    axs[3].grid(True)

    fig.suptitle("EE position: actual vs integrated from ee_vel")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # if out_path is not None:
    #     plt.savefig(out_path, dpi=160)
    #     print(f"Saved comparison plot to {out_path}")
    # else:
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Integrate ee_vel from franka CSV and compare to actual ee_pos.")
    parser.add_argument("--csv", type=str, default="franka_data/ee_collection_run3.csv", help="Path to ee_collection_run1.csv")
    parser.add_argument("--out", type=str, default="franka_vel_integral_compare.png", help="Output plot path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    df = load_franka_csv(args.csv)
    t_s = compute_time_seconds(df["t_ns"].to_numpy())

    vel = np.stack(
        [df["ee_vel_x"].to_numpy(), df["ee_vel_y"].to_numpy(), df["ee_vel_z"].to_numpy()],
        axis=1,
    ).astype(np.float64)
    pos = np.stack(
        [df["ee_pos_x"].to_numpy(), df["ee_pos_y"].to_numpy(), df["ee_pos_z"].to_numpy()],
        axis=1,
    ).astype(np.float64)

    pos_delta = integrate_velocity(t_s, vel)
    pos_int = pos[0:1] + pos_delta

    dt = np.diff(t_s)
    print(f"num samples: {len(t_s)}")
    print(f"dt min/mean/max: {dt.min():.6f}s / {dt.mean():.6f}s / {dt.max():.6f}s")
    print(f"final integrated displacement: {pos_delta[-1]}" )
    print(f"final actual displacement: {pos[-1] - pos[0]}" )
    print(f"final error magnitude: {np.linalg.norm(pos_int[-1] - pos[-1]):.6f} m")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plot_position_comparison(t_s, pos, pos_int, None if args.show else args.out)


if __name__ == "__main__":
    main()
