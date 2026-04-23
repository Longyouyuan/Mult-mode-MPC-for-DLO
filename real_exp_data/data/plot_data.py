from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_axes_equal_3d(ax, xs, ys, zs):
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return

    x_min, x_max = np.nanmin(xs), np.nanmax(xs)
    y_min, y_max = np.nanmin(ys), np.nanmax(ys)
    z_min, z_max = np.nanmin(zs), np.nanmax(zs)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0.0:
        max_range = 1e-3

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        # matplotlib < 3.3: draw an invisible cubic bounding box
        corners = np.array([[x_mid - half, y_mid - half, z_mid - half],
                            [x_mid + half, y_mid + half, z_mid + half]])
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], alpha=0.0)


def plot_goal_vs_real_node_3d(rope, goal_traj, goal_point, obs=None):
    if rope is None or goal_traj is None:
        print("Missing data, skipping 3D goal-vs-rope plot.")
        return

    px_name = 'p{}_x'.format(goal_point)
    py_name = 'p{}_y'.format(goal_point)
    pz_name = 'p{}_z'.format(goal_point)

    if px_name not in rope.dtype.names or py_name not in rope.dtype.names or pz_name not in rope.dtype.names:
        print("Rope node {} is not present in rope_state.csv, skipping 3D plot.".format(goal_point))
        return

    goal_x = np.asarray(goal_traj['goal_x'])
    goal_y = np.asarray(goal_traj['goal_y'])
    goal_z = np.asarray(goal_traj['goal_z'])
    real_x = np.asarray(rope[px_name])
    real_y = np.asarray(rope[py_name])
    real_z = np.asarray(rope[pz_name])

    goal_mask = np.isfinite(goal_x) & np.isfinite(goal_y) & np.isfinite(goal_z)
    real_mask = np.isfinite(real_x) & np.isfinite(real_y) & np.isfinite(real_z)

    goal_x = goal_x[goal_mask]
    goal_y = goal_y[goal_mask]
    goal_z = goal_z[goal_mask]
    real_x = real_x[real_mask]
    real_y = real_y[real_mask]
    real_z = real_z[real_mask]

    if goal_x.size == 0 or real_x.size == 0:
        print("Goal or real trajectory is empty, skipping 3D plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(goal_x, goal_y, goal_z, linewidth=2.5, color='tab:blue', label='goal traj')
    ax.plot(real_x, real_y, real_z, linewidth=2.0, color='tab:orange', label='real rope node {}'.format(goal_point))

    ax.scatter(goal_x[0], goal_y[0], goal_z[0], color='tab:blue', marker='o', s=45, label='goal start')
    ax.scatter(goal_x[-1], goal_y[-1], goal_z[-1], color='tab:blue', marker='x', s=55, label='goal end')
    ax.scatter(real_x[0], real_y[0], real_z[0], color='tab:orange', marker='o', s=45, label='real start')
    ax.scatter(real_x[-1], real_y[-1], real_z[-1], color='tab:orange', marker='x', s=55, label='real end')

    # 10 evenly-spaced time markers for real rope node (goal_point)
    n_marks = 10
    ridxs = np.linspace(0, real_x.size - 1, n_marks, dtype=int)
    ax.scatter(real_x[ridxs], real_y[ridxs], real_z[ridxs],
               color='tab:orange', marker='D', s=40, zorder=6, label='node {} time marks'.format(goal_point))
    for k, idx in enumerate(ridxs):
        ax.text(real_x[idx], real_y[idx], real_z[idx], ' {}'.format(k), fontsize=7, color='tab:orange')

    all_x = np.concatenate((goal_x, real_x))
    all_y = np.concatenate((goal_y, real_y))
    all_z = np.concatenate((goal_z, real_z))

    if goal_point != 0 and 'p0_x' in rope.dtype.names:
        n0_x = np.asarray(rope['p0_x'])
        n0_y = np.asarray(rope['p0_y'])
        n0_z = np.asarray(rope[pz_name])
        n0_mask = np.isfinite(n0_x) & np.isfinite(n0_y) & np.isfinite(n0_z)
        n0_x, n0_y, n0_z = n0_x[n0_mask], n0_y[n0_mask], n0_z[n0_mask]
        if n0_x.size > 0:
            ax.plot(n0_x, n0_y, n0_z, linewidth=1.5, color='tab:green', label='rope node 0 (EE)')
            ax.scatter(n0_x[0], n0_y[0], n0_z[0], color='tab:green', marker='o', s=35)
            ax.scatter(n0_x[-1], n0_y[-1], n0_z[-1], color='tab:green', marker='x', s=45)
            # 10 evenly-spaced time markers
            n_marks = 10
            idxs = np.linspace(0, n0_x.size - 1, n_marks, dtype=int)
            ax.scatter(n0_x[idxs], n0_y[idxs], n0_z[idxs],
                       color='tab:red', marker='D', s=40, zorder=6, label='node 0 time marks')
            for k, idx in enumerate(idxs):
                ax.text(n0_x[idx], n0_y[idx], n0_z[idx], ' {}'.format(k), fontsize=7, color='tab:green')
            all_x = np.concatenate((all_x, n0_x))
            all_y = np.concatenate((all_y, n0_y))
            all_z = np.concatenate((all_z, n0_z))

    if obs is not None:
        ox = np.asarray(obs['obs_x'])
        oy = np.asarray(obs['obs_y'])
        oz = np.asarray(obs['obs_z'])
        omask = np.isfinite(ox) & np.isfinite(oy) & np.isfinite(oz)
        ox, oy, oz = ox[omask], oy[omask], oz[omask]
        if ox.size > 0:
            ax.plot(ox, oy, oz, linewidth=1.8, color='tab:purple', label='obstacle traj')
            ax.scatter(ox[0], oy[0], oz[0], color='tab:purple', marker='o', s=35)
            ax.scatter(ox[-1], oy[-1], oz[-1], color='tab:purple', marker='x', s=45)
            all_x = np.concatenate((all_x, ox))
            all_y = np.concatenate((all_y, oy))
            all_z = np.concatenate((all_z, oz))
    set_axes_equal_3d(ax, all_x, all_y, all_z)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Goal Trajectory vs Real Rope Node {}'.format(goal_point))
    ax.legend()
    ax.grid(True)
    fig.tight_layout()



def build_goal_traj_from_paper_csv(df):
    goal_prefix = "goal_traj" if "goal_traj_x" in df.columns else "goal_pos"
    return np.rec.fromarrays(
        [
            df[f"{goal_prefix}_x"].to_numpy(),
            df[f"{goal_prefix}_y"].to_numpy(),
            df[f"{goal_prefix}_z"].to_numpy(),
        ],
        names=["goal_x", "goal_y", "goal_z"],
    )


def build_obs_from_paper_csv(df):
    if not {"obs_x", "obs_y", "obs_z"}.issubset(df.columns):
        return None
    return np.rec.fromarrays(
        [
            df["obs_x"].to_numpy(),
            df["obs_y"].to_numpy(),
            df["obs_z"].to_numpy(),
        ],
        names=["obs_x", "obs_y", "obs_z"],
    )


def build_rope_state(df, rope_indices):
    names = []
    arrays = []
    for idx in rope_indices:
        for axis in ("x", "y", "z"):
            names.append(f"p{idx}_{axis}")
            arrays.append(df[f"rope_p{idx}_{axis}"].to_numpy())
    return np.rec.fromarrays(arrays, names=names)


def main():
    # csv_path = "./sin_2.5s/paper_data_sin_1.csv"
    # csv_path = "./eight_4s/paper_data_eight_2.csv"
    csv_path = "./ablation/paper_data_continuous_eight_ablation.csv"
    # csv_path = "./continuous_eight/paper_data_continuous_eight_1.csv"
    df = pd.read_csv(csv_path)

    rope_indices = sorted(
        int(col.split("_")[1][1:])
        for col in df.columns
        if col.startswith("rope_p") and col.endswith("_x")
    )
    if not rope_indices:
        raise ValueError(f"No rope node columns found in {csv_path}")

    goal_traj = build_goal_traj_from_paper_csv(df)
    obs = build_obs_from_paper_csv(df)
    rope = build_rope_state(df, rope_indices)
    last_idx = rope_indices[-1]

    plot_goal_vs_real_node_3d(rope, goal_traj, last_idx, obs=obs)
    plt.show()


if __name__ == "__main__":
    main()
