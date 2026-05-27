from pathlib import Path

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Obstacle geometry in meters
OBS_HEIGHT = 0.14       # 14 cm
OBS_DIAMETER = 0.055    # 5.5 cm
OBS_RADIUS = OBS_DIAMETER / 2.0
OBS_MARGIN_DIAMETER = 0.10  # 7 cm including safety margin
OBS_MARGIN_RADIUS = OBS_MARGIN_DIAMETER / 2.0
ROPE_TIP_COLOR = (0.94, 0.54, 0.14)
JUMP_SPIKE_COLOR = (0.88, 0.10, 0.12)
FROZEN_SPIKE_COLOR = JUMP_SPIKE_COLOR
SPIKE_MIN_JUMP = 0.02
SPIKE_JUMP_FACTOR = 5.0
SPIKE_MIN_RESIDUAL = 0.015
SPIKE_RESIDUAL_FACTOR = 3.5
SPIKE_FREEZE_FACTOR = 0.12
OBSTACLE_SIDE_COLOR = (0.68, 0.46, 0.24, 0.52)
OBSTACLE_TOP_COLOR = (0.82, 0.60, 0.34, 0.74)
OBSTACLE_BOTTOM_COLOR = (0.56, 0.36, 0.16, 0.52)
OBSTACLE_EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)
OBSTACLE_MARGIN_COLOR = (0.62, 0.38, 0.16, 0.95)
OBSTACLE_ZORDER = 5
ROPE_TIP_ZORDER = 40

# Global run switches when launching this file directly.
DEFAULT_CSV_PATH = ["./eight_obs/paper_data_continuous_eight_obs_1.csv",
                    "./eight_obs/paper_data_continuous_eight_obs_2.csv",
                    "./eight_obs/paper_data_continuous_eight_obs_3.csv",
                    "./eight_obs/paper_data_continuous_eight_obs_4.csv",
                    "./eight_obs/paper_data_continuous_eight_obs_5.csv",
                    "./eight_obs/paper_data_continuous_eight_obs--1.csv",
                    "./eight_obs/paper_data_continuous_eight_obs--2.csv",
                    "./eight_obs/paper_data_continuous_eight_obs--3.csv",
                    "./eight_obs/paper_data_continuous_eight_obs--5.csv",
                    ]
# DEFAULT_CSV_PATH = "./eight_obs/paper_data_continuous_eight_obs_1.csv"

DEFAULT_BATCH_FIRST_LEGEND = True
DEFAULT_SAVE_MP4 = True
DEFAULT_REPAIR_ROPE_TIP = False

ACTIVE_ANIMATION = None


class ObstacleLegendHandle:
    pass


class HandlerObstacleLegend(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        center = (xdescent + 0.5 * width, ydescent + 0.5 * height - 5.5)
        symbol_size = min(width, height)
        inner_radius = 0.34 * symbol_size
        outer_radius = 0.74 * symbol_size

        inner_circle = Circle(
            center,
            inner_radius,
            facecolor=OBSTACLE_TOP_COLOR,
            edgecolor=OBSTACLE_EDGE_COLOR,
            linewidth=1.2,
            transform=trans,
        )
        outer_circle = Circle(
            center,
            outer_radius,
            facecolor="none",
            edgecolor=OBSTACLE_MARGIN_COLOR,
            linewidth=1.6,
            linestyle=(0, (2.5, 2.0)),
            transform=trans,
        )
        return [inner_circle, outer_circle]


def set_axes_equal_3d(ax, xs, ys, zs, padding=0.03):
    """Make x/y/z axes use the same scale so the cylinder is not distorted."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)

    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not np.any(mask):
        return

    x_min, x_max = np.nanmin(xs[mask]), np.nanmax(xs[mask])
    y_min, y_max = np.nanmin(ys[mask]), np.nanmax(ys[mask])
    z_min, z_max = np.nanmin(zs[mask]), np.nanmax(zs[mask])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0:
        max_range = 1e-3

    max_range += 2.0 * padding
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
        pass


def set_top_down_view(ax):
    ax.view_init(elev=90, azim=180)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass


def build_goal_traj_from_paper_csv(df):
    """Compatible with goal_traj_x/y/z or goal_pos_x/y/z."""
    goal_prefix = "goal_traj" if "goal_traj_x" in df.columns else "goal_pos"
    required = [f"{goal_prefix}_{a}" for a in ("x", "y", "z")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing goal columns: {missing}")

    return np.column_stack([
        df[f"{goal_prefix}_x"].to_numpy(),
        df[f"{goal_prefix}_y"].to_numpy(),
        df[f"{goal_prefix}_z"].to_numpy(),
    ])


def build_obs_from_paper_csv(df):
    if not {"obs_x", "obs_y", "obs_z"}.issubset(df.columns):
        return None
    return np.column_stack([
        df["obs_x"].to_numpy(),
        df["obs_y"].to_numpy(),
        df["obs_z"].to_numpy(),
    ])


def build_time_from_paper_csv(df):
    if "time" not in df.columns:
        return None
    return df["time"].to_numpy(dtype=float)


def get_rope_indices(df):
    indices = sorted(
        int(col.split("_")[1][1:])
        for col in df.columns
        if col.startswith("rope_p") and col.endswith("_x")
    )
    if not indices:
        raise ValueError("No rope node columns found. Expected columns like rope_p0_x, rope_p0_y, rope_p0_z.")
    return indices


def build_rope_tip(df, goal_point=None):
    """Return the selected rope point. By default, use the last rope node as rope tip."""
    rope_indices = get_rope_indices(df)
    if goal_point is None:
        goal_point = rope_indices[-1]

    required = [f"rope_p{goal_point}_{a}" for a in ("x", "y", "z")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing rope point columns for p{goal_point}: {missing}")

    return np.column_stack([
        df[f"rope_p{goal_point}_x"].to_numpy(),
        df[f"rope_p{goal_point}_y"].to_numpy(),
        df[f"rope_p{goal_point}_z"].to_numpy(),
    ]), goal_point


def make_cylinder_mesh(center, radius=OBS_RADIUS, height=OBS_HEIGHT, n_theta=40, n_z=2, n_r=20):
    """Create a vertical cylinder with closed top and bottom caps."""
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(cz - height / 2.0, cz + height / 2.0, n_z)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_side = cx + radius * np.cos(theta_grid)
    y_side = cy + radius * np.sin(theta_grid)

    radial = np.linspace(0.0, radius, n_r)
    theta_cap, radial_cap = np.meshgrid(theta, radial)
    x_cap = cx + radial_cap * np.cos(theta_cap)
    y_cap = cy + radial_cap * np.sin(theta_cap)
    z_top = np.full_like(x_cap, cz + height / 2.0)
    z_bottom = np.full_like(x_cap, cz - height / 2.0)

    return (x_side, y_side, z_grid), (x_cap, y_cap, z_top), (x_cap, y_cap, z_bottom)


def finite_rows(*arrays):
    """Keep frames where all provided xyz arrays are finite."""
    n = min(a.shape[0] for a in arrays if a is not None)
    mask = np.ones(n, dtype=bool)
    for arr in arrays:
        if arr is not None:
            mask &= np.all(np.isfinite(arr[:n]), axis=1)
    return [None if arr is None else arr[:n][mask] for arr in arrays]


def build_default_gif_path(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    output_dir = csv_path.parent if output_dir is None else Path(output_dir)
    return output_dir / f"{csv_path.stem}.gif"


def build_default_mp4_path(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    output_dir = csv_path.parent if output_dir is None else Path(output_dir)
    return output_dir / f"{csv_path.stem}.mp4"


def normalize_csv_paths(csv_arg):
    if isinstance(csv_arg, (str, Path)):
        return [str(csv_arg)]
    return [str(csv_path) for csv_path in csv_arg]


def close_animation_figure(animation_obj):
    figure = getattr(animation_obj, "_fig", None)
    if figure is not None:
        plt.close(figure)


def get_animation_axis_limits(animation_obj):
    figure = getattr(animation_obj, "_fig", None)
    if figure is None or not figure.axes:
        return None

    ax = figure.axes[0]
    return {
        "x": ax.get_xlim(),
        "y": ax.get_ylim(),
        "z": ax.get_zlim(),
    }


def apply_axis_limits_3d(ax, axis_limits):
    ax.set_xlim(*axis_limits["x"])
    ax.set_ylim(*axis_limits["y"])
    ax.set_zlim(*axis_limits["z"])

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


def resample_xyz_series(source_times, series, target_times):
    if series is None:
        return None

    series = np.asarray(series, dtype=float)
    return np.column_stack([
        np.interp(target_times, source_times, series[:, dim])
        for dim in range(series.shape[1])
    ])


def resample_mask_series(source_times, mask, target_times):
    if mask is None:
        return None

    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0 or len(target_times) == 0:
        return np.zeros(len(target_times), dtype=bool)
    if len(source_times) != len(mask):
        raise ValueError("source_times and mask must have the same length.")
    if len(source_times) == 1:
        return np.full(len(target_times), bool(mask[0]), dtype=bool)

    right_idx = np.searchsorted(source_times, target_times, side="left")
    right_idx = np.clip(right_idx, 0, len(source_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(source_times) - 1)

    choose_left = np.abs(target_times - source_times[left_idx]) <= np.abs(source_times[right_idx] - target_times)
    nearest_idx = np.where(choose_left, left_idx, right_idx)
    return mask[nearest_idx]


def resample_animation_series(time_values, target_fps, *series):
    elapsed_time = np.asarray(time_values, dtype=float)
    series = [None if values is None else np.asarray(values, dtype=float) for values in series]

    if len(elapsed_time) < 2:
        return series, elapsed_time

    valid_mask = np.isfinite(elapsed_time)
    elapsed_time = elapsed_time[valid_mask]
    series = [None if values is None else values[valid_mask] for values in series]
    if len(elapsed_time) < 2:
        return series, elapsed_time

    increasing_mask = np.ones(len(elapsed_time), dtype=bool)
    increasing_mask[1:] = np.diff(elapsed_time) > 1e-9
    elapsed_time = elapsed_time[increasing_mask]
    series = [None if values is None else values[increasing_mask] for values in series]
    if len(elapsed_time) < 2 or elapsed_time[-1] <= 0:
        return series, elapsed_time

    target_time = np.arange(0.0, elapsed_time[-1] + 1.0 / target_fps, 1.0 / target_fps)
    if target_time[-1] > elapsed_time[-1]:
        target_time[-1] = elapsed_time[-1]

    resampled = [
        resample_xyz_series(elapsed_time, values, target_time)
        if values is not None else None
        for values in series
    ]
    return resampled, target_time


def compute_gif_frame_durations_ms(elapsed_time, fallback_step_s):
    elapsed_time = np.asarray(elapsed_time, dtype=float)
    if len(elapsed_time) == 0:
        return [max(10, int(round(fallback_step_s * 1000.0)))]

    if len(elapsed_time) == 1:
        return [max(10, int(round(fallback_step_s * 1000.0)))]

    desired_step_cs = np.empty(len(elapsed_time), dtype=float)
    desired_step_cs[:-1] = np.diff(elapsed_time) * 100.0
    desired_step_cs[-1] = fallback_step_s * 100.0

    durations_ms = []
    carry = 0.0
    for step_cs in desired_step_cs:
        quantized_cs = max(1, int(round(step_cs + carry)))
        carry = step_cs + carry - quantized_cs
        durations_ms.append(quantized_cs * 10)
    return durations_ms


def capture_animation_frame(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return Image.fromarray(rgba)


def save_animation_gif(fig, update, n_frames, save_path, elapsed_time, gif_fps):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    durations_ms = compute_gif_frame_durations_ms(elapsed_time, 1.0 / max(gif_fps, 1))
    update(0)
    first_frame = capture_animation_frame(fig)

    def iter_remaining_frames():
        for frame_idx in range(1, n_frames):
            update(frame_idx)
            yield capture_animation_frame(fig)

    first_frame.save(
        save_path,
        save_all=True,
        append_images=iter_remaining_frames(),
        duration=durations_ms,
        loop=0,
    )


def compute_step_norms(points):
    """Return frame-to-frame displacement norms, keeping NaN where a step is not finite."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.empty(0, dtype=float)

    step = np.full(len(points) - 1, np.nan, dtype=float)
    finite_step = np.all(np.isfinite(points[:-1]), axis=1) & np.all(np.isfinite(points[1:]), axis=1)
    if np.any(finite_step):
        deltas = points[1:][finite_step] - points[:-1][finite_step]
        step[finite_step] = np.linalg.norm(deltas, axis=1)
    return step


def interpolate_invalid_points(points, invalid_mask):
    """Linearly fill invalid rope-tip samples using surrounding valid frames."""
    points = np.asarray(points, dtype=float)
    repaired = points.copy()
    invalid_mask = np.asarray(invalid_mask, dtype=bool)

    if len(points) == 0 or not np.any(invalid_mask):
        return repaired

    valid_mask = ~invalid_mask & np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(valid_mask) < 2:
        return repaired

    frame_index = np.arange(len(points), dtype=float)
    for dim in range(points.shape[1]):
        repaired[invalid_mask, dim] = np.interp(
            frame_index[invalid_mask],
            frame_index[valid_mask],
            points[valid_mask, dim],
        )
    return repaired


def local_consistent_step(points, start, direction, horizon=3):
    """Estimate the local motion scale after a candidate jump using nearby steps."""
    steps = []
    idx = start
    while 0 <= idx + direction < len(points) and len(steps) < horizon:
        p0 = points[idx]
        p1 = points[idx + direction]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            steps.append(np.linalg.norm(p1 - p0))
        idx += direction
    return float(np.median(steps)) if steps else np.nan


def extend_release_tail(points, invalid_mask, frozen_mask, start_idx, jump_threshold):
    """Extend a plateau repair into one or two release frames when the jump settles immediately."""
    step = compute_step_norms(points)
    end_idx = start_idx

    while end_idx < len(points) - 1:
        current_jump = step[end_idx]
        if not np.isfinite(current_jump) or current_jump <= jump_threshold:
            break

        future_consistent = local_consistent_step(points, end_idx + 1, +1)
        if np.isfinite(future_consistent) and future_consistent < jump_threshold * 0.6:
            invalid_mask[end_idx] = True
            frozen_mask[end_idx] = True
            end_idx += 1
            continue
        break


def mark_flat_jump_runs(points, invalid_mask, frozen_mask, freeze_tolerance, jump_threshold, residual_threshold):
    """Mark frozen plateaus that are inconsistent with the surrounding motion."""
    n_points = len(points)

    def maybe_mark_run(run_start, run_end):
        if run_start is None or run_end <= run_start:
            return

        plateau = points[run_start:run_end + 1]
        if not np.all(np.isfinite(plateau)):
            return

        plateau_value = np.mean(plateau, axis=0)
        left = run_start - 1
        right = run_end + 1
        left_valid = left >= 0 and np.all(np.isfinite(points[left]))
        right_valid = right < n_points and np.all(np.isfinite(points[right]))
        entry_jump = np.linalg.norm(plateau_value - points[left]) if left_valid else 0.0
        exit_jump = np.linalg.norm(points[right] - plateau_value) if right_valid else 0.0

        if left_valid and right_valid:
            bridge_value = 0.5 * (points[left] + points[right])
            bridge_residual = np.linalg.norm(plateau_value - bridge_value)
            if (
                entry_jump > jump_threshold
                and exit_jump > jump_threshold
                and bridge_residual > residual_threshold
            ):
                invalid_mask[run_start:run_end + 1] = True
                frozen_mask[run_start:run_end + 1] = True
                return

        run_length = run_end - run_start + 1
        if run_length < 5:
            return

        if right_valid and exit_jump > jump_threshold:
            forward_step = local_consistent_step(points, right, +1)
            if np.isfinite(forward_step) and forward_step < jump_threshold * 0.6:
                stale_start = min(run_end, run_start + 1)
                invalid_mask[stale_start:run_end + 1] = True
                frozen_mask[stale_start:run_end + 1] = True
                extend_release_tail(points, invalid_mask, frozen_mask, right, jump_threshold)
                return

        if left_valid and entry_jump > jump_threshold:
            backward_step = local_consistent_step(points, left, -1)
            if np.isfinite(backward_step) and backward_step < jump_threshold * 0.6:
                stale_end = max(run_start, run_end - 1)
                invalid_mask[run_start:stale_end + 1] = True
                frozen_mask[run_start:stale_end + 1] = True

    run_start = None
    for idx in range(1, n_points):
        if not np.all(np.isfinite(points[idx - 1:idx + 1])):
            maybe_mark_run(run_start, idx - 1)
            run_start = None
            continue

        if np.linalg.norm(points[idx] - points[idx - 1]) <= freeze_tolerance:
            if run_start is None:
                run_start = idx - 1
        else:
            maybe_mark_run(run_start, idx - 1)
            run_start = None

    maybe_mark_run(run_start, n_points - 1)
    return invalid_mask


def mark_bridge_spikes(points, invalid_mask, jump_spike_mask, jump_threshold, residual_threshold):
    """Mark isolated frames that bridge two large jumps but are locally inconsistent."""
    step = compute_step_norms(points)

    for idx in range(1, len(points) - 1):
        if invalid_mask[idx] or not np.all(np.isfinite(points[idx - 1:idx + 2])):
            continue

        prev_jump = step[idx - 1]
        next_jump = step[idx]
        if not (np.isfinite(prev_jump) and np.isfinite(next_jump)):
            continue
        if prev_jump <= jump_threshold or next_jump <= jump_threshold:
            continue

        before = step[idx - 2] if idx - 2 >= 0 else 0.0
        after = step[idx + 1] if idx + 1 < len(step) else 0.0
        if (
            (np.isfinite(before) and before > jump_threshold * 0.6)
            or (np.isfinite(after) and after > jump_threshold * 0.6)
        ):
            continue

        midpoint = 0.5 * (points[idx - 1] + points[idx + 1])
        if np.linalg.norm(points[idx] - midpoint) > residual_threshold:
            invalid_mask[idx] = True
            jump_spike_mask[idx] = True

    return invalid_mask


def detect_rope_tip_spikes(
    rope_tip,
    rolling_window=7,
    max_passes=4,
    min_jump=SPIKE_MIN_JUMP,
    jump_factor=SPIKE_JUMP_FACTOR,
    min_residual=SPIKE_MIN_RESIDUAL,
    residual_factor=SPIKE_RESIDUAL_FACTOR,
):
    """Classify rope-tip sensor spikes without modifying the original signal."""
    rope_tip = np.asarray(rope_tip, dtype=float)
    n_points = len(rope_tip)
    invalid_mask = ~np.all(np.isfinite(rope_tip), axis=1)
    jump_spike_mask = np.zeros(n_points, dtype=bool)
    frozen_spike_mask = np.zeros(n_points, dtype=bool)

    if n_points == 0:
        stats = {
            "num_points": 0,
            "corrected_points": 0,
            "jump_spikes": 0,
            "frozen_spikes": 0,
            "typical_step": 0.0,
            "jump_threshold": min_jump,
            "residual_threshold": min_residual,
        }
        return invalid_mask, jump_spike_mask, frozen_spike_mask, stats

    step = compute_step_norms(rope_tip)
    valid_steps = step[np.isfinite(step) & (step > 1e-9)]
    typical_step = float(np.median(valid_steps)) if len(valid_steps) else 0.0
    jump_threshold = max(min_jump, jump_factor * typical_step)
    residual_threshold = max(min_residual, residual_factor * typical_step)
    freeze_tolerance = max(1e-4, SPIKE_FREEZE_FACTOR * typical_step)
    rolling_window = max(3, int(rolling_window))
    if rolling_window % 2 == 0:
        rolling_window += 1

    invalid_mask = mark_flat_jump_runs(
        rope_tip,
        invalid_mask,
        frozen_spike_mask,
        freeze_tolerance=freeze_tolerance,
        jump_threshold=jump_threshold,
        residual_threshold=residual_threshold,
    )
    invalid_mask = mark_bridge_spikes(
        rope_tip,
        invalid_mask,
        jump_spike_mask,
        jump_threshold=jump_threshold,
        residual_threshold=residual_threshold,
    )

    step_prev = np.zeros(n_points, dtype=float)
    step_next = np.zeros(n_points, dtype=float)
    if len(step):
        finite_step = np.nan_to_num(step, nan=0.0)
        step_prev[1:] = finite_step
        step_next[:-1] = finite_step

    for _ in range(max_passes):
        repaired = interpolate_invalid_points(rope_tip, invalid_mask)
        baseline = np.column_stack([
            pd.Series(repaired[:, axis]).rolling(
                rolling_window,
                center=True,
                min_periods=1,
            ).median().to_numpy()
            for axis in range(repaired.shape[1])
        ])
        residual = np.linalg.norm(rope_tip - baseline, axis=1)

        new_invalid = (
            ~invalid_mask
            & np.all(np.isfinite(rope_tip), axis=1)
            & np.isfinite(residual)
            & (residual > residual_threshold)
            & ((step_prev > jump_threshold) | (step_next > jump_threshold))
        )
        if not np.any(new_invalid):
            break
        invalid_mask |= new_invalid
        jump_spike_mask |= new_invalid

    stats = {
        "num_points": n_points,
        "corrected_points": int(np.count_nonzero(invalid_mask)),
        "jump_spikes": int(np.count_nonzero(jump_spike_mask)),
        "frozen_spikes": int(np.count_nonzero(frozen_spike_mask)),
        "typical_step": typical_step,
        "jump_threshold": jump_threshold,
        "residual_threshold": residual_threshold,
    }
    return invalid_mask, jump_spike_mask, frozen_spike_mask, stats


def repair_rope_tip_occlusions(
    rope_tip,
    rolling_window=7,
    max_passes=4,
    min_jump=SPIKE_MIN_JUMP,
    jump_factor=SPIKE_JUMP_FACTOR,
    min_residual=SPIKE_MIN_RESIDUAL,
    residual_factor=SPIKE_RESIDUAL_FACTOR,
):
    """Detect short occlusion-induced jumps and replace them by linear interpolation."""
    rope_tip = np.asarray(rope_tip, dtype=float)
    invalid_mask, _jump_spike_mask, _frozen_spike_mask, stats = detect_rope_tip_spikes(
        rope_tip,
        rolling_window=rolling_window,
        max_passes=max_passes,
        min_jump=min_jump,
        jump_factor=jump_factor,
        min_residual=min_residual,
        residual_factor=residual_factor,
    )
    repaired = interpolate_invalid_points(rope_tip, invalid_mask)
    return repaired, invalid_mask, stats


def make_trail_collection(
    points,
    color=ROPE_TIP_COLOR,
    linewidth=3.0,
    jump_mask=None,
    frozen_mask=None,
):
    """Make fading trail segments for the recent rope-tip history."""
    if len(points) < 2:
        return Line3DCollection([], linewidths=linewidth)

    jump_mask = np.zeros(len(points), dtype=bool) if jump_mask is None else np.asarray(jump_mask, dtype=bool)
    frozen_mask = np.zeros(len(points), dtype=bool) if frozen_mask is None else np.asarray(frozen_mask, dtype=bool)
    if len(jump_mask) != len(points) or len(frozen_mask) != len(points):
        raise ValueError("Trail masks must have the same length as the trail points.")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(0.50, 1.0, len(segments))
    colors = []
    for idx, alpha in enumerate(alphas):
        if jump_mask[idx] or jump_mask[idx + 1]:
            segment_color = JUMP_SPIKE_COLOR
        elif frozen_mask[idx] or frozen_mask[idx + 1]:
            segment_color = FROZEN_SPIKE_COLOR
        else:
            segment_color = color
        colors.append((segment_color[0], segment_color[1], segment_color[2], alpha))

    collection = Line3DCollection(segments, colors=colors, linewidths=linewidth)
    return collection


def make_circle_line(center, radius, z_level, n_theta=240):
    cx, cy, _ = center
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    z = np.full_like(theta, z_level)
    return x, y, z


def add_obstacle_cylinder(ax, center):
    cylinder_artists = []
    surface_colors = [OBSTACLE_SIDE_COLOR, OBSTACLE_TOP_COLOR, OBSTACLE_BOTTOM_COLOR]

    for (x_cyl, y_cyl, z_cyl), surface_color in zip(make_cylinder_mesh(center), surface_colors):
        surface = ax.plot_surface(
            x_cyl,
            y_cyl,
            z_cyl,
            color=surface_color,
            linewidth=0,
            antialiased=True,
            shade=True,
            zorder=OBSTACLE_ZORDER,
        )
        cylinder_artists.append(surface)

    top_z = center[2] + OBS_HEIGHT / 2.0 + 1e-4
    bottom_z = center[2] - OBS_HEIGHT / 2.0 + 5e-5

    x_top, y_top, z_top = make_circle_line(center, OBS_RADIUS, top_z)
    top_outline, = ax.plot(
        x_top,
        y_top,
        z_top,
        color=OBSTACLE_EDGE_COLOR,
        linewidth=1.6,
        zorder=OBSTACLE_ZORDER + 1,
    )
    cylinder_artists.append(top_outline)

    x_bottom, y_bottom, z_bottom = make_circle_line(center, OBS_RADIUS, bottom_z)
    bottom_outline, = ax.plot(
        x_bottom,
        y_bottom,
        z_bottom,
        color=OBSTACLE_EDGE_COLOR,
        linewidth=1.0,
        alpha=0.65,
        zorder=OBSTACLE_ZORDER,
    )
    cylinder_artists.append(bottom_outline)

    x_margin, y_margin, z_margin = make_circle_line(center, OBS_MARGIN_RADIUS, top_z + 1e-4)
    margin_ring, = ax.plot(
        x_margin,
        y_margin,
        z_margin,
        color=OBSTACLE_MARGIN_COLOR,
        linewidth=1.8,
        linestyle=(0, (4, 3)),
        zorder=OBSTACLE_ZORDER + 2,
    )
    cylinder_artists.append(margin_ring)

    return cylinder_artists


def build_legend(
    ax,
    goal_line,
    goal_point_artist,
    rope_tip_artist,
    connector_line,
    include_obstacle,
    include_jump_spikes=False,
    include_frozen_spikes=False,
):
    handles = [goal_line, goal_point_artist, rope_tip_artist]
    labels = ["goal traj", "goal point", "rope tip"]
    handler_map = None

    if include_jump_spikes or include_frozen_spikes:
        handles.append(
            Line2D([], [], color=JUMP_SPIKE_COLOR, linewidth=3.0)
        )
        labels.append("perception error")

    if include_obstacle:
        handles.append(ObstacleLegendHandle())
        labels.append("obstacle + margin")
        handler_map = {ObstacleLegendHandle: HandlerObstacleLegend()}

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.90),
        handler_map=handler_map,
        handlelength=2.2,
        handleheight=1.6,
    )


def animate_goal_rope_tracking(
    csv_path,
    goal_point=None,
    trail_len=80,
    interval=33,
    stride=1,
    save_path=None,
    fps=30,
    dpi=160,
    repair_rope_tip=True,
    save_gif=False,
    gif_dir=None,
    gif_fps=60,
    save_mp4=False,
    mp4_dir=None,
    show_legend=True,
    axis_limits=None,
):
    df = pd.read_csv(csv_path)

    goal = build_goal_traj_from_paper_csv(df)
    rope_tip, selected_goal_point = build_rope_tip(df, goal_point=goal_point)
    obs = build_obs_from_paper_csv(df)
    time_values = build_time_from_paper_csv(df)

    if repair_rope_tip:
        rope_tip, _rope_tip_invalid_mask, rope_tip_repair_stats = repair_rope_tip_occlusions(rope_tip)
        if rope_tip_repair_stats["corrected_points"]:
            corrected_ratio = 100.0 * rope_tip_repair_stats["corrected_points"] / rope_tip_repair_stats["num_points"]
            print(
                "Interpolated "
                f"{rope_tip_repair_stats['corrected_points']} suspicious rope-tip frames "
                f"({corrected_ratio:.1f}%) using occlusion jump repair."
            )

    _rope_tip_invalid_mask, jump_spike_mask, frozen_spike_mask, _rope_tip_spike_stats = detect_rope_tip_spikes(rope_tip)

    goal, rope_tip, obs, time_values, jump_spike_mask, frozen_spike_mask = finite_rows(
        goal,
        rope_tip,
        obs,
        None if time_values is None else time_values[:, None],
        jump_spike_mask[:, None],
        frozen_spike_mask[:, None],
    )
    if time_values is not None:
        time_values = time_values[:, 0]
    jump_spike_mask = jump_spike_mask[:, 0].astype(bool)
    frozen_spike_mask = frozen_spike_mask[:, 0].astype(bool)

    if stride > 1:
        goal = goal[::stride]
        rope_tip = rope_tip[::stride]
        if obs is not None:
            obs = obs[::stride]
        if time_values is not None:
            time_values = time_values[::stride]
        jump_spike_mask = jump_spike_mask[::stride]
        frozen_spike_mask = frozen_spike_mask[::stride]

    n_frames = min(len(goal), len(rope_tip), len(jump_spike_mask), len(frozen_spike_mask))
    if obs is not None:
        n_frames = min(n_frames, len(obs))
    if time_values is not None:
        n_frames = min(n_frames, len(time_values))

    goal = goal[:n_frames]
    rope_tip = rope_tip[:n_frames]
    if obs is not None:
        obs = obs[:n_frames]
    if time_values is not None:
        time_values = time_values[:n_frames]
    jump_spike_mask = jump_spike_mask[:n_frames]
    frozen_spike_mask = frozen_spike_mask[:n_frames]

    if n_frames <= 2:
        raise ValueError("Need more than 2 frames after alignment to trim the first and last frame.")

    goal = goal[1:-1]
    rope_tip = rope_tip[1:-1]
    if obs is not None:
        obs = obs[1:-1]
    if time_values is not None:
        time_values = time_values[1:-1]
    jump_spike_mask = jump_spike_mask[1:-1]
    frozen_spike_mask = frozen_spike_mask[1:-1]
    n_frames = len(goal)

    elapsed_time_values = None
    if time_values is not None and len(time_values):
        elapsed_time_values = time_values - time_values[0]

    resolved_save_path = None if save_path is None else Path(save_path)
    if save_gif and save_mp4:
        raise ValueError("save_gif and save_mp4 cannot both be True.")

    if resolved_save_path is None:
        if save_gif:
            resolved_save_path = build_default_gif_path(csv_path, output_dir=gif_dir)
        elif save_mp4:
            resolved_save_path = build_default_mp4_path(csv_path, output_dir=mp4_dir)

    save_suffix = None if resolved_save_path is None else resolved_save_path.suffix.lower()
    if save_suffix is not None and save_suffix not in {".gif", ".mp4"}:
        raise ValueError("save_path must end with .gif or .mp4")
    if save_gif and save_suffix not in {None, ".gif"}:
        raise ValueError("save_gif requires a .gif output path.")
    if save_mp4 and save_suffix not in {None, ".mp4"}:
        raise ValueError("save_mp4 requires a .mp4 output path.")

    save_is_gif = save_suffix == ".gif"
    save_is_mp4 = save_suffix == ".mp4"
    export_fps = gif_fps if save_is_gif else (fps if save_is_mp4 else None)

    if export_fps is not None:
        if elapsed_time_values is not None and len(elapsed_time_values) >= 2:
            source_elapsed_time_values = elapsed_time_values.copy()
            resampled_series, elapsed_time_values = resample_animation_series(
                elapsed_time_values,
                export_fps,
                goal,
                rope_tip,
                obs,
            )
            goal, rope_tip, obs = resampled_series
            jump_spike_mask = resample_mask_series(source_elapsed_time_values, jump_spike_mask, elapsed_time_values)
            frozen_spike_mask = resample_mask_series(source_elapsed_time_values, frozen_spike_mask, elapsed_time_values)
        else:
            elapsed_time_values = np.arange(n_frames, dtype=float) / max(export_fps, 1)
        n_frames = len(goal)

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False

    def format_elapsed_time(frame_idx):
        if elapsed_time_values is not None and len(elapsed_time_values):
            return f"t = {elapsed_time_values[frame_idx]:.2f} s"

        title_time_step = (1.0 / max(fps, 1)) if resolved_save_path else (interval / 1000.0)
        return f"t = {frame_idx * title_time_step:.2f} s"

    # Static full trajectories
    goal_line, = ax.plot(goal[:, 0], goal[:, 1], goal[:, 2],
                         color="tab:blue", linewidth=2.4, label="goal traj")
    # Dynamic artists
    goal_point_artist = ax.scatter([], [], [], s=180, color="limegreen",
                                   edgecolor="black", linewidth=0.7, depthshade=False,
                                   label="current goal point")
    rope_tip_artist = ax.scatter([], [], [], s=95, color=ROPE_TIP_COLOR,
                                 edgecolor="black", linewidth=0.6, depthshade=False,
                                 label="current rope tip")
    connector_line, = ax.plot([], [], [], color="gray", linewidth=1.2,
                              alpha=0.55, label="tracking error")
    trail_collection = None
    cylinder_artists = []

    rope_tip_artist.set_zorder(ROPE_TIP_ZORDER + 2)
    goal_point_artist.set_zorder(ROPE_TIP_ZORDER + 1)

    # Axis range includes obstacle radius/height.
    all_xyz = [goal, rope_tip]
    if obs is not None:
        obs_pad = np.vstack([
            obs,
            obs + np.array([OBS_MARGIN_RADIUS, OBS_MARGIN_RADIUS, OBS_HEIGHT / 2.0]),
            obs - np.array([OBS_MARGIN_RADIUS, OBS_MARGIN_RADIUS, OBS_HEIGHT / 2.0]),
        ])
        all_xyz.append(obs_pad)
    all_xyz = np.vstack(all_xyz)
    if axis_limits is None:
        set_axes_equal_3d(ax, all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2], padding=0.04)
    else:
        apply_axis_limits_3d(ax, axis_limits)
    set_top_down_view(ax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(format_elapsed_time(0), y=0.8)
    ax.grid(True)
    if show_legend:
        build_legend(
            ax,
            goal_line,
            goal_point_artist,
            rope_tip_artist,
            connector_line,
            include_obstacle=obs is not None,
            include_jump_spikes=np.any(jump_spike_mask),
            include_frozen_spikes=np.any(frozen_spike_mask),
        )

    def set_scatter_3d(scatter, xyz):
        scatter._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])

    def update(frame):
        nonlocal trail_collection, cylinder_artists

        g = goal[frame]
        r = rope_tip[frame]

        set_scatter_3d(goal_point_artist, g)
        set_scatter_3d(rope_tip_artist, r)

        if jump_spike_mask[frame] or frozen_spike_mask[frame]:
            rope_tip_artist.set_facecolor([FROZEN_SPIKE_COLOR])
        else:
            rope_tip_artist.set_facecolor([ROPE_TIP_COLOR])

        connector_line.set_data([r[0], g[0]], [r[1], g[1]])
        connector_line.set_3d_properties([r[2], g[2]])

        # Rope-tip fading history
        start = max(0, frame - trail_len + 1)
        trail_points = rope_tip[start:frame + 1]
        trail_jump_mask = jump_spike_mask[start:frame + 1]
        trail_frozen_mask = frozen_spike_mask[start:frame + 1]
        if trail_collection is not None:
            trail_collection.remove()
        if len(trail_points) >= 2:
            trail_collection = make_trail_collection(
                trail_points,
                color=ROPE_TIP_COLOR,
                linewidth=3.0,
                jump_mask=trail_jump_mask,
                frozen_mask=trail_frozen_mask,
            )
            trail_collection.set_zorder(ROPE_TIP_ZORDER + 1)
            ax.add_collection3d(trail_collection)
        else:
            trail_collection = None

        if obs is not None:
            for artist in cylinder_artists:
                artist.remove()
            cylinder_artists = add_obstacle_cylinder(ax, obs[frame])

        ax.set_title(format_elapsed_time(frame), y=0.9)

        artists = [goal_point_artist, rope_tip_artist, connector_line]
        if trail_collection is not None:
            artists.append(trail_collection)
        artists.extend(cylinder_artists)
        return artists

    animation_interval = 1000.0 / max(export_fps, 1) if export_fps is not None else interval

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=animation_interval,
        blit=False,
        repeat=True,
    )

    # Trigger an initial draw so matplotlib starts the animation lifecycle
    # before control returns to the caller.
    fig.canvas.draw()

    if resolved_save_path is not None:
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_is_gif:
            save_animation_gif(
                fig,
                update,
                n_frames,
                resolved_save_path,
                elapsed_time_values,
                gif_fps,
            )
        else:
            ani.save(
                str(resolved_save_path),
                writer=animation.FFMpegWriter(fps=fps),
                dpi=dpi,
            )

        print(f"Saved animation to: {resolved_save_path}")

    return ani


def main():
    global ACTIVE_ANIMATION

    parser = argparse.ArgumentParser(description="Animate 3D goal tracking and obstacle avoidance.")
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSV_PATH,
        help="Input CSV path(s).",
    )
    parser.add_argument(
        "--goal-point",
        type=int,
        default=None,
        help="Rope point index to visualize. Default: last rope node, treated as rope tip.",
    )
    parser.add_argument(
        "--trail-len",
        type=int,
        default=80,
        help="Number of previous frames shown as rope-tip residual trail.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame to speed up rendering.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=33,
        help="Delay between frames in milliseconds for interactive playback.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional custom output path ending with .mp4 or .gif.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Export FPS for mp4. Use --save-mp4 --fps 60 for real-time 60 fps video. GIF export is always saved at real-time 60 fps.",
    )
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a real-time GIF named after the input CSV if --save is not provided.",
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Optional output directory for the auto-named GIF. Default: the CSV directory.",
    )
    parser.add_argument(
        "--save-mp4",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_MP4,
        help="Save an MP4 named after the input CSV if --save is not provided.",
    )
    parser.add_argument(
        "--mp4-dir",
        type=str,
        default=None,
        help="Optional output directory for the auto-named MP4. Default: the CSV directory.",
    )
    parser.add_argument(
        "--batch-first-legend",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BATCH_FIRST_LEGEND,
        help="When batch saving multiple inputs, show the legend only on the first animation.",
    )
    parser.add_argument(
        "--repair-rope-tip",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REPAIR_ROPE_TIP,
        help="Enable or disable in-memory rope-tip occlusion repair before plotting.",
    )
    args = parser.parse_args()

    csv_paths = normalize_csv_paths(args.csv)
    if len(csv_paths) > 1 and args.save is not None:
        raise ValueError("--save cannot be used with multiple CSV inputs. Use --mp4-dir or --gif-dir instead.")

    batch_export_enabled = len(csv_paths) > 1 and (args.save_mp4 or args.save_gif)
    if batch_export_enabled:
        reference_axis_limits = None
        if args.save_mp4 and len(csv_paths) > 1:
            reference_animation = animate_goal_rope_tracking(
                csv_path=csv_paths[1],
                goal_point=args.goal_point,
                trail_len=args.trail_len,
                interval=args.interval,
                stride=args.stride,
                save_path=None,
                fps=args.fps,
                repair_rope_tip=args.repair_rope_tip,
                save_gif=False,
                gif_dir=args.gif_dir,
                save_mp4=False,
                mp4_dir=args.mp4_dir,
                show_legend=False,
                axis_limits=None,
            )
            reference_axis_limits = get_animation_axis_limits(reference_animation)
            close_animation_figure(reference_animation)

        last_animation = None
        for csv_idx, csv_path in enumerate(csv_paths):
            last_animation = animate_goal_rope_tracking(
                csv_path=csv_path,
                goal_point=args.goal_point,
                trail_len=args.trail_len,
                interval=args.interval,
                stride=args.stride,
                save_path=None,
                fps=args.fps,
                repair_rope_tip=args.repair_rope_tip,
                save_gif=args.save_gif,
                gif_dir=args.gif_dir,
                save_mp4=args.save_mp4,
                mp4_dir=args.mp4_dir,
                show_legend=not args.batch_first_legend or csv_idx == 0,
                axis_limits=reference_axis_limits,
            )
            close_animation_figure(last_animation)

        ACTIVE_ANIMATION = last_animation
        return ACTIVE_ANIMATION

    if len(csv_paths) > 1:
        warnings.warn(
            "Multiple CSV inputs were provided while batch export is disabled; only the first file will be rendered.",
            RuntimeWarning,
        )

    ACTIVE_ANIMATION = animate_goal_rope_tracking(
        csv_path=csv_paths[0],
        goal_point=args.goal_point,
        trail_len=args.trail_len,
        interval=args.interval,
        stride=args.stride,
        save_path=args.save,
        fps=args.fps,
        repair_rope_tip=args.repair_rope_tip,
        save_gif=args.save_gif,
        gif_dir=args.gif_dir,
        save_mp4=args.save_mp4,
        mp4_dir=args.mp4_dir,
        show_legend=True,
        axis_limits=None,
    )

    if args.save is None and not args.save_gif and not args.save_mp4:
        plt.show()

    return ACTIVE_ANIMATION


if __name__ == "__main__":
    main()
