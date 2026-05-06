import copy
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import *
from real_rope_SPiD.rope_franka import RopeFranka


# === C++-matched trajectory generation (trajectory_utils.h) ===
def cpp_half_dense_then_uniform(N, ratio=0.5, sharpness=2.0, start=0.0, end=2.0, device=None, dtype=torch.float32):
    """Exact Python rewrite of trajectory_utils::halfDenseThenUniform."""
    if N <= 1:
        return torch.tensor([start], device=device, dtype=dtype)

    n_dense = int(N * ratio)  # C++ static_cast<int>: truncate toward zero for positive N
    n_dense = max(2, min(n_dense, N))
    n_uniform = N - n_dense

    base = np.exp(sharpness)
    x_dense = []
    for i in range(n_dense):
        t = 0.0 if n_dense == 1 else float(i) / float(n_dense - 1)
        val = ratio * (base ** t - 1.0) / (base - 1.0)
        x_dense.append(val)

    final_spacing = (x_dense[n_dense - 1] - x_dense[n_dense - 2]) if n_dense >= 2 else ratio
    x_uniform = [x_dense[-1] + final_spacing * float(i) for i in range(1, n_uniform + 1)]
    x = np.asarray(x_dense + x_uniform, dtype=np.float64)
    scale = (end - start) / x[-1]
    x = x * scale + start
    return torch.tensor(x, device=device, dtype=dtype)


def cpp_extend_last_point(traj, repeat):
    """Exact Python rewrite of trajectory_utils::extendLastPoint."""
    repeat = int(repeat)
    if repeat <= 0 or traj.numel() == 0:
        return traj
    return torch.cat([traj, traj[-1:].repeat(repeat, 1)], dim=0)


def cpp_eight_trajectory(points, scale_x=0.25, scale_y=0.45, z0=0.2, loops=1, bias=None):
    """Exact Python rewrite of trajectory_utils::eightTrajectory."""
    device = points.device
    dtype = points.dtype
    if points.numel() == 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    if bias is None:
        bias = torch.zeros(3, device=device, dtype=dtype)
    else:
        bias = torch.as_tensor(bias, device=device, dtype=dtype)

    last = points[-1]
    a = -0.2
    u = points / last * (2.0 * np.pi * float(loops))
    t = u + a * torch.sin(2.0 * u)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    denom = 1.0 + sin_t * sin_t

    y = -float(scale_x) * cos_t / denom
    x = float(scale_y) * (sin_t * cos_t) / denom
    z = torch.full_like(x, float(z0))
    traj = torch.stack([x, y, z], dim=-1)

    start_shift = traj[0].clone()
    start_shift[2] = 0.0
    traj = traj - start_shift + bias
    return traj.to(dtype=torch.float32)


def cpp_eight_trajectory_continuous(N, loops=5, ramp_fraction=0.1, scale_x=0.25, scale_y=0.45, z0=0.2, bias=None, device=None):
    """Exact Python rewrite of trajectory_utils::eightTrajectoryContinuous."""
    N = int(N)
    if N <= 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    dtype = torch.float64
    if bias is None:
        bias_t = torch.zeros(3, device=device, dtype=dtype)
    else:
        bias_t = torch.as_tensor(bias, device=device, dtype=dtype)

    total_angle = 2.0 * np.pi * float(loops)
    n_ramp = max(1, int(N * ramp_fraction))

    u_vals = np.zeros((N,), dtype=np.float64)
    cum = 0.0
    for i in range(1, N):
        if i < n_ramp:
            speed = 0.5 * (1.0 - np.cos(np.pi * float(i) / float(n_ramp)))
        else:
            speed = 1.0
        cum += speed
        u_vals[i] = cum
    if cum > 0.0:
        u_vals *= total_angle / cum

    u = torch.tensor(u_vals, device=device, dtype=dtype)
    a = -0.2
    t = u + a * torch.sin(2.0 * u)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    denom = 1.0 + sin_t * sin_t
    y = -float(scale_x) * cos_t / denom
    x = float(scale_y) * (sin_t * cos_t) / denom
    z = torch.full_like(x, float(z0))
    traj = torch.stack([x, y, z], dim=-1)

    start_shift = traj[0].clone()
    start_shift[2] = 0.0
    traj = traj - start_shift + bias_t
    return traj.to(dtype=torch.float32)


def cpp_egg_trajectory(points, scale_x=0.25, scale_y=0.35, k=0.3, z0=0.2, bias=None):
    """Exact Python rewrite of trajectory_utils::eggTrajectory."""
    device = points.device
    dtype = points.dtype
    if points.numel() == 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    if bias is None:
        bias = torch.zeros(3, device=device, dtype=dtype)
    else:
        bias = torch.as_tensor(bias, device=device, dtype=dtype)

    p0 = points[0]
    pn = points[-1]
    range_v = pn - p0
    t = torch.where(range_v > 0.0, (points - p0) / range_v, torch.zeros_like(points))
    theta = t * (2.0 * np.pi)
    theta_shift = theta - np.pi / 2.0
    r = 1.0 - float(k) * torch.cos(theta_shift)
    x = float(scale_x) * r * torch.cos(theta_shift)
    y = float(scale_y) * r * torch.sin(theta_shift)
    z = torch.full_like(x, float(z0))
    traj = torch.stack([x, y, z], dim=-1)

    min_y_idx = int(torch.argmin(traj[:, 1]).item())
    traj = torch.cat([traj[min_y_idx:], traj[:min_y_idx]], dim=0)

    start_shift = traj[0].clone()
    start_shift[2] = 0.0
    traj = traj - start_shift + bias
    return traj.to(dtype=torch.float32)


def cpp_sin_trajectory(points, width=0.45, length=2.0, z0=0.2, bias=None):
    """Exact Python rewrite of trajectory_utils::sinTrajectory."""
    device = points.device
    dtype = points.dtype
    if points.numel() == 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    if bias is None:
        bias = torch.zeros(3, device=device, dtype=dtype)
    else:
        bias = torch.as_tensor(bias, device=device, dtype=dtype)

    p0 = points[0]
    pn = points[-1]
    range_v = torch.where((pn - p0) > 0.0, pn - p0, torch.ones_like(pn))
    t = (points - p0) / range_v
    x = torch.sin(3.0 * points) * float(width)
    y = t * float(length)
    z = torch.full_like(x, float(z0))
    traj = torch.stack([x, y, z], dim=-1)

    start_shift = traj[0].clone()
    start_shift[2] = 0.0
    traj = traj - start_shift + bias
    return traj.to(dtype=torch.float32)


class Controller(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_width, max_action):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(input_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, action_dim)

    def forward(self, batch_pos, batch_vel, goal, goal_pos, goal_vel, smart=False):
        batch_pos = batch_pos.clone()
        batch_vel = batch_vel.clone()
        goal = goal.clone()
        goal_pos = goal_pos.clone()
        goal_vel = goal_vel.clone()

        if smart:
            shift = batch_pos[:, 0:1, 0:3].detach().clone()
            batch_pos = batch_pos - shift
            goal = goal - shift[:, :, :]
            goal_pos = goal_pos - shift[:, 0, :]

        if (
            (not torch.isfinite(batch_vel).all())
            or (not torch.isfinite(batch_pos).all())
            or (not torch.isfinite(goal).all())
            or (not torch.isfinite(goal_pos).all())
            or (not torch.isfinite(goal_vel).all())
        ):
            print("Find nan in nn input!")

        state = torch.cat([
            batch_pos.view(batch_pos.shape[0], -1),
            batch_vel.view(batch_vel.shape[0], -1),
            goal.reshape(goal.shape[0], -1),
            goal_pos,
            goal_vel,
        ], dim=-1)
        state = F.relu(self.l1(state))
        state = F.relu(self.l2(state))
        state = F.relu(self.l3(state))
        action = self.max_action * torch.tanh(self.l4(state))

        head_goal = goal[:, 0, :].clone()
        head_goal[:, 2] += L
        heuristic_action = (head_goal - batch_pos[:, 0, :]) * 5.0
        action = torch.clamp(heuristic_action + action, -self.max_action, self.max_action)

        if not torch.isfinite(action).all():
            print("Find nan in nn action!")

        return action


def build_trajectory_suite(device, ctr_period):
    """Build the original Python training trajectory suite, but with C++ trajectory formulas.

    Only trajectory generation is changed:
      - keep original Python durations: eight=4.0s, egg=3.5s, sin=2.5s
      - keep original point counts from each duration and ctr_period
      - use trajectory_utils.h formulas for halfDenseThenUniform/eight/egg/sin
      - shift each trajectory so its first point matches the initial rope bottom/tip point
    """
    traj_specs = []
    bias = initial_rope_bottom_point(device)

    eight_time = 4.0
    eight_points = cpp_half_dense_then_uniform(
        N=int(1000 / ctr_period * eight_time) + 1,
        ratio=0.3,
        sharpness=2.0,
        start=0.0,
        end=2.0,
        device=device,
        dtype=torch.float32,
    )
    traj_specs.append((
        "eight_4.0s",
        cpp_eight_trajectory(
            eight_points,
            scale_x=0.45 * 2.0 * 0.3,
            scale_y=0.65 * 2.0 * 0.35,
            z0=0.0,
            loops=1,
            bias=bias,
        ),
        eight_time,
    ))

    egg_time = 3.5
    egg_points = cpp_half_dense_then_uniform(
        N=int(1000 / ctr_period * egg_time) + 1,
        ratio=0.2,
        sharpness=2.0,
        start=0.0,
        end=2.0,
        device=device,
        dtype=torch.float32,
    )
    traj_specs.append((
        "egg_3.5s",
        cpp_egg_trajectory(
            egg_points,
            scale_x=0.3 / 2.0,
            scale_y=0.6 / 2.0,
            k=0.3,
            z0=0.0,
            bias=bias,
        ),
        egg_time,
    ))

    sin_time = 2.5
    sin_width = 0.3 / 2.0 * 1.2
    sin_length = 0.6
    sin_points = cpp_half_dense_then_uniform(
        N=int(1000 / ctr_period * sin_time) + 1,
        ratio=0.3,
        sharpness=2.0,
        start=0.0,
        end=2.0,
        device=device,
        dtype=torch.float32,
    )
    traj_specs.append((
        "sin_2.5s",
        cpp_sin_trajectory(
            sin_points,
            width=sin_width,
            length=sin_length,
            z0=0.0,
            bias=bias,
        ),
        sin_time,
    ))

    return traj_specs


def trajectory_metrics(traj, total_time):
    if traj.shape[0] < 2:
        return 0.0, 0.0

    deltas = traj[1:] - traj[:-1]
    step_dist = torch.linalg.norm(deltas, dim=-1)
    total_length = float(step_dist.sum().detach().cpu())

    sample_dt = float(total_time) / float(traj.shape[0] - 1)
    max_speed = float((step_dist / sample_dt).max().detach().cpu())
    return total_length, max_speed


def plot_trajectory_suite(traj_specs, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    summary_lines = []

    for name, traj, total_time in traj_specs:
        traj_cpu = traj.detach().cpu()
        total_length, max_speed = trajectory_metrics(traj_cpu, total_time)
        summary_lines.append(f"{name}: length={total_length:.4f} m, vmax={max_speed:.4f} m/s")

        label = f"{name} | L={total_length:.3f}m | vmax={max_speed:.3f}m/s"
        ax.plot(traj_cpu[:, 0], traj_cpu[:, 1], linewidth=2.0, label=label)
        ax.scatter(traj_cpu[0, 0], traj_cpu[0, 1], marker="o", s=35)
        ax.scatter(traj_cpu[-1, 0], traj_cpu[-1, 1], marker="x", s=45)

    print("[Trajectory Suite]")
    for line in summary_lines:
        print("  " + line)

    ax.set_title("Training Goal Trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160)
        print(f"[Trajectory Suite] saved plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def pad_trajectory_batch(traj_specs, device):
    traj_names = [item[0] for item in traj_specs]
    traj_times = [item[2] for item in traj_specs]
    traj_lengths = torch.tensor([item[1].shape[0] for item in traj_specs], device=device, dtype=torch.long)
    max_length = int(traj_lengths.max().item())

    padded_goals = torch.zeros((len(traj_specs), max_length, 3), device=device)
    for idx, (_, traj, _) in enumerate(traj_specs):
        length = traj.shape[0]
        padded_goals[idx, :length] = traj
        padded_goals[idx, length:] = traj[-1]

    return traj_names, traj_times, padded_goals, traj_lengths


def make_initial_state(batch_size, device):
    positions = initial_rope_positions(batch_size, device)
    velocities = torch.zeros((batch_size, N + 1, 3), device=device)
    return positions, velocities


def build_goal_window_batch(padded_goals, traj_lengths, step, pre_length):
    goal_batch = []
    batch_size = padded_goals.shape[0]

    for batch_idx in range(batch_size):
        traj_len = int(traj_lengths[batch_idx].item())
        traj = padded_goals[batch_idx, :traj_len]

        if step >= traj_len:
            goal = traj[-1:].repeat(pre_length, 1)
        elif step + pre_length <= traj_len:
            goal = traj[step:step + pre_length]
        else:
            remain = traj_len - step
            last_goal = traj[-1:].repeat(pre_length - remain, 1)
            goal = torch.cat([traj[step:], last_goal], dim=0)

        goal_batch.append(goal)

    return torch.stack(goal_batch, dim=0)


def snapshot_module_state(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def make_nonfinite_rollout_result(batch_size, device, history, failure_step, failure_reason):
    nan_scalar = torch.tensor(float('nan'), device=device)
    return {
        'loss': nan_scalar,
        'avg_track_error': nan_scalar,
        'avg_control_penalty': nan_scalar,
        'per_traj_sqerr_sum': torch.zeros((batch_size,), device=device),
        'per_traj_valid_steps': torch.zeros((batch_size,), device=device),
        'positions_history': history,
        'is_finite': False,
        'failure_step': failure_step,
        'failure_reason': failure_reason,
    }


def rollout_trajectory_batch(
    rope_model,
    controller,
    padded_goals,
    traj_lengths,
    rollout_steps,
    record_history=False,
):
    batch_size = padded_goals.shape[0]
    positions, velocities = make_initial_state(batch_size, rope_model.device)
    goal_pos = positions[:, 0, :].clone()
    goal_vel = velocities[:, 0, :].clone()
    franka_hidden = None
    filter_ctl = torch.zeros((batch_size, 3), device=rope_model.device)

    history = []
    if record_history:
        history.append(positions.detach().clone().cpu())

    track_error_sum = torch.zeros((), device=rope_model.device)
    per_traj_sqerr_sum = torch.zeros((batch_size,), device=rope_model.device)
    per_traj_valid_steps = torch.zeros((batch_size,), device=rope_model.device)
    control_penalty_sum = torch.zeros((), device=rope_model.device)
    valid_count = 0

    for step in range(1, rollout_steps + 1):
        active_mask = step < traj_lengths
        if not torch.any(active_mask):
            break

        if not torch.isfinite(positions).all() or not torch.isfinite(velocities).all():
            return make_nonfinite_rollout_result(
                batch_size=batch_size,
                device=rope_model.device,
                history=history,
                failure_step=step,
                failure_reason='state_before_control',
            )

        curr_goal = build_goal_window_batch(padded_goals, traj_lengths, step, pre_length)
        batch_ctl = controller(
            positions,
            velocities,
            curr_goal,
            goal_pos,
            goal_vel,
            smart=smart,
        )
        if not torch.isfinite(batch_ctl).all():
            return make_nonfinite_rollout_result(
                batch_size=batch_size,
                device=rope_model.device,
                history=history,
                failure_step=step,
                failure_reason='controller_output',
            )

        filter_candidate = alpha * batch_ctl + (1.0 - alpha) * filter_ctl
        if not torch.isfinite(filter_candidate).all():
            return make_nonfinite_rollout_result(
                batch_size=batch_size,
                device=rope_model.device,
                history=history,
                failure_step=step,
                failure_reason='filtered_control',
            )

        (
            next_positions,
            next_velocities,
            next_goal_pos,
            next_goal_vel,
            next_franka_hidden,
        ) = rope_model.simulation_for_ctr(
            positions.clone(),
            velocities.clone(),
            filter_candidate,
            goal_pos=goal_pos,
            goal_vel=goal_vel,
            hidden_state=franka_hidden,
            ctr_period=ctr_period,
            control_mode=mode,
            return_goal_state=True,
        )
        if not torch.isfinite(next_positions).all() or not torch.isfinite(next_velocities).all():
            return make_nonfinite_rollout_result(
                batch_size=batch_size,
                device=rope_model.device,
                history=history,
                failure_step=step,
                failure_reason='rope_franka_simulation',
            )

        active_mask_pos = active_mask.view(-1, 1, 1)
        active_mask_vec = active_mask.view(-1, 1)
        positions = torch.where(active_mask_pos, next_positions, positions)
        velocities = torch.where(active_mask_pos, next_velocities, velocities)
        goal_pos = torch.where(active_mask_vec, next_goal_pos, goal_pos)
        goal_vel = torch.where(active_mask_vec, next_goal_vel, goal_vel)
        franka_hidden = (
            next_franka_hidden if franka_hidden is None
            else torch.where(active_mask_vec, next_franka_hidden, franka_hidden)
        )
        filter_ctl = torch.where(active_mask_vec, filter_candidate, filter_ctl)

        goal_point = curr_goal[:, 0, :]
        sqerr = ((positions[:, -1, :] - goal_point) ** 2).sum(dim=-1) * 100.0
        if not torch.isfinite(sqerr).all():
            return make_nonfinite_rollout_result(
                batch_size=batch_size,
                device=rope_model.device,
                history=history,
                failure_step=step,
                failure_reason='tracking_error',
            )

        track_error_sum = track_error_sum + sqerr[active_mask].sum()
        control_penalty_sum = control_penalty_sum + (batch_ctl[active_mask] ** 2).sum()
        per_traj_sqerr_sum = per_traj_sqerr_sum + torch.where(active_mask, sqerr, torch.zeros_like(sqerr))
        per_traj_valid_steps = per_traj_valid_steps + active_mask.float()
        valid_count += int(active_mask.sum().item())

        if record_history:
            history.append(positions.detach().clone().cpu())

    avg_track_error = track_error_sum / max(valid_count, 1)
    avg_control_penalty = control_penalty_sum / max(valid_count, 1)
    loss = avg_track_error * 15.0 + control_penalty_weight * avg_control_penalty

    return {
        'loss': loss,
        'avg_track_error': avg_track_error,
        'avg_control_penalty': avg_control_penalty,
        'per_traj_sqerr_sum': per_traj_sqerr_sum,
        'per_traj_valid_steps': per_traj_valid_steps,
        'positions_history': history,
        'is_finite': True,
        'failure_step': None,
        'failure_reason': None,
    }


def summarize_rollout_metrics(stage_label, traj_names, rollout_result):
    if not rollout_result.get('is_finite', True):
        print(
            f"[{stage_label}] non-finite rollout "
            f"step={rollout_result.get('failure_step')} source={rollout_result.get('failure_reason')}"
        )
        return

    valid_steps = torch.clamp(rollout_result['per_traj_valid_steps'], min=1.0)
    per_traj_mse_m2 = (rollout_result['per_traj_sqerr_sum'] / valid_steps) / 100.0
    per_traj_rmse_cm = torch.sqrt(per_traj_mse_m2) * 100.0

    mean_rmse_cm = per_traj_rmse_cm.mean().item()
    worst_idx = int(torch.argmax(per_traj_rmse_cm).item())
    worst_name = traj_names[worst_idx]
    worst_rmse_cm = per_traj_rmse_cm[worst_idx].item()
    aggregate_loss = rollout_result['loss'].item()

    print(
        f"[{stage_label}] loss={aggregate_loss:.4f} "
        f"mean_rmse={mean_rmse_cm:.2f}cm worst={worst_name}:{worst_rmse_cm:.2f}cm"
    )


def plot_final_tracking_animation(positions_history, goal_traj, dt, record_interval, title, rope_length):
    if isinstance(positions_history, torch.Tensor):
        positions_np = positions_history.detach().cpu().numpy()
    else:
        positions_np = np.asarray(positions_history)

    if isinstance(goal_traj, torch.Tensor):
        goal_np = goal_traj.detach().cpu().numpy()
    else:
        goal_np = np.asarray(goal_traj)

    num_frames = min(positions_np.shape[0], goal_np.shape[0])
    if num_frames <= 0:
        print("[Final Visualization] skipped animation because there are no frames to show")
        return None

    positions_np = positions_np[:num_frames]
    goal_np = goal_np[:num_frames]

    merged_points = np.concatenate((positions_np.reshape(-1, 3), goal_np.reshape(-1, 3)), axis=0)
    mins = merged_points.min(axis=0)
    maxs = merged_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 0.5 * float(rope_length), 1e-3)
    radius *= 1.08

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    def animate(frame_idx):
        ax.clear()

        rope_frame = positions_np[frame_idx]
        goal_frame = goal_np[frame_idx]

        ax.plot(
            rope_frame[:, 0], rope_frame[:, 1], rope_frame[:, 2],
            'o-', lw=1.5, color='crimson', markersize=3, label='Rope'
        )
        ax.plot(
            goal_np[:frame_idx + 1, 0], goal_np[:frame_idx + 1, 1], goal_np[:frame_idx + 1, 2],
            '--', lw=1.2, color='green', alpha=0.75, label='Goal traj'
        )
        ax.scatter(
            rope_frame[-1, 0], rope_frame[-1, 1], rope_frame[-1, 2],
            color='royalblue', s=40, label='Tip'
        )
        ax.scatter(
            goal_frame[0], goal_frame[1], goal_frame[2],
            color='green', s=80, marker='*', label='Goal'
        )

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(f"{title} | t={frame_idx * dt * record_interval:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)
        if frame_idx == 0:
            ax.legend(loc='upper right')

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)
    plt.show()
    return ani


def visualize_final_rollout(traj_names, padded_goals, traj_lengths, rollout_result, preferred_index=None):
    if not rollout_result.get('is_finite', True):
        print("[Final Visualization] skipped because final rollout is non-finite")
        return

    history = rollout_result.get('positions_history', [])
    if not history:
        print("[Final Visualization] skipped because positions_history was not recorded")
        return

    positions_history = torch.stack(history, dim=1)
    if preferred_index is None:
        vis_indices = list(range(len(traj_names)))
    else:
        vis_indices = [int(np.clip(preferred_index, 0, len(traj_names) - 1))]

    for order_idx, vis_idx in enumerate(vis_indices, start=1):
        traj_len = int(traj_lengths[vis_idx].item())
        vis_positions = positions_history[vis_idx, :traj_len]
        vis_goal = padded_goals[vis_idx, :traj_len].detach().cpu()
        vis_name = traj_names[vis_idx]

        print(
            f"[Final Visualization] showing trajectory {order_idx}/{len(vis_indices)}: "
            f"{vis_name} (index={vis_idx}, frames={traj_len})"
        )
        plot_tip_vs_goal_and_error(vis_positions, vis_goal, dt, ctr_period, tip_idx=-1)
        plot_final_tracking_animation(vis_positions, vis_goal, dt, ctr_period, vis_name, L)


# === Real rope + Franka parameters ===
N = 8
segment_lengths = [0.0973, 0.0996, 0.0993, 0.0986, 0.0990, 0.0993, 0.0975, 0.1160]
L = float(sum(segment_lengths))
rope_top_position = [0.48, -0.3, 0.45]
mass = 12.8 / 1000 / N
tip_extra_mass = 15.0 / 1000
k = 0.12
damping = [0.2] * N
k_bend = [0.0006712] * (N - 1)
damping_bend = [0.000401] * (N - 1)
air_drag = 0.3006
g = 10.27
dt = 0.001

ctr_period = 10
pre_length = 75
mode = 'vel'
max_action = 1.5
hidden_width = 512
alpha = 0.1
smart = True
learning_rate = 1e-3
stage_lr_decay = 0.8  # 每进入下一个 rollout stage，基础学习率乘一次
min_stage_learning_rate = 1e-4
epoch_lr_gamma = 0.99  # 每个 epoch 结束后，学习率再乘 0.9
weight_decay = 0.0
control_penalty_weight = 0.0
gradient_clip_norm = 0.01
epochs_per_stage = 30
# Keep SPiD method hyperparameters aligned with controller/SPiD_tracking/training_SPiD_2.py.
rollout_curriculum = list(range(40, 401, 40))
resume_model_name = None
franka_ckpt_name = "best_model.pt"
plot_training_goals = True
show_final_rollout_visualization = True
final_visualize_index = None  # None means visualize all trajectories sequentially.


def initial_rope_positions(batch_size, device):
    top = torch.tensor(rope_top_position, device=device, dtype=torch.float32)
    lengths = torch.tensor(segment_lengths, device=device, dtype=torch.float32)
    z_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.float32),
        torch.cumsum(lengths, dim=0),
    ])

    positions = top.view(1, 1, 3).expand(batch_size, N + 1, 3).clone()
    positions[:, :, 2] = top[2] - z_offsets.view(1, N + 1)
    return positions


def initial_rope_bottom_point(device):
    return initial_rope_positions(1, device)[0, -1].clone()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")

    script_dir = Path(__file__).resolve().parent
    franka_ckpt_path = script_dir / franka_ckpt_name

    rope_model = RopeFranka(
        franka_ckpt_path=franka_ckpt_path,
        N=N,
        L=L,
        segment_lengths=segment_lengths,
        mass=mass,
        tip_extra_mass=tip_extra_mass,
        k=k,
        k_bend=k_bend,
        damping=damping,
        damping_bend=damping_bend,
        air_drag=air_drag,
        g=g,
        dt=dt,
        device=device,
        rope_mode='ctr',
        ctr_period=ctr_period,
        control_mode=mode,
        freeze_franka=True,
    )

    controller = Controller(
        input_dim=(N + 1) * 3 * 2 + 3 * pre_length + 3 + 3,
        action_dim=3,
        hidden_width=hidden_width,
        max_action=max_action,
    ).to(device)

    if resume_model_name is not None:
        resume_path = Path(resume_model_name)
        if not resume_path.is_absolute():
            resume_path = script_dir / resume_model_name
        controller.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Loaded initial weights: {resume_path}")

    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)

    traj_specs = build_trajectory_suite(device, ctr_period)
    initial_top = initial_rope_positions(1, device)[0, 0].detach().cpu().tolist()
    initial_bottom = initial_rope_bottom_point(device).detach().cpu().tolist()
    print(f"[Initial Rope] top={initial_top} bottom={initial_bottom} length={L:.4f} m")
    plot_trajectory_suite(
        traj_specs,
        save_path=script_dir / "training_goal_trajectories.png",
        show=plot_training_goals,
    )
    traj_names, traj_times, padded_goals, traj_lengths = pad_trajectory_batch(traj_specs, device)
    max_control_steps = int(traj_lengths.max().item()) - 1

    print("=" * 100)
    print(f"[Setup] batch_size={len(traj_specs)} trajectories={len(traj_specs)} device={device}")
    print(f"[Setup] rope_franka_ckpt={franka_ckpt_path}")
    print(f"[Setup] ctr_period={ctr_period} pre_length={pre_length} mode={mode} max_action={max_action}")
    print(f"[Setup] alpha={alpha} lr={learning_rate:.2e} epochs_per_stage={epochs_per_stage}")
    print(
        f"[Setup] stage_lr_decay={stage_lr_decay:.2f} min_stage_lr={min_stage_learning_rate:.2e} "
        f"epoch_lr_gamma={epoch_lr_gamma:.2f}"
    )
    print(f"[Setup] rollout_curriculum={rollout_curriculum}")
    print(f"[Setup] max_control_steps={max_control_steps}")
    print("=" * 100)

    for stage_idx, rollout_steps in enumerate(rollout_curriculum):
        stage_rollout = min(rollout_steps, max_control_steps)
        stage_base_lr = max(learning_rate * (stage_lr_decay ** stage_idx), min_stage_learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = stage_base_lr
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=epoch_lr_gamma,
        )

        print()
        print(
            f"[Stage {stage_idx + 1}/{len(rollout_curriculum)} Start] "
            f"rollout={stage_rollout} epochs={epochs_per_stage} base_lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        best_stage_state = None
        best_stage_optimizer_state = None
        best_stage_loss = None
        best_stage_epoch = None
        stage_hit_nonfinite = False

        for epoch_idx in range(epochs_per_stage):
            t0 = time.perf_counter()
            epoch_start_state = snapshot_module_state(controller)
            epoch_start_optimizer_state = copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad()

            rollout_result = rollout_trajectory_batch(
                rope_model,
                controller,
                padded_goals,
                traj_lengths,
                stage_rollout,
                record_history=False,
            )
            loss = rollout_result['loss']

            if rollout_result['is_finite'] and torch.isfinite(loss):
                loss_value = loss.item()
                if best_stage_loss is None or loss_value < best_stage_loss:
                    best_stage_loss = loss_value
                    best_stage_state = epoch_start_state
                    best_stage_optimizer_state = epoch_start_optimizer_state
                    best_stage_epoch = epoch_idx + 1

            if (not rollout_result['is_finite']) or (not torch.isfinite(loss)):
                stage_hit_nonfinite = True
                if best_stage_state is not None:
                    controller.load_state_dict(best_stage_state)
                    optimizer.load_state_dict(best_stage_optimizer_state)
                print(
                    f"[WARN][Stage {stage_idx + 1}/{len(rollout_curriculum)}] "
                    f"[Epoch {epoch_idx + 1}/{epochs_per_stage}] non-finite loss "
                    f"step={rollout_result.get('failure_step')} source={rollout_result.get('failure_reason')}"
                )
                if best_stage_epoch is not None:
                    print(
                        f"[Stage {stage_idx + 1}/{len(rollout_curriculum)}] restored best epoch "
                        f"{best_stage_epoch}/{epochs_per_stage} loss={best_stage_loss:.4f}"
                    )
                break

            if loss.grad_fn is None:
                print(
                    f"[WARN][Stage {stage_idx + 1}/{len(rollout_curriculum)}] "
                    f"[Epoch {epoch_idx + 1}/{epochs_per_stage}] loss detached from graph"
                )
                scheduler.step()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            elapsed = time.perf_counter() - t0
            summarize_rollout_metrics(
                f"Epoch End][Stage {stage_idx + 1}/{len(rollout_curriculum)}][Epoch {epoch_idx + 1}/{epochs_per_stage}",
                traj_names,
                rollout_result,
            )
            print(f"[Epoch Time] {elapsed:.2f}s current_lr={optimizer.param_groups[0]['lr']:.2e}")

        if best_stage_state is None:
            raise RuntimeError(
                f"Stage {stage_idx + 1} produced no finite rollout. "
                f"Reduce learning_rate or shorten rollout_curriculum near {stage_rollout}."
            )

        controller.load_state_dict(best_stage_state)
        optimizer.load_state_dict(best_stage_optimizer_state)
        print(
            f"[Stage {stage_idx + 1}/{len(rollout_curriculum)}] using best epoch "
            f"{best_stage_epoch}/{epochs_per_stage} loss={best_stage_loss:.4f} for evaluation/save"
        )

        with torch.no_grad():
            stage_eval = rollout_trajectory_batch(
                rope_model,
                controller,
                padded_goals,
                traj_lengths,
                stage_rollout,
                record_history=False,
            )
        summarize_rollout_metrics(
            f"Stage End][Stage {stage_idx + 1}/{len(rollout_curriculum)}][rollout {stage_rollout}",
            traj_names,
            stage_eval,
        )
        if stage_hit_nonfinite:
            print(f"[Stage {stage_idx + 1}/{len(rollout_curriculum)}] stopped early after non-finite rollout")

        ckpt_name = script_dir / f"ckpt_real_rope_franka_stage{stage_idx + 1}_rollout{stage_rollout}.pt"
        torch.save(controller.state_dict(), ckpt_name)
        print(f"[Stage {stage_idx + 1}/{len(rollout_curriculum)} End] checkpoint={ckpt_name.name}")

    with torch.no_grad():
        final_eval = rollout_trajectory_batch(
            rope_model,
            controller,
            padded_goals,
            traj_lengths,
            max_control_steps,
            record_history=show_final_rollout_visualization,
        )
    summarize_rollout_metrics("Final Eval][full rollout", traj_names, final_eval)
    if show_final_rollout_visualization:
        visualize_final_rollout(
            traj_names,
            padded_goals,
            traj_lengths,
            final_eval,
            preferred_index=final_visualize_index,
        )

    final_model_name = script_dir / (
        f"trained_real_rope_franka_controller_cp{ctr_period}_pre{pre_length}_{mode}_ma{max_action}_N{N}_L{L:.3f}.pt"
    )
    torch.save(controller.state_dict(), final_model_name)
    print(f"[Save] final model saved: {final_model_name.name}")


if __name__ == '__main__':
    main()
