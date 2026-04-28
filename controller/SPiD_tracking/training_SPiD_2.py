import copy
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.rope import Rope
from common.utils import *


class Controller(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_width, max_action):
		super().__init__()
		self.max_action = max_action
		self.l1 = nn.Linear(input_dim, hidden_width)
		self.l2 = nn.Linear(hidden_width, hidden_width)
		self.l3 = nn.Linear(hidden_width, hidden_width)
		self.l4 = nn.Linear(hidden_width, action_dim)

	def forward(self, batch_pos, batch_vel, goal, smart=False):
		batch_pos = batch_pos.clone()
		batch_vel = batch_vel.clone()
		goal = goal.clone()

		if smart:
			shift = batch_pos[:, 0:1, 0:3].detach().clone()
			batch_pos = batch_pos - shift
			goal = goal - shift[:, :, :]

		if torch.isnan(batch_vel).any() or torch.isnan(batch_pos).any():
			print("Find nan in nn input!")

		state = torch.cat([
			batch_pos.view(batch_pos.shape[0], -1),
			batch_vel.view(batch_vel.shape[0], -1),
			goal.reshape(goal.shape[0], -1),
		], dim=-1)
		state = F.relu(self.l1(state))
		state = F.relu(self.l2(state))
		state = F.relu(self.l3(state))
		action = self.max_action * torch.tanh(self.l4(state))

		head_goal = goal[:, 0, :].clone()
		head_goal[:, 2] += L
		heuristic_action = (head_goal - batch_pos[:, 0, :]) * 5.0
		action = torch.clamp(heuristic_action + action, -self.max_action, self.max_action)

		if torch.isnan(action).any():
			print("Find nan in nn action!")

		return action


def build_trajectory_suite(device, ctr_period):
	traj_specs = []

	for total_time in [4.0, 5.0, 6.0]:
		total_horizon_t = int(1000 / ctr_period * total_time)
		points = half_dense_then_uniform(
			N=total_horizon_t + 1,
			ratio=0.3,
			sharpness=2.0,
			mode='exp',
			interval=(0.0, 2.0),
			plot=False,
		)
		traj_specs.append((f"sin_{total_time:.1f}s", sin_traj(points, width=0.45 * 2.0, plot=False, device=device), total_time))
		traj_specs.append((f"egg_{total_time:.1f}s", egg_traj(points, scale_x=0.38 * 2.0, scale_y=0.52 * 2.0, plot=False, device=device), total_time))
		traj_specs.append((f"eight_{total_time:.1f}s", eight_traj(points, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2, loops=1, plot=False, device=device), total_time))

	for total_time in [12.0, 15.0, 18.0]:
		total_horizon_t = int(1000 / ctr_period * total_time)
		for drawn_name in ['SpongeBob', 'flower', 'PatrickStar']:
			traj_specs.append((
				f"{drawn_name}_{total_time:.1f}s",
				build_goal_traj_from_drawn(
					drawn_path=str(REPO_ROOT / 'my_trajs' / f'{drawn_name}.npy'),
					total_horizon=total_horizon_t,
					device=device,
					z0=0.2,
					scale_x=3.0,
					scale_y=3.0,
					keep_aspect=False,
					sigma=0.0,
					uniform_M=1000,
					ratio=0.3,
					sharpness=2.0,
					interval=(0.0, 2.0),
				),
				total_time,
			))

	return traj_specs


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
	positions = torch.zeros((batch_size, N + 1, 3), device=device)
	positions[:, :, 2] = torch.linspace(0.2 + L, 0.2, steps=N + 1, device=device).unsqueeze(0)
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
		batch_ctl = controller(positions, velocities, curr_goal, smart=smart)
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

		next_positions, next_velocities = rope_model.simulation_for_ctr(
			positions.clone(),
			velocities.clone(),
			filter_candidate,
			ctr_period=ctr_period,
			mode=mode,
		)
		if not torch.isfinite(next_positions).all() or not torch.isfinite(next_velocities).all():
			return make_nonfinite_rollout_result(
				batch_size=batch_size,
				device=rope_model.device,
				history=history,
				failure_step=step,
				failure_reason='rope_simulation',
			)

		active_mask_pos = active_mask.view(-1, 1, 1)
		positions = torch.where(active_mask_pos, next_positions, positions)
		velocities = torch.where(active_mask_pos, next_velocities, velocities)
		filter_ctl = torch.where(active_mask.unsqueeze(-1), filter_candidate, filter_ctl)

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

	result = {
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
	return result


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


# === Parameters ===
N = 20
L = 0.05 * N
mass = (0.0025 * 40 / N) * L
k = 0.46
damping = [0.2] * N
k_bend = [0.0006712] * (N - 1)
damping_bend = [0.000401] * (N - 1)
air_drag = 0.2206
g = 10.07
dt = 0.001

ctr_period = 10
pre_length = 75
mode = 'vel'
max_action = 1.5
hidden_width = 512
alpha = 0.1
smart = True
learning_rate = 1e-7  # 3e-5
stage_lr_decay = 0.7
min_stage_learning_rate = 1e-8
epoch_lr_gamma = 0.8
weight_decay = 0.0
control_penalty_weight = 0.0
gradient_clip_norm = 0.0001
epochs_per_stage = 30
rollout_curriculum = list(range(1200, 1801, 200))
resume_model_name = "ckpt_directfit_stage5_rollout1200.pt"


def main():
	torch.manual_seed(0)
	np.random.seed(0)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	rope_model = Rope(
		N=N,
		L=L,
		mass=mass,
		k=k,
		k_bend=k_bend,
		damping=damping,
		damping_bend=damping_bend,
		air_drag=air_drag,
		g=g,
		dt=dt,
		device=device,
		mode='ctr',
	)

	controller = Controller(
		input_dim=(N + 1) * 3 * 2 + 3 * pre_length,
		action_dim=3,
		hidden_width=hidden_width,
		max_action=max_action,
	).to(device)

	if resume_model_name is not None:
		resume_path = Path(resume_model_name)
		if not resume_path.is_absolute():
			resume_path = Path(__file__).resolve().parent / resume_model_name
		controller.load_state_dict(torch.load(resume_path, map_location=device))
		print(f"Loaded initial weights: {resume_path}")

	optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)

	traj_specs = build_trajectory_suite(device, ctr_period)
	traj_names, traj_times, padded_goals, traj_lengths = pad_trajectory_batch(traj_specs, device)
	max_control_steps = int(traj_lengths.max().item()) - 1

	print("=" * 100)
	print(f"[Setup] batch_size={len(traj_specs)} trajectories={len(traj_specs)} device={device}")
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

		ckpt_name = Path(__file__).resolve().parent / f"ckpt_directfit_stage{stage_idx + 1}_rollout{stage_rollout}.pt"
		torch.save(controller.state_dict(), ckpt_name)
		print(f"[Stage {stage_idx + 1}/{len(rollout_curriculum)} End] checkpoint={ckpt_name.name}")

	with torch.no_grad():
		final_eval = rollout_trajectory_batch(
			rope_model,
			controller,
			padded_goals,
			traj_lengths,
			max_control_steps,
			record_history=False,
		)
	summarize_rollout_metrics("Final Eval][full rollout", traj_names, final_eval)

	final_model_name = Path(__file__).resolve().parent / (
		f"trained_directfit_controller_cp{ctr_period}_pre{pre_length}_{mode}_ma{max_action}_N{N}_L{L}_direct18.pt"
	)
	torch.save(controller.state_dict(), final_model_name)
	print(f"[Save] final model saved: {final_model_name.name}")


if __name__ == '__main__':
	main()
