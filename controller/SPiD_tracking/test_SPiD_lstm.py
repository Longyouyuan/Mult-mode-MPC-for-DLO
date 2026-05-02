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


class UniLSTMController(nn.Module):
	"""
	Single-direction LSTM controller for SPiD path tracking.

	Inputs:
		batch_pos: [B, N + 1, 3]
		batch_vel: [B, N + 1, 3]
		goal:      [B, pre_length, 3]

	Design:
		1. Encode current rope state by MLP.
		2. Encode future goal window by a single-direction LSTM.
		3. Fuse state feature + goal sequence feature.
		4. Output residual action, then add heuristic action.
	"""
	def __init__(
		self,
		state_dim,
		action_dim,
		state_hidden_width,
		lstm_hidden_width,
		fusion_hidden_width,
		max_action,
		lstm_layers=1,
		dropout=0.0,
	):
		super().__init__()
		self.max_action = max_action
		self.lstm_hidden_width = lstm_hidden_width
		self.lstm_layers = lstm_layers

		self.state_encoder = nn.Sequential(
			nn.Linear(state_dim, state_hidden_width),
			nn.ReLU(),
			nn.Linear(state_hidden_width, state_hidden_width),
			nn.ReLU(),
		)

		self.goal_encoder = nn.LSTM(
			input_size=3,
			hidden_size=lstm_hidden_width,
			num_layers=lstm_layers,
			batch_first=True,
			bidirectional=False,
			dropout=dropout if lstm_layers > 1 else 0.0,
		)

		goal_feature_dim = lstm_hidden_width
		self.fusion = nn.Sequential(
			nn.Linear(state_hidden_width + goal_feature_dim, fusion_hidden_width),
			nn.ReLU(),
			nn.Linear(fusion_hidden_width, fusion_hidden_width),
			nn.ReLU(),
			nn.Linear(fusion_hidden_width, action_dim),
		)

	def forward(self, batch_pos, batch_vel, goal, smart=False):
		batch_pos = batch_pos.clone()
		batch_vel = batch_vel.clone()
		goal = goal.clone()

		if smart:
			shift = batch_pos[:, 0:1, 0:3].detach().clone()
			batch_pos = batch_pos - shift
			goal = goal - shift[:, :, :]

		if torch.isnan(batch_vel).any() or torch.isnan(batch_pos).any() or torch.isnan(goal).any():
			print("Find nan in nn input!")

		state = torch.cat([
			batch_pos.reshape(batch_pos.shape[0], -1),
			batch_vel.reshape(batch_vel.shape[0], -1),
		], dim=-1)
		state_feature = self.state_encoder(state)

		goal_output, _ = self.goal_encoder(goal)
		# UniLSTM final sequence feature. This is the last output after scanning the goal window.
		goal_feature = goal_output[:, -1, :]

		fused_feature = torch.cat([state_feature, goal_feature], dim=-1)
		nn_action = self.max_action * torch.tanh(self.fusion(fused_feature))

		head_goal = goal[:, 0, :].clone()
		head_goal[:, 2] += L
		heuristic_action = (head_goal - batch_pos[:, 0, :]) * 5.0
		action = torch.clamp(heuristic_action + nn_action, -self.max_action, self.max_action)

		if torch.isnan(action).any():
			print("Find nan in nn action!")

		return action


def resolve_model_path(default_name):
	# Prefer exact configured name.
	exact_matches = list(REPO_ROOT.rglob(default_name))
	if exact_matches:
		return max(exact_matches, key=lambda path: path.stat().st_mtime)

	# Then prefer UniLSTM direct-fit checkpoints.
	unilstm_patterns = [
		"trained_unilstm_controller*.pt",
		"ckpt_unilstm_stage*_rollout*.pt",
		"trained_lstm_controller*.pt",
		"ckpt_lstm_stage*_rollout*.pt",
	]
	for pattern in unilstm_patterns:
		matches = list(REPO_ROOT.rglob(pattern))
		if matches:
			return max(matches, key=lambda path: path.stat().st_mtime)

	# Optional fallback for your previous naming, if you renamed files manually.
	fallback_matches = list(REPO_ROOT.rglob("*unilstm*.pt")) + list(REPO_ROOT.rglob("*lstm*.pt"))
	if fallback_matches:
		return max(fallback_matches, key=lambda path: path.stat().st_mtime)

	raise FileNotFoundError(f"No UniLSTM controller checkpoint found for {default_name}")


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


def evaluate_trajectory(rope_model, controller, traj_name, goal_traj, total_time):
	start_time = time.perf_counter()
	with torch.no_grad():
		positions_history = rope_model.simulate_with_controller_for_path_tracking2(
			controller,
			N=N,
			Seg_L=L / N,
			path=goal_traj,
			record_interval=1,
			ctr_period=ctr_period,
			mode=mode,
			alpha=alpha,
			smart=smart,
			path_fitting=True,
			pre_length=pre_length,
		)
	elapsed = time.perf_counter() - start_time

	positions_history = positions_history.squeeze(0)
	tip_history = positions_history[:, -1, :]
	goal_cpu = goal_traj.detach().cpu()[:tip_history.shape[0]]

	error = tip_history - goal_cpu
	squared_error = (error ** 2).sum(dim=-1)
	error_norm = torch.linalg.norm(error, dim=-1)

	mse_m2 = squared_error.mean().item()
	rmse_cm = (mse_m2 ** 0.5) * 100.0
	mean_err_cm = error_norm.mean().item() * 100.0
	max_err_cm = error_norm.max().item() * 100.0
	training_style_loss = mse_m2 * 1500.0

	result = {
		'traj_name': traj_name,
		'total_time': total_time,
		'frames': tip_history.shape[0],
		'mse_m2': mse_m2,
		'rmse_cm': rmse_cm,
		'mean_err_cm': mean_err_cm,
		'max_err_cm': max_err_cm,
		'training_style_loss': training_style_loss,
		'elapsed_s': elapsed,
		'positions_history': positions_history,
		'goal_traj': goal_cpu,
		'squared_error_sum': squared_error.sum().item(),
	}
	return result


def print_result(result):
	print(
		f"[Eval] {result['traj_name']:<18} frames={result['frames']:>4} "
		f"rmse={result['rmse_cm']:>6.2f}cm mean={result['mean_err_cm']:>6.2f}cm "
		f"max={result['max_err_cm']:>6.2f}cm loss={result['training_style_loss']:>8.4f} "
		f"time={result['elapsed_s']:.2f}s"
	)


def print_summary(results):
	total_frames = sum(item['frames'] for item in results)
	total_squared_error = sum(item['squared_error_sum'] for item in results)
	aggregate_mse_m2 = total_squared_error / max(total_frames, 1)
	aggregate_rmse_cm = (aggregate_mse_m2 ** 0.5) * 100.0
	aggregate_loss = aggregate_mse_m2 * 1500.0
	mean_rmse_cm = float(np.mean([item['rmse_cm'] for item in results]))
	worst_result = max(results, key=lambda item: item['rmse_cm'])

	print("=" * 100)
	print(f"[Summary] trajectories={len(results)} total_frames={total_frames}")
	print(f"[Summary] aggregate_rmse={aggregate_rmse_cm:.2f}cm aggregate_loss={aggregate_loss:.4f}")
	print(f"[Summary] mean_per_traj_rmse={mean_rmse_cm:.2f}cm")
	print(
		f"[Summary] worst_traj={worst_result['traj_name']} "
		f"rmse={worst_result['rmse_cm']:.2f}cm max={worst_result['max_err_cm']:.2f}cm"
	)
	print("=" * 100)


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
record_interval = 1
ctr_period = 10
pre_length = 75
mode = 'vel'
max_action = 1.5
alpha = 0.1
smart = True
visualize_indices = list(range(18))

# These must match your UniLSTM training script.
state_hidden_width = 256
lstm_hidden_width = 128
fusion_hidden_width = 256
lstm_layers = 1
lstm_dropout = 0.0

# Change this to your final UniLSTM checkpoint if needed.
model_name = f'trained_unilstm_controller_cp{ctr_period}_pre{pre_length}_{mode}_ma{max_action}_N{N}_L{L}_direct18.pt'
# Examples:
# model_name = 'trained_lstm_controller_cp10_pre75_vel_ma1.5_N20_L1.0_direct18.pt'
# model_name = 'ckpt_unilstm_stage8_rollout500.pt'


def main():
	# CPU is safer for testing/debugging. Change to cuda only if you want faster evaluation.
	device = torch.device('cpu')
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

	model_path = resolve_model_path(model_name)
	controller = UniLSTMController(
		state_dim=(N + 1) * 3 * 2,
		action_dim=3,
		state_hidden_width=state_hidden_width,
		lstm_hidden_width=lstm_hidden_width,
		fusion_hidden_width=fusion_hidden_width,
		max_action=max_action,
		lstm_layers=lstm_layers,
		dropout=lstm_dropout,
	).to(device)

	state_dict = torch.load(model_path, map_location=device)
	if isinstance(state_dict, dict) and 'state_dict' in state_dict:
		state_dict = state_dict['state_dict']
	if isinstance(state_dict, dict) and 'model_state' in state_dict:
		state_dict = state_dict['model_state']
	controller.load_state_dict(state_dict)
	controller.eval()
	print(f"Loaded UniLSTM controller: {model_path}")

	traj_specs = build_trajectory_suite(device, ctr_period)
	print(f"Evaluating {len(traj_specs)} trajectories generated like training_SPiD.py")

	results = []
	for traj_name, goal_traj, total_time in traj_specs:
		result = evaluate_trajectory(rope_model, controller, traj_name, goal_traj, total_time)
		print_result(result)
		results.append(result)

	print_summary(results)

	for idx in visualize_indices:
		if idx < 0 or idx >= len(results):
			continue
		result = results[idx]
		print(f"[Visualize] idx={idx} traj={result['traj_name']}")
		plot_tip_vs_goal_and_error(result['positions_history'], result['goal_traj'], dt, record_interval, tip_idx=-1)
		plot_animation_3d_for_path_tracking(result['positions_history'], result['goal_traj'], dt, record_interval=record_interval, L=L, batch_idx=0)


if __name__ == '__main__':
	main()
