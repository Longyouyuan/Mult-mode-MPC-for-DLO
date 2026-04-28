from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Mujoco_env.mj_utils import *
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

        s = torch.cat([
            batch_pos.view(batch_pos.shape[0], -1),
            batch_vel.view(batch_vel.shape[0], -1),
            goal.reshape(goal.shape[0], -1)
        ], dim=-1)
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        s = F.relu(self.l3(s))
        a2 = self.max_action * torch.tanh(self.l4(s))

        head_goal = goal[:, 0, :].clone()
        head_goal[:, 2] += L
        a_heur = (head_goal - batch_pos[:, 0, :]) * 5.0
        a2 = torch.clamp(a_heur + a2, -self.max_action, self.max_action)

        if torch.isnan(a2).any():
            print("Find nan in nn action!")

        return a2


def resolve_model_path(default_name):
    exact_matches = list(REPO_ROOT.rglob(default_name))
    if exact_matches:
        return max(exact_matches, key=lambda path: path.stat().st_mtime)

    stage_matches = list(REPO_ROOT.rglob("ckpt_stage*_seg*.pt"))
    if stage_matches:
        return max(stage_matches, key=lambda path: path.stat().st_mtime)

    fallback_matches = list(REPO_ROOT.rglob("trained_tracking_controller_cp*_ctr_smart_right.pt"))
    if fallback_matches:
        return max(fallback_matches, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(f"No trained SPiD controller checkpoint found for {default_name}")


def pad_goal_window(goal, horizon):
    if goal.shape[0] >= horizon:
        return goal[:horizon].unsqueeze(0)

    last_goal = goal[-1:, :].repeat(horizon - goal.shape[0], 1)
    return torch.cat([goal, last_goal], dim=0).unsqueeze(0)


# === mujoco ===
model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "Mujoco_env" / "cable_show.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 4)

# 修改仿真步长（单位：秒）/积分器
dt = 0.001
model.opt.timestep = dt
model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

# 获取所有 body 的名称
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]

# 获取绳子节点的 body 名称（例如，名称以 'B_' 开头）
cable_body_indices = [i for i, name in enumerate(body_names) if name and name.startswith('B_')]

data.ctrl[0] = 0
data.ctrl[1] = 0
data.ctrl[2] = 0

# === Parameters ===
N = 20
P = N + 1
L = 0.05 * N
mass = (0.0025 * 40 / N) * L
k = 0.46
damping = [0.2] * N
k_bend = [0.0006712] * (N - 1)
damping_bend = [0.000401] * (N - 1)
air_drag = 0.2206
g = 10.07
T_task = 6.0
total_steps = int(T_task / dt)
mode = 'vel'

ctr_period = 10
pre_length = 75
hidden_width = 512
max_action = 1.5
alpha = 0.1
smart = True
record_interval = 10
last_point_repeat = 100 * 0
total_horizon = int(total_steps / ctr_period) + last_point_repeat

model_name = f'trained_tracking_controller_cp{ctr_period}_pre{pre_length}_{mode}_ma{max_action}_N{N}_L{L}_ctr_smart_right.pt'

# === Setup device ===
device = torch.device('cpu')
print(f"Using device: {device}")

model_path = resolve_model_path(model_name)
controller = Controller(
    input_dim=(N + 1) * 3 * 2 + 3 * pre_length,
    action_dim=3,
    hidden_width=hidden_width,
    max_action=max_action,
).to(device)

state_dict = torch.load(model_path, map_location=device)
if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
if isinstance(state_dict, dict) and 'model_state' in state_dict:
    state_dict = state_dict['model_state']
controller.load_state_dict(state_dict)
controller.eval()
print(f"Loaded controller: {model_path}")

# === 目标轨迹（保留原逻辑）===

# === 1. Generate sin/egg/eight Trajs ===
points = half_dense_then_uniform(
    N=total_horizon + 1 - last_point_repeat, ratio=0.3, sharpness=2.0,
    mode='exp', interval=(0.0, 2.0), plot=False
)

# Goal_traj = eight_traj(points, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2, loops=1, plot=True, device=device)
# Goal_traj = egg_traj(points, scale_x=0.38 * 2.0, scale_y=0.52 * 2.0, plot=False, device=device)
Goal_traj = sin_traj(points, width=0.45 * 2.0, plot=False, device=device)
# Goal_traj = torch.vstack((torch.sin(4 * points) * 0.25, points,
#                           torch.ones(total_horizon + 1 - last_point_repeat) * 0.2)).T.to(device)
plot_goal_traj(Goal_traj, T_task)

# # === 2. Draw Traj ===
#
# Goal_traj = build_goal_traj_from_drawn(
#     drawn_path=str(REPO_ROOT / 'my_trajs' / 'PatrickStar.npy'),
#     total_horizon=total_horizon - last_point_repeat,
#     device=device,
#     z0=0.2,
#     scale_x=3.0,
#     scale_y=3.0,
#     keep_aspect=False,
#     sigma=0.0,
#     uniform_M=1000,
#     ratio=0.3,
#     sharpness=2.0,
#     interval=(0.0, 2.0)
# )
# plot_goal_traj(Goal_traj, T_task)

Goal_traj = extend_last_point(Goal_traj, N=last_point_repeat)

# === Init state（保留原逻辑）===
pos = torch.zeros((1, P, 3), device=device)
pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
vel = torch.zeros((1, P, 3), device=device)

with viewer.launch_passive(model, data) as viewer:
    cam_setting(viewer, fixed=True)
    viewer.showinfo = True

    goal_his = []
    slider_pos = []
    rope_top = []

    action_history = []
    vel_history = []
    pos_history = []
    time_record = []

    node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + node * 6 + 3))

    pcf = PositionCommandFilter(np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], dt)
    Goal_np = Goal_traj.detach().cpu().numpy()
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'slider')
    filtered_action = np.zeros(3, dtype=np.float32)

    task_t_start = time.perf_counter()

    for i in range(total_horizon):
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + node * 3] = data.xpos[1:1 + node][::-1].reshape(-1)
        mj_state[0, 1 + node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)

        global_goal_tracking = Goal_np[i + 1]
        model.site_pos[1][[0, 1, 2]] = global_goal_tracking

        if i < (total_horizon - pre_length):
            goal = Goal_traj[i + 1:i + 1 + pre_length, :]
        else:
            goal = Goal_traj[i + 1:, :]

        controller_goal = pad_goal_window(goal, pre_length)

        t0 = time.perf_counter()
        with torch.no_grad():
            raw_action = controller(pos, vel, controller_goal, smart=smart).squeeze(0).cpu().numpy()
        action = alpha * raw_action + (1.0 - alpha) * filtered_action
        filtered_action = action.copy()
        t1 = time.perf_counter()
        time_record.append(t1 - t0)

        for j in range(ctr_period):
            if mode == 'vel':
                pos_target = pcf.input_velocity(action)
            elif mode == 'acc':
                pos_target = pcf.input_acceleration(action)

            data.ctrl[0] = pos_target[0]
            data.ctrl[1] = pos_target[1]
            data.ctrl[2] = pos_target[2]
            global_goal_action_pos = pos_target + np.array([0.0, 0.0, 1.2])
            model.site_pos[0][[0, 1, 2]] = global_goal_action_pos

            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0.0, 0.5 * 9.81])
            mujoco.mj_step(model, data)

        viewer.sync()

        goal_his.append(global_goal_action_pos.copy())
        slider_pos.append(data.xpos[slider_id].copy())
        rope_top.append(data.site_xpos[-1].copy())

        action_history.append(action.copy())
        vel_history.append(vel.clone())
        pos_history.append(pos.clone())

task_t_end = time.perf_counter()
print("average time on inference: ", torch.tensor(time_record).mean(), "Desired time:", dt * ctr_period)
print("Task time:", T_task, "   Spent time:", task_t_end - task_t_start)

# === 动画/轨迹（保留）===
pos_history = torch.cat(pos_history, dim=0)
plot_tip_vs_goal_and_error(pos_history, Goal_traj[1:], dt, ctr_period, tip_idx=-1)

# === action 曲线（保留）===
action_history_np = np.array(action_history)
time_axis = np.arange(action_history_np.shape[0]) * ctr_period * dt

plt.figure(figsize=(10, 6))
plt.plot(time_axis, action_history_np[:, 0], label='Action X', color='r')
plt.plot(time_axis, action_history_np[:, 1], label='Action Y', color='g')
plt.plot(time_axis, action_history_np[:, 2], label='Action Z', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Action Value')
plt.title('Action XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === velocity 曲线（保留）===
vel_history_np = np.stack([
    v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    for v in vel_history
], axis=0)[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(time_axis, vel_history_np[:, 0, 0], label='Velocity X', color='r', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 0, 1], label='Velocity Y', color='g', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 0, 2], label='Velocity Z', color='b', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Value')
plt.title('Top Velocity XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# pos 轨迹
tip_pos_history = pos_history[:, 0, :]
plt.figure(figsize=(10, 6))
plt.plot(time_axis, tip_pos_history[:, 0].cpu().numpy(), label='Tip X', color='m')
plt.plot(time_axis, tip_pos_history[:, 1].cpu().numpy(), label='Tip Y', color='c')
plt.plot(time_axis, tip_pos_history[:, 2].cpu().numpy(), label='Tip Z', color='y')
plt.xlabel('Time (s)')
plt.ylabel('top Position')
plt.title('Top Point Position XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== Check low-level controller performance ==========================
time = list(range(len(goal_his)))

goal_array = np.array(goal_his)
slider_pos_array = np.array(slider_pos)
rope_top_array = np.array(rope_top)
ctr_history = np.array(action_history)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(time, goal_array[:, 0], label='Goal', color='blue', linestyle='-')
ax1.plot(time, slider_pos_array[:, 0], label='Slider', color='green', linestyle='--')
ax1.plot(time, rope_top_array[:, 0], label='Rope Top', color='red', linestyle=':')
ax1.set_title('X-axis Position Tracking')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('X Position')
ax1.grid(True)
ax1.legend()

ax2.plot(time, goal_array[:, 1], label='Goal', color='blue', linestyle='-')
ax2.plot(time, slider_pos_array[:, 1], label='Slider', color='green', linestyle='--')
ax2.plot(time, rope_top_array[:, 1], label='Rope Top', color='red', linestyle=':')
ax2.set_title('Y-axis Position Tracking')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Y Position')
ax2.grid(True)
ax2.legend()

ax3.plot(time, goal_array[:, 2], label='Goal', color='blue', linestyle='-')
ax3.plot(time, slider_pos_array[:, 2], label='Slider', color='green', linestyle='--')
ax3.plot(time, rope_top_array[:, 2], label='Rope Top', color='red', linestyle=':')
ax3.set_title('Z-axis Position Tracking')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Z Position')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()
