import time
import struct
import zmq
from collections import deque

import torch
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from common.utils import *
from common.rope_warp_4_baserope import WarpRope, N, P
from controller.M2PC import Planner, cost_fn

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# === Parameters（保持 Sampling_MPC.py 风格）===
L = 0.8
segment_lengths = [0.0973, 0.0996, 0.0993, 0.0986, 0.0990, 0.0993, 0.0975, 0.1160]
mass = 12.8 / 1000 / N
tip_extra_mass = 15 / 1000
k = 10000 * 0.12
damping = 0.2
bending_k = 0.0006712
bending_damping = 0.000401
air_drag = 0.3006 / 1000
g = 10.27
mode = 'acc'

dt = 0.001
T_task = 5.0
total_steps = int(T_task / dt)

ctr_period = 25
horizon = 30  # 20 doesn't work; 25 can work

# ===== 多模态参数 =====
n_sample = 400     # 总采样数
m_modes = 1        # 模态数（单模态=1）
assert n_sample % m_modes == 0

n_improve = 10
noise_scale = 1.8  # 0.25 looks good for naive case, 1.5更好for naive case？
action_dim = 3
limits = torch.tensor([-5.0, 5.0])
total_horizon = int(total_steps / ctr_period)

# diversity 超参
top_k_good = 200
beta = 5.0
wJ = 1.0

visualization = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === warp_4 rope：rollout（batch = m_modes + n_sample）===
rope = WarpRope(
    batch_size=m_modes + n_sample,
    L=L,
    segment_lengths=segment_lengths,
    mass=mass,
    tip_extra_mass=tip_extra_mass,
    k=k,
    damping=damping,
    bending_k=bending_k,
    bending_damping=bending_damping,
    air_drag=air_drag,
    g=g,
    dt=dt,
    max_record_steps=horizon * ctr_period,
    record_interval=ctr_period,
    ctr_period=ctr_period,
    mode=mode
)

# === warp_4 rope：执行（batch=1）===
rope_exec = WarpRope(
    batch_size=1,
    L=L,
    segment_lengths=segment_lengths,
    mass=mass,
    tip_extra_mass=tip_extra_mass,
    k=k,
    damping=damping,
    bending_k=bending_k,
    bending_damping=bending_damping,
    air_drag=air_drag,
    g=g,
    dt=dt,
    max_record_steps=ctr_period,
    record_interval=ctr_period,
    ctr_period=ctr_period,
    mode=mode
)

# ================= 初始状态 =================
pos = torch.zeros((1, P, 3), device=device)

# 底端（固定端）
pos[0, 0] = torch.tensor([0.0, 0.0, 1.2], device=device)

# 按真实长度往下累加
for i in range(1, P):
    pos[0, i, 0] = 0.0
    pos[0, i, 1] = 0.0
    pos[0, i, 2] = pos[0, i - 1, 2] - segment_lengths[i - 1]

vel = torch.zeros((1, P, 3), device=device)

# === 目标轨迹（保留原逻辑）===

# === 1. Generate sin/egg/eight Trajs ===
points = half_dense_then_uniform(
    N=total_horizon + 1, ratio=0.3, sharpness=2.0,
    mode='exp', interval=(0.0, 2.0), plot=False
)

Goal_traj = eight_traj(points, scale_x=0.45*2.0*0.3, scale_y=0.65*2.0*0.3, z0=0.0, loops=1, plot=False, device=device,bias=pos[0, -1])
# Goal_traj = egg_traj(points, scale_x=0.38*2.0, scale_y=0.52*2.0, plot=False, device=device)
# Goal_traj = sin_traj(points, width=0.45*2.0, plot=False, device=device)
# plot_goal_traj(Goal_traj, T_task)  # uncomment for visualization

# # === 2. Draw Traj ===
# Goal_traj = build_goal_traj_from_drawn(
#     drawn_path="../my_trajs/SpongeBob.npy",
#     total_horizon=total_horizon,
#     device=device,
#     z0=0.2,
#     scale_x=3.0,
#     scale_y=3.0,
#     keep_aspect=False,  # 允许非等比缩放（你说可能不是方形）
#     sigma=2.0,
#     uniform_M=1000,  # 越大越均匀/越平滑（但太大也没必要）
#     ratio=0.3,
#     sharpness=2.0,
#     interval=(0.0, 2.0)
# )
# plot_goal_traj(Goal_traj, T_task)

planner = Planner(
    rope, cost_fn, dt, ctr_period, horizon,
    n_sample, n_improve, noise_scale, action_dim,
    limits=limits, device=device, mode=mode,
    m_modes=m_modes, top_k_good=top_k_good, beta=beta, wJ=wJ
)

if visualization:
    viz = LiveMPCVisualizer(goal_traj=Goal_traj, margin=0.05)

action_history = []
pos_history = []
vel_history = []
time_record = []
pos_history.append(pos.clone())

stored_actions = None
# stored_actions = np.load('action_history.npy')

# === warm start（保留原逻辑；但要初始化多模态 seeds）===
if stored_actions is None:
    vel_start = (Goal_traj[1:1 + horizon, :] - Goal_traj[:horizon, :]) / (T_task / horizon)
    vel_start = vel_start.to(device)

    # 初始化所有模态 seeds = vel_start (+小扰动让模态更容易分开)
    planner.seeds[:] = vel_start.unsqueeze(0).repeat(m_modes, 1, 1)
    if m_modes > 1:
        planner.seeds[1:] += 0.05 * torch.randn_like(planner.seeds[1:])

    goal = Goal_traj[1:1 + horizon, :]

    planner.n_improve = n_improve * 100
    planner.improve_policy(pos, vel, goal)
    planner.n_improve = n_improve

# === MPC loop（保留原逻辑）===
for i in range(total_horizon):
    if i % 100 == 0:
        print('i: ', i)

    if stored_actions is None:
        if i < (total_horizon - horizon):
            goal = Goal_traj[i + 1:i + 1 + horizon, :]
        else:
            goal = Goal_traj[i + 1:, :]  # 尾段短的也没事：内部会 padding

        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal)
        action = planner.get_action()
        t1 = time.perf_counter()
        time_record.append(t1 - t0)
        # print("Time spent:", t1 - t0, "Desired time:", dt * ctr_period)

    else:
        action = torch.from_numpy(stored_actions[i]).to(device)

    # visualization
    if visualization:
        if i % 2 == 0:
            k_cand = 100
            cand_tip_traj, cand_cost, m_mode_trajs = planner.get_cand(k_cand=k_cand)

            if cand_tip_traj is not None:
                rope_now = pos[0]  # (P,3)
                viz.update(
                    rope_pos=rope_now,
                    cur_i=i,
                    cand_tip_traj=cand_tip_traj,
                    cand_cost=cand_cost,
                    m_mode_trajs=m_mode_trajs,
                    title=f"MPC step {i} | {k_cand} candidates | modes={m_modes}"
                )

    # ---- 执行 chosen action（batch=1 rope_exec）----
    action_seq = action.view(1, 1, 3)  # (1,1,3)
    rope_exec.set_state_and_action(pos, vel, action_seq)
    _ = rope_exec.simulate(steps=ctr_period)

    # 更新状态
    pos = wp.to_torch(rope_exec.pos)
    vel = wp.to_torch(rope_exec.vel)

    pos_history.append(pos.clone())
    action_history.append(action.detach().cpu().numpy())
    vel_history.append(vel.clone().detach().cpu().numpy()[0, 0])

    # Update policy
    planner.update_policy()


if visualization:
    viz.close()

print("time: ", torch.tensor(time_record).mean())

# === 动画/轨迹（保留）===
pos_history = torch.cat(pos_history, dim=0)
anim = plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)
plot_tip_vs_goal_and_error(pos_history, Goal_traj, dt, ctr_period)

# === action 曲线（保留）===
action_history_np = np.array(action_history)
np.save("openloop_action_baserope_eight.npy", action_history_np)
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

# === velocity 曲线（保留）===
vel_history_np = np.array(vel_history)
plt.figure(figsize=(10, 6))
plt.plot(time_axis, vel_history_np[:, 0], label='Velocity X', color='r', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 1], label='Velocity Y', color='g', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 2], label='Velocity Z', color='b', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Value')
plt.title('Velocity XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# pos 轨迹
tip_pos_history = pos_history[1:, 0, :]
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
