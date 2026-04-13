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

def precompute_trajectory_from_actions(
    acc_seq,
    init_pos,
    dt,
    alpha,
    vel_min,
    vel_max,
    pos_min,
    pos_max,
    hold_steps=25,
):
    num_actions = acc_seq.shape[0]
    num_steps = num_actions * hold_steps

    filtered_acc_seq = np.zeros((num_steps, 3), dtype=np.float32)
    vel_seq = np.zeros((num_steps, 3), dtype=np.float32)
    pos_seq = np.zeros((num_steps, 3), dtype=np.float32)

    acc_f = np.zeros(3, dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    pos = init_pos.astype(np.float32).copy()

    step_idx = 0

    for action_idx in range(num_actions):
        acc_goal = acc_seq[action_idx]

        for _ in range(hold_steps):
            # low-pass filter
            acc_f = alpha * acc_f + (1.0 - alpha) * acc_goal
            filtered_acc_seq[step_idx] = acc_f

            # integrate velocity
            vel += acc_f * dt
            vel = np.clip(vel, vel_min, vel_max)

            # integrate position
            pos += vel * dt

            # boundary handling
            for j in range(3):
                if pos[j] >= pos_max[j] and vel[j] > 0:
                    pos[j] = pos_max[j]
                    vel[j] = 0.0
                elif pos[j] <= pos_min[j] and vel[j] < 0:
                    pos[j] = pos_min[j]
                    vel[j] = 0.0

            vel_seq[step_idx] = vel
            pos_seq[step_idx] = pos
            step_idx += 1

    return filtered_acc_seq, vel_seq, pos_seq

rate_hz = 1000.0
dt = 1.0 / rate_hz
record_interval = 30
alpha = 0.8

# Workspace limits
# pos_min = np.array([0.33, -0.35, 0.25], dtype=np.float32)  # x y z
# pos_max = np.array([0.63,  0.35, 0.45], dtype=np.float32)
# vel_min = np.array([-0.3, -1.0, -0.3], dtype=np.float32)  # x y z
# vel_max = np.array([ 0.3,  1.0,  0.3], dtype=np.float32)
init_pos = np.array([0.48, -0.30, 0.35], dtype=np.float32)

pos_min = np.array([-100, -100, -100], dtype=np.float32)  # x y z
pos_max = np.array([100,  100, 100], dtype=np.float32)
vel_min = np.array([-100, -100, -100], dtype=np.float32)  # x y z
vel_max = np.array([100,  100,  100], dtype=np.float32)
# init_pos = np.array([0.0, -0.0, 1.2], dtype=np.float32)

offline_actions = np.load("openloop_action_baserope_eight.npy")
ctr_period = 25
num_steps = offline_actions.shape[0] * ctr_period
total_time = num_steps / rate_hz

# Precompute whole trajectory offline
filtered_acc_seq, vel_seq, pos_seq = precompute_trajectory_from_actions(
    acc_seq=offline_actions,
    init_pos=init_pos,
    dt=dt,
    alpha=alpha,
    vel_min=vel_min,
    vel_max=vel_max,
    pos_min=pos_min,
    pos_max=pos_max,
    hold_steps=ctr_period,
)

# === action 曲线（保留）===
time_axis = np.arange(pos_seq.shape[0]) * ctr_period * dt

plt.figure(figsize=(10, 6))
plt.plot(offline_actions[:, 0], label='Action X', color='r')
plt.plot(offline_actions[:, 1], label='Action Y', color='g')
plt.plot(offline_actions[:, 2], label='Action Z', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Action Value')
plt.title('Action XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

vel_seq = vel_seq[ctr_period - 1::ctr_period]
pos_seq = pos_seq[ctr_period - 1::ctr_period]

plt.figure(figsize=(10, 6))
plt.plot(vel_seq[:, 0], label='Vel X', color='r')
plt.plot( vel_seq[:, 1], label='Vel Y', color='g')
plt.plot( vel_seq[:, 2], label='Vel Z', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Vel Value')
plt.title('Vel XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.plot( pos_seq[:, 0], label='Pos X', color='r')
plt.plot(pos_seq[:, 1], label='Pos Y', color='g')
plt.plot( pos_seq[:, 2], label='Pos Z', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Pos Value')
plt.title('Pos XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


