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


# ================= ZMQ =================
STATE_HDR = struct.Struct("<iqII")
ACT_HDR   = struct.Struct("<iqf")


def now_ns():
    return time.perf_counter_ns()

point_follow = 0

class LiveRopeViz:
    def __init__(self, goal_traj, margin=0.08, tip_index=-1):
        self.goal = self._np(goal_traj)
        self.margin = margin
        self.tip_index = tip_index

        plt.ion()
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.plot(
            self.goal[:, 0],
            self.goal[:, 1],
            self.goal[:, 2],
            '--',
            linewidth=1.5,
            alpha=0.6,
            label='Goal Traj'
        )

        self.rope_line, = self.ax.plot(
            [], [], [], '-o',
            linewidth=2.0,
            markersize=4,
            label='Rope'
        )

        self.goal_point = self.ax.scatter([], [], [], s=60, c='r', marker='x', label='Current Goal')
        self.tip_point = self.ax.scatter([], [], [], s=40, c='g', marker='o', label='Tip')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.grid(True)

        self._set_axes_from_goal()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _np(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _set_axes_from_goal(self):
        xyz_min = self.goal.min(axis=0)
        xyz_max = self.goal.max(axis=0)

        xyz_min = np.minimum(xyz_min, np.array([-0.2, -0.8, 0.0]))
        xyz_max = np.maximum(xyz_max, np.array([0.2,  0.8, 1.3]))

        xlim = (xyz_min[0] - self.margin, xyz_max[0] + self.margin)
        ylim = (xyz_min[1] - self.margin, xyz_max[1] + self.margin)
        zlim = (xyz_min[2] - self.margin, xyz_max[2] + self.margin)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)

        try:
            self.ax.set_box_aspect([
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                zlim[1] - zlim[0],
            ])
        except Exception:
            pass

    def update(self, rope_pos, cur_goal, step=None):
        rope = self._np(rope_pos)
        if rope.ndim == 3:
            rope = rope[0]

        goal = self._np(cur_goal)

        self.rope_line.set_data(rope[:, 0], rope[:, 1])
        self.rope_line.set_3d_properties(rope[:, 2])

        self.goal_point.remove()
        self.goal_point = self.ax.scatter(
            [goal[0]], [goal[1]], [goal[2]],
            s=60, c='r', marker='x', label='Current Goal'
        )

        tip = rope[self.tip_index]
        self.tip_point.remove()
        self.tip_point = self.ax.scatter(
            [tip[0]], [tip[1]], [tip[2]],
            s=40, c='g', marker='o', label='Tip'
        )

        if step is not None:
            self.ax.set_title(f'Rope Tracking | step {step}')

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()


ctx = zmq.Context.instance()
server_ip = "10.240.21.20"

pub = ctx.socket(zmq.PUB)
pub.connect(f"tcp://{server_ip}:4532")

sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{server_ip}:6001")

time.sleep(0.2)

visualization = True

# ================= 参数 =================
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
T_task = 10.0
total_steps = int(T_task / dt)

ctr_period = 25
horizon = 30
H = horizon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

# ================= 固定步数延迟参数 =================
# 每个 action 延迟 delay_steps 个 control step 执行
delay_steps = 0   # 例如 4 -> 4 * 25ms = 100ms

# 初始填充 0 动作
initial_action = np.zeros(3, dtype=np.float32)
action_buffer = deque([initial_action.copy() for _ in range(delay_steps)])

# ================= rope（执行）=================
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

# ================= 目标轨迹 =================
last_point_repeat = 100 * 0
total_horizon = int(total_steps / ctr_period) + last_point_repeat

points = half_dense_then_uniform(
    N=total_horizon + 1 - last_point_repeat,
    ratio=0.3,
    sharpness=2.0,
    mode='exp',
    interval=(0.0, 2.0),
    plot=True
)

Goal_traj = eight_traj(
    points,
    scale_x=0.45 * 2.0 * 0.3,
    scale_y=0.65 * 2.0 * 0.3,
    z0=0.0,
    loops=1,
    plot=False,
    device=device,
    bias=pos[0, point_follow]
)

plot_goal_traj(Goal_traj, T_task)

# # === 2. Draw Traj ===
# Goal_traj = build_goal_traj_from_drawn(
#     drawn_path="../my_trajs/SpongeBob.npy",
#     total_horizon=total_horizon-last_point_repeat,
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

if visualization:
    viz = LiveRopeViz(Goal_traj, margin=0.05, tip_index=-1)

# ================= warm up =================
goal = Goal_traj[1:1 + horizon].cpu().numpy()

frames = [
    STATE_HDR.pack(-1, now_ns(), P, H),
    pos.detach().cpu().numpy().reshape(-1).astype(np.float32).tobytes(),
    vel.detach().cpu().numpy().reshape(-1).astype(np.float32).tobytes(),
    goal.reshape(-1).astype(np.float32).tobytes(),
]
pub.send_multipart(frames)

time.sleep(2.0)

# ================= 主循环 =================
pos_history = []
vel_history = []
raw_action_history = []
applied_action_history = []
time_record = []
actuator = FirstOrderActuator(dim=3, tau=0.20, dt=ctr_period * dt, init=np.zeros(3))
for i in range(int(total_steps / ctr_period)):
    if i % 50 == 0:
        print("i:", i)

    # ===== goal =====
    if i < (len(Goal_traj) - horizon):
        goal = Goal_traj[i + 1:i + 1 + horizon]
    else:
        goal = Goal_traj[i + 1:]

    # ===== send state =====
    t0 = time.perf_counter()

    frames = [
        STATE_HDR.pack(i, now_ns(), P, H),
        pos.detach().cpu().numpy().reshape(-1).astype(np.float32).tobytes(),
        vel.detach().cpu().numpy().reshape(-1).astype(np.float32).tobytes(),
        goal.detach().cpu().numpy().reshape(-1).astype(np.float32).tobytes(),
    ]
    pub.send_multipart(frames)

    # ===== recv action =====
    rep = sub.recv_multipart()
    t1 = time.perf_counter()
    time_record.append(t1 - t0)

    if len(rep) != 2:
        print("[client] bad reply")
        continue

    step_ref, t_state_ns, compute_ms = ACT_HDR.unpack(rep[0])
    raw_action = np.frombuffer(rep[1], dtype=np.float32).copy()

    total_time = (now_ns() - t_state_ns) / 1e6
    if total_time > 25.0:
        print(
            f"Total time: {total_time:.3f}ms, "
            f"Compute={compute_ms:.3f}ms, "
            f"Network delay={total_time - compute_ms:.3f}ms"
        )

    # ===== 固定 n-step 延迟 =====
    action_buffer.append(raw_action)
    delayed_action = action_buffer.popleft().copy()

    applied_action = actuator.update(delayed_action)

    # ===== 执行延迟后的 action =====
    action_seq = torch.from_numpy(applied_action).to(device).view(1, 1, 3)

    rope_exec.set_state_and_action(pos, vel, action_seq)
    rope_exec.simulate(steps=ctr_period)

    pos = wp.to_torch(rope_exec.pos)
    vel = wp.to_torch(rope_exec.vel)

    if visualization:
        viz.update(pos, Goal_traj[i + 1], step=i)

    pos_history.append(pos.clone())
    vel_history.append(vel.clone())
    raw_action_history.append(raw_action.copy())
    applied_action_history.append(applied_action.copy())

# ================= 结果 =================
if visualization:
    viz.close()

print("avg time:", np.mean(time_record))
print("delay_steps:", delay_steps)
print("equivalent delay time:", delay_steps * ctr_period * dt, "seconds")

pos_history = torch.cat(pos_history, dim=0)

plot_tip_vs_goal_and_error(pos_history, Goal_traj, dt, ctr_period, tip_idx=point_follow)

# ================= action =================
raw_action_history_np = np.array(raw_action_history)
applied_action_history_np = np.array(applied_action_history)
time_axis = np.arange(len(applied_action_history_np)) * ctr_period * dt

plt.figure(figsize=(10, 6))
plt.plot(time_axis, raw_action_history_np[:, 0], '--', label='Raw Action X', alpha=0.7)
plt.plot(time_axis, raw_action_history_np[:, 1], '--', label='Raw Action Y', alpha=0.7)
plt.plot(time_axis, raw_action_history_np[:, 2], '--', label='Raw Action Z', alpha=0.7)
plt.plot(time_axis, applied_action_history_np[:, 0], label='Applied Action X')
plt.plot(time_axis, applied_action_history_np[:, 1], label='Applied Action Y')
plt.plot(time_axis, applied_action_history_np[:, 2], label='Applied Action Z')
plt.xlabel('Time (s)')
plt.ylabel('Action')
plt.title('Raw vs Applied Action')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================= velocity =================
vel_np = torch.stack(vel_history).detach().cpu().numpy()[:, 0, 0]

plt.figure(figsize=(10, 6))
plt.plot(time_axis, vel_np[:, 0], label='Velocity X')
plt.plot(time_axis, vel_np[:, 1], label='Velocity Y')
plt.plot(time_axis, vel_np[:, 2], label='Velocity Z')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Velocity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================= tip pos =================
tip_pos_history = pos_history[:, point_follow, :].detach().cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(time_axis, tip_pos_history[:, 0], label='Tip X')
plt.plot(time_axis, tip_pos_history[:, 1], label='Tip Y')
plt.plot(time_axis, tip_pos_history[:, 2], label='Tip Z')
plt.xlabel('Time (s)')
plt.ylabel('Tip Position')
plt.title('Tip Position XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()