import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import imageio_ffmpeg
import matplotlib as mpl
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


class RopeDataset(Dataset):
    def __init__(self, position_data, velocity_data, total_samples=None):
        if total_samples is None:
            self.position_data = position_data
            self.velocity_data = velocity_data
        else:
            self.position_data = position_data[:total_samples]
            self.velocity_data = velocity_data[:total_samples]

    def __len__(self):
        return len(self.position_data)

    def __getitem__(self, idx):
        return idx, self.position_data[idx], self.velocity_data[idx]


class RopeDataset2(Dataset):
    def __init__(self, position_data, velocity_data, time, total_samples=None):
        if total_samples is None:
            self.position_data = position_data
            self.velocity_data = velocity_data
            self.time = time
        else:
            self.position_data = position_data[:total_samples]
            self.velocity_data = velocity_data[:total_samples]
            self.time = time[:total_samples]

    def __len__(self):
        return len(self.position_data)

    def __getitem__(self, idx):
        return idx, self.position_data[idx], self.velocity_data[idx], self.time[idx]


class PositionCommandFilter:
    def __init__(self, initial_pos=np.array([0.0, 0.0, 0.0]), initial_vel=np.array([0.0, 0.0, 0.0]), dt=0.001):
        self.pos = initial_pos
        self.vel = initial_vel
        self.dt = dt

    def input_acceleration2(self, acc_cmd, max_vel=1.5):
        self.vel += acc_cmd * self.dt
        self.pos += self.vel * self.dt
        return self.pos  # 平滑后的目标位置

    def input_acceleration(self, acc_cmd, max_vel=1.5):
        if acc_cmd[0] * self.vel[0] < 0:
            self.vel[0] /= 1.002
        if acc_cmd[1] * self.vel[1] < 0:
            self.vel[1] /= 1.002

        self.vel += acc_cmd * self.dt

        self.vel = np.clip(self.vel, -max_vel, max_vel)
        self.pos += self.vel * self.dt
        return self.pos  # 平滑后的目标位置

    def input_velocity(self, vel_cmd):
        self.vel = vel_cmd
        self.pos += self.vel * self.dt
        return self.pos  # 平滑后的目标位置


def plot_animation_two_ropes_3d_real(positions1, positions2, t, L=1.0, repeat=True,
                                name1="Real Rope", name2="Our Rope", batch_idx=0, save_path=None):
    """ Render 3D animation showing two ropes in one plot. """
    pos1_np = positions1[batch_idx].cpu().numpy()
    pos2_np = positions2[batch_idx].cpu().numpy()

    # pos1_np = positions1
    # pos2_np = positions2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)

    def animate(i):
        ax.clear()
        p1 = pos1_np[i]
        p2 = pos2_np[i]

        ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], 'o-', lw=1.5, color='crimson', label=name1, markersize=2)
        ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], 'o-', lw=1.5, color='navy', label=name2, markersize=2)

        ax.set_xlim(-1.0, 2.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0, 2.0)

        # ax.set_title(f"Time = {t[i]:.3f}s")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # ❗用 2D 坐标写时间文本，贴在坐标轴的左上角，不用 set_title 了
        ax.text2D(
            0.4, 0.80,  # (x, y) in axes coords, 左上角
            f"Time = {t[i]:.3f}s",
            transform=ax.transAxes,
            fontsize=12
        )

        # ❗把 legend 放到图内右上角，靠近绳子
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.90, 0.80),
            fontsize=10
        )

        ax.grid(True)

    num_frames = min(pos1_np.shape[0], pos2_np.shape[0])
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, repeat=repeat)

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=50)
        print("Animation saved.")

    # 计算动画总时长（秒）
    total_duration = (num_frames * 60) / 1000  # interval是毫秒，转换为秒

    if repeat == False:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation_3d_real(positions, t, L=1.0, repeat=True, batch_idx=0, save_path=None, interval=20):
    """ Renders 3D animation of rope motion """
    positions_np = positions[batch_idx].cpu().numpy()  # shape: (frames, nodes, 3)
    # positions_np = positions

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)

    def animate(i):
        ax.clear()
        pos = positions_np[i]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', lw=1.5, color='crimson', markersize=3)

        # 顶端点（假设是最后一个点，也可以改成 pos[0]）
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='blue', s=30, label='Tip')

        # XY平面影子
        # ax.plot(pos[:, 0], pos[:, 1], np.zeros_like(pos[:, 2]),
        #         'o--', lw=1, color='gray', alpha=0.5, label='Shadow (XY)', markersize=3)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.5, L + 0.5)
        ax.set_title(f"Time = {t[i]:.3f}s")
        # ax.set_title(f"Time = {t * i * 10:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)

    num_frames = positions_np.shape[0]
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=interval)

    if save_path:
        print(f"Saving 3D animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=30)
        print("Animation saved.")

    total_duration = (num_frames * 60) / 50

    if not repeat:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation_3d_traj(
        positions, target_traj, dt, record_interval, L=1.2, repeat=True, save_path=None
):
    """
    Renders 3D animation of rope motion with a highlighed target trajectory point.

    Args:
        positions: Tensor [batch, frames, nodes, 3]
        target_traj: Tensor [frames, 3] – target positions at each time
    """
    positions_np = positions.cpu().numpy()  # [frames, nodes, 3]
    target_np = target_traj.cpu().numpy()  # [frames, 3]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        pos = positions_np[i]  # 当前帧的绳子
        target = target_np[i]  # 当前帧的目标点

        # 绘制绳子
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', lw=1.5, color='crimson', markersize=3)
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='blue', s=30, label='Tip')

        # 当前目标点（高亮）
        ax.scatter(target[0], target[1], target[2], color='green', s=40, label='Target')

        # 可选：画出整条目标轨迹（透明灰色）
        ax.plot(target_np[:, 0], target_np[:, 1], target_np[:, 2],
                '--', lw=1, color='gray', alpha=0.5, label='Target Trajectory')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 2.0)
        ax.set_zlim(0.0, L)
        ax.set_title(f"Time = {i * dt * record_interval:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)
        ax.legend()

    num_frames = positions_np.shape[0]
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)

    if save_path:
        print(f"Saving 3D animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=30)
        print("Animation saved.")

    total_duration = (num_frames * 60) / 1000

    if not repeat:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation_3d_for_path_tracking(
    positions,
    goal_traj,
    dt,
    record_interval=1,
    L=1.0,
    batch_idx=0
):
    """
    positions: (batch, frames, nodes, 3) 或 (frames, nodes, 3)
    goal_traj: (frames, 3) 的目标轨迹（随时间变化）
    """

    # ---- 处理 positions ----
    if isinstance(positions, torch.Tensor):
        if positions.dim() == 4:  # (batch, frames, nodes, 3)
            positions_np = positions[batch_idx].detach().cpu().numpy()
        elif positions.dim() == 3:  # (frames, nodes, 3)
            positions_np = positions.detach().cpu().numpy()
        else:
            raise ValueError(f"positions 维度不对: {positions.shape}")
    else:
        positions_np = positions  # 已经是 numpy 的话直接用

    num_frames = positions_np.shape[0]

    # ---- 处理 goal_traj ----
    if isinstance(goal_traj, torch.Tensor):
        goal_np = goal_traj.detach().cpu().numpy()
    else:
        goal_np = goal_traj

    # 允许 (frames, 3) 或 (frames, 1, 3)
    if goal_np.ndim == 3 and goal_np.shape[1] == 1:
        goal_np = goal_np[:, 0, :]  # (frames,1,3) → (frames,3)

    assert goal_np.shape[0] >= num_frames, \
        f"goal_traj 帧数 {goal_np.shape[0]} 少于 positions 帧数 {num_frames}"

    goal_np = goal_np[:num_frames]  # 对齐帧数：取最前面的 num_frames 帧

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()

        # 当前绳子
        pos = positions_np[i]          # (nodes, 3)
        gx, gy, gz = goal_np[i].tolist()

        # 绳子
        ax.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            'o-', lw=1.5, color='crimson', markersize=3, label='Rope'
        )

        # 顶端（如果你的控制点是底端，就把 0 换成 -1）
        ax.scatter(
            pos[0, 0], pos[0, 1], pos[0, 2],
            color='blue', s=40, label='Top'
        )

        # 目标轨迹（从 0 到当前帧的轨迹）
        ax.plot(
            goal_np[:i+1, 0],
            goal_np[:i+1, 1],
            goal_np[:i+1, 2],
            '--', lw=1.5, color='green', label='Goal traj' if i == 0 else ""
        )

        # 当前目标点
        ax.scatter(
            gx, gy, gz,
            color='green', s=80, marker='*', label='Goal (current)'
        )

        # 轴范围
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.0, 2.0)
        ax.set_zlim(0.0, L + 0.2)

        ax.set_title(f"Time = {i * dt * record_interval:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)

        # 为了不每一帧都重复 legend，可以只在前几帧需要时显示
        if i == 0:
            ax.legend(loc='upper right')

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)
    plt.show()

    return ani

def plot_animation_3d_for_point_tracking(positions, goal, dt, record_interval=1, L=1.0, batch_idx=0):
    """
    positions: (batch, frames, nodes, 3)
    goal: (1,3) 或 (3,) → 固定目标点
    """
    # 处理 positions
    if positions.dim() == 4:
        positions_np = positions[batch_idx].cpu().numpy()  # (frames, nodes,3)
    else:
        positions_np = positions.cpu().numpy()

    num_frames = positions_np.shape[0]

    # 处理 goal → 固定目标点 (3,)
    if isinstance(goal, torch.Tensor):
        goal = goal.cpu().numpy()

    goal = goal.reshape(-1)  # 允许 (1,3) 或 (3,)
    assert goal.shape[0] == 3

    gx, gy, gz = goal.tolist()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        pos = positions_np[i]

        # 绳子
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                'o-', lw=1.5, color='crimson', markersize=3)

        # 顶端（若你底端为控制点，就改 -1）
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
                   color='blue', s=40, label='Top')

        # 目标点（固定）
        ax.scatter(gx, gy, gz, color='green', s=80, marker='*', label='Goal')

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(0.0, L + 0.2)

        ax.set_title(f"Time = {i * dt * record_interval:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)
    plt.show()

    return ani



def plot_animation_3d(positions, dt, record_interval, L, repeat=True, batch_idx=0, save_path=None):
    """ Renders 3D animation of rope motion """
    positions_np = positions[batch_idx].cpu().numpy()  # shape: (frames, nodes, 3)
    # positions_np = positions

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        pos = positions_np[i]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', lw=1.5, color='crimson', markersize=3)

        # 顶端点（假设是最后一个点，也可以改成 pos[0]）
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='blue', s=30, label='Tip')

        # XY平面影子
        # ax.plot(pos[:, 0], pos[:, 1], np.zeros_like(pos[:, 2]),
        #         'o--', lw=1, color='gray', alpha=0.5, label='Shadow (XY)', markersize=3)

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(0.0, L + 0.2)
        ax.set_title(f"Time = {i * dt * record_interval:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)

    num_frames = positions_np.shape[0]
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)

    if save_path:
        print(f"Saving 3D animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=30)
        print("Animation saved.")

    total_duration = (num_frames * 60) / 1000

    if not repeat:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani

def plot_animation_two_ropes_3d_2(positions1, positions2, dt, record_interval, L, repeat=True,
                                name1="Mujoco Rope", name2="Our Rope", batch_idx=0, save_path=None):
    """ Render 3D animation showing two ropes in one plot. """
    pos1_np = positions1[batch_idx].cpu().numpy()
    pos2_np = positions2[batch_idx].cpu().numpy()

    fig = plt.figure(figsize=(6, 6))          # 稍微小一点的画布
    ax = fig.add_subplot(111, projection='3d')

    # 调整体布局，减少大白边
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    def animate(i):
        ax.cla()  # 和 ax.clear() 一样，这里用 cla 简写

        p1 = pos1_np[i]
        p2 = pos2_np[i]

        ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], 'o-', lw=1.5,
                color='crimson', label=name1, markersize=2)
        ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], 'o-', lw=1.5,
                color='navy', label=name2, markersize=2)

        # ax.set_xlim(-1.0, 1.0)
        # ax.set_ylim(-1.0, 2.0)
        # ax.set_zlim(0.0, 2.0)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 2.0)

        # # 视角（这里是从 +Y 方向看 XZ 平面，你原来的：elev=0, azim=-90）
        # ax.view_init(elev=0, azim=-90)  # 稍微有点俯视，不那么扁

        # 坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ❗用 2D 坐标写时间文本，贴在坐标轴的左上角，不用 set_title 了
        ax.text2D(
            0.4, 0.80,                    # (x, y) in axes coords, 左上角
            f"Time = {i * dt * record_interval:.2f}s",
            transform=ax.transAxes,
            fontsize=12
        )

        # ❗把 legend 放到图内右上角，靠近绳子
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.90, 0.80),
            fontsize=10
        )

        ax.grid(True)

    num_frames = min(pos1_np.shape[0], pos2_np.shape[0])
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, repeat=repeat)

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=30)
        print("Animation saved.")

    # 计算动画总时长（秒）
    total_duration = (num_frames * 20) / 1000  # interval=20 ms

    if repeat is False:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation_two_ropes_3d(positions1, positions2, dt, record_interval, L, repeat=True,
                                name1="Mujoco Rope", name2="My Rope", batch_idx=0, save_path=None):
    """ Render 3D animation showing two ropes in one plot. """
    pos1_np = positions1[batch_idx].detach().cpu().numpy()
    pos2_np = positions2[batch_idx].detach().cpu().numpy()

    # pos1_np = positions1
    # pos2_np = positions2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        p1 = pos1_np[i]
        p2 = pos2_np[i]

        ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], 'o-', lw=1.5, color='crimson', label=name1, markersize=2)
        ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], 'o-', lw=1.5, color='navy', label=name2, markersize=2)

        ax.set_xlim(-2.0, 1.0)
        ax.set_ylim(-1.0, 2.0)
        ax.set_zlim(0.0, L + 0.2)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 2.0)
        ax.set_zlim(0.0, L + 0.5)

        # ax.view_init(elev=0, azim=-90)

        ax.set_title(f"Time = {i * dt * record_interval:.2f}s")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.grid(True)

    num_frames = min(pos1_np.shape[0], pos2_np.shape[0])
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, repeat=repeat)

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer="ffmpeg", fps=30)
        print("Animation saved.")

    # 计算动画总时长（秒）
    total_duration = (num_frames * 60) / 1000  # interval是毫秒，转换为秒

    if repeat == False:
        plt.show(block=False)
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation(positions, dt, record_interval, L, batch_idx=0, repeat=True, save_path=None):
    """ Renders animation """
    positions_np = positions[batch_idx].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        ax.clear()
        pos = positions_np[i]
        ax.plot(pos[:, 0], pos[:, 1], 'o-', lw=1.5, color='crimson', markersize=3)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.1, L + 0.1)
        ax.invert_yaxis()
        # Adjust title time calculation based on how history was recorded
        # If initial state (t=0) is included, time is i * dt * record_interval
        ax.set_title(f"Time = {i * dt * record_interval:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect('equal')

    num_frames = positions_np.shape[0]
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer="pillow", fps=30)
        print("Animation saved.")

    # 计算动画总时长（秒）
    total_duration = (num_frames * 60) / 1000  # interval是毫秒，转换为秒

    # 使用定时器在动画播放完成后关闭窗口
    if repeat == False:
        plt.show(block=False)  # 非阻塞显示
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()

    return ani


def plot_animation_two_ropes(positions1, positions2, dt, record_interval, L, repeat=True,
                             name1="Mujoco Rope", name2="My Rope", batch_idx=0, save_path=None):
    """ Render animation showing two ropes in one plot. """
    pos1_np = positions1[batch_idx].cpu().numpy()
    pos2_np = positions2[batch_idx].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        ax.clear()
        p1 = pos1_np[i]
        p2 = pos2_np[i]

        ax.plot(p1[:, 0], p1[:, 1], 'o-', lw=1.5, color='crimson', label=name1, markersize=3)
        ax.plot(p2[:, 0], p2[:, 1], 'o-', lw=1.5, color='navy', label=name2, markersize=3)

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.1, L + 0.1)
        ax.invert_yaxis()
        ax.set_title(f"Time = {i * dt * record_interval:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend()

    num_frames = min(pos1_np.shape[0], pos2_np.shape[0])
    ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, repeat=repeat)

    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer="pillow", fps=30)
        print("Animation saved.")

    # 计算动画总时长（秒）
    total_duration = (num_frames * 60) / 1000  # interval是毫秒，转换为秒

    # 使用定时器在动画播放完成后关闭窗口
    if repeat == False:
        plt.show(block=False)  # 非阻塞显示
        plt.pause(total_duration)
        plt.close()
    else:
        plt.show()
    return ani


def sin_traj(points, width=0.45, plot=False, device=None):
    """
    鸡蛋底端作为起点，且从起点开始点密集→后面稀疏。
    points 本身的非均匀性会直接体现在轨迹上。
    """

    if device is None:
        device = points.device

    traj = torch.vstack((torch.sin(4 * points) * width, points, torch.ones(len(points)) * 0.2)).T.to(device)

    if plot:
        plt.figure(figsize=(4, 4))
        plt.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), '-o', markersize=3)
        plt.scatter([0], [0], color='red', s=50, label='start / bottom (0,0)')
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.legend()
        plt.title("Egg trajectory (dense near start)")
        plt.tight_layout()
        plt.show()

    return traj


def eight_traj(points, scale_x=0.25, scale_y=0.45,
               z0=0.2, loops=1, plot=False, device="cpu"):
    """
    生成一个从 (0,0,z0) 开始的 3D 八字形（lemniscate）轨迹，
    并通过时间重参数化，让上下转弯处走得更慢，中间走得更快。
    """
    device = torch.device(device)
    points = points.to(device)

    # 1. 先生成线性的 u ∈ [0, 2π * loops]
    u = points / points[-1] * (2 * math.pi * loops)

    # 2. 对 u 做非线性 warp：上下慢，中间快
    a = -0.2  # 负的，让上下转弯处时间变慢
    t = u + a * torch.sin(2 * u)

    # 3. Lemniscate of Bernoulli 公式（你改过的 x,y 方向）
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    denom = 1.0 + sin_t**2

    y = -scale_x * cos_t / denom
    x =  scale_y * (sin_t * cos_t) / denom
    z = torch.ones_like(t) * z0

    traj = torch.stack([x, y, z], dim=-1)  # (T, 3)

    # 4. 平移使起点变为 (0, 0, z0)
    start = traj[0].clone()
    start[2] = 0.0
    traj = traj - start

    if plot:
        plt.figure(figsize=(4, 4))
        plt.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), '-o', markersize=3)
        plt.scatter([0], [0], color='red', s=50, label='start (0,0)')
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.legend()
        plt.title("Eight trajectory (slow at turns, fast in middle)")
        plt.tight_layout()
        plt.show()

    return traj


def egg_traj(points,
             scale_x=0.25, scale_y=0.35,
             k=0.3, z_val=0.2,
             plot=False, device=None):
    """
    鸡蛋底端作为起点，且从起点开始点密集→后面稀疏。
    points 本身的非均匀性会直接体现在轨迹上。
    """

    if device is None:
        device = points.device

    # 归一化到 [0, 1]，再映射到 [0, 2π]
    t = (points - points[0]) / (points[-1] - points[0])
    theta = t * 2 * torch.pi          # t=0 -> theta=0

    # 让 theta=0 对应鸡蛋底端：theta_shift = theta - π/2
    theta_shift = theta - torch.pi / 2

    # 鸡蛋极坐标半径
    r = 1.0 - k * torch.cos(theta_shift)

    # 转为笛卡尔坐标
    x = scale_x * r * torch.cos(theta_shift)
    y = scale_y * r * torch.sin(theta_shift)

    # 把第一个点平移到 (0,0)
    x0, y0 = x[0].clone(), y[0].clone()
    x = x - x0
    y = y - y0
    # 强制数值上完全 0
    x[0] = 0.0
    y[0] = 0.0

    z = torch.ones_like(x) * z_val
    traj = torch.stack((x, y, z), dim=-1).to(device)

    if plot:
        plt.figure(figsize=(4, 4))
        plt.plot(x.cpu(), y.cpu(), '-o', markersize=3)
        plt.scatter([0], [0], color='red', s=50, label='start / bottom (0,0)')
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.legend()
        plt.title("Egg trajectory (dense near start)")
        plt.tight_layout()
        plt.show()

    return traj


def egg_traj2(points,
             scale_x=0.25, scale_y=0.35,
             k=0.3, z_val=0.2,
             curv_strength=2.0,   # 曲率加权强度，0 表示不管曲率
             plot=False, device=None):
    """
    鸡蛋型轨迹：
    - 底端作为起点，并被平移到 (0,0,z_val)
    - 使用 points 的非均匀分布（起点可更密）
    - 同时在曲率高的地方进一步加密采样

    Args:
        points: (N,) 任意单调递增时间/参数
        scale_x, scale_y: 鸡蛋整体大小
        k: 控制蛋的尖钝
        z_val: z 固定值
        curv_strength: 曲率权重强度，建议 0~5 之间
        plot: 是否画 2D 轨迹
        device: 输出设备
    """

    if device is None:
        device = points.device

    # ---------- 1. 归一化 points 到 [0, 1] ----------
    t = (points - points[0]) / (points[-1] - points[0])
    t = t.clamp(0.0, 1.0)

    # ---------- 2. 在 [0, 2π] 上高分辨率采样鸡蛋曲线 ----------
    M = 2000  # 曲率估计用的密集采样点数
    phi_dense = torch.linspace(0.0, 2 * torch.pi, M, device=device)  # 这里 phi=0 就是底端

    # 对应之前的写法：theta_shift = phi - π/2
    theta_shift_dense = phi_dense - torch.pi / 2

    r_dense = 1.0 - k * torch.cos(theta_shift_dense)
    x_dense = scale_x * r_dense * torch.cos(theta_shift_dense)
    y_dense = scale_y * r_dense * torch.sin(theta_shift_dense)

    # ---------- 3. 用差分近似曲率 κ(phi) ----------
    dphi = phi_dense[1] - phi_dense[0]

    # 一阶导 (中心差分)
    x1 = (x_dense[2:] - x_dense[:-2]) / (2 * dphi)
    y1 = (y_dense[2:] - y_dense[:-2]) / (2 * dphi)

    # 二阶导
    x2 = (x_dense[2:] - 2 * x_dense[1:-1] + x_dense[:-2]) / (dphi ** 2)
    y2 = (y_dense[2:] - 2 * y_dense[1:-1] + y_dense[:-2]) / (dphi ** 2)

    # 曲率公式 κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    num = torch.abs(x1 * y2 - y1 * x2)
    denom = (x1 ** 2 + y1 ** 2).clamp(min=1e-8) ** 1.5
    kappa = num / denom

    # 补到长度 M
    curv = torch.zeros_like(phi_dense)
    curv[1:-1] = kappa
    curv[0] = curv[1]
    curv[-1] = curv[-2]

    # 归一化曲率到 [0,1]
    if curv.max() > 0:
        curv = curv / curv.max()
    else:
        curv = torch.zeros_like(curv)

    # ---------- 4. 曲率权重 + 累积分布 C(phi) ----------
    # w 越大，对应区域采样越密
    w = 1.0 + curv_strength * curv   # curv_strength=0 就是均匀
    C = torch.cumsum(w, dim=0)
    C = (C - C[0]) / (C[-1] - C[0])  # 归一化到 [0,1]

    # ---------- 5. 用 t 通过 C 的“反函数”取出 phi ----------
    # searchsorted 找到每个 t 落在 C 的哪个区间
    idx = torch.searchsorted(C, t)
    idx = torch.clamp(idx, 1, M - 1)

    idx0 = idx - 1
    idx1 = idx

    C0 = C[idx0]
    C1 = C[idx1]
    phi0 = phi_dense[idx0]
    phi1 = phi_dense[idx1]

    # 线性插值：在 (C0, C1) 区间中找到对应 t 的 phi
    alpha = (t - C0) / (C1 - C0 + 1e-8)
    phi = phi0 + alpha * (phi1 - phi0)

    # ---------- 6. 使用 phi 生成最终鸡蛋轨迹 ----------
    theta_shift = phi - torch.pi / 2
    r = 1.0 - k * torch.cos(theta_shift)

    x = scale_x * r * torch.cos(theta_shift)
    y = scale_y * r * torch.sin(theta_shift)

    # 把起点平移到 (0,0)，保证底端 = 起点 = (0,0)
    x0, y0 = x[0].clone(), y[0].clone()
    x = x - x0
    y = y - y0
    x[0] = 0.0
    y[0] = 0.0

    z = torch.ones_like(x) * z_val
    traj = torch.stack((x, y, z), dim=-1).to(device)

    # ---------- 7. 可视化 ----------
    if plot:
        plt.figure(figsize=(4, 4))
        plt.plot(x.cpu(), y.cpu(), 'o-', markersize=3)
        plt.scatter([0], [0], color='red', s=50, label='start / bottom (0,0)')
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.legend()
        plt.title(f"Egg trajectory (curv_strength={curv_strength})")
        plt.tight_layout()
        plt.show()

    return traj


def half_dense_then_uniform(N=100, ratio=0.5, sharpness=2.0, mode='exp', interval=(0.0, 2.0), plot=True):
    """
    非对称采样：前半段密集→稀疏，后半段匀速采样，spacing在0.5处连续。

    Args:
        N (int): 总采样点数
        ratio (float): 非线性密集采样段占比（如0.5表示前一半）
        sharpness (float): 非线性变化控制强度（越大越陡）
        mode (str): 'exp' | 'sqrt' | 'poly'
        plot (bool): 是否可视化

    Returns:
        torch.Tensor: 非均匀采样点，shape=(N,)
    """
    start, end = interval
    n_dense = int(N * ratio)
    n_uniform = N - n_dense

    t_dense = torch.linspace(0, 1, steps=n_dense)

    # 非线性函数：密集→稀疏，映射到 [0, ratio]
    if mode == 'exp':
        base = torch.exp(torch.tensor(sharpness))
        x_dense = ratio * (base ** t_dense - 1) / (base - 1)
    elif mode == 'sqrt':
        x_dense = ratio * t_dense**0.5
    elif mode == 'poly':
        x_dense = ratio * t_dense**(1 / sharpness)
    else:
        raise ValueError("Unsupported mode")

    # 计算 0.5 处 spacing，用它延长均匀段
    final_spacing = x_dense[-1] - x_dense[-2]
    x_uniform = x_dense[-1] + final_spacing * torch.arange(1, n_uniform + 1)

    x = torch.cat([x_dense, x_uniform], dim=0)

    x = x * ((end - start) / x[-1]) + start

    if plot:
        plt.figure(figsize=(7, 3))
        plt.plot(x.numpy(), torch.arange(N), 'o-')
        plt.title(f"Asymmetric Sampling: dense→uniform (mode={mode})")
        plt.xlabel("Sample value"); plt.ylabel("Index")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return x

def rolling_window(position_data, steps=1, size=1, start=1):
    """
        steps       # 窗口长度
        size       # 窗口数量
        start   # 从第 start 个时间步开始滑窗
    """
    T, *rest = position_data.shape
    assert T >= start + steps + size - 1, f"Insufficient time steps: T={T}, required T ≥ start + k + m - 1"

    stride0, stride1, stride2 = position_data.stride()
    base = position_data[start:]  # 从 start 开始滑窗
    return base.as_strided((size, steps, *rest), (stride0, stride0, stride1, stride2))


def rolling_window_custom_starts(position_data, steps=1, starts=None):
    """
    position_data: Tensor of shape (T, ...)
    steps: Window length
    starts: 1D array or list of starting indices for each window

    Returns:
        Tensor of shape (len(starts), steps, ...)
    """
    T, *rest = position_data.shape
    starts = torch.as_tensor(starts, device=position_data.device)
    assert torch.all(starts + steps <= T), "Some windows exceed data length!"

    # 生成时间步索引矩阵，每一行是一个窗口的起点+步长偏移
    idx_matrix = starts.unsqueeze(1) + torch.arange(steps, device=position_data.device).unsqueeze(0)  # Shape: (len(starts), steps)

    # 高效索引
    return position_data[idx_matrix]  # Shape: (len(starts), steps, *rest)


def compile_model(model):
    # Attempt to compile the model with torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Attempting to compile the model with torch.compile (mode='reduce-overhead')...")
        # Using a specific backend like 'inductor' can sometimes offer more speedups
        # For CPU, Triton might not be available by default with standard pip install of PyTorch on all OSes.
        # Inductor is a more general backend.
        # You might need to install nightly PyTorch for some advanced backends or features.
        try:
            # mode="max-autotune" can give better performance but takes longer to compile.
            # mode="reduce-overhead" is a good starting point.
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Failed to compile model: {e}. Running without compilation.")
    else:
        print("torch.compile not available (requires PyTorch 2.0+). Running without compilation.")


def mj_data_to_my_data(N, sampled_data, device=torch.device('cpu')):
    diff = int(40 / N)

    position_data = sampled_data[:, 1:1 + 41*3]  # (sample_num, node_num*3)
    match_node_index = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(0, position_data.shape[1] // 3, diff)])
    position_data = position_data[:, match_node_index]  # (sample_num, (N+1)*3)
    position_data = position_data.reshape(position_data.shape[0], -1, 3)  # (sample_num, N+1, 3)
    position_data = torch.from_numpy(position_data).to(device)

    velocity_data = sampled_data[:, 1+41*3:1 + 41*3 * 2]  # (sample_num, node_num*3)
    velocity_data = velocity_data[:, match_node_index]  # (sample_num, (N+1)*3)
    velocity_data = velocity_data.reshape(velocity_data.shape[0], -1, 3)  # (sample_num, N+1, 3)
    velocity_data = torch.from_numpy(velocity_data).to(device)

    control_sequence = sampled_data[:-1, -3:]
    control_sequence = torch.from_numpy(control_sequence).to(device)

    return position_data, velocity_data, control_sequence


def mj_data_to_my_data_short(N, sampled_data, device=torch.device('cpu')):
    diff = int(12 / N)

    position_data = sampled_data[:, 1:1 + 13*3]  # (sample_num, node_num*3)
    match_node_index = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(0, position_data.shape[1] // 3, diff)])
    position_data = position_data[:, match_node_index]  # (sample_num, (N+1)*3)
    position_data = position_data.reshape(position_data.shape[0], -1, 3)  # (sample_num, N+1, 3)
    position_data = torch.from_numpy(position_data).to(device)

    velocity_data = sampled_data[:, 1+13*3:1 + 13*3 * 2]  # (sample_num, node_num*3)
    velocity_data = velocity_data[:, match_node_index]  # (sample_num, (N+1)*3)
    velocity_data = velocity_data.reshape(velocity_data.shape[0], -1, 3)  # (sample_num, N+1, 3)
    velocity_data = torch.from_numpy(velocity_data).to(device)

    control_sequence = sampled_data[:-1, -3:]
    control_sequence = torch.from_numpy(control_sequence).to(device)

    return position_data, velocity_data, control_sequence


def make_times_monotonic(pos, times, merge_duplicates=True):
    pos = np.asarray(pos)
    times = np.asarray(times, dtype=float)
    if pos.ndim != 3 or pos.shape[0] != times.shape[0]:
        raise ValueError("pos 需为 (T, N, 3)，且 pos.shape[0] == times.shape[0]")

    T, N, D = pos.shape
    valid = np.isfinite(times) & np.isfinite(pos.reshape(T, -1)).all(axis=1)
    times = times[valid]
    pos = pos[valid]
    if times.size == 0:
        raise ValueError("清理后无有效样本。")

    order = np.argsort(times, kind="mergesort")
    times_sorted = times[order]
    pos_sorted = pos[order]

    if not merge_duplicates:
        return pos_sorted, times_sorted

    pos_flat = pos_sorted.reshape(len(times_sorted), -1)
    uniq_t, inv, counts = np.unique(times_sorted, return_inverse=True, return_counts=True)
    accum = np.zeros((len(uniq_t), pos_flat.shape[1]), dtype=pos_flat.dtype)
    np.add.at(accum, inv, pos_flat)
    pos_flat_mean = accum / counts[:, None]

    pos_out = pos_flat_mean.reshape(len(uniq_t), N, D)
    times_out = uniq_t
    return pos_out, times_out


def resample_uniform(pos, times, dt=None, kind="cubic"):
    if dt is None:
        dt = np.median(np.diff(times))
    dt = 0.01  # 固定 0.01 s
    t_uniform = np.arange(times[0], times[-1], dt)

    T, N, D = pos.shape
    pos_flat = pos.reshape(T, -1)
    f = interp1d(times, pos_flat, kind=kind, axis=0, fill_value="extrapolate")
    pos_uniform = f(t_uniform).reshape(len(t_uniform), N, D)
    return pos_uniform, t_uniform


def compute_velocity(pos, times, window_length=41, polyorder=3):
    T, N, D = pos.shape
    dt = float(np.mean(np.diff(times)))
    pos_flat = pos.reshape(T, -1)

    pos_flat = savgol_filter(
        pos_flat, window_length=41, polyorder=polyorder,
        deriv=0, delta=dt, axis=0
    )
    vel_flat = savgol_filter(
        pos_flat, window_length=window_length, polyorder=polyorder,
        deriv=1, delta=dt, axis=0
    )
    pos = pos_flat.reshape(T, N, D)
    vel = vel_flat.reshape(T, N, D)
    return pos, vel


def get_goal_traj(total_time=2.5, curve_type='sin', ctr_period=10, device=torch.device('cpu')):
    total_horizon = int(1000 / ctr_period * total_time)
    points = half_dense_then_uniform(
        N=total_horizon + 1,
        ratio=0.3,
        sharpness=2.0,
        mode='exp',
        interval=(0.0, 2.0),
        plot=False
    )

    if curve_type == 'sin':
        Goal_traj = sin_traj(points, width=0.45, plot=False, device=device)
    elif curve_type == 'egg':
        Goal_traj = egg_traj(points, scale_x=0.38, scale_y=0.52, plot=False, device=device)
    else:
        Goal_traj = eight_traj(points, scale_x=0.45 * 1.0, scale_y=0.65 * 1.0, z0=0.2, loops=1, plot=False, device=device)

    return Goal_traj



if __name__ == "__main__":
    position_data = torch.arange(20).reshape(20, 1)  # (T=20, 1)

    steps = 4
    starts = [0, 2, 5, 10]  # 指定滑窗起点

    windows = rolling_window_custom_starts(position_data, steps, starts)
    print(windows.squeeze(-1))
