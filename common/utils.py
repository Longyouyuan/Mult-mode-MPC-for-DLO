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

    def input_acceleration(self, acc_cmd, max_vel=np.inf):
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
        positions: Tensor [frames, nodes, 3]
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


class LiveMPCVisualizer:
    def __init__(self, goal_traj, refresh_pause=0.001, margin=0.05):
        """
        goal_traj: (T,3) torch/numpy，初始化时传入，用它确定坐标轴范围
        refresh_pause: plt.pause 的刷新间隔
        margin: 给范围加一点边距（相对最大跨度的比例）
        """
        plt.ion()
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.refresh_pause = refresh_pause

        self.goal_traj = None
        self._axis_center = None  # (3,)
        self._axis_half = None    # float

        self.set_goal(goal_traj, margin=margin)  # ✅ 初始化就设 goal，并计算轴范围

    def _to_np(self, x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def set_goal(self, goal_traj, margin=0.05):
        """
        根据 goal_traj 设定坐标轴范围：
        - zmin 强制为 0
        - 三轴等比例：用最大跨度决定 half range
        """
        goal_np = self._to_np(goal_traj)
        if goal_np is None or goal_np.ndim != 2 or goal_np.shape[1] != 3:
            raise ValueError("goal_traj must be (T,3)")

        self.goal_traj = goal_np

        mins = goal_np.min(axis=0)
        maxs = goal_np.max(axis=0)

        # ✅ z 最小值固定为 0
        mins[2] = 0.0

        center = 0.5 * (mins + maxs)
        span = (maxs - mins)
        half = 0.5 * float(np.max(span))

        # 给一点边距，避免贴边（按最大跨度比例）
        half = half * (1.0 + float(margin))
        half = max(half, 1e-6)

        self._axis_center = center
        self._axis_half = half

    def _apply_equal_axis(self):
        c = self._axis_center
        h = self._axis_half
        ax = self.ax
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)

        # ✅ z 最小值为 0（并且仍保持等比例：如果 c[2]-h < 0，则把 z 向上推）
        zmin = c[2] - h
        zmax = c[2] + h
        if zmin < 0.0:
            shift = -zmin
            zmin += shift
            zmax += shift
        ax.set_zlim(zmin, zmax)

    def update(self, rope_pos, cur_i=0,
               cand_tip_traj=None, cand_cost=None, m_mode_trajs=None,
               title=""):
        """
        rope_pos:      (P,3) torch/numpy
        cur_i:         int, current mpc step index
        cand_tip_traj: (K,H,3) torch/numpy
        cand_cost:     (K,) torch/numpy
        m_mode_trajs:   (m modes,H,3) torch/numpy
        """
        rope_np = self._to_np(rope_pos)
        goal_np = self.goal_traj

        cand_np = self._to_np(cand_tip_traj)
        cost_np = self._to_np(cand_cost)
        best_np = self._to_np(m_mode_trajs)

        ax = self.ax
        ax.clear()

        # ---- axes style ----
        self._apply_equal_axis()  # ✅ 每帧按 goal 设定好的等比例范围
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.grid(True)

        # ---- plot rope ----
        if rope_np is not None:
            ax.plot(rope_np[:, 0], rope_np[:, 1], rope_np[:, 2],
                    'o-', lw=1.5, color='crimson', markersize=1.5, label='Rope')

            # Tip：你注释写“TOP tip = first particle”，这里继续用 0
            ax.scatter(rope_np[0, 0], rope_np[0, 1], rope_np[0, 2],
                       color='blue', s=30, label='Tip')

        # ---- goal traj + current target ----
        if goal_np is not None and goal_np.shape[0] > 0:
            ax.plot(goal_np[:, 0], goal_np[:, 1], goal_np[:, 2],
                    '--', lw=2.0, color='gray', alpha=0.5, label='Target Trajectory')

            gi = int(np.clip(cur_i, 0, goal_np.shape[0] - 1))
            target = goal_np[gi]
            ax.scatter(target[0], target[1], target[2],
                       color='green', s=40, label='Target')

        # ---- candidates (low reward=red, high=green) ----
        if cand_np is not None and cand_np.shape[0] > 0:
            cmap = plt.get_cmap("RdYlGn")

            if cost_np is not None and cost_np.shape[0] == cand_np.shape[0]:
                reward = -cost_np
                rmin, rmax = float(reward.min()), float(reward.max())
                denom = (rmax - rmin) if (rmax - rmin) > 1e-8 else 1.0
                w = (reward - rmin) / denom  # [0,1]
            else:
                w = np.full((cand_np.shape[0],), 0.5, dtype=np.float32)

            for k in range(cand_np.shape[0]):
                tr = cand_np[k]
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                        lw=1.0, alpha=0.25, color=cmap(float(w[k])))

        # ---- best trajectories ----
        if best_np is not None and best_np.shape[0] > 0:
            # 用绿色突出 best（也符合你“好=绿”的语义）
            for j in range(best_np.shape[0]):
                tr = best_np[j]
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                        lw=1.5, alpha=0.8, color='blue',
                        label='Best' if j == 0 else None)

        # ---- legend (clean) ----
        handles, labels = ax.get_legend_handles_labels()
        if cand_np is not None and cand_np.shape[0] > 0:
            from matplotlib.lines import Line2D
            cand_proxy = Line2D([0], [0], color='gray', lw=1.5, alpha=0.35)
            handles.append(cand_proxy)
            labels.append(f"Candidates (K={cand_np.shape[0]})")

        ax.legend(handles, labels, loc='best')

        plt.pause(self.refresh_pause)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def _remove_near_duplicates(xy_np, eps=1e-6):
    """去掉连续重复/极近点，避免弧长=0导致 searchsorted 出问题"""
    if xy_np.shape[0] < 2:
        return xy_np
    dif = xy_np[1:] - xy_np[:-1]
    seg = np.linalg.norm(dif, axis=1)
    keep = np.ones((xy_np.shape[0],), dtype=bool)
    keep[1:] = seg > eps
    out = xy_np[keep]
    return out


def preprocess_start_and_scale(
    xy_np,
    start_to_zero=True,
    scale_x=1.0,
    scale_y=1.0,
    keep_aspect=False
):
    """
    start_to_zero: 起点平移到 (0,0)（和绳子初始对齐常用）
    scale_x/scale_y: 最终轨迹在 x/y 方向的 peak-to-peak 跨度（单位 m）
    keep_aspect: True => 保持形状比例，用统一缩放因子（取满足两者的较小者）
    """
    xy = xy_np.astype(np.float32).copy()

    # 1) 起点对齐到 (0,0)
    if start_to_zero:
        xy -= xy[0:1]

    # 2) 计算当前跨度
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)  # (2,)

    # 3) 缩放（不再 center，避免破坏“起点=0”）
    if keep_aspect:
        sx = float(scale_x) / float(span[0])
        sy = float(scale_y) / float(span[1])
        s = min(sx, sy)
        xy *= s
    else:
        xy[:, 0] *= float(scale_x) / float(span[0])
        xy[:, 1] *= float(scale_y) / float(span[1])

    return xy


def uniform_resample_by_arclength(xy_np, M=800):
    """
    把手绘 polyline 按弧长均匀重采样成 M 个点（解决“有些段稀疏”）
    返回: (M,2) float32
    """
    xy = _remove_near_duplicates(xy_np)
    if xy.shape[0] < 2:
        raise ValueError("Too few points after removing duplicates.")

    # cumulative arclength
    dif = xy[1:] - xy[:-1]
    seg = np.linalg.norm(dif, axis=1).astype(np.float32)
    s = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(seg)], axis=0)  # (K,)
    total = float(s[-1])
    if total < 1e-8:
        raise ValueError("Total length is ~0; drawing is degenerate.")

    s01 = s / total  # normalize to [0,1]
    u = np.linspace(0.0, 1.0, num=int(M), dtype=np.float32)

    # piecewise linear interpolation
    idx = np.searchsorted(s01, u, side="right") - 1
    idx = np.clip(idx, 0, len(s01) - 2)

    s0 = s01[idx]
    s1 = s01[idx + 1]
    w = (u - s0) / (s1 - s0 + 1e-8)

    p0 = xy[idx]
    p1 = xy[idx + 1]
    out = p0 + (p1 - p0) * w[:, None]
    return out.astype(np.float32)


def sample_with_schedule_from_uniform(xy_uniform_np, points_t, device="cpu"):
    """
    xy_uniform_np: (M,2) 已经是“弧长均匀参数化”的轨迹
    points_t: (T,) torch, in [0, t_end]（前密后疏）
    逻辑：把 points_t 归一化到 [0,1]，当作弧长参数，从 uniform轨迹上取点
    返回: (T,2) torch
    """
    device = torch.device(device)
    xy = torch.tensor(xy_uniform_np, dtype=torch.float32, device=device)  # (M,2)
    M = xy.shape[0]

    t_end = float(points_t[-1].item())
    u = (points_t / (t_end + 1e-8)).clamp(0.0, 1.0)  # (T,)

    # u -> index in [0, M-1]
    pos = u * (M - 1)
    i0 = torch.floor(pos).long().clamp(0, M - 2)
    i1 = i0 + 1
    w = (pos - i0.float()).unsqueeze(1)  # (T,1)

    p0 = xy[i0]
    p1 = xy[i1]
    out = p0 + (p1 - p0) * w
    return out


def smooth_gaussian(xy_np, sigma=2.0, radius=None):
    """
    xy_np: (M,2)
    sigma: 越大越平滑
    radius: 核半径，默认 3*sigma
    """
    if sigma is None or sigma <= 0:
        return xy_np

    if radius is None:
        radius = int(max(1, round(3.0 * float(sigma))))
    radius = int(radius)

    t = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(t ** 2) / (2 * float(sigma) ** 2))
    kernel /= (kernel.sum() + 1e-8)

    out = np.zeros_like(xy_np, dtype=np.float32)
    for d in range(2):
        v = xy_np[:, d]
        v_pad = np.pad(v, (radius, radius), mode="edge")
        out[:, d] = np.convolve(v_pad, kernel, mode="valid")
    return out



def build_goal_traj_from_drawn(
    drawn_path,
    total_horizon,
    device="cpu",
    z0=0.2,

    # --- scaling ---
    scale_x=1.0,
    scale_y=1.0,
    keep_aspect=False,
    sigma=2.0,

    # --- uniform fix for sparse segments ---
    uniform_M=800,

    # --- schedule ---
    ratio=0.3,
    sharpness=2.0,
    interval=(0.0, 2.0)
):
    """
    输出:
      Goal_traj: (total_horizon+1, 3) torch on device
    """
    xy_raw = np.load(drawn_path).astype(np.float32)  # (K,2)

    # 1) 起点对齐到 (0,0) + 缩放到指定长宽
    xy_scaled = preprocess_start_and_scale(
        xy_raw,
        start_to_zero=True,
        scale_x=scale_x,
        scale_y=scale_y,
        keep_aspect=keep_aspect
    )

    # ✅ 新增：平滑一步（调这个 sigma）
    xy_scaled = smooth_gaussian(xy_scaled, sigma=sigma)

    # 2) 先均匀重采样（解决“有些点之间很稀疏”）
    xy_uniform = uniform_resample_by_arclength(xy_scaled, M=uniform_M)  # (M,2)

    # 3) 前密后疏 schedule（长度 total_horizon+1）
    points = half_dense_then_uniform(
        N=int(total_horizon) + 1,
        ratio=ratio,
        sharpness=sharpness,
        mode="exp",
        interval=interval,
        plot=False
    ).to(device)

    # 4) 用 schedule 从“均匀轨迹”上取点 => (T,2)
    xy_goal = sample_with_schedule_from_uniform(xy_uniform, points, device=device)

    # 5) 加 z 维
    z = torch.ones((xy_goal.shape[0], 1), device=xy_goal.device, dtype=xy_goal.dtype) * float(z0)
    Goal_traj = torch.cat([xy_goal[:, 0:1], xy_goal[:, 1:2], z], dim=1)  # (T,3)

    return Goal_traj


def draw_traj_xy(
    save_path="drawn_traj.npy",
    xlim=(-0.5, 0.5),
    ylim=(-0.5, 0.5),
    figsize=(7, 7),
    dpi=120
):
    """
    鼠标左键按住拖动绘制一条连续轨迹（单位：米）
    xlim / ylim: 画布物理范围（米），可非正方形
    """
    pts = []
    drawing = {"on": False}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ✅ 关键：画布范围由你指定（物理坐标）
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # 是否强制等比例（建议开，防止形状被拉伸）
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Draw trajectory in XY (meters)\n"
        f"x∈[{xlim[0]}, {xlim[1]}], y∈[{ylim[0]}, {ylim[1]}]\n"
        f"LMB drag=draw | Enter=save | Backspace=clear | Esc=quit"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)

    line, = ax.plot([], [], "-", lw=2)

    def redraw():
        if len(pts) < 2:
            line.set_data([], [])
        else:
            arr = np.asarray(pts, dtype=np.float32)
            line.set_data(arr[:, 0], arr[:, 1])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        drawing["on"] = True
        pts.append([event.xdata, event.ydata])
        redraw()

    def on_move(event):
        if not drawing["on"] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        pts.append([event.xdata, event.ydata])
        redraw()

    def on_release(event):
        if event.button == 1:
            drawing["on"] = False

    def on_key(event):
        if event.key == "enter":
            if len(pts) < 2:
                print("[draw] Too few points, not saved.")
                plt.close(fig)
                return
            arr = np.asarray(pts, dtype=np.float32)
            np.save(save_path, arr)
            print(f"[draw] Saved: {save_path}, shape={arr.shape}")
            plt.close(fig)

        elif event.key == "backspace":
            pts.clear()
            redraw()

        elif event.key == "escape":
            print("[draw] Quit without saving.")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return save_path


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


def plot_tip_vs_goal_and_error(
    pos_history, goal_traj, dt, record_interval, tip_idx=-1
):
    """
    pos_history: Tensor (T,P,3) or (1,T,P,3)
    goal_traj:   Tensor (T,3)
    dt:          physics dt
    record_interval: steps per recorded frame
    tip_idx:     index of tip particle (default -1)
    """

    # ---------- normalize shapes ----------
    if isinstance(pos_history, torch.Tensor):
        ph = pos_history.detach()
    else:
        ph = torch.as_tensor(pos_history)

    if ph.dim() == 4:        # (B,T,P,3)
        ph = ph[0]
    assert ph.dim() == 3, f"pos_history should be (T,P,3), got {ph.shape}"

    if isinstance(goal_traj, torch.Tensor):
        gt = goal_traj.detach()
    else:
        gt = torch.as_tensor(goal_traj)

    if gt.dim() == 3:        # (1,T,3)
        gt = gt[0]
    assert gt.dim() == 2, f"goal_traj should be (T,3), got {gt.shape}"

    # ---------- align length ----------
    T = min(ph.shape[0], gt.shape[0])
    ph = ph[:T]
    gt = gt[:T]

    # ---------- tip & error ----------
    tip = ph[:, tip_idx, :]                 # (T,3)
    err = tip - gt                          # (T,3)
    err_norm = torch.linalg.norm(err, dim=1)

    # ---------- time axis ----------
    t = torch.arange(T, device=tip.device) * (dt * record_interval)

    tip_np = tip.cpu().numpy()
    gt_np = gt.cpu().numpy()
    errn_np = err_norm.cpu().numpy()
    t_np = t.cpu().numpy()

    # ================= plotting =================
    fig = plt.figure(figsize=(12, 5))

    # -------- subplot 1: 3D trajectory --------
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')

    ax0.plot(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
             '--', lw=2, label='Goal')
    ax0.plot(tip_np[:, 0], tip_np[:, 1], tip_np[:, 2],
             '-', lw=2, label='Tip')

    ax0.scatter(gt_np[0, 0], gt_np[0, 1], gt_np[0, 2],
                s=40, label='Goal start')
    ax0.scatter(tip_np[0, 0], tip_np[0, 1], tip_np[0, 2],
                s=40, label='Tip start')

    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")
    ax0.set_title("Tip trajectory vs Goal")
    ax0.legend()
    ax0.grid(True)

    # ---- enforce equal XYZ scale ----
    all_pts = np.concatenate([tip_np, gt_np], axis=0)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    half = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    half = max(half, 1e-6)  # 防止退化

    ax0.set_xlim(cx - half, cx + half)
    ax0.set_ylim(cy - half, cy + half)
    ax0.set_zlim(cz - half, cz + half)

    # -------- subplot 2: ||error|| --------
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(t_np, errn_np, lw=2)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("||error||")
    ax1.set_title("Tracking error norm over time")
    ax1.grid(True)

    plt.tight_layout()
    plt.show()


def plot_goal_traj(goal_traj, T_task, title="Goal Trajectory",
                   color='gray', scatter=True, equal_xy=True):
    """
    goal_traj: (frames,3)
    T_task: total time (s)
    equal_xy: if True, keep equal scale in XY plane
    """
    if isinstance(goal_traj, torch.Tensor):
        goal_np = goal_traj.detach().cpu().numpy()
    else:
        goal_np = np.asarray(goal_traj)

    # --- length ---
    diffs = goal_np[1:] - goal_np[:-1]                 # (F-1,3)
    seg_lengths = np.linalg.norm(diffs, axis=1)         # (F-1,)
    total_length = float(seg_lengths.sum())

    # --- max speed ---
    # assume uniform sampling over T_task across (frames-1) segments
    if goal_np.shape[0] >= 2:
        dt = float(T_task) / float(goal_np.shape[0] - 1)
        speeds = seg_lengths / dt                       # (F-1,)
        max_speed = float(speeds.max())
    else:
        dt = 0.0
        max_speed = 0.0

    title2 = (f"{title} (Length: {total_length:.3f} m, "
              f"Est. Time: {T_task:.1f} s, Max speed: {max_speed:.3f} m/s)")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(goal_np[:, 0], goal_np[:, 1], goal_np[:, 2],
            lw=2, color=color, label='Goal Trajectory')
    if scatter:
        ax.scatter(goal_np[:, 0], goal_np[:, 1], goal_np[:, 2],
                   color=color, s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title2)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # ---- make XY plane equal scale ----
    if equal_xy:
        # set equal aspect for x and y by matching ranges
        x_min, x_max = goal_np[:, 0].min(), goal_np[:, 0].max()
        y_min, y_max = goal_np[:, 1].min(), goal_np[:, 1].max()
        z_min, z_max = 0.0, 1.0

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)

        half = 0.5 * max((x_max - x_min), (y_max - y_min)) * 1.2  # 1.2 for margin
        # avoid degenerate
        half = max(half, 1e-6)

        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)

        # z 不强制等比例，只给个合理范围（你也可以按需固定）
        ax.set_zlim(z_min, z_max)

    plt.show()



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
