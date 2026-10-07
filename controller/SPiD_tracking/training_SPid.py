import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rope import Rope
from common.utils import *
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import math
import random

# torch.manual_seed(0)


class Controller(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_width, max_action):
        super(Controller, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(input_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, action_dim)

        # # Initialize weights for better training results
        # nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.l2.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.l3.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.l4.weight)

    def forward(self, batch_pos, batch_vel, goal, smart=False):
        batch_pos = batch_pos.clone()  # 新 tensor，独立内存 很关键！
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
        a2 = self.max_action * torch.tanh(self.l4(s))  # [-max, max]

        head_goal = goal[:, 0, :].clone()
        head_goal[:, 2] += L
        a_heur = (head_goal - batch_pos[:, 0, :]) * 5.0
        a2 = torch.clamp(a_heur + a2, -self.max_action, self.max_action)

        if torch.isnan(a2).any():
            print("Find nan in nn action!")

        # goal_head = goal.clone()
        # goal_head[:, 2] = goal_head[:, 2] + 0.4
        # err = goal_head - batch_pos[:, 0]  # 误差
        # a = 5.0 * err  # 线性关系
        # a = torch.clamp(a, min=-max_action, max=max_action)

        return a2


def data_augmentation(position_data, velocity_data, device, N):
    shifts = torch.tensor(np.arange(-1.0, 1.1, 0.1), device=device)  # -1.0 to 1.0 with 0.1 intervals
    position_data = position_data.unsqueeze(0).repeat(len(shifts), 1, 1, 1)
    velocity_data = velocity_data.unsqueeze(0).repeat(len(shifts), 1, 1, 1)
    position_data[:, :, :, 0] += shifts.view(-1, 1, 1)
    position_data = position_data.reshape(-1, position_data.shape[2], position_data.shape[3])
    velocity_data = velocity_data.reshape(-1, velocity_data.shape[2], velocity_data.shape[3])

    num_angles = 20
    angles = torch.arange(num_angles) * (2 * torch.pi / num_angles)
    # 构造 10 个 (3x3) 的旋转矩阵，shape = (10, 3, 3)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    R = torch.stack([
        torch.stack([cos_a, -sin_a, zeros], dim=1),
        torch.stack([sin_a, cos_a, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1),
    ], dim=1)  # shape (10, 3, 3)

    # pos shape: (batch, N, 3)
    # expand pos -> (num_angles, batch, N, 3)
    pos_exp = position_data.unsqueeze(0).expand(num_angles, -1, -1, -1)
    vel_exp = velocity_data.unsqueeze(0).expand(num_angles, -1, -1, -1)

    # 旋转矩阵也扩展到和 pos 匹配：R -> (num_angles, 1, 3, 3)
    R_exp = R[:, None, :, :].to(device)

    # 旋转操作：[..., 3] @ [3, 3] -> [num_angles, batch, N, 3]
    pos_aug = torch.matmul(pos_exp, R_exp)
    vel_aug = torch.matmul(vel_exp, R_exp)

    # reshape 合并 batch：shape (batch * num_angles, N, 3)
    pos_aug = pos_aug.reshape(-1, N+1, 3)
    vel_aug = vel_aug.reshape(-1, N+1, 3)

    return pos_aug, vel_aug


def smart_data_augmentation(position_data, velocity_data, device, N):
    num_angles = 20
    angles = torch.arange(num_angles) * (2 * torch.pi / num_angles)
    # 构造 10 个 (3x3) 的旋转矩阵，shape = (10, 3, 3)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    R = torch.stack([
        torch.stack([cos_a, -sin_a, zeros], dim=1),
        torch.stack([sin_a, cos_a, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1),
    ], dim=1)  # shape (10, 3, 3)

    # pos shape: (batch, N, 3)
    # expand pos -> (num_angles, batch, N, 3)
    pos_exp = position_data.unsqueeze(0).expand(num_angles, -1, -1, -1)
    vel_exp = velocity_data.unsqueeze(0).expand(num_angles, -1, -1, -1)

    # 旋转矩阵也扩展到和 pos 匹配：R -> (num_angles, 1, 3, 3)
    R_exp = R[:, None, :, :].to(device)

    # 旋转操作：[..., 3] @ [3, 3] -> [num_angles, batch, N, 3]
    pos_aug = torch.matmul(pos_exp, R_exp)
    vel_aug = torch.matmul(vel_exp, R_exp)

    # reshape 合并 batch：shape (batch * num_angles, N, 3)
    pos_aug = pos_aug.reshape(-1, N+1, 3)
    vel_aug = vel_aug.reshape(-1, N+1, 3)

    return pos_aug, vel_aug


def data_split(position_data, velocity_data, ratio=0.8):
    # Step 1: 获取总样本数
    num_samples = position_data.shape[0]

    # Step 2: 随机打乱索引
    indices = torch.randperm(num_samples)

    # Step 3: 划分 8:2
    split = int(ratio * num_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Step 4: 使用索引选择样本
    pos_train = position_data[train_idx]
    vel_train = velocity_data[train_idx]
    pos_val = position_data[val_idx]
    vel_val = velocity_data[val_idx]

    return pos_train, vel_train, pos_val, vel_val


def make_traj_segments(traj_list=[], seg_len=50, step=1):
    """
    traj_list: [traj1, traj2, ...]，每个 traj 形状是 (T, 3)
    seg_len:   每个小段的长度，比如 30
    step:      滑动窗口步长，比如 5 / 10 / 30（30 就是不重叠）

    返回：
        segments: (batch, seg_len, 3)
    """
    segments = []

    for traj in traj_list:
        # 保证是 2D: (T, 3)
        assert traj.dim() == 2 and traj.shape[1] == 3, \
            f"traj shape must be (T, 3), got {traj.shape}"

        T = traj.shape[0]
        if T < seg_len:
            # 太短了，没法切一个完整窗口，直接跳过或你也可以选择 pad
            continue

        # unfold 在 dim=0（时间轴）上做滑动窗口
        windows = traj.unfold(0, seg_len, step)  # (num_windows, 3, seg_len)
        windows = windows.permute(0, 2, 1)  # (num_windows, seg_len, 3)
        segments.append(windows)

    if len(segments) == 0:
        raise ValueError("No trajectory is long enough for the given seg_len.")

    segments = torch.cat(segments, dim=0)  # (batch, seg_len, 3)

    # 平移，使得每段起点 = (0,0,0)
    starts = segments[:, 0:1, :]      # 取每段第一个点
    segments = segments - starts      # 自动 broadcast 到整个段

    return segments


def random_rotation_matrix(device=None, dtype=torch.float32):
    """
    生成一个绕 z 轴的随机旋转矩阵 (3,3)
    """
    device = device or torch.device("cpu")

    theta = 2 * math.pi * torch.rand(1, device=device, dtype=dtype)

    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.tensor([
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.]
    ], device=device, dtype=dtype)

    return R  # (3,3)


def visualize_trajs_3d(trajs, title="Trajectories", figsize=(6, 6)):
    """
    trajs: (B, T, 3)  或 (T, 3)
           B 条轨迹段，每段长度 T

    自动支持单条轨迹 (T,3)
    """
    if trajs.dim() == 2:
        trajs = trajs.unsqueeze(0)  # (1, T, 3)

    B, T, _ = trajs.shape

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for b in range(B):
        xs = trajs[b, :, 0].cpu().numpy()
        ys = trajs[b, :, 1].cpu().numpy()
        zs = trajs[b, :, 2].cpu().numpy()

        ax.plot(xs, ys, zs, label=f"traj {b}", linewidth=2)
        ax.scatter(xs[0], ys[0], zs[0], s=50, c='red')  # 起点
        ax.scatter(xs[-1], ys[-1], zs[-1], s=30, c='green')  # 终点

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.tight_layout()
    plt.show()


def sample_targets_traj(traj_seg_bag, batch_pos, noise_std=0.0, plot=False):
    """
    traj_seg_bag: (M, T, 3)
    batch_pos:    (B, N+1, 3)
    返回:
        goal_traj: (B, T, 3)
    """
    device = batch_pos.device
    dtype = batch_pos.dtype

    assert traj_seg_bag.dim() == 3 and traj_seg_bag.shape[-1] == 3, \
        f"traj_seg_bag shape must be (M, T, 3), got {traj_seg_bag.shape}"

    M, T, _ = traj_seg_bag.shape
    B = batch_pos.shape[0]
    if M == 0:
        raise ValueError("traj_seg_bag is empty.")

    traj_seg_bag = traj_seg_bag.to(device=device, dtype=dtype)

    # 1) 随机挑选 B 条轨迹段: (B, T, 3)
    idx = torch.randint(low=0, high=M, size=(B,), device=device)
    segs = traj_seg_bag[idx]   # (B, T, 3)

    # 2) 绕 z 轴随机旋转
    # 生成 B 个角度: (B, 1, 1)
    theta = 2 * math.pi * torch.rand(B, 1, 1, device=device, dtype=dtype)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x = segs[..., 0:1]  # (B,T,1)
    y = segs[..., 1:2]  # (B,T,1)
    z = segs[..., 2:3]  # (B,T,1)

    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    z_rot = z  # z 不变

    segs_rot = torch.cat([x_rot, y_rot, z_rot], dim=-1)  # (B,T,3)

    # 3) 对齐到每条 rope 的 tip 附近
    tip = batch_pos[:, -1:, :]   # (B, 1, 3)

    # 每条轨迹起点: (B, 1, 3)
    start = segs_rot[:, 0:1, :]

    translation = tip - start    # (B,1,3)
    if noise_std > 0.0:
        translation = translation + noise_std * torch.randn_like(translation)

    goal_traj = segs_rot + translation  # (B,T,3)

    # 4) 可视化（debug 用）
    if plot:
        # 只画前 min(B,50) 条
        visualize_trajs_3d(goal_traj[:min(B, 50)], title="Sampled Goal Trajectories")

    return goal_traj


# === Parameters ===
N = 20  # Number of segment
L = 0.05 * N  # Total length (m)
mass = (0.0025 * 40 / N) * L  # Mass per segment (kg)
k = 0.46  # Spring stiffness
damping = [0.2] * N   # Damping coefficient
k_bend = [0.0006712] * (N-1)  # Bending stiffness
damping_bend = [0.000401] * (N-1)  # Bending damping coefficient
air_drag = 0.2206  # Drag coefficient
g = 10.07  # Gravitational acceleration
dt = 0.001  # Time step (s)
T = 10.0  # Simulation duration (s)
total_steps = int(T / dt)
record_interval = 10

# === Setup device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Comment this out
# device = torch.device('cpu')  # Force CPU
print(f"Using device: {device}")

smart = True

# === Create model instance ===
model = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, air_drag=air_drag, g=g, dt=dt, device=device, mode='ctr')

# === sampled data processing ===
sampled_data_all = []
# sampled_data = np.load('../../data/data_for_controller_training.npy').astype(np.float32)  # (sample_num, 1+node_num*6+2)
# sampled_data_all.append(np.concatenate([sampled_data[:500], sampled_data[500::1]]))

sampled_data = np.load(REPO_ROOT / 'data' / 'fixed_tip_pos_low.npy').astype(np.float32)
sampled_data_all.append(sampled_data[::10])

sampled_data_all.append(np.load(REPO_ROOT / 'data' / 'DAgger_01.npy').astype(np.float32))


sampled_data = np.vstack(sampled_data_all)

position_data, velocity_data, _ = mj_data_to_my_data(N, sampled_data, device)
pos_init = position_data[0:1]
vel_init = velocity_data[0:1]

# === Data augmentation ===
if smart:
    position_data, velocity_data = smart_data_augmentation(position_data, velocity_data, device, N)
else:
    position_data, velocity_data = data_augmentation(position_data, velocity_data, device, N)

position_data = torch.zeros((5000, N+1, 3), device=device)
position_data[:, :, 2] = torch.linspace(0.2 + L, 0.2, steps=N+1)
velocity_data = torch.zeros((5000, N+1, 3), device=device)

# position_data = torch.vstack((position_data, position_data2))
# velocity_data = torch.vstack((velocity_data, velocity_data2))

# === Data split for training and validation ===
pos_train, vel_train, pos_val, vel_val = data_split(position_data, velocity_data, ratio=0.8)

# === 创建数据集和数据加载器 ===
ctr_period = 10
pre_length = 75
init_val_seg_len = 100  # only used for initial validation; training uses curriculum stages
learning_rate = 0.0001
mode = 'vel'
max_action = 1.5
# mode = 'acc'
# max_action = 10.0

batch_size = 250
dataset = RopeDataset(pos_train, vel_train)  # Remove total_samples parameter to use all data
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # 每个 epoch 都会打乱数据
    num_workers=0,  # 可以根据需要调整工作进程数
    drop_last=True  # 丢弃最后一个不完整的 batch
)
dataset = RopeDataset(pos_val, vel_val)  # Remove total_samples parameter to use all data
dataloader_val = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # 每个 epoch 都会打乱数据
    num_workers=0,  # 可以根据需要调整工作进程数
    drop_last=True  # 丢弃最后一个不完整的 batch
)


# Remove unused sample data
del position_data, velocity_data, pos_train, vel_train, pos_val, vel_val, dataset


controller = Controller(input_dim=(N+1)*3*2+3*pre_length, action_dim=3, hidden_width=512, max_action=max_action).to(device)
# # L=0.1
model_name = SCRIPT_DIR / 'trained_tracking_controller_cp10_pre75_vel_ma1.5_N20_L1.0_ctr_smart_right.pt'
controller.load_state_dict(torch.load(model_name))
# # L=0.2
optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=0e-5)

# === Curriculum Learning ===
# Each traj_seg_length must be > pre_length so the look-ahead window is not pure padding.
# With pre_length=75: stages start at 80 and grow to 400.
curriculum = [
    {'traj_seg_length': 80,  'epochs': 0},
    {'traj_seg_length': 150, 'epochs': 0},
    {'traj_seg_length': 250, 'epochs': 0},
    {'traj_seg_length': 400, 'epochs': 3},
]

# with torch.no_grad():
#     positions_history = model.simulate_with_controller(pos_init, vel_init, controller, steps=10000, record_interval=10,
#                                                        mode=mode)
#     plot_animation_3d(positions_history, 0.001, 10, L, batch_idx=0)


alpha = 0.1
gamma = 1.0  # 越接近1，越平；越小，越重视晚期（末段）
tbptt_k = 400  # TBPTT: 每 K 个控制步截断梯度链

train_samples = len(dataloader.dataset)
val_samples = len(dataloader_val.dataset)
num_train_batches = len(dataloader)
num_val_batches = len(dataloader_val)
total_curriculum_epochs = sum(stage['epochs'] for stage in curriculum)
curriculum_summary = ' -> '.join([f"seg{s['traj_seg_length']}({s['epochs']}ep)" for s in curriculum])

print("=" * 90)
print(f"[Setup] device={device} train_samples={train_samples} val_samples={val_samples} train_batches={num_train_batches} val_batches={num_val_batches}")
print(f"[Setup] ctr_period={ctr_period} pre_length={pre_length} init_val_seg_len={init_val_seg_len} mode={mode} max_action={max_action}")
print(f"[Setup] alpha={alpha} gamma={gamma} tbptt_k={tbptt_k} batch_size={batch_size}")
print(f"[Setup] curriculum ({total_curriculum_epochs} epochs total): {curriculum_summary}")
print("=" * 90)

traj_list = []

# === sin / egg / eight: 4s / 5s / 6s × 3 shapes = 9 trajectories ===
for total_time in [4.0, 5.0, 6.0]:
    total_horizon_t = int(1000 / ctr_period * total_time)
    points = half_dense_then_uniform(N=total_horizon_t + 1, ratio=0.3, sharpness=2.0, mode='exp', interval=(0.0, 2.0), plot=False)
    traj_list.append(sin_traj(points, width=0.45*2.0, plot=False, device=device))
    traj_list.append(egg_traj(points, scale_x=0.38*2.0, scale_y=0.52*2.0, plot=False, device=device))
    traj_list.append(eight_traj(points, scale_x=0.45*2.0, scale_y=0.65*2.0, z0=0.2, loops=1, plot=False, device=device))

# === drawn trajectories: 12s / 15s / 18s × 3 shapes = 9 trajectories ===
for total_time in [12.0, 15.0, 18.0]:
    total_horizon_t = int(1000 / ctr_period * total_time)
    for drawn_name in ['SpongeBob', 'flower', 'PatrickStar']:
        traj_list.append(build_goal_traj_from_drawn(
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
            interval=(0.0, 2.0)
        ))

# Initial validation (seg_len=init_val_seg_len, before any training)
traj_seg_bag = make_traj_segments(traj_list=traj_list, seg_len=init_val_seg_len, step=1)

positions_history = model.simulate_with_controller_for_path_tracking2(controller, N=N, Seg_L=L/N, path=traj_list[0], record_interval=1, ctr_period=ctr_period, mode=mode, alpha=alpha, smart=smart, pre_length=pre_length)
# plot_animation_3d_for_path_tracking(positions_history, traj_list[0], dt, record_interval=10, L=L, batch_idx=0)

loss_val_his = []
for _, batch_pos, batch_vel in dataloader_val:
    with torch.no_grad():
        track_error = 0.0
        batch_goal_traj = sample_targets_traj(traj_seg_bag, batch_pos, noise_std=0.0)
        filter_ctl = torch.zeros((batch_size, 3), device=device)
        for j in range(1, init_val_seg_len):
            if j+pre_length <= init_val_seg_len:
                curr_goal = batch_goal_traj[:, j:j+pre_length]
            else:
                remain = init_val_seg_len - j
                last_goal = batch_goal_traj[:, -1:].repeat(1, pre_length - remain, 1)
                curr_goal = torch.cat([batch_goal_traj[:, j:], last_goal], dim=1)
            batch_ctl = controller(batch_pos, batch_vel, curr_goal, smart=smart)
            filter_ctl = alpha*batch_ctl + (1-alpha)*filter_ctl
            batch_pos, batch_vel = model.simulation_for_ctr(batch_pos, batch_vel, filter_ctl, ctr_period=ctr_period, mode=mode)
            curr_goal_point = curr_goal[:, 0, :]
            step_err = ((batch_pos[:, -1, :] - curr_goal_point) ** 2).sum(dim=-1).mean() * 100
            w = gamma ** (init_val_seg_len - 1 - j)
            track_error += w * step_err
        loss_val_his.append((track_error / init_val_seg_len * 15).item())
print(f"[Initial Val] seg_len={init_val_seg_len} mean_loss={np.mean(loss_val_his):.4f}")


domain_randomization = False
model.dr = domain_randomization

for stage_idx, stage in enumerate(curriculum):
    traj_seg_length = stage['traj_seg_length']
    epochs = stage['epochs']
    stage_epoch_offset = sum(item['epochs'] for item in curriculum[:stage_idx])

    seg_step = max(1, traj_seg_length // 20)  # adaptive step: keep bag size manageable
    traj_seg_bag = make_traj_segments(traj_list=traj_list, seg_len=traj_seg_length, step=seg_step)

    # LR: linearly from 5e-4 (short segs) down to 5e-6 (long segs), clipped to [5e-6, 5e-4]
    t = min(max((traj_seg_length - 10) / (400 - 10), 0.0), 1.0)
    lr = 5e-5 * (1 - t) + 5e-7 * t
    for g in optimizer.param_groups:
        g['lr'] = lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    print()
    print(f"[Stage {stage_idx+1}/{len(curriculum)} Start] seg_len={traj_seg_length} step={seg_step} epochs={epochs} lr={lr:.2e} global_epoch_range={stage_epoch_offset + 1}-{stage_epoch_offset + epochs}/{total_curriculum_epochs}")

    for i in range(epochs):
        loss_his = []
        iteration = 0
        for _, batch_pos, batch_vel in dataloader:
            iteration += 1
            control_penalty = 0
            track_error = 0.0

            if domain_randomization:
                model.k_tensor = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
                model.dx_tensor = (1 * (torch.randn(batch_size, 1, 1).clamp(-2, 2)/2) * L / N / 20 + L/N).to(device)
                model.mass_tensor = (1 * (torch.randn(batch_size, 1).clamp(-2, 2)/2) * mass / 4 + mass).to(device)
                model.k_bend_tensor = (1 * (torch.randn(batch_size, 1).clamp(-2, 2)/2) * k_bend[0] / 2 + k_bend[0]).to(device)
                model.damping_bend_tensor = (1 * (torch.randn(batch_size, 1, 1).clamp(-2, 2) / 2) * damping_bend[0] / 2 + damping_bend[0]).to(device)

            batch_goal_traj = sample_targets_traj(traj_seg_bag, batch_pos, noise_std=0.0)
            filter_ctl = torch.zeros((batch_size, 3), device=device)

            for j in range(1, traj_seg_length):
                # TBPTT: truncate gradient every tbptt_k steps
                if j % tbptt_k == 0:
                    batch_pos = batch_pos.detach()
                    batch_vel = batch_vel.detach()
                    filter_ctl = filter_ctl.detach()

                if j+pre_length <= traj_seg_length:
                    curr_goal = batch_goal_traj[:, j:j+pre_length]
                else:
                    remain = traj_seg_length - j
                    last_goal = batch_goal_traj[:, -1:].repeat(1, pre_length - remain, 1)
                    curr_goal = torch.cat([batch_goal_traj[:, j:], last_goal], dim=1)

                batch_ctl = controller(batch_pos, batch_vel, curr_goal, smart=smart)
                control_penalty += torch.mean(batch_ctl ** 2)
                filter_ctl = alpha * batch_ctl + (1 - alpha) * filter_ctl
                batch_pos, batch_vel = model.simulation_for_ctr(batch_pos, batch_vel, filter_ctl, ctr_period=ctr_period, mode=mode)

                curr_goal_point = curr_goal[:, 0, :]
                step_err = ((batch_pos[:, -1, :] - curr_goal_point) ** 2).sum(dim=-1).mean() * 100
                w = gamma ** (traj_seg_length - 1 - j)
                track_error += w * step_err

                if torch.isnan(batch_vel).any():
                    print(f"[WARN][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs}][Batch {iteration}/{num_train_batches}] NaN detected in batch_vel")
                if not torch.isfinite(batch_pos).all() or not torch.isfinite(batch_vel).all():
                    print(f"[WARN][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs}][Batch {iteration}/{num_train_batches}] non-finite pos/vel in rollout")
                    break

            loss = track_error / traj_seg_length * 15 + 0.00 * control_penalty
            loss_his.append(loss.item())

            if not torch.isfinite(loss):
                print(f"[WARN][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs}][Batch {iteration}/{num_train_batches}] non-finite loss={loss.item()}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            for name, p in controller.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print(f"[WARN][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs}][Batch {iteration}/{num_train_batches}] non-finite grad in {name}")
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=0.1)
            optimizer.step()

        # validation
        loss_val_his = []
        if domain_randomization:
            model.register_buffer('dx_tensor', torch.tensor(L / N, dtype=torch.float32))
            model.register_buffer('mass_tensor', torch.tensor(mass, dtype=torch.float32))
            model.register_buffer('k_bend_tensor', torch.tensor(k_bend, dtype=torch.float32))
            model.register_buffer('damping_bend_tensor', torch.tensor(damping_bend, dtype=torch.float32).unsqueeze(1))
            model.to(model.device)

        for _, batch_pos, batch_vel in dataloader_val:
            with torch.no_grad():
                track_error = 0.0
                batch_goal_traj = sample_targets_traj(traj_seg_bag, batch_pos, noise_std=0.0)
                filter_ctl = torch.zeros((batch_size, 3), device=device)
                for j in range(1, traj_seg_length):
                    if j + pre_length <= traj_seg_length:
                        curr_goal = batch_goal_traj[:, j:j + pre_length]
                    else:
                        remain = traj_seg_length - j
                        last_goal = batch_goal_traj[:, -1:].repeat(1, pre_length - remain, 1)
                        curr_goal = torch.cat([batch_goal_traj[:, j:], last_goal], dim=1)
                    batch_ctl = controller(batch_pos, batch_vel, curr_goal, smart=smart)
                    filter_ctl = alpha * batch_ctl + (1 - alpha) * filter_ctl
                    batch_pos, batch_vel = model.simulation_for_ctr(batch_pos, batch_vel, filter_ctl, ctr_period=ctr_period, mode=mode)
                    curr_goal_point = curr_goal[:, 0, :]
                    step_err = ((batch_pos[:, -1, :] - curr_goal_point) ** 2).sum(dim=-1).mean() * 100
                    w = gamma ** (traj_seg_length - 1 - j)
                    track_error += w * step_err
                loss_val_his.append((track_error / traj_seg_length * 15).item())

        global_epoch = stage_epoch_offset + i + 1
        mean_train_loss = np.mean(loss_his) if loss_his else float('nan')
        mean_val_loss = np.mean(loss_val_his) if loss_val_his else float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch End][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs} | Global {global_epoch}/{total_curriculum_epochs}] train_loss={mean_train_loss:.4f} val_loss={mean_val_loss:.4f} lr={current_lr:.2e}")
        scheduler.step()

        # if (i+1) % 20 == 0:
        #     print(f"[Visualize][Stage {stage_idx+1}/{len(curriculum)}][Epoch {i+1}/{epochs}] rendering sampled and reference rollouts")
        #     with torch.no_grad():
        #         for kkk in range(3):
        #             rand = random.randint(0, len(batch_goal_traj) - 1)
        #             positions_history = model.simulate_with_controller_for_path_tracking2(controller, N=N, Seg_L=L / N,
        #                                                                                   path=batch_goal_traj[rand], record_interval=1,
        #                                                                                   ctr_period=ctr_period, mode=mode,
        #                                                                                   alpha=alpha, smart=smart, path_fitting=True,
        #                                                                                   pre_length=pre_length)
        #             plot_animation_3d_for_path_tracking(positions_history, batch_goal_traj[rand], dt, record_interval=1, L=L * 2, batch_idx=0)
        #
        #         for traj in traj_list:
        #             positions_history1 = model.simulate_with_controller_for_path_tracking2(controller, N=N, Seg_L=L / N,
        #                                                                                    path=traj, record_interval=1,
        #                                                                                    ctr_period=ctr_period, mode=mode,
        #                                                                                    alpha=alpha, smart=smart,
        #                                                                                    path_fitting=True,
        #                                                                                    pre_length=pre_length)
        #             plot_animation_3d_for_path_tracking(positions_history1, traj, dt, record_interval=10, L=L * 2, batch_idx=0)

    # Save checkpoint at end of each curriculum stage
    ckpt_name = SCRIPT_DIR / f'ckpt_stage{stage_idx+1}_seg{traj_seg_length}.pt'
    torch.save(controller.state_dict(), ckpt_name)
    print(f"[Stage {stage_idx+1}/{len(curriculum)} End] checkpoint={ckpt_name}")

# Save trained model
print("[Save] saving final model...")
model_name = SCRIPT_DIR / f'trained_tracking_controller_cp{ctr_period}_pre{pre_length}_{mode}_ma{max_action}_N{N}_L{L}_ctr_smart_right.pt'
torch.save(controller.state_dict(), model_name)
print(f"[Save] final model saved: {model_name}")





