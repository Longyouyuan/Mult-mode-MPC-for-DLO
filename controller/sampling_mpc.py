import torch
from common.rope import *
from common.utils import *
import matplotlib.pyplot as plt
import numpy as np


class Policy:
    def __init__(self, action_dim, horizon, dt, limits=None, device='cpu'):
        self.action_dim = action_dim
        self.horizon = horizon
        self.dt = dt
        self.steps = int(horizon / dt)
        self.device = device
        self.parameters = torch.zeros((horizon, action_dim), device=self.device)
        self.limits = limits.to(device)

    def action(self):
        return self.clamp(self.parameters[0])

    def clamp(self, actions):
        if self.limits is None:
            return actions
        low, high = self.limits
        return torch.clamp(actions, low, high)

    def update_policy(self):
        self.parameters = torch.vstack((self.parameters[1:, :], self.parameters[-1:, :]))


class Planner:
    def __init__(self, rope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve, noise_scale, action_dim,
                 limits=None, device='cpu', mode='vel'):
        self.rope = rope
        self.cost_fn = cost_fn
        self.dt = dt
        self.ctr_period = ctr_period
        self.horizon = horizon
        self.n_sample = n_sample
        self.n_improve = n_improve
        self.noise_scale = noise_scale
        self.action_dim = action_dim
        self.limits = limits
        self.policy = Policy(action_dim, horizon, dt, limits, device=device)
        self.mode = mode
        self.device = device

    def rollout(self, pos, vel, batch_ctr_parameter, len):
        pos = pos.repeat(batch_ctr_parameter.shape[0], 1, 1)
        vel = vel.repeat(batch_ctr_parameter.shape[0], 1, 1)
        batch_traj = self.rope.sampling_forward(pos, vel, batch_ctr_parameter, horizion=len,
                                                ctr_period=self.ctr_period, mode=self.mode)
        return batch_traj[:, :, -1]

    def improve_policy(self, pos, vel, len, goal):
        num = 0
        for i in range(self.n_improve):
            noise = torch.randn((self.n_sample, len, self.action_dim), device=self.device) * self.noise_scale
            batch_ctr_parameter = noise + self.policy.parameters[0:len, :].unsqueeze(0)
            batch_ctr_parameter = torch.cat([self.policy.parameters[0:len, :].unsqueeze(0), batch_ctr_parameter], dim=0)
            batch_ctr_parameter = self.policy.clamp(batch_ctr_parameter)
            with torch.no_grad():
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter, len)
            cost = self.cost_fn(batch_traj, goal)
            idx = cost.argmin()
            self.policy.parameters[0:len, :] = batch_ctr_parameter[idx]

            if i == 0:
                min = cost[idx]
                num += 1
            elif cost[idx] < min:
                num += 1
        # print('valid n_improve', num)

    def get_action(self):
        return self.policy.action()


def cost_fn(batch_traj, goal):
    err = batch_traj - goal.unsqueeze(0)
    cost = torch.sum(torch.abs(err), dim=(1, 2))
    return cost


def half_dense_then_uniform(N=100, ratio=0.5, sharpness=2.0, mode='exp', interval=(0.0, 2.0), plot=True):
    """
    非对称采样：前半段密集→稀疏，后半段匀速采样，spacing在0.5处连续。

    Args:
        N (int): 总采样点数
        ratio (float): 非线性密集采样段占比（如0.5表示前一半）
        sharpness (float): 非线性变化控制强度（越大越陡）
        mode (str): 'exp' | 'sqrt' | 'poly'
        interval=(0.0, 2.0): 2m长
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


def plot_goal_traj(goal_traj, T_task, title="Goal Trajectory", color='gray', scatter=True):
    """
    可视化三维目标轨迹。

    Args:
        goal_traj (Tensor): shape=(T, 3) 的三维轨迹点
        title (str): 图标题
        color (str): 线条颜色
        scatter (bool): 是否标出每个点
    """
    # 如果是 torch tensor，先转为 numpy
    if isinstance(goal_traj, torch.Tensor):
        goal_traj = goal_traj.cpu().numpy()

    # 计算轨迹总长度
    diffs = goal_traj[1:] - goal_traj[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = seg_lengths.sum()
    # 在标题中展示总长度
    title = f"{title} (Length: {total_length:.3f} m, Est. Time: {T_task:.1f} s)"

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(goal_traj[:, 0], goal_traj[:, 1], goal_traj[:, 2],
            lw=2, color=color, label='Goal Trajectory')

    if scatter:
        ax.scatter(goal_traj[:, 0], goal_traj[:, 1], goal_traj[:, 2],
                   color=color, s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === Parameters ===
    N = 20  # Number of segments
    L = 1.0  # Total length (m)
    mass = 0.0025 * 40 / N  # Mass per segment (kg)
    k = 0.46  # Spring stiffness
    damping = [0.2] * N  # Damping coefficient
    k_bend = [0.0006712] * (N - 1)  # Bending stiffness
    damping_bend = [0.000401] * (N - 1)  # Bending damping coefficient
    air_drag = 0.2206  # Drag coefficient
    g = 10.07  # Gravitational acceleration
    dt = 0.001  # Time step (s)
    T_task = 2.0  # Task duration (s)
    total_steps = int(T_task / dt)
    mode = 'acc'

    ctr_period = 25  # 40 Hz
    horizon = 30  # Pridiction horizon*ctr_period ms
    n_sample = 1280
    n_improve = 4
    noise_scale = 0.25
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)

    # === Setup device ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Comment this out
    # device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")

    # === Create model instance ===
    rope = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, air_drag=air_drag,
                 g=g, dt=dt, device=device, mode='ctr')

    # === Generate a example trajectory ===
    # # straight line
    # points = torch.linspace(0, 2, steps=total_horizon+1)
    # points = 0.25 * points**2
    # Goal_traj = torch.vstack((torch.zeros(total_horizon+1), points,
    #                           torch.ones(total_horizon+1)*0.2)).T.to(device)

    points = half_dense_then_uniform(N=total_horizon + 1, ratio=0.5, sharpness=2.0,
                                     mode='exp', interval=(0.0, 2.0), plot=True)
    Goal_traj = torch.vstack((torch.sin(4*points)*0.25, points,
                              torch.ones(total_horizon + 1) * 0.2)).T.to(device)
    plot_goal_traj(Goal_traj, T_task)

    planner = Planner(rope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve, noise_scale, action_dim,
                      limits=limits, device=device, mode=mode)

    # sampled_data = np.load('../../data/static_init_with_xyz_drive.npy').astype(np.float32)[0:1]
    # pos, vel, _ = mj_data_to_my_data(N, sampled_data, device)

    pos = torch.zeros((1, 21, 3), device=device)
    pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=21)
    vel = torch.zeros((1, 21, 3), device=device)
    # planner.policy.parameters[:, 1] = torch.ones(horizon) * 0.2

    action_history = []
    pos_history = []
    vel_history = []
    pos_history.append(pos)

    stored_actions = None
    # stored_actions = np.load('action_history.npy')

    if stored_actions is None:
        # warm up
        vel_start = (Goal_traj[1:1 + horizon, :] - Goal_traj[:horizon, :])/(T_task/horizon)
        planner.policy.parameters = 1.0 * vel_start.to(device)
        goal = Goal_traj[1:1 + horizon, :]
        planner.n_improve = n_improve * 100
        planner.improve_policy(pos, vel, goal.shape[0], goal)
        planner.n_improve = n_improve

    for i in range(total_horizon):
        print('i: ', i)
        if stored_actions is None:
            if i < (total_horizon-horizon):
                goal = Goal_traj[i+1:i+1+horizon, :]
                planner.improve_policy(pos, vel, goal.shape[0], goal)
            else:
                goal = Goal_traj[i + 1:, :]
                planner.improve_policy(pos, vel, goal.shape[0], goal)

            # 执行 / 更新policy.parameter
            action = planner.get_action()
        else:
            action = torch.from_numpy(stored_actions[i]).to(device)
        # action = torch.tensor([0.0, 0.5, 0.0], device=device)
        pos, vel = rope.simulation_for_ctr(pos, vel, action, ctr_period=ctr_period, mode=mode)

        planner.policy.update_policy()

        pos_history.append(pos)
        action_history.append(action.cpu().numpy())
        vel_history.append(vel.cpu().numpy()[0, 0])

    # np.save("action_history_with_simple_model.npy", np.array(action_history))

    pos_history = torch.cat(pos_history)
    plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)

    # === Draw action/velocity ===
    action_history_np = np.array(action_history)  # shape: (total_horizon, 3)
    time = np.arange(action_history_np.shape[0]) * ctr_period * dt

    plt.figure(figsize=(10, 6))
    plt.plot(time, action_history_np[:, 0], label='Action X', color='r')
    plt.plot(time, action_history_np[:, 1], label='Action Y', color='g')
    plt.plot(time, action_history_np[:, 2], label='Action Z', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Action Value')
    plt.title('Action XYZ over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 画出vel_history的曲线
    vel_history_np = np.array(vel_history)  # shape: (total_horizon, 3)
    plt.figure(figsize=(10, 6))
    plt.plot(time, vel_history_np[:, 0], label='Velocity X', color='r', linestyle='--')
    plt.plot(time, vel_history_np[:, 1], label='Velocity Y', color='g', linestyle='--')
    plt.plot(time, vel_history_np[:, 2], label='Velocity Z', color='b', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Value')
    plt.title('Velocity XYZ over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ccc = 1

