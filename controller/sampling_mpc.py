import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.utils import *
from common.rope_warp_4 import WarpRope, N, P


class Policy:
    def __init__(self, action_dim, horizon, dt, limits=None, device='cpu'):
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.steps = int(horizon / dt)  # 保持兼容（实际不用）
        self.device = device
        self.parameters = torch.zeros((self.horizon, self.action_dim), device=self.device, dtype=torch.float32)
        self.limits = limits.to(device) if limits is not None else None

    def action(self):
        return self.clamp(self.parameters[0])

    def clamp(self, actions):
        if self.limits is None:
            return actions
        low, high = self.limits
        return torch.clamp(actions, low, high)

    def update_policy(self):
        # in-place receding-horizon shift
        self.parameters[:-1, :].copy_(self.parameters[1:, :].clone())
        self.parameters[-1:, :].copy_(self.parameters[-2:-1, :])


class Planner:
    """
    结构照 Sampling_MPC.py，但更高效：
      - 预分配 noise/candidate，避免 torch.cat
      - 全程只用固定 horizon（无 len_）
      - 如果 goal 不足 horizon：重复最后一个 goal 点 padding 到 horizon
      - WarpRope 只创建一次：batch=n_sample+1、frames=horizon
    """
    def __init__(self, rope: WarpRope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve,
                 noise_scale, action_dim, limits=None, device='cpu', mode='acc'):
        self.rope = rope
        self.cost_fn = cost_fn
        self.dt = float(dt)
        self.ctr_period = int(ctr_period)
        self.horizon = int(horizon)
        self.n_sample = int(n_sample)
        self.n_improve = int(n_improve)
        self.noise_scale = float(noise_scale)
        self.action_dim = int(action_dim)
        self.limits = limits
        self.policy = Policy(action_dim, horizon, dt, limits, device=device)
        self.mode = mode
        self.device = device

        self.batch = self.n_sample + 1  # baseline + noise samples

        # prealloc buffers (GPU)
        self._noise = torch.empty((self.n_sample, self.horizon, self.action_dim),
                                  device=self.device, dtype=torch.float32)
        self._cand = torch.empty((self.batch, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)

        # ---- for visualization cache ----
        self.last_tip_traj = None  # (B, H, 3)
        self.last_cost = None  # (B,)
        self.best_traj = None

    @torch.no_grad()
    def _pad_goal_to_horizon(self, goal: torch.Tensor) -> torch.Tensor:
        """
        goal: (T,3), T can be <= horizon
        return: (horizon,3) by repeating the last point if needed
        """
        goal = goal.to(self.device, dtype=torch.float32).contiguous()
        T = goal.shape[0]
        if T == self.horizon:
            return goal
        if T <= 0:
            raise ValueError("goal length must be >= 1")
        last = goal[-1:, :].expand(self.horizon - T, -1)
        return torch.cat([goal, last], dim=0)

    @torch.no_grad()
    def _build_candidates_full_horizon(self):
        """
        输出 (batch, horizon, action_dim)
        cand[0] = baseline policy
        cand[1:] = policy + noise
        """
        # baseline
        self._cand[0].copy_(self.policy.parameters)

        # noise full horizon
        self._noise.normal_(0.0, self.noise_scale)

        # noisy samples
        self._cand[1:].copy_(self.policy.parameters.unsqueeze(0))
        self._cand[1:].add_(self._noise)

        # clamp in-place
        if self.policy.limits is not None:
            low, high = self.policy.limits
            self._cand.clamp_(float(low.item()), float(high.item()))

        return self._cand

    @torch.no_grad()
    def rollout(self, pos, vel, batch_ctr_parameter_full):
        """
        pos, vel: (1,P,3)
        batch_ctr_parameter_full: (batch, horizon, 3)
        return: (batch, horizon, 3) end-effector positions
        """
        self.rope.set_state_and_action(pos, vel, batch_ctr_parameter_full)
        traj = self.rope.simulate(steps=self.horizon * self.ctr_period)  # (B, horizon, P, 3)
        return traj[:, :, -1, :]  # 只取末端 (B,horizon,3)

    def improve_policy(self, pos, vel, goal):
        # ✅ 无 len_：goal 不足就 padding
        goal = self._pad_goal_to_horizon(goal)  # (horizon,3)

        for _ in range(self.n_improve):
            batch_ctr_parameter = self._build_candidates_full_horizon()
            with torch.no_grad():

                # t0 = time.perf_counter()
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter)  # (B,horizon,3)
                # t1 = time.perf_counter()
                # print("time for rollout:", t1 - t0)

            cost = self.cost_fn(batch_traj, goal)  # (B,)

            if False:
                idx = int(cost.argmin().item())
                self.policy.parameters.copy_(batch_ctr_parameter[idx])
            else:
                # ===== MPPI 风格：对“整个 action 序列”做 softmin 加权平均 =====
                lam = 0.3 * (cost.std() + 1e-6)

                J = cost
                J_min = J.min()
                # w_k ∝ exp(-(J_k - J_min)/λ)
                w = torch.exp(-(J - J_min) / lam)  # (B,)
                w = w / (w.sum() + 1e-8)  # 归一化

                # 直接对控制序列加权平均: (H, action_dim)
                new_params = (w.view(-1, 1, 1) * batch_ctr_parameter).sum(dim=0)

                self.policy.parameters.copy_(new_params)

        # ✅ cache for visualization (save the LAST improve iteration)
        self.last_tip_traj = batch_traj.detach()
        self.last_cost = cost.detach()

    @torch.no_grad()
    def get_cand(self, top_k=100, best_k=1):
        """
        Returns:
          top_tip_traj: (K,H,3)  K=min(top_k,B)
          top_cost:     (K,)
          best_k_traj: (best_k,) indices in the ORIGINAL batch (B,)
        """
        if self.last_tip_traj is None or self.last_cost is None:
            return None, None, None

        B = self.last_cost.shape[0]
        K = min(int(top_k), int(B))
        best_k = min(int(best_k), int(B))

        # sort by cost ascending
        sort_idx = torch.argsort(self.last_cost)
        top_idx = sort_idx[:K]

        top_idx = torch.randperm(B, device=self.last_cost.device)[:K]
        top_tip_traj = self.last_tip_traj[top_idx]  # (K,H,3)
        top_cost = self.last_cost[top_idx]  # (K,)

        # best_k trajectories (for future multi-modal you can pass best_k>1)
        best_indices = sort_idx[:best_k]  # (best_k,)
        best_k_traj = self.last_tip_traj[best_indices]
        best_k_traj = self.last_tip_traj[0:1]

        return top_tip_traj, top_cost, best_k_traj

    def get_action(self):
        return self.policy.action()


def cost_fn(batch_traj, goal_h):
    # batch_traj: (B,horizon,3), goal_h: (horizon,3)
    err = batch_traj - goal_h.unsqueeze(0)
    return torch.sum(torch.abs(err), dim=(1, 2))


if __name__ == "__main__":
    # === Parameters（保持 Sampling_MPC.py 风格）===
    L = 1.0
    mass = 0.0025 * 40 / N
    k = 10000 * 0.46
    damping = 0.2
    bending_k = 0.0006712
    bending_damping = 0.000401
    air_drag = 0.2206 / 1000
    g = 10.07
    dt = 0.001
    T_task = 2.5 * 5.0
    total_steps = int(T_task / dt)
    mode = 'acc'  # 或 'vel'

    ctr_period = 25
    horizon = 30  # 20 doesn't work; 25 can work
    n_sample = 400
    n_improve = 10
    noise_scale = 1.50  # 0.25 looks good for naive case
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)  # 当total_steps不能整除时，会把多余的截断

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === warp_4 rope：rollout（batch = n_sample+1）===
    rope = WarpRope(
        batch_size=n_sample + 1,
        L=L,
        mass=mass,
        k=k,
        damping=damping,
        bending_k=bending_k * 0.0,
        bending_damping=bending_damping,
        air_drag=air_drag,
        g=g,
        dt=dt,
        max_record_steps=horizon * ctr_period,
        record_interval=ctr_period,   # 每个控制 tick 记录一次 => frames=horizon
        ctr_period=ctr_period,
        mode=mode
    )

    # === warp_4 rope：执行（batch=1）===
    rope_exec = WarpRope(
        batch_size=1,
        L=L,
        mass=mass,
        k=k,
        damping=damping,
        bending_k=bending_k,
        bending_damping=bending_damping,
        air_drag=air_drag,
        g=g,
        dt=dt,
        max_record_steps=ctr_period,
        record_interval=ctr_period,   # frames=1
        ctr_period=ctr_period,
        mode=mode
    )

    # === 目标轨迹（保留原逻辑）===

    # # === 1. Generate sin/egg/eight Trajs ===
    # points = half_dense_then_uniform(
    #     N=total_horizon + 1, ratio=0.3, sharpness=2.0,
    #     mode='exp', interval=(0.0, 2.0), plot=False
    # )
    #
    # Goal_traj = eight_traj(points, scale_x=0.45*3.0, scale_y=0.65*3.0, z0=0.2, loops=1, plot=False, device=device)
    # # Goal_traj = egg_traj(points, scale_x=0.38*2.0, scale_y=0.52*2.0, plot=False, device=device)
    # # Goal_traj = sin_traj(points, width=0.45*2.0, plot=False, device=device)
    # plot_goal_traj(Goal_traj, T_task)

    # === 2. Draw Traj ===
    Goal_traj = build_goal_traj_from_drawn(
        drawn_path="../my_trajs/my_draw.npy",
        total_horizon=total_horizon,
        device=device,
        z0=0.2,
        scale_x=3.0,
        scale_y=3.0,
        keep_aspect=False,  # 允许非等比缩放（你说可能不是方形）
        sigma=2.0,
        uniform_M=1000,  # 越大越均匀/越平滑（但太大也没必要）
        ratio=0.3,
        sharpness=2.0,
        interval=(0.0, 2.0)
    )
    plot_goal_traj(Goal_traj, T_task)

    planner = Planner(
        rope, cost_fn, dt, ctr_period, horizon,
        n_sample, n_improve, noise_scale, action_dim,
        limits=limits, device=device, mode=mode
    )

    # viz = LiveMPCVisualizer(goal_traj=Goal_traj, margin=0.05)

    # === Init state（保留原逻辑）===
    pos = torch.zeros((1, P, 3), device=device)
    pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
    vel = torch.zeros((1, P, 3), device=device)

    action_history = []
    pos_history = []
    vel_history = []
    time_record = []
    pos_history.append(pos.clone())

    stored_actions = None
    # stored_actions = np.load('action_history.npy')

    # === warm start（保留原逻辑；goal 不足会自动 padding）===
    if stored_actions is None:
        vel_start = (Goal_traj[1:1 + horizon, :] - Goal_traj[:horizon, :]) / (T_task / horizon)
        planner.policy.parameters.copy_(vel_start.to(device))

        goal = Goal_traj[1:1 + horizon, :]
        planner.n_improve = n_improve * 100
        planner.improve_policy(pos, vel, goal)
        planner.n_improve = n_improve

    # === MPC loop（保留原逻辑）===
    import warp as wp
    for i in range(total_horizon):
        if i % 10 == 0:
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

        # # visulization
        # if i % 1 == 0:  # 每2步刷新一次
        #     topK = 100
        #     bestK = 1  # 以后多模态你改成 3/5 都行
        #     cand_tip_traj, cand_cost, best_k_traj = planner.get_cand(top_k=topK, best_k=bestK)
        #
        #     if cand_tip_traj is not None:
        #         rope_now = pos[0]  # (P,3) 这里假设 pos shape 是 (1,P,3)
        #         viz.update(
        #             rope_pos=rope_now,
        #             cur_i=i,
        #             cand_tip_traj=cand_tip_traj,
        #             cand_cost=cand_cost,
        #             best_k_traj=best_k_traj,
        #             title=f"MPC step {i} | top{topK} candidates"
        #         )

        # ---- 执行 chosen action（batch=1 rope_exec）----
        action_seq = action.view(1, 1, 3)  # (B=1,T=1,3)
        rope_exec.set_state_and_action(pos, vel, action_seq)
        _ = rope_exec.simulate(steps=ctr_period)

        # 更新状态（从 warp 内部 pos/vel 拿出来）
        pos = wp.to_torch(rope_exec.pos)
        vel = wp.to_torch(rope_exec.vel)

        planner.policy.update_policy()

        pos_history.append(pos.clone())
        action_history.append(action.detach().cpu().numpy())
        vel_history.append(vel.clone().detach().cpu().numpy()[0, 0])

    # viz.close()

    print("time: ", torch.tensor(time_record).mean())

    # === 动画/轨迹（保留）===
    pos_history = torch.cat(pos_history, dim=0)
    anim = plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)
    plot_tip_vs_goal_and_error(pos_history, Goal_traj, dt, ctr_period)

    # === action 曲线（保留）===
    action_history_np = np.array(action_history)
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

    # === velocity 曲线（保留）===
    vel_history_np = np.array(vel_history)
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

    # 绘制 pos_history 顶端点轨迹
    tip_pos_history = pos_history[1:, 0, :]  # 假设最后一个点是顶端点，形状: (steps, 3)
    plt.figure(figsize=(10, 6))
    plt.plot(time, tip_pos_history[:, 0].cpu().numpy(), label='Tip X', color='m')
    plt.plot(time, tip_pos_history[:, 1].cpu().numpy(), label='Tip Y', color='c')
    plt.plot(time, tip_pos_history[:, 2].cpu().numpy(), label='Tip Z', color='y')
    plt.xlabel('Time (s)')
    plt.ylabel('Tip Position')
    plt.title('Tip (End Point) Position XYZ over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
