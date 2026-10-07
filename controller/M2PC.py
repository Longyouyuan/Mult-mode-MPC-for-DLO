import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import *
from common.rope_warp_4_baserope import WarpRope, N, P
import warnings

# ============================================================
# Multi-Modal Planner
# ============================================================


class Planner:
    """
    多模态采样 MPC（pairwise-diversity + Greedy）
    - n_sample: 总采样数
    - m_modes: 模态数 m，要求 n_sample % m == 0
    - good pool: top_k_good（只用这个）
    - 可行性：假设都可行（不筛 feasible）
    - 特征：只用末端轨迹（不标准化）
    - 更新：所有模态都 receding-horizon shift（满足你“次模态也更新”的要求）
    """

    def __init__(self, rope: WarpRope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve,
                 noise_scale, action_dim, limits=None, device='cpu', mode='acc',
                 m_modes=1, top_k_good=200, beta=1.0, wJ=1.0, standard_m2pc=True,
                 prior_weight=1.0):

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
        self.mode = mode
        self.standard_m2pc = standard_m2pc
        self.device = device

        # 多模态
        self.m = int(m_modes)
        assert self.n_sample % self.m == 0, "n_sample % m_modes must == 0"
        self.Nk = self.n_sample // self.m

        # good pool & diversity
        self.top_k_good = int(top_k_good)
        self.beta = float(beta)
        self.wJ = float(wJ)
        self.prior_weight = float(prior_weight)
        if self.prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative")
        if self.noise_scale <= 0.0 and self.prior_weight > 0.0:
            raise ValueError("noise_scale must be positive when prior_weight > 0")

        # batch = m 个 seeds + n_sample 个扰动样本
        self.batch = self.m + self.n_sample

        # 多模态 action seeds: (m, H, 3)
        self.seeds = torch.zeros((self.m, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)
        self.J_star = torch.zeros(self.m, device=self.device, dtype=torch.float32)

        # buffers
        self._noise = torch.empty((self.n_sample, self.horizon, self.action_dim),
                                  device=self.device, dtype=torch.float32)
        self._cand = torch.empty((self.batch, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)

        # cache for visualization
        self.last_tip_traj = None  # (B,H,3)
        self.last_cost = None      # (B,)
        self.last_log_prior = None # (B,)
        self.traj_seeds = None # (m,H,3)
        dev_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]
        self._profile_uses_cuda = bool(torch.cuda.is_available() and dev_type == "cuda")
        self.last_profile = None
        self.profile_history = []

    def _sync_profile_timer(self):
        if self._profile_uses_cuda:
            torch.cuda.synchronize()

    def reset_profile(self):
        self.last_profile = None
        self.profile_history.clear()

    def get_profile_history(self):
        return list(self.profile_history)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _pad_goal_to_horizon(self, goal: torch.Tensor) -> torch.Tensor:
        goal = goal.to(self.device, dtype=torch.float32).contiguous()
        T = goal.shape[0]
        if T == self.horizon:
            return goal
        if T <= 0:
            raise ValueError("goal length must be >= 1")
        last = goal[-1:, :].expand(self.horizon - T, -1)
        return torch.cat([goal, last], dim=0)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _build_candidates_full_horizon(self):
        """
        cand:
          [0:m]        = seeds
          [m: m+Ns]    = seeds[k] + noise (each mode has Nk samples)
        """
        # seeds
        self._cand[:self.m].copy_(self.seeds)

        # noise
        self._noise.normal_(0.0, self.noise_scale)

        # noisy samples per mode
        for k in range(self.m):
            s = k * self.Nk
            e = (k + 1) * self.Nk
            self._cand[self.m + s:self.m + e].copy_(self.seeds[k].unsqueeze(0))
            self._cand[self.m + s:self.m + e].add_(self._noise[s:e])

        # clamp
        if self.limits is not None:
            low, high = self.limits
            self._cand.clamp_(float(low.item()), float(high.item()))

        return self._cand

    @staticmethod
    def _minmax_norm(v, eps=1e-8):
        vmin = v.min()
        vmax = v.max()
        if (vmax - vmin) < 1e-12:
            return torch.zeros_like(v)
        return (v - vmin) / (vmax - vmin + eps)

    @torch.no_grad()
    def _gaussian_mixture_log_prior(self, candidates):
        """Log density under the equally weighted Gaussian warm-start mixture.

        BM2PC uses only a few modes, so the direct broadcast distance kernel is
        faster than a small GEMM on CUDA while keeping the temporary tensor tiny.
        """
        flat = candidates.reshape(candidates.shape[0], -1)
        centers = self.seeds.reshape(self.m, -1)
        diff = flat[:, None, :] - centers[None, :, :]
        sq_dist = (diff * diff).sum(dim=2)

        variance = self.noise_scale * self.noise_scale
        log_components = -0.5 * sq_dist / variance
        log_normalizer = -0.5 * flat.shape[1] * math.log(2.0 * math.pi * variance)
        return torch.logsumexp(log_components, dim=1) - math.log(self.m) + log_normalizer

    # ------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, pos, vel, batch_ctr_parameter_full, full=False):
        """
        pos, vel: (1,P,3)
        batch_ctr_parameter_full: (B,H,3)
        return: (B,H,3) end-effector positions
        """
        self.rope.set_state_and_action(pos, vel, batch_ctr_parameter_full)
        traj = self.rope.simulate(steps=self.horizon * self.ctr_period)  # (B,H,P,3)

        if full is False:
            return traj[:, :, -1, :]  # (B,H,3) 只取末端  # careful about here
        else:
            return traj  # (B,H,P,3) 所有

    # @torch.no_grad()
    # def _greedy_select(self, cost_g, tip_traj_g):
    #     """
    #     cost_g: (K,)
    #     tip_traj_g: (K,H,3)
    #     return selected indices in [0..K-1], size=m
    #     """
    #
    #     feat = tip_traj_g.reshape(tip_traj_g.shape[0], -1)
    #     dist2 = torch.cdist(feat, feat, p=2) ** 2  # (K,K)
    #
    #     # init with minimum cost
    #     first = int(0)  # first = int(torch.argmin(cost_g).item())
    #     selected = [first]
    #
    #     div_sum = dist2[:, first].clone()
    #
    #     while len(selected) < self.m:
    #         score = -self.wJ * cost_g + self.beta * div_sum
    #         score[selected] = -1e18
    #
    #         nxt = int(torch.argmax(score).item())
    #         selected.append(nxt)
    #         div_sum += dist2[:, nxt]
    #
    #     return torch.tensor(selected, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=1e5):
        """
        cost_g_org: (K,)
        tip_traj_g_org: (K,H,3)
        return selected indices in [0..K-1], size=m
        """

        # 统计“正常候选”的数量（因为已排序，所以就是从头开始连续的一段）
        valid_n = int((cost_g_org < thr).sum().item())

        # 0个：全碰撞
        if valid_n == 0:
            warnings.warn(
                f"[greedy_select] valid_n=0 (all cost >= {thr:.2e}). "
                f"Fallback: return first {self.m} indices.",
                category=UserWarning
            )
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        # 1-m个：可能不够
        elif valid_n <= self.m:
            return torch.arange(self.m, device=self.device, dtype=torch.long)

        cost_g = cost_g_org[:valid_n]
        tip_traj_g = tip_traj_g_org[:valid_n]
        log_prior_g = None
        if self.prior_weight > 0.0 and log_prior_g_org is not None:
            log_prior_g = log_prior_g_org[:valid_n]

        feat = tip_traj_g.reshape(valid_n, -1)  # (K, H*3)

        # Lazy column computation: only compute distance to each selected point
        # instead of the full (K,K) matrix. Saves ~300x FLOPs for small m.
        first_score = -self.wJ * self._minmax_norm(cost_g)
        if log_prior_g is not None:
            first_score = first_score + self.prior_weight * log_prior_g
        first = int(torch.argmax(first_score).item())
        selected = [first]

        diff = feat - feat[first]              # (K, d)
        div_sum = (diff * diff).sum(dim=1)     # (K,) squared distances to first point

        c_norm = self._minmax_norm(cost_g)
        score = torch.empty(valid_n, device=feat.device, dtype=feat.dtype)
        while len(selected) < self.m:
            d_norm = self._minmax_norm(div_sum)

            score.copy_(-self.wJ * c_norm + self.beta * d_norm)
            if log_prior_g is not None:
                score.add_(self.prior_weight * log_prior_g)
            score[selected] = -1e18

            nxt = int(torch.argmax(score).item())
            selected.append(nxt)
            diff = feat - feat[nxt]
            div_sum = div_sum + (diff * diff).sum(dim=1)

        return torch.tensor(selected, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------
    @torch.no_grad()
    def improve_policy(self, pos, vel, goal, Obs=None):
        self._sync_profile_timer()
        improve_t0 = time.perf_counter()
        goal = self._pad_goal_to_horizon(goal)  # (H,3)
        rollout_time = 0.0
        cost_time = 0.0
        multimodal_time = 0.0

        for _ in range(self.n_improve):
            batch_ctr_parameter = self._build_candidates_full_horizon()  # (B,H,3)
            log_prior = None
            if self.prior_weight > 0.0:
                log_prior = self._gaussian_mixture_log_prior(batch_ctr_parameter)

            if Obs is None:
                self._sync_profile_timer()
                t0 = time.perf_counter()
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter)     # (B,H,3)
                self._sync_profile_timer()
                rollout_time += time.perf_counter() - t0

                self._sync_profile_timer()
                t0 = time.perf_counter()
                cost = self.cost_fn(batch_traj, goal)                        # (B,)
                self._sync_profile_timer()
                cost_time += time.perf_counter() - t0
            else:
                self._sync_profile_timer()
                t0 = time.perf_counter()
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter, full=True)  # (B,H,P，3)
                self._sync_profile_timer()
                rollout_time += time.perf_counter() - t0

                self._sync_profile_timer()
                t0 = time.perf_counter()
                cost = self.cost_fn(batch_traj, goal, Obs=Obs)
                self._sync_profile_timer()
                cost_time += time.perf_counter() - t0
                batch_traj = batch_traj[:, :, -1, :]

            self._sync_profile_timer()
            t0 = time.perf_counter()
            if self.m == 1:
                selection_score = -self.wJ * self._minmax_norm(cost)
                if log_prior is not None:
                    selection_score = selection_score + self.prior_weight * log_prior
                idx = int(torch.argmax(selection_score).item())
                if self.standard_m2pc:
                    self.seeds.copy_(batch_ctr_parameter[idx])
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

                    self.seeds.copy_(new_params)
            else:
                # 选择最佳的 self.top_k_good 个候选人进池子
                sort_idx = torch.argsort(cost)
                good_idx = sort_idx[:self.top_k_good]
                # good_idx = torch.topk(cost, self.top_k_good, largest=False).indices  # 听说这个可能不保证排序结果

                traj_g = batch_traj[good_idx]              # (K,H,3)
                cost_g = cost[good_idx]                    # (K,)
                log_prior_g = log_prior[good_idx] if log_prior is not None else None
                # ctr_g = batch_ctr_parameter[good_idx]      # (K,H,3)

                # greedy select m modes
                sel = self._greedy_select(cost_g, traj_g, log_prior_g)  # (m,)
                # U_star = batch_ctr_parameter[good_idx[sel]]                      # (m,H,3)

                # update seeds
                self.seeds.copy_(batch_ctr_parameter[good_idx[sel]])
            self._sync_profile_timer()
            multimodal_time += time.perf_counter() - t0

        # cache for visualization: show the last sampling batch
        self.last_tip_traj = batch_traj.detach()
        self.last_cost = cost.detach()
        self.last_log_prior = log_prior.detach() if log_prior is not None else None
        if self.m == 1:
            self.traj_seeds = batch_traj[idx:idx+1]
        else:
            self.traj_seeds = traj_g[sel]
            J_star = cost_g[sel]  # (m,)
            self.J_star.copy_(J_star)

        self._sync_profile_timer()
        improve_time = time.perf_counter() - improve_t0
        profile = {
            "improve": improve_time,
            "rollout": rollout_time,
            "cost": cost_time,
            "multimodal": multimodal_time,
            "overhead": improve_time - rollout_time - cost_time - multimodal_time,
        }
        self.last_profile = profile
        self.profile_history.append(profile)

    # ------------------------------------------------------------
    @torch.no_grad()
    def update_policy(self):
        """
        你要求：次模态也更新（跟主模态一样）
        => 所有 seeds 都做 receding-horizon shift
        """
        self.seeds[:, :-1, :].copy_(self.seeds[:, 1:, :].clone())
        self.seeds[:, -1:, :].copy_(self.seeds[:, -2:-1, :])

    # ------------------------------------------------------------
    def get_action(self, rule='greedy'):
        # 执行本轮选中的“best mode”的一个动作

        if self.m == 1:
            return self.seeds[0, 0]

        if rule == 'greedy':
            # 选 cost 最小的模态
            best = int(torch.argmin(self.J_star).item())

        elif rule == 'sample':
            # Boltzmann sampling: p ∝ exp(-J / temp)
            j = self.J_star - self.J_star.min()  # 提高数值稳定性
            temp = 1.0 * self.J_star.std() + 1e-6  # temp过小-->greedy / temp过大-->random
            p = torch.softmax(-j / temp, dim=0)
            best = int(torch.multinomial(p, 1).item())

        elif rule == 'random':
            # 均匀随机选一个模态
            best = int(torch.randint(self.m, (1,), device=self.device).item())

        else:
            raise ValueError(f"Unknown rule: {rule}")

        return self.seeds[best, 0]

    # ------------------------------------------------------------
    @torch.no_grad()
    def get_cand(self, k_cand=100, rule='random'):
        """
        返回候选轨迹用于可视化

        Returns:
            k_tip_traj: (K,H,3)
            k_cost:     (K,)
            traj_seeds: (m,H,3) or (1,H,3)
        """
        if self.last_tip_traj is None or self.last_cost is None:
            return None, None, None

        B = self.last_cost.shape[0]
        K = min(int(k_cand), B)

        if rule == 'random':
            k_idx = torch.randperm(B, device=self.last_cost.device)[:K]
        elif rule == 'best':
            k_idx = torch.topk(self.last_cost, K, largest=False).indices
        else:
            raise ValueError(f"Unknown rule: {rule}")

        k_tip_traj = self.last_tip_traj[k_idx]
        k_cost = self.last_cost[k_idx]

        return k_tip_traj, k_cost, self.traj_seeds


@torch.no_grad()
def collide_two_cylinders(pts, c1, c2, radius, half_h, margin=0.0):
    r = radius + margin
    h = half_h + margin
    r2 = r * r

    def _hit(center):
        dx = pts[..., 0] - center[0]
        dy = pts[..., 1] - center[1]
        dz = pts[..., 2] - center[2]
        return ((dx*dx + dy*dy) <= r2) & (dz.abs() <= h)   # (B,H,P)

    hit = _hit(c1) | _hit(c2)
    return hit.any(dim=2).any(dim=1)  # (B,)


def cost_fn(batch_traj, goal_h, Obs=None):
    if Obs is None and batch_traj.dim() == 3:
        err = batch_traj - goal_h.unsqueeze(0)
        return torch.sum(torch.abs(err), dim=(1, 2))
    elif Obs is not None and batch_traj.dim() == 4:
        margin = 0.02
        # collision_cost = float("inf")
        collision_cost = 1e8

        tip = batch_traj[:, :, -1, :]  # (B,H,3)
        err = tip - goal_h.unsqueeze(0)
        w = torch.tensor([1.8, 1.8, 1.0], device=err.device)
        J_track = torch.sum(torch.abs(err) * w, dim=(1, 2))
        # J_track = torch.sum(torch.abs(err), dim=(1, 2))  # (B,)

        Obs_t = torch.as_tensor(Obs, device=tip.device, dtype=tip.dtype)
        r0, h0 = Obs_t[0, 0], Obs_t[0, 1]
        c1, c2 = Obs_t[1, :], Obs_t[2, :]

        hit = collide_two_cylinders(batch_traj, c1, c2, r0, h0, margin=margin)

        # 碰撞淘汰（argmin => +inf）
        # J = torch.where(hit, torch.full_like(J_track, float(collision_cost)), J_track)
        J = torch.where(hit, J_track + collision_cost, J_track)
        return J
    else:
        raise ValueError("Cost_fn doesn't know if there is Obstacle")


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
    T_task = 2.5 * 4.0
    total_steps = int(T_task / dt)
    mode = 'acc'  # 或 'vel'

    ctr_period = 25
    horizon = 30  # 20 doesn't work; 25 can work

    # ===== 多模态参数 =====
    n_sample = 400     # 总采样数
    m_modes = 1        # 模态数（单模态=1）
    assert n_sample % m_modes == 0

    n_improve = 5
    noise_scale = 1.8  # 0.25 looks good for naive case, 1.5更好for naive case？
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)

    # diversity 超参
    top_k_good = 200
    beta = 5000.0
    wJ = 1000.0

    visualization = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === warp_4 rope：rollout（batch = m_modes + n_sample）===
    rope = WarpRope(
        batch_size=m_modes + n_sample,
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
        record_interval=ctr_period,
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
        record_interval=ctr_period,
        ctr_period=ctr_period,
        mode=mode
    )

    # === 目标轨迹（保留原逻辑）===

    # === 1. Generate sin/egg/eight Trajs ===
    points = half_dense_then_uniform(
        N=total_horizon + 1, ratio=0.3, sharpness=2.0,
        mode='exp', interval=(0.0, 2.0), plot=False
    )

    Goal_traj = eight_traj(points, scale_x=0.45*3.0, scale_y=0.65*3.0, z0=0.2, loops=1, plot=False, device=device)
    # Goal_traj = egg_traj(points, scale_x=0.38*2.0, scale_y=0.52*2.0, plot=False, device=device)
    # Goal_traj = sin_traj(points, width=0.45*2.0, plot=False, device=device)
    plot_goal_traj(Goal_traj, T_task)

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
    import warp as wp
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

        # Update policy
        planner.update_policy()

        pos_history.append(pos.clone())
        action_history.append(action.detach().cpu().numpy())
        vel_history.append(vel.clone().detach().cpu().numpy()[0, 0])

    if visualization:
        viz.close()

    print("time: ", torch.tensor(time_record).mean())

    # === 动画/轨迹（保留）===
    pos_history = torch.cat(pos_history, dim=0)
    anim = plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)
    plot_tip_vs_goal_and_error(pos_history, Goal_traj, dt, ctr_period)

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
    plt.show()

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
