import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.utils import *
from common.rope_warp_4_baserope import WarpRope, N, P
import warnings

# ============================================================
# DBSCAN Baseline Planner
# ============================================================


class Planner:
    """
    PyTorch GPU DBSCAN baseline for sampling MPC.

    和 M2PC.py 保持同一套外部调用方式：
    - improve_policy(pos, vel, goal, Obs=None)
    - update_policy()
    - get_action()
    - get_cand(k_cand=100, rule='random')

    主要区别：
    - 不再使用 m_modes；DBSCAN 不需要预先指定模态数量。
    - batch = 1 个当前 seed + n_sample 个扰动样本。
    - 每轮先按 cost 排序取 top_k_good 个 good candidates。
    - 如果有障碍物，继续过滤掉碰撞 candidates，即 cost >= collision_thr 的样本。
    - 对 good 且无碰撞的末端轨迹做 PyTorch DBSCAN（torch.cdist，GPU-native）。
    - 从 DBSCAN 的非噪声 cluster 中选平均 cost 最低的 cluster，再取该 cluster 内 cost 最低的 candidate 更新 seed。
      如果 DBSCAN 没有找到有效 cluster，则 fallback 到无碰撞 good pool 中 cost 最低的 candidate。
    """

    def __init__(self, rope: WarpRope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve,
                 noise_scale, action_dim, limits=None, device='cpu', mode='acc',
                 top_k_good=200, dbscan_eps=3.0, dbscan_min_samples=5,
                 collision_thr=1e5, normalize_dbscan_feat=True,
                 standard_m2pc=True, **kwargs):
        """
        Args:
            rope: WarpRope rollout model. Its batch_size should be 1 + n_sample.
            cost_fn: same as M2PC.py.
            top_k_good: first keep this many lowest-cost candidates before collision filtering.
            dbscan_eps: DBSCAN radius in feature space. If normalize_dbscan_feat=True, this is on normalized features.
            dbscan_min_samples: minimum samples for a core point.
            collision_thr: candidates with cost >= collision_thr are treated as collision / invalid.
                           This matches the current cost_fn where collision adds 1e8.
            normalize_dbscan_feat: normalize trajectory feature dimensions before DBSCAN.
            standard_m2pc: kept only for interface compatibility; DBSCAN baseline always updates one seed.
            **kwargs: absorbs m_modes/beta/wJ if old scripts accidentally pass them.
        """
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

        self.m = 1
        self.top_k_good = int(top_k_good)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.collision_thr = float(collision_thr)
        self.normalize_dbscan_feat = bool(normalize_dbscan_feat)

        # batch = 1 seed + n_sample noisy candidates.
        self.batch = 1 + self.n_sample

        self.seeds = torch.zeros((1, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)
        self.J_star = torch.zeros(1, device=self.device, dtype=torch.float32)

        self._noise = torch.empty((self.n_sample, self.horizon, self.action_dim),
                                  device=self.device, dtype=torch.float32)
        self._cand = torch.empty((self.batch, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)

        self.last_tip_traj = None
        self.last_cost = None
        self.traj_seeds = None
        self.last_dbscan_labels = None
        self.last_good_idx = None
        self.last_dbscan_scale = None
        self.last_dbscan_core_fraction = None
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

    @torch.no_grad()
    def _build_candidates_full_horizon(self):
        """
        cand:
          [0]          = current seed
          [1:1+Ns]     = seed + noise
        """
        self._cand[0:1].copy_(self.seeds)
        self._noise.normal_(0.0, self.noise_scale)
        self._cand[1:].copy_(self.seeds[0].unsqueeze(0))
        self._cand[1:].add_(self._noise)

        if self.limits is not None:
            low, high = self.limits
            self._cand.clamp_(float(low.item()), float(high.item()))

        return self._cand

    @torch.no_grad()
    def rollout(self, pos, vel, batch_ctr_parameter_full, full=False):
        self.rope.set_state_and_action(pos, vel, batch_ctr_parameter_full)
        traj = self.rope.simulate(steps=self.horizon * self.ctr_period)
        if full is False:
            return traj[:, :, -1, :]
        else:
            return traj

    @torch.no_grad()
    def _make_dbscan_feature(self, tip_traj_g: torch.Tensor) -> torch.Tensor:
        feat = tip_traj_g.reshape(tip_traj_g.shape[0], -1).contiguous()
        if self.normalize_dbscan_feat and feat.shape[0] > 1:
            mean = feat.mean(dim=0, keepdim=True)
            std = feat.std(dim=0, keepdim=True).clamp_min(1e-6)
            feat = (feat - mean) / std
        return feat

    @torch.no_grad()
    def _dbscan_torch(self, feat: torch.Tensor):
        """
        PyTorch / GPU-native DBSCAN.

        Returns:
            labels: (K,), int64 tensor on self.device. -1 means noise.

        Implementation details:
        - torch.cdist builds the eps-neighborhood graph on GPU.
        - Core points are connected components of the core-only graph.
        - Border points are assigned to the nearest neighboring core point.
        - No sklearn / numpy / CPU clustering dependency.
        """
        K = int(feat.shape[0])
        labels = torch.full((K,), -1, device=feat.device, dtype=torch.long)
        if K == 0 or K < self.dbscan_min_samples:
            return labels

        # Full eps-neighborhood graph. For top_k_good ~= 200 this is cheap.
        dist = torch.cdist(feat, feat, p=2)                       # (K,K), GPU if feat is GPU

        # Keep lightweight DBSCAN diagnostics for eps tuning and to reduce the
        # effect of tiny pairwise asymmetries when inspecting cluster scales.
        dist_sym = 0.5 * (dist + dist.transpose(0, 1))
        pairwise_upper = dist_sym.triu(diagonal=1)
        pairwise_upper = pairwise_upper[pairwise_upper > 0]
        if pairwise_upper.numel() > 0:
            self.last_dbscan_scale = pairwise_upper.median()
        else:
            self.last_dbscan_scale = dist.new_tensor(0.0)

        neighbors = dist <= self.dbscan_eps                       # (K,K)
        is_core = neighbors.sum(dim=1) >= self.dbscan_min_samples # (K,)
        self.last_dbscan_core_fraction = is_core.float().mean()

        core_idx = torch.where(is_core)[0]
        C = int(core_idx.numel())
        if C == 0:
            return labels

        # Connected components among core points.
        # Label propagation: every core point repeatedly takes the minimum label
        # among its adjacent core neighbors until convergence.
        core_adj = neighbors.index_select(0, core_idx).index_select(1, core_idx)  # (C,C)
        comp = torch.arange(C, device=feat.device, dtype=torch.long)
        big = torch.full((C, C), C + 1, device=feat.device, dtype=torch.long)

        for _ in range(C):
            prev = comp
            comp_mat = comp.view(1, C).expand(C, C)
            new_comp = torch.where(core_adj, comp_mat, big).min(dim=1).values
            comp = new_comp
            if bool(torch.equal(comp, prev)):
                break

        # Compress arbitrary component ids to 0..num_clusters-1.
        _, inv = torch.unique(comp, sorted=True, return_inverse=True)
        labels[core_idx] = inv

        # Assign border points to nearest adjacent core point's cluster.
        border_idx = torch.where(~is_core)[0]
        if border_idx.numel() > 0:
            border_to_core = neighbors.index_select(0, border_idx).index_select(1, core_idx)
            has_core_neighbor = border_to_core.any(dim=1)

            if bool(has_core_neighbor.any()):
                valid_border_idx = border_idx[has_core_neighbor]
                valid_mask = border_to_core[has_core_neighbor]
                d_bc = dist.index_select(0, valid_border_idx).index_select(1, core_idx)

                inf = torch.full_like(d_bc, float('inf'))
                d_bc = torch.where(valid_mask, d_bc, inf)
                nearest_core_local = torch.argmin(d_bc, dim=1)
                nearest_core_global = core_idx[nearest_core_local]
                labels[valid_border_idx] = labels[nearest_core_global]

        return labels

    @torch.no_grad()
    def _select_by_dbscan(self, cost_g: torch.Tensor, tip_traj_g: torch.Tensor):
        """
        cost_g: (K,) already sorted ascending and already collision-free.
        tip_traj_g: (K,H,3)

        Returns:
            best_local_idx: scalar index into good non-collision pool
            labels: (K,)
            rep_local_idx: representative local indices, one per discovered cluster
        """
        K = int(cost_g.shape[0])
        if K == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return None, empty, empty
        if K < self.dbscan_min_samples:
            return (torch.tensor(0, device=self.device, dtype=torch.long),
                    torch.full((K,), -1, device=self.device, dtype=torch.long),
                    torch.tensor([0], device=self.device, dtype=torch.long))

        feat = self._make_dbscan_feature(tip_traj_g)
        labels = self._dbscan_torch(feat)
        cluster_ids = torch.unique(labels[labels >= 0])

        if cluster_ids.numel() == 0:
            return (torch.tensor(0, device=self.device, dtype=torch.long),
                    labels,
                    torch.tensor([0], device=self.device, dtype=torch.long))

        best_cluster = None
        best_cluster_score = None
        rep_indices = []

        for cid in cluster_ids.tolist():
            mask = labels == int(cid)
            idxs = torch.where(mask)[0]
            c_cost = cost_g[idxs]
            local_best = idxs[torch.argmin(c_cost)]
            rep_indices.append(local_best)

            # Cluster score: mean cost. Change to c_cost.min() if you prefer the most aggressive baseline.
            score = c_cost.mean()
            if best_cluster_score is None or float(score.item()) < float(best_cluster_score.item()):
                best_cluster_score = score
                best_cluster = int(cid)

        best_mask = labels == best_cluster
        best_idxs = torch.where(best_mask)[0]
        best_local_idx = best_idxs[torch.argmin(cost_g[best_idxs])]
        rep_local_idx = torch.stack(rep_indices).to(device=self.device, dtype=torch.long)
        return best_local_idx, labels, rep_local_idx

    @torch.no_grad()
    def improve_policy(self, pos, vel, goal, Obs=None):
        self._sync_profile_timer()
        improve_t0 = time.perf_counter()
        goal = self._pad_goal_to_horizon(goal)

        batch_traj = None
        cost = None
        best_full_idx = None
        rep_full_idx = None
        rollout_time = 0.0
        cost_time = 0.0
        multimodal_time = 0.0

        for _ in range(self.n_improve):
            batch_ctr_parameter = self._build_candidates_full_horizon()

            if Obs is None:
                self._sync_profile_timer()
                t0 = time.perf_counter()
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter)
                self._sync_profile_timer()
                rollout_time += time.perf_counter() - t0

                self._sync_profile_timer()
                t0 = time.perf_counter()
                cost = self.cost_fn(batch_traj, goal)
                self._sync_profile_timer()
                cost_time += time.perf_counter() - t0
            else:
                self._sync_profile_timer()
                t0 = time.perf_counter()
                batch_traj_full = self.rollout(pos, vel, batch_ctr_parameter, full=True)
                self._sync_profile_timer()
                rollout_time += time.perf_counter() - t0

                self._sync_profile_timer()
                t0 = time.perf_counter()
                cost = self.cost_fn(batch_traj_full, goal, Obs=Obs)
                self._sync_profile_timer()
                cost_time += time.perf_counter() - t0
                batch_traj = batch_traj_full[:, :, -1, :]

            self._sync_profile_timer()
            t0 = time.perf_counter()
            sort_idx = torch.argsort(cost)
            K0 = min(self.top_k_good, int(sort_idx.numel()))
            good_idx = sort_idx[:K0]

            # 先选 top_k_good，再过滤碰撞；当前 cost_fn 中碰撞样本 cost 会加 1e8。
            valid_mask = cost[good_idx] < self.collision_thr
            good_valid_idx = good_idx[valid_mask]

            if good_valid_idx.numel() == 0:
                warnings.warn(
                    f"[DBSCAN_MPC] No collision-free candidate in top_k_good={self.top_k_good}. "
                    f"Fallback to global argmin, even if it may be collision.",
                    category=UserWarning
                )
                best_full_idx = sort_idx[0]
                rep_full_idx = best_full_idx.view(1)
                self.last_dbscan_labels = torch.empty(0, device=self.device, dtype=torch.long)
                self.last_good_idx = good_valid_idx.detach()
            else:
                traj_g = batch_traj[good_valid_idx]
                cost_g = cost[good_valid_idx]

                best_local_idx, labels, rep_local_idx = self._select_by_dbscan(cost_g, traj_g)
                if best_local_idx is None:
                    best_full_idx = sort_idx[0]
                    rep_full_idx = best_full_idx.view(1)
                else:
                    best_full_idx = good_valid_idx[best_local_idx]
                    rep_full_idx = good_valid_idx[rep_local_idx]

                self.last_dbscan_labels = labels.detach()
                self.last_good_idx = good_valid_idx.detach()

            self.seeds.copy_(batch_ctr_parameter[best_full_idx].view(1, self.horizon, self.action_dim))
            self.J_star[0] = cost[best_full_idx]
            self._sync_profile_timer()
            multimodal_time += time.perf_counter() - t0

        self.last_tip_traj = batch_traj.detach()
        self.last_cost = cost.detach()
        if rep_full_idx is not None:
            self.traj_seeds = batch_traj[rep_full_idx].detach()
        elif best_full_idx is not None:
            self.traj_seeds = batch_traj[best_full_idx:best_full_idx + 1].detach()
        else:
            self.traj_seeds = None

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

    @torch.no_grad()
    def update_policy(self):
        self.seeds[:, :-1, :].copy_(self.seeds[:, 1:, :].clone())
        self.seeds[:, -1:, :].copy_(self.seeds[:, -2:-1, :])

    def get_action(self, rule='greedy'):
        return self.seeds[0, 0]

    @torch.no_grad()
    def get_cand(self, k_cand=100, rule='random'):
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

    # ===== DBSCAN baseline 参数 =====
    n_sample = 400     # 扰动采样数；总 batch = 1 + n_sample

    n_improve = 5
    noise_scale = 1.8  # 0.25 looks good for naive case, 1.5更好for naive case？
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)

    # PyTorch DBSCAN 超参
    top_k_good = 200
    dbscan_eps = 3.0
    dbscan_min_samples = 5
    collision_thr = 1e5

    visualization = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === warp_4 rope：rollout（batch = 1 + n_sample）===
    rope = WarpRope(
        batch_size=1 + n_sample,
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
        top_k_good=top_k_good,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        collision_thr=collision_thr,
        normalize_dbscan_feat=True
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

        # 初始化 DBSCAN baseline 的单个 seed。
        planner.seeds[:] = vel_start.unsqueeze(0)

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
                        title=f"Torch-DBSCAN MPC step {i} | {k_cand} candidates"
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
