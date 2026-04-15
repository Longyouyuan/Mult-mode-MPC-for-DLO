import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.utils import *
from warp_rope_franka_model import WarpRopeFranka, N, P, load_runner

import warnings
import warp as wp


class Planner:
    def __init__(self, rope: WarpRopeFranka, cost_fn, dt, ctr_period, horizon, n_sample, n_improve,
                 noise_scale, action_dim, limits=None, device='cpu', mode='acc',
                 m_modes=1, top_k_good=200, beta=1.0, wJ=1.0, standard_m2pc=True):

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

        self.m = int(m_modes)
        assert self.n_sample % self.m == 0, "n_sample % m_modes must == 0"
        self.Nk = self.n_sample // self.m

        self.top_k_good = int(top_k_good)
        self.beta = float(beta)
        self.wJ = float(wJ)

        self.batch = self.m + self.n_sample

        self.seeds = torch.zeros((self.m, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)
        self.J_star = torch.zeros(self.m, device=self.device, dtype=torch.float32)

        self._noise = torch.empty((self.n_sample, self.horizon, self.action_dim),
                                  device=self.device, dtype=torch.float32)
        self._cand = torch.empty((self.batch, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)

        self.last_tip_traj = None
        self.last_cost = None
        self.traj_seeds = None

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
        self._cand[:self.m].copy_(self.seeds)
        self._noise.normal_(0.0, self.noise_scale)

        for k in range(self.m):
            s = k * self.Nk
            e = (k + 1) * self.Nk
            self._cand[self.m + s:self.m + e].copy_(self.seeds[k].unsqueeze(0))
            self._cand[self.m + s:self.m + e].add_(self._noise[s:e])

        if self.limits is not None:
            low, high = self.limits
            self._cand.clamp_(float(low.item()), float(high.item()))

        return self._cand

    @torch.no_grad()
    def rollout(self, pos, vel, batch_ctr_parameter_full, goal_pos, goal_vel, full=False):
        self.rope.set_state_and_action(pos, vel, batch_ctr_parameter_full, goal_pos, goal_vel)
        out = self.rope.simulate(steps=self.horizon * self.ctr_period)
        rope_traj = out["rope_traj"]  # (B,H,P,3)
        self.real_pos_1_batch = out["real_traj"][:, 0, :] 

        if full is False:
            return rope_traj[:, :, -1, :]   # last rope node
        else:
            return rope_traj

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, thr=1e5):
        valid_n = int((cost_g_org < thr).sum().item())

        if valid_n == 0:
            warnings.warn(
                f"[greedy_select] valid_n=0 (all cost >= {thr:.2e}). "
                f"Fallback: return first {self.m} indices.",
                category=UserWarning
            )
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        elif valid_n <= self.m:
            return torch.arange(self.m, device=self.device, dtype=torch.long)

        def minmax_norm(v, eps=1e-8):
            vmin = v.min()
            vmax = v.max()
            if (vmax - vmin) < 1e-12:
                return torch.zeros_like(v)
            return (v - vmin) / (vmax - vmin + eps)

        cost_g = cost_g_org[:valid_n]
        tip_traj_g = tip_traj_g_org[:valid_n]

        feat = tip_traj_g.reshape(valid_n, -1)
        dist2 = torch.cdist(feat, feat, p=2) ** 2

        first = 0
        selected = [first]
        div_sum = dist2[:, first].clone()

        c_norm = minmax_norm(cost_g)
        while len(selected) < self.m:
            d_norm = minmax_norm(div_sum)
            score = -self.wJ * c_norm + self.beta * d_norm
            score[selected] = -1e18

            nxt = int(torch.argmax(score).item())
            selected.append(nxt)
            div_sum += dist2[:, nxt]

        return torch.tensor(selected, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def improve_policy(self, pos, vel, goal, goal_pos, goal_vel, Obs=None):
        goal = self._pad_goal_to_horizon(goal)

        for _ in range(self.n_improve):
            batch_ctr_parameter = self._build_candidates_full_horizon()

            if Obs is None:
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter, goal_pos, goal_vel)
                cost = self.cost_fn(batch_traj, goal)
            else:
                batch_traj = self.rollout(pos, vel, batch_ctr_parameter, goal_pos, goal_vel, full=True)
                cost = self.cost_fn(batch_traj, goal, Obs=Obs)
                batch_traj = batch_traj[:, :, -1, :]

            if self.m == 1:
                idx = int(cost.argmin().item())
                if self.standard_m2pc:
                    self.seeds.copy_(batch_ctr_parameter[idx])
                else:
                    lam = 0.3 * (cost.std() + 1e-6)
                    J = cost
                    J_min = J.min()
                    w = torch.exp(-(J - J_min) / lam)
                    w = w / (w.sum() + 1e-8)
                    new_params = (w.view(-1, 1, 1) * batch_ctr_parameter).sum(dim=0)
                    self.seeds.copy_(new_params)
            else:
                sort_idx = torch.argsort(cost)
                good_idx = sort_idx[:self.top_k_good]

                traj_g = batch_traj[good_idx]
                cost_g = cost[good_idx]
                ctr_g = batch_ctr_parameter[good_idx]

                sel = self._greedy_select(cost_g, traj_g)
                U_star = ctr_g[sel]
                self.seeds.copy_(U_star)

        self.last_tip_traj = batch_traj.detach()
        self.last_cost = cost.detach()
        if self.m == 1:
            self.traj_seeds = batch_traj[idx:idx + 1]
            self.real_pos_1 = self.real_pos_1_batch[idx]
        else:
            self.traj_seeds = traj_g[sel]
            J_star = cost_g[sel]
            self.J_star.copy_(J_star)

    @torch.no_grad()
    def update_policy(self):
        self.seeds[:, :-1, :].copy_(self.seeds[:, 1:, :].clone())
        self.seeds[:, -1:, :].copy_(self.seeds[:, -2:-1, :])

    def get_action(self, rule='greedy'):
        if self.m == 1:
            return self.seeds[0, 0]

        if rule == 'greedy':
            best = int(torch.argmin(self.J_star).item())
        elif rule == 'sample':
            j = self.J_star - self.J_star.min()
            temp = 1.0 * self.J_star.std() + 1e-6
            p = torch.softmax(-j / temp, dim=0)
            best = int(torch.multinomial(p, 1).item())
        elif rule == 'random':
            best = int(torch.randint(self.m, (1,), device=self.device).item())
        else:
            raise ValueError(f"Unknown rule: {rule}")

        return self.seeds[best, 0]

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
        return ((dx * dx + dy * dy) <= r2) & (dz.abs() <= h)

    hit = _hit(c1) | _hit(c2)
    return hit.any(dim=2).any(dim=1)


def cost_fn(batch_traj, goal_h, Obs=None):
    if Obs is None and batch_traj.dim() == 3:
        err = batch_traj - goal_h.unsqueeze(0)
        return torch.sum(torch.abs(err), dim=(1, 2))
    elif Obs is not None and batch_traj.dim() == 4:
        margin = 0.02
        collision_cost = 1e8

        tip = batch_traj[:, :, -1, :]
        err = tip - goal_h.unsqueeze(0)
        w = torch.tensor([1.8, 1.8, 1.0], device=err.device)
        J_track = torch.sum(torch.abs(err) * w, dim=(1, 2))

        Obs_t = torch.as_tensor(Obs, device=tip.device, dtype=tip.dtype)
        r0, h0 = Obs_t[0, 0], Obs_t[0, 1]
        c1, c2 = Obs_t[1, :], Obs_t[2, :]

        hit = collide_two_cylinders(batch_traj, c1, c2, r0, h0, margin=margin)
        J = torch.where(hit, J_track + collision_cost, J_track)
        return J
    else:
        raise ValueError("Cost_fn doesn't know if there is Obstacle")


if __name__ == "__main__":
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
    mode = 'acc'

    ctr_period = 25
    horizon = 30

    n_sample = 32
    m_modes = 1
    assert n_sample % m_modes == 0

    n_improve = 10
    noise_scale = 1.8
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)

    top_k_good = 200
    beta = 5.0
    wJ = 0.0

    visualization = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    franka_runner = load_runner("best_model.pt", device="cuda")

    rope = WarpRopeFranka(
        franka_runner=franka_runner,
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
        device="cuda"
    )

    rope_exec = WarpRopeFranka(
        franka_runner=franka_runner,
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
        device="cuda"
    )

    Goal_traj = build_goal_traj_from_drawn(
        drawn_path="../my_trajs/SpongeBob.npy",
        total_horizon=total_horizon,
        device=device,
        z0=0.2,
        scale_x=3.0,
        scale_y=3.0,
        keep_aspect=False,
        sigma=2.0,
        uniform_M=1000,
        ratio=0.3,
        sharpness=2.0,
        interval=(0.0, 2.0)
    )
    plot_goal_traj(Goal_traj, T_task)

    planner = Planner(
        rope, cost_fn, dt, ctr_period, horizon,
        n_sample, n_improve, noise_scale, action_dim,
        limits=limits, device=device, mode=mode,
        m_modes=m_modes, top_k_good=top_k_good, beta=beta, wJ=wJ
    )

    if visualization:
        viz = LiveMPCVisualizer(goal_traj=Goal_traj, margin=0.05)

    pos = torch.zeros((1, P, 3), device=device)
    pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
    vel = torch.zeros((1, P, 3), device=device)

    goal_pos = pos[:, 0, :].clone()
    goal_vel = vel[:, 0, :].clone()

    action_history = []
    pos_history = []
    vel_history = []
    goal_pos_history = []
    goal_vel_history = []
    time_record = []

    pos_history.append(pos.clone())
    goal_pos_history.append(goal_pos.clone())
    goal_vel_history.append(goal_vel.clone())

    stored_actions = None

    if stored_actions is None:
        vel_start = (Goal_traj[1:1 + horizon, :] - Goal_traj[:horizon, :]) / (T_task / horizon)
        vel_start = vel_start.to(device)

        planner.seeds[:] = vel_start.unsqueeze(0).repeat(m_modes, 1, 1)
        if m_modes > 1:
            planner.seeds[1:] += 0.05 * torch.randn_like(planner.seeds[1:])

        goal = Goal_traj[1:1 + horizon, :]
        planner.n_improve = n_improve * 100
        planner.improve_policy(pos, vel, goal, goal_pos, goal_vel)
        planner.n_improve = n_improve

    for i in range(total_horizon):
        if i % 100 == 0:
            print('i: ', i)

        if stored_actions is None:
            if i < (total_horizon - horizon):
                goal = Goal_traj[i + 1:i + 1 + horizon, :]
            else:
                goal = Goal_traj[i + 1:, :]

            t0 = time.perf_counter()
            planner.improve_policy(pos, vel, goal, goal_pos, goal_vel)
            action = planner.get_action()
            t1 = time.perf_counter()
            time_record.append(t1 - t0)
        else:
            action = torch.from_numpy(stored_actions[i]).to(device)

        if visualization:
            if i % 2 == 0:
                k_cand = 100
                cand_tip_traj, cand_cost, m_mode_trajs = planner.get_cand(k_cand=k_cand)

                if cand_tip_traj is not None:
                    rope_now = pos[0]
                    viz.update(
                        rope_pos=rope_now,
                        cur_i=i,
                        cand_tip_traj=cand_tip_traj,
                        cand_cost=cand_cost,
                        m_mode_trajs=m_mode_trajs,
                        title=f"MPC step {i} | {k_cand} candidates | modes={m_modes}"
                    )

        action_seq = action.view(1, 1, 3)
        prev_goal_pos = goal_pos.clone()

        rope_exec.set_state_and_action(pos, vel, action_seq, goal_pos, goal_vel)
        out_exec = rope_exec.simulate(steps=ctr_period)

        pos = wp.to_torch(rope_exec.pos)
        vel = wp.to_torch(rope_exec.vel)

        goal_pos = out_exec["goal_traj"][:, -1, :].contiguous()
        goal_vel = goal_vel + (ctr_period * dt) * action.view(1, 3)

        planner.update_policy()

        pos_history.append(pos.clone())
        action_history.append(action.detach().cpu().numpy())
        vel_history.append(vel.clone().detach().cpu().numpy()[0, -1])
        goal_pos_history.append(goal_pos.clone())
        goal_vel_history.append(goal_vel.clone())

    if visualization:
        viz.close()

    print("time: ", torch.tensor(time_record).mean())

    pos_history = torch.cat(pos_history, dim=0)
    plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)
    plot_tip_vs_goal_and_error(pos_history, Goal_traj, dt, ctr_period)

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

    tip_pos_history = pos_history[1:, -1, :]
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, tip_pos_history[:, 0].cpu().numpy(), label='Tip X', color='m')
    plt.plot(time_axis, tip_pos_history[:, 1].cpu().numpy(), label='Tip Y', color='c')
    plt.plot(time_axis, tip_pos_history[:, 2].cpu().numpy(), label='Tip Z', color='y')
    plt.xlabel('Time (s)')
    plt.ylabel('last Position')
    plt.title('Last Point Position XYZ over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
