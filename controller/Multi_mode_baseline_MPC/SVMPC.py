import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.utils import *
from common.rope_warp_4_baserope import WarpRope, N, P

# ============================================================
# Stein Variational Model Predictive Control baseline
# Architecture / interface mirrors M2PC.py.
# ============================================================


class Planner:
    """
    SV-MPC / SVGD baseline for open-loop control sequences.

    Compatible with M2PC.Planner constructor and public methods:
      - improve_policy(pos, vel, goal, Obs=None)
      - update_policy()
      - get_action(rule='greedy'|'sample'|'random'|'weighted')
      - get_cand(k_cand=100, rule='random'|'best')

    Interpretation:
      - m_modes is the number of SVGD particles.
      - n_sample is the total number of Monte-Carlo samples used for likelihood-gradient
        estimation; n_sample must be divisible by m_modes.
      - self.seeds:       (m,H,D) current representative particles.
      - self.J_star:      (m,) cost of those representative particles only.
      - self.traj_seeds:  (m,H,3) rollout trajectories of representative particles.

    The default implementation intentionally uses a weak/uniform prior
    (prior_weight=0.0) because a strong Gaussian prior centered at the weighted
    mean can collapse all particles and make the controller drift in one direction.
    """

    def __init__(self, rope: WarpRope, cost_fn, dt, ctr_period, horizon, n_sample, n_improve,
                 noise_scale, action_dim, limits=None, device='cpu', mode='acc',
                 m_modes=4, top_k_good=200, beta=1.0, wJ=1.0, standard_m2pc=True,
                 alpha=1.0, lambda_=1.0, svgd_step_size=0.25,
                 grad_noise_scale=None, prior_noise_scale=None,
                 likelihood_type='EU', elite_frac=0.1,
                 kernel_h=None, kernel_min_h=1e-6,
                 prior_weight=0.0,
                 use_weighted_average=False,
                 update_prior_mean=False,
                 update_prior_cov=False,
                 min_weight_temp=1e-6):

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

        # In SV-MPC, m means number of Stein particles.
        self.m = int(m_modes)
        assert self.m >= 1, "m_modes must be >= 1"
        assert self.n_sample % self.m == 0, "n_sample % m_modes must == 0"
        self.Nk = self.n_sample // self.m

        # Kept only for constructor compatibility with M2PC.py.
        self.top_k_good = int(top_k_good)
        self.beta = float(beta)
        self.wJ = float(wJ)

        # SV-MPC hyperparameters.
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.svgd_step_size = float(svgd_step_size)
        self.grad_noise_scale = float(self.noise_scale if grad_noise_scale is None else grad_noise_scale)
        self.prior_noise_scale = float(self.noise_scale if prior_noise_scale is None else prior_noise_scale)
        self.likelihood_type = str(likelihood_type).upper()
        assert self.likelihood_type in ("EU", "PLC")
        self.elite_frac = float(elite_frac)
        self.kernel_h = kernel_h
        self.kernel_min_h = float(kernel_min_h)
        self.prior_weight = float(prior_weight)
        self.use_weighted_average = bool(use_weighted_average)
        self.update_prior_mean = bool(update_prior_mean)
        self.update_prior_cov = bool(update_prior_cov)
        self.min_weight_temp = float(min_weight_temp)

        # batch = m representative particles + n_sample local samples.
        self.batch = self.m + self.n_sample

        self.seeds = torch.zeros((self.m, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)
        self.J_star = torch.full((self.m,), float("inf"), device=self.device, dtype=torch.float32)
        self.particle_cost = self.J_star.clone()
        self.particle_weights = torch.full((self.m,), 1.0 / self.m, device=self.device, dtype=torch.float32)
        self.best_idx = torch.tensor(0, device=self.device, dtype=torch.long)
        self.best_sequence = torch.zeros((self.horizon, self.action_dim), device=self.device, dtype=torch.float32)

        # Optional Gaussian prior. Disabled by default via prior_weight=0.0.
        self.prior_mean = torch.zeros_like(self.seeds)
        self.prior_var = torch.full_like(self.seeds, max(self.prior_noise_scale ** 2, 1e-6))

        self._noise = torch.empty((self.n_sample, self.horizon, self.action_dim),
                                  device=self.device, dtype=torch.float32)
        self._cand = torch.empty((self.batch, self.horizon, self.action_dim),
                                 device=self.device, dtype=torch.float32)
        self._eval_cand = torch.empty_like(self._cand)

        self.last_tip_traj = None
        self.last_cost = None
        self.traj_seeds = None
        dev_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]
        self._profile_uses_cuda = bool(torch.cuda.is_available() and dev_type == "cuda")
        self._active_profile = None
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
    def _build_svgd_batch(self):
        """
        cand:
          [0:m]              = current representative particles
          [m:m+n_sample]     = Nk Gaussian samples around each particle
        """
        self._cand[:self.m].copy_(self.seeds)
        self._noise.normal_(0.0, self.grad_noise_scale)

        for k in range(self.m):
            s = k * self.Nk
            e = (k + 1) * self.Nk
            self._cand[self.m + s:self.m + e].copy_(self.seeds[k].unsqueeze(0))
            self._cand[self.m + s:self.m + e].add_(self._noise[s:e])

        if self.limits is not None:
            low, high = self.limits
            self._cand.clamp_(float(low.item()), float(high.item()))

        return self._cand

    # ------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, pos, vel, batch_ctr_parameter_full, full=False):
        self.rope.set_state_and_action(pos, vel, batch_ctr_parameter_full)
        traj = self.rope.simulate(steps=self.horizon * self.ctr_period)
        if full is False:
            return traj[:, :, -1, :]
        return traj

    # ------------------------------------------------------------
    @torch.no_grad()
    def _rollout_and_cost_full_batch(self, pos, vel, ctr_batch, goal, Obs=None):
        profile = self._active_profile

        if Obs is None:
            self._sync_profile_timer()
            t0 = time.perf_counter()
            tip_traj = self.rollout(pos, vel, ctr_batch)
            self._sync_profile_timer()
            if profile is not None:
                profile["rollout"] += time.perf_counter() - t0

            self._sync_profile_timer()
            t0 = time.perf_counter()
            cost = self.cost_fn(tip_traj, goal)
            self._sync_profile_timer()
            if profile is not None:
                profile["cost"] += time.perf_counter() - t0
        else:
            self._sync_profile_timer()
            t0 = time.perf_counter()
            full_traj = self.rollout(pos, vel, ctr_batch, full=True)
            self._sync_profile_timer()
            if profile is not None:
                profile["rollout"] += time.perf_counter() - t0

            self._sync_profile_timer()
            t0 = time.perf_counter()
            cost = self.cost_fn(full_traj, goal, Obs=Obs)
            self._sync_profile_timer()
            if profile is not None:
                profile["cost"] += time.perf_counter() - t0
            tip_traj = full_traj[:, :, -1, :]
        return tip_traj, cost

    # ------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_controls(self, pos, vel, controls, goal, Obs=None):
        """Evaluate K controls using the fixed-size WarpRope batch."""
        K = int(controls.shape[0])
        if K > self.batch:
            raise ValueError(f"K={K} exceeds fixed rollout batch={self.batch}")
        self._eval_cand[:K].copy_(controls)
        if K < self.batch:
            self._eval_cand[K:].copy_(controls[0].unsqueeze(0).expand(self.batch - K, -1, -1))
        tip_traj, cost = self._rollout_and_cost_full_batch(pos, vel, self._eval_cand, goal, Obs=Obs)
        return tip_traj[:K], cost[:K]

    # ------------------------------------------------------------
    @torch.no_grad()
    def _sample_weights(self, sample_cost):
        """Weights over Nk local samples for each particle."""
        if self.likelihood_type == "EU":
            temp = max(self.lambda_, self.min_weight_temp)
            score = -self.alpha * sample_cost / temp
            score = score - score.max(dim=1, keepdim=True).values
            w = torch.exp(score)
            return w / (w.sum(dim=1, keepdim=True) + 1e-12)

        # PLC / CEM-like elite indicator.
        n_elite = max(1, int(np.ceil(self.elite_frac * self.Nk)))
        elite_idx = torch.topk(sample_cost, n_elite, largest=False, dim=1).indices
        w = torch.zeros_like(sample_cost)
        w.scatter_(1, elite_idx, 1.0 / float(n_elite))
        return w

    # ------------------------------------------------------------
    @torch.no_grad()
    def _likelihood_gradient(self, sample_actions, sample_cost):
        """
        Score-function estimate for grad_theta log E[L].

        sample_actions: (m,Nk,H,D)
        sample_cost:    (m,Nk)
        return:         (m,H,D)
        """
        theta = self.seeds.unsqueeze(1)
        diff = sample_actions - theta
        var = max(self.grad_noise_scale ** 2, 1e-12)
        w = self._sample_weights(sample_cost)
        return (w[:, :, None, None] * diff).sum(dim=1) / var

    # ------------------------------------------------------------
    @torch.no_grad()
    def _prior_gradient(self):
        # Correct Gaussian log-prior gradient: grad log N(theta; mu, var) = -(theta-mu)/var.
        if self.prior_weight == 0.0:
            return torch.zeros_like(self.seeds)
        return -(self.seeds - self.prior_mean) / (self.prior_var + 1e-12)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _median_kernel_bandwidth(self, x_flat):
        if self.kernel_h is not None:
            return torch.as_tensor(float(self.kernel_h), device=self.device, dtype=x_flat.dtype)
        if self.m <= 1:
            return torch.as_tensor(1.0, device=self.device, dtype=x_flat.dtype)

        x2 = (x_flat * x_flat).sum(dim=1, keepdim=True)
        dist2 = x2 + x2.T - 2.0 * (x_flat @ x_flat.T)
        dist2.clamp_(min=0.0)
        mask = ~torch.eye(self.m, dtype=torch.bool, device=self.device)
        vals = dist2[mask]
        med = vals.median() if vals.numel() > 0 else torch.as_tensor(1.0, device=self.device, dtype=x_flat.dtype)
        h = med / max(np.log(float(self.m) + 1.0), 1e-6)
        return torch.clamp(h, min=self.kernel_min_h)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _svgd_phi(self, grad_log_posterior):
        if self.m == 1:
            return grad_log_posterior

        x = self.seeds.reshape(self.m, -1)
        score = grad_log_posterior.reshape(self.m, -1)
        h = self._median_kernel_bandwidth(x)

        x_j = x[:, None, :]      # (j,1,D)
        x_i = x[None, :, :]      # (1,i,D)
        diff_ji = x_j - x_i
        dist2 = (diff_ji * diff_ji).sum(dim=2)
        Kji = torch.exp(-dist2 / h)  # K[j,i]

        attraction = Kji.T @ score
        repulsion = ((-2.0 / h) * Kji[:, :, None] * diff_ji).sum(dim=0)
        return ((attraction + repulsion) / float(self.m)).reshape_as(self.seeds)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _particle_weights_from_cost(self, particle_cost):
        temp = max(self.lambda_, self.min_weight_temp)
        score = -self.alpha * particle_cost / temp
        score = score - score.max()
        w = torch.exp(score)
        return w / (w.sum() + 1e-12)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _maybe_update_prior_from_particles(self):
        if not self.update_prior_mean:
            return
        w = self.particle_weights.view(self.m, 1, 1)
        mean_single = (w * self.seeds).sum(dim=0, keepdim=True)
        self.prior_mean.copy_(mean_single.expand_as(self.seeds))

        if self.update_prior_cov:
            diff = self.seeds - mean_single
            var_single = (w * diff * diff).sum(dim=0, keepdim=True)
            var_single.clamp_(min=1e-6, max=max(self.noise_scale ** 2 * 25.0, 1e-6))
            self.prior_var.copy_(var_single.expand_as(self.seeds))

    # ------------------------------------------------------------
    @torch.no_grad()
    def improve_policy(self, pos, vel, goal, Obs=None):
        self._sync_profile_timer()
        improve_t0 = time.perf_counter()
        profile = {"rollout": 0.0, "cost": 0.0, "multimodal": 0.0}
        self._active_profile = profile

        goal = self._pad_goal_to_horizon(goal)

        # If external warm-start writes self.seeds, initialize optional prior around it once.
        if torch.all(self.prior_mean == 0) and torch.any(self.seeds != 0):
            self.prior_mean.copy_(self.seeds)

        last_tip_traj = None
        last_cost = None

        for _ in range(self.n_improve):
            batch_ctr_parameter = self._build_svgd_batch()
            batch_tip_traj, cost = self._rollout_and_cost_full_batch(pos, vel, batch_ctr_parameter, goal, Obs=Obs)

            self._sync_profile_timer()
            t0 = time.perf_counter()
            sample_actions = batch_ctr_parameter[self.m:self.m + self.n_sample].reshape(
                self.m, self.Nk, self.horizon, self.action_dim
            )
            sample_cost = cost[self.m:self.m + self.n_sample].reshape(self.m, self.Nk)

            grad_lik = self._likelihood_gradient(sample_actions, sample_cost)
            grad_prior = self._prior_gradient()
            phi = self._svgd_phi(grad_lik + self.prior_weight * grad_prior)

            self.seeds.add_(self.svgd_step_size * phi)
            if self.limits is not None:
                low, high = self.limits
                self.seeds.clamp_(float(low.item()), float(high.item()))

            last_tip_traj = batch_tip_traj.detach()
            last_cost = cost.detach()
            self._sync_profile_timer()
            profile["multimodal"] += time.perf_counter() - t0

        # Final cost of representative particles only.
        particle_traj, particle_cost = self._evaluate_controls(pos, vel, self.seeds, goal, Obs=Obs)

        self._sync_profile_timer()
        t0 = time.perf_counter()
        self.particle_cost = particle_cost.detach().clone()
        self.J_star.copy_(self.particle_cost)
        self.particle_weights = self._particle_weights_from_cost(self.particle_cost).detach().clone()
        self.best_idx = torch.argmin(self.J_star)

        if self.use_weighted_average:
            self.best_sequence.copy_((self.particle_weights.view(self.m, 1, 1) * self.seeds).sum(dim=0))
        else:
            self.best_sequence.copy_(self.seeds[int(self.best_idx.item())])

        self._maybe_update_prior_from_particles()

        self.last_tip_traj = last_tip_traj
        self.last_cost = last_cost
        self.traj_seeds = particle_traj.detach()
        self._sync_profile_timer()
        profile["multimodal"] += time.perf_counter() - t0

        self._sync_profile_timer()
        improve_time = time.perf_counter() - improve_t0
        profile["improve"] = improve_time
        profile["overhead"] = improve_time - profile["rollout"] - profile["cost"] - profile["multimodal"]
        self.last_profile = profile.copy()
        self.profile_history.append(self.last_profile)
        self._active_profile = None

    # ------------------------------------------------------------
    @torch.no_grad()
    def update_policy(self):
        self.seeds[:, :-1, :].copy_(self.seeds[:, 1:, :].clone())
        self.seeds[:, -1:, :].copy_(self.seeds[:, -2:-1, :])

        self.prior_mean[:, :-1, :].copy_(self.prior_mean[:, 1:, :].clone())
        self.prior_mean[:, -1:, :].copy_(self.prior_mean[:, -2:-1, :])

        self.prior_var[:, :-1, :].copy_(self.prior_var[:, 1:, :].clone())
        self.prior_var[:, -1:, :].copy_(self.prior_var[:, -2:-1, :])

        self.best_sequence[:-1, :].copy_(self.best_sequence[1:, :].clone())
        self.best_sequence[-1:, :].copy_(self.best_sequence[-2:-1, :])

    # ------------------------------------------------------------
    def get_action(self, rule='greedy'):
        if self.m == 1:
            return self.seeds[0, 0]

        if rule == 'greedy':
            # MAP / minimum-cost representative particle.
            if self.use_weighted_average:
                return self.best_sequence[0]
            best = int(torch.argmin(self.J_star).item())
        elif rule == 'sample':
            best = int(torch.multinomial(self.particle_weights, 1).item())
        elif rule == 'random':
            best = int(torch.randint(self.m, (1,), device=self.device).item())
        elif rule == 'weighted':
            return (self.particle_weights.view(self.m, 1, 1) * self.seeds).sum(dim=0)[0]
        else:
            raise ValueError(f"Unknown rule: {rule}")

        return self.seeds[best, 0]

    # ------------------------------------------------------------
    @torch.no_grad()
    def get_cand(self, k_cand=100, rule='random'):
        if self.last_tip_traj is None or self.last_cost is None:
            return None, None, None

        # Visualize MC samples rather than the m representative slots.
        if self.last_tip_traj.shape[0] >= self.m + self.n_sample:
            pool_traj = self.last_tip_traj[self.m:self.m + self.n_sample]
            pool_cost = self.last_cost[self.m:self.m + self.n_sample]
        else:
            pool_traj = self.last_tip_traj
            pool_cost = self.last_cost

        B = pool_cost.shape[0]
        K = min(int(k_cand), B)
        if rule == 'random':
            k_idx = torch.randperm(B, device=pool_cost.device)[:K]
        elif rule == 'best':
            k_idx = torch.topk(pool_cost, K, largest=False).indices
        else:
            raise ValueError(f"Unknown rule: {rule}")

        return pool_traj[k_idx], pool_cost[k_idx], self.traj_seeds


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
        w = torch.tensor([1.8, 1.8, 1.0], device=err.device, dtype=err.dtype)
        J_track = torch.sum(torch.abs(err) * w, dim=(1, 2))

        Obs_t = torch.as_tensor(Obs, device=tip.device, dtype=tip.dtype)
        r0, h0 = Obs_t[0, 0], Obs_t[0, 1]
        c1, c2 = Obs_t[1, :], Obs_t[2, :]

        hit = collide_two_cylinders(batch_traj, c1, c2, r0, h0, margin=margin)
        return torch.where(hit, J_track + collision_cost, J_track)
    else:
        raise ValueError("Cost_fn doesn't know if there is Obstacle")


if __name__ == "__main__":
    # === Parameters, kept close to M2PC.py ===
    L = 1.0
    mass = 0.0025 * 40 / N
    k = 10000 * 0.46
    damping = 0.2
    bending_k = 0.0006712
    bending_damping = 0.000401
    air_drag = 0.2206 / 1000
    g = 10.07
    dt = 0.001
    T_task = 2.5 * 2
    total_steps = int(T_task / dt)
    mode = 'acc'

    ctr_period = 25
    horizon = 30

    n_particles = 3
    samples_per_particle = 133
    n_sample = n_particles * samples_per_particle
    m_modes = n_particles
    assert n_sample % m_modes == 0

    n_improve = 5
    noise_scale = 1.8
    action_dim = 3
    limits = torch.tensor([-15.0, 15.0])
    total_horizon = int(total_steps / ctr_period)

    top_k_good = 200
    beta = 5.0
    wJ = 0.0

    alpha = 5.0
    lambda_ = 1.0
    svgd_step_size = 0.5
    likelihood_type = 'EU'

    visualization = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    points = half_dense_then_uniform(
        N=total_horizon + 1, ratio=0.3, sharpness=2.0,
        mode='exp', interval=(0.0, 2.0), plot=False
    )
    Goal_traj = eight_traj(points, scale_x=0.45 * 3.0, scale_y=0.65 * 3.0,
                           z0=0.2, loops=1, plot=False, device=device)
    plot_goal_traj(Goal_traj, T_task)

    planner = Planner(
        rope, cost_fn, dt, ctr_period, horizon,
        n_sample, n_improve, noise_scale, action_dim,
        limits=limits, device=device, mode=mode,
        m_modes=m_modes, top_k_good=top_k_good, beta=beta, wJ=wJ,
        alpha=alpha, lambda_=lambda_, svgd_step_size=svgd_step_size,
        likelihood_type=likelihood_type,
        prior_weight=0.0,
        update_prior_mean=False,
        use_weighted_average=False
    )

    if visualization:
        viz = LiveMPCVisualizer(goal_traj=Goal_traj, margin=0.05)

    pos = torch.zeros((1, P, 3), device=device)
    pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
    vel = torch.zeros((1, P, 3), device=device)

    action_history = []
    pos_history = []
    vel_history = []
    time_record = []
    pos_history.append(pos.clone())

    stored_actions = None

    if stored_actions is None:
        vel_start = (Goal_traj[1:1 + horizon, :] - Goal_traj[:horizon, :]) / (T_task / horizon)
        vel_start = vel_start.to(device)

        planner.seeds[:] = vel_start.unsqueeze(0).repeat(m_modes, 1, 1)
        if m_modes > 1:
            planner.seeds[1:] += 0.05 * torch.randn_like(planner.seeds[1:])
        planner.prior_mean.copy_(planner.seeds)

        goal = Goal_traj[1:1 + horizon, :]
        planner.n_improve = n_improve * 20
        planner.improve_policy(pos, vel, goal)
        planner.n_improve = n_improve

    import warp as wp
    for i in range(total_horizon):
        if i % 100 == 0:
            print('i: ', i)

        if stored_actions is None:
            if i < (total_horizon - horizon):
                goal = Goal_traj[i + 1:i + 1 + horizon, :]
            else:
                goal = Goal_traj[i + 1:, :]

            t0 = time.perf_counter()
            planner.improve_policy(pos, vel, goal)
            action = planner.get_action(rule='greedy')
            t1 = time.perf_counter()
            time_record.append(t1 - t0)
        else:
            action = torch.from_numpy(stored_actions[i]).to(device)

        if visualization and i % 2 == 0:
            k_cand = 100
            cand_tip_traj, cand_cost, m_mode_trajs = planner.get_cand(k_cand=k_cand)
            if cand_tip_traj is not None:
                viz.update(
                    rope_pos=pos[0],
                    cur_i=i,
                    cand_tip_traj=cand_tip_traj,
                    cand_cost=cand_cost,
                    m_mode_trajs=m_mode_trajs,
                    title=f"SV-MPC step {i} | samples={k_cand} | particles={m_modes}"
                )

        action_seq = action.view(1, 1, 3)
        rope_exec.set_state_and_action(pos, vel, action_seq)
        _ = rope_exec.simulate(steps=ctr_period)

        pos = wp.to_torch(rope_exec.pos)
        vel = wp.to_torch(rope_exec.vel)

        planner.update_policy()

        pos_history.append(pos.clone())
        action_history.append(action.detach().cpu().numpy())
        vel_history.append(vel.clone().detach().cpu().numpy()[0, 0])

    if visualization:
        viz.close()

    print("time: ", torch.tensor(time_record).mean())

    pos_history = torch.cat(pos_history, dim=0)
    anim = plot_animation_3d_traj(pos_history, Goal_traj, dt, ctr_period, L=1.2)
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
