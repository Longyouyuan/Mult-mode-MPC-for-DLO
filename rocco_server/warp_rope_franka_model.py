
import time
import numpy as np
import torch
import warp as wp
from warp_franka_model import load_runner
from common.utils import *

wp.init()
wp.set_module_options({"enable_backward": False})

# ===== Fixed rope size =====
N = 5
P = N + 1  # nodes
S = N      # spring segments
B = N - 1  # bending triplets

TILE_THREADS = 32 * 2

# ===== Franka model sizes =====
H = wp.constant(64)
R = wp.constant(64)
X_DIM = wp.constant(12)
OUT_DIM = wp.constant(3)
ONE = wp.constant(1)


@wp.func
def gravity_air_force_mass(v: wp.vec3, mg_i: float, air_drag: float) -> wp.vec3:
    return -(wp.vec3(0.0, 0.0, mg_i) + air_drag * v)


@wp.func
def linear_spring_damping(
        pi: wp.vec3, pj: wp.vec3,
        vi: wp.vec3, vj: wp.vec3,
        k: float, damping: float, rest_length: float
) -> wp.vec3:
    dx = pj - pi
    l = wp.length(dx)
    d = dx / (l + 1.0e-8)

    rel_v = vj - vi
    proj_v = wp.dot(rel_v, d)

    return (k * (l - rest_length) + damping * proj_v) * d


@wp.func
def bending_spring_damper(
        p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
        v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
        bending_k: float, bending_damping: float
) -> wp.mat33:
    dx1 = p1 - p0
    l1 = wp.length(dx1)
    d1 = dx1 / (l1 + 1.0e-8)

    dx2 = p2 - p1
    l2 = wp.length(dx2)
    d2 = dx2 / (l2 + 1.0e-8)

    cr = wp.cross(d1, d2)

    F0 = wp.vec3(0.0, 0.0, 0.0)
    F1 = wp.vec3(0.0, 0.0, 0.0)
    F2 = wp.vec3(0.0, 0.0, 0.0)

    if bending_k != 0.0:
        c = wp.dot(d1, d2)
        c = wp.clamp(c, -1.0 + 1.0e-8, 1.0 - 1.0e-8)
        beta = wp.acos(c)

        sb = wp.sin(beta)
        scale = bending_k * beta / (sb + 1.0e-8)
        common = scale * cr

        pre = -(wp.cross(d1, common)) / (l1 + 1.0e-8)
        aft = -(wp.cross(d2, common)) / (l2 + 1.0e-8)

        F0 = F0 + (-pre)
        F1 = F1 + (pre + aft)
        F2 = F2 + (-aft)

    if bending_damping != 0.0:
        c1 = cr

        grad_d1 = wp.cross(d1, c1)
        norm_grad_d1 = wp.length(grad_d1)
        dbeta_de1 = grad_d1 / (norm_grad_d1 * l1 + 1.0e-8)

        grad_d2 = wp.cross(d2, -c1)
        norm_grad_d2 = wp.length(grad_d2)
        dbeta_de2 = grad_d2 / (norm_grad_d2 * l2 + 1.0e-8)

        vel_diff1 = v1 - v0
        vel_diff2 = v2 - v1
        dbeta_dt = wp.dot(dbeta_de1, vel_diff1) + wp.dot(dbeta_de2, vel_diff2)

        tmp = bending_damping * dbeta_dt
        D1 = tmp * dbeta_de1
        D2 = -tmp * dbeta_de2

        F0 = F0 + D1
        F1 = F1 + (-(D1 + D2))
        F2 = F2 + D2

    return wp.matrix_from_cols(F0, F1, F2)


@wp.func
def mat_col0(m: wp.mat33) -> wp.vec3:
    return wp.vec3(m[0, 0], m[1, 0], m[2, 0])


@wp.func
def mat_col1(m: wp.mat33) -> wp.vec3:
    return wp.vec3(m[0, 1], m[1, 1], m[2, 1])


@wp.func
def mat_col2(m: wp.mat33) -> wp.vec3:
    return wp.vec3(m[0, 2], m[1, 2], m[2, 2])


@wp.func
def integrate_one(v: wp.vec3, f: wp.vec3, inv_mass: float, dt: float) -> wp.vec3:
    return v + dt * inv_mass * f


# ------------------------------
# Franka scalar math
# ------------------------------
@wp.func
def sigmoid_sum3(a: float, b: float, c: float) -> float:
    s = a + b + c
    return 1.0 / (1.0 + wp.exp(-s))


@wp.func
def tanh_fma(a: float, b: float, c: float) -> float:
    s = a + b + c
    ex = wp.exp(s)
    enx = wp.exp(-s)
    return (ex - enx) / (ex + enx)


@wp.func
def relu_sum3(a: float, b: float, c: float) -> float:
    return wp.max(a + b + c, 0.0)


@wp.func
def relu_sum(a: float, b: float) -> float:
    return wp.max(a + b, 0.0)


@wp.func
def gru_mix(z_val: float, n_val: float, h_val: float) -> float:
    return n_val + z_val * (h_val - n_val)


@wp.kernel
def simulate_tiled_franka(
        pos: wp.array2d(dtype=wp.vec3),
        vel: wp.array2d(dtype=wp.vec3),
        f: wp.array2d(dtype=wp.vec3),

        control: wp.array2d(dtype=wp.vec3),
        ctr_period: int,

        traj: wp.array3d(dtype=wp.vec3),
        record_interval: int,
        max_frames: int,

        # explicit robot states
        goal_pos0: wp.array(dtype=wp.vec3),
        goal_vel0: wp.array(dtype=wp.vec3),

        # optional logs for debugging / training
        goal_traj: wp.array2d(dtype=wp.vec3),
        real_traj: wp.array2d(dtype=wp.vec3),

        node_mass: wp.array(dtype=wp.float32),
        node_mg: wp.array(dtype=wp.float32),
        node_inv_mass: wp.array(dtype=wp.float32),

        k: float,
        damping: float,
        bending_k: float,
        bending_damping: float,
        air_drag: float,
        rest_length_arr: wp.array(dtype=wp.float32),

        # franka weights
        z0: wp.array(dtype=wp.float32, ndim=2),

        w_ir: wp.array(dtype=wp.float32, ndim=2),
        w_iz: wp.array(dtype=wp.float32, ndim=2),
        w_in: wp.array(dtype=wp.float32, ndim=2),

        w_hr: wp.array(dtype=wp.float32, ndim=2),
        w_hz: wp.array(dtype=wp.float32, ndim=2),
        w_hn: wp.array(dtype=wp.float32, ndim=2),

        b_r:  wp.array(dtype=wp.float32, ndim=2),
        b_z:  wp.array(dtype=wp.float32, ndim=2),
        b_in: wp.array(dtype=wp.float32, ndim=2),
        b_hn: wp.array(dtype=wp.float32, ndim=2),

        fc1_w_h: wp.array(dtype=wp.float32, ndim=2),
        fc1_w_x: wp.array(dtype=wp.float32, ndim=2),
        fc1_b: wp.array(dtype=wp.float32, ndim=2),

        fc2_w: wp.array(dtype=wp.float32, ndim=2),
        fc2_b: wp.array(dtype=wp.float32, ndim=2),

        fc3_w: wp.array(dtype=wp.float32, ndim=2),
        fc3_b: wp.array(dtype=wp.float32, ndim=2),

        dt: float,
        franka_dt: float,
        steps: int
):
    b = wp.tid()

    mg_tile = wp.tile_load(node_mg, P)
    inv_mass_tile = wp.tile_load(node_inv_mass, P)
    mass0 = node_mass[0]

    # robot commanded state
    gpos = goal_pos0[b]
    gvel = goal_vel0[b]

    # real ee state: loop-carried, init from rope node 0
    rpos = pos[b, 0]
    rvel = vel[b, 0]

    # initial hidden state for the franka rollout cell
    h = wp.tile_load(z0, shape=(H, ONE), offset=(0, 0))

    # time-invariant weights/biases: load once outside loop
    w_ir_t = wp.tile_load(w_ir, shape=(H, X_DIM), offset=(0, 0))
    w_iz_t = wp.tile_load(w_iz, shape=(H, X_DIM), offset=(0, 0))
    w_in_t = wp.tile_load(w_in, shape=(H, X_DIM), offset=(0, 0))

    w_hr_t = wp.tile_load(w_hr, shape=(H, H), offset=(0, 0))
    w_hz_t = wp.tile_load(w_hz, shape=(H, H), offset=(0, 0))
    w_hn_t = wp.tile_load(w_hn, shape=(H, H), offset=(0, 0))

    b_r_t  = wp.tile_load(b_r,  shape=(H, ONE), offset=(0, 0))
    b_z_t  = wp.tile_load(b_z,  shape=(H, ONE), offset=(0, 0))
    b_in_t = wp.tile_load(b_in, shape=(H, ONE), offset=(0, 0))
    b_hn_t = wp.tile_load(b_hn, shape=(H, ONE), offset=(0, 0))

    fc1_wh_t = wp.tile_load(fc1_w_h, shape=(R, H), offset=(0, 0))
    fc1_wx_t = wp.tile_load(fc1_w_x, shape=(R, X_DIM), offset=(0, 0))
    fc1_b_t = wp.tile_load(fc1_b, shape=(R, ONE), offset=(0, 0))

    fc2_w_t = wp.tile_load(fc2_w, shape=(R, R), offset=(0, 0))
    fc2_b_t = wp.tile_load(fc2_b, shape=(R, ONE), offset=(0, 0))

    fc3_w_t = wp.tile_load(fc3_w, shape=(OUT_DIM, R), offset=(0, 0))
    fc3_b_t = wp.tile_load(fc3_b, shape=(OUT_DIM, ONE), offset=(0, 0))

    rest_tile = wp.tile_load(rest_length_arr, S)

    for s in range(steps):
        idx = s // ctr_period

        # --------------------------------------------------
        # A) rope forces on current state
        # --------------------------------------------------
        # 1) gravity + air drag
        v_tile = wp.tile_load(vel[b], P)
        ga_tile = wp.tile_map(gravity_air_force_mass, v_tile, mg_tile, air_drag)
        wp.tile_store(f[b], ga_tile)

        # 2) stretch springs
        p_i = wp.tile_load(pos[b], S, offset=0)
        p_ip = wp.tile_load(pos[b], S, offset=1)
        v_i = wp.tile_load(vel[b], S, offset=0)
        v_ip = wp.tile_load(vel[b], S, offset=1)

        seg_f = wp.tile_map(
            linear_spring_damping,
            p_i, p_ip, v_i, v_ip,
            k, damping, rest_tile
        )
        wp.tile_atomic_add(f[b], seg_f, offset=0)
        wp.tile_atomic_add(f[b], -seg_f, offset=1)

        # 3) bending spring + damping
        if (bending_k != 0.0) or (bending_damping != 0.0):
            bp0 = wp.tile_load(pos[b], B, offset=0)
            bp1 = wp.tile_load(pos[b], B, offset=1)
            bp2 = wp.tile_load(pos[b], B, offset=2)

            bv0 = wp.tile_load(vel[b], B, offset=0)
            bv1 = wp.tile_load(vel[b], B, offset=1)
            bv2 = wp.tile_load(vel[b], B, offset=2)

            bendM = wp.tile_map(
                bending_spring_damper,
                bp0, bp1, bp2,
                bv0, bv1, bv2,
                bending_k, bending_damping
            )

            b0 = wp.tile_map(mat_col0, bendM)
            b1 = wp.tile_map(mat_col1, bendM)
            b2 = wp.tile_map(mat_col2, bendM)

            wp.tile_atomic_add(f[b], b0, offset=0)
            wp.tile_atomic_add(f[b], b1, offset=1)
            wp.tile_atomic_add(f[b], b2, offset=2)

        # --------------------------------------------------
        # B) Franka model: predict real ee acceleration
        #    using current rpos/rvel/gpos/gvel
        # --------------------------------------------------
        x = wp.tile_zeros(shape=(X_DIM, ONE), dtype=wp.float32)
        x[0, 0] = rpos[0]
        x[1, 0] = rpos[1]
        x[2, 0] = rpos[2]
        x[3, 0] = rvel[0]
        x[4, 0] = rvel[1]
        x[5, 0] = rvel[2]
        x[6, 0] = gpos[0]
        x[7, 0] = gpos[1]
        x[8, 0] = gpos[2]
        x[9, 0] = gvel[0]
        x[10, 0] = gvel[1]
        x[11, 0] = gvel[2]

        ir = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        iz = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        inn = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hr = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hz = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hn = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)

        wp.tile_matmul(w_ir_t, x, ir)
        wp.tile_matmul(w_iz_t, x, iz)
        wp.tile_matmul(w_in_t, x, inn)
        wp.tile_matmul(w_hr_t, h, hr)
        wp.tile_matmul(w_hz_t, h, hz)
        wp.tile_matmul(w_hn_t, h, hn)

        r = wp.tile_map(sigmoid_sum3, ir, hr, b_r_t)
        z = wp.tile_map(sigmoid_sum3, iz, hz, b_z_t)
        r_hn_b = wp.tile_map(wp.mul, r, hn + b_hn_t)
        n = wp.tile_map(tanh_fma, inn, b_in_t, r_hn_b)
        h = wp.tile_map(gru_mix, z, n, h)

        h1 = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        h1x = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        h2 = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        acc = wp.tile_zeros(shape=(OUT_DIM, ONE), dtype=wp.float32)

        wp.tile_matmul(fc1_wh_t, h, h1)
        wp.tile_matmul(fc1_wx_t, x, h1x)
        h1 = wp.tile_map(relu_sum3, h1, h1x, fc1_b_t)

        wp.tile_matmul(fc2_w_t, h1, h2)
        h2 = wp.tile_map(relu_sum, h2, fc2_b_t)

        wp.tile_matmul(fc3_w_t, h2, acc)
        acc = acc + fc3_b_t

        racc = wp.vec3(acc[0, 0], acc[1, 0], acc[2, 0])

        # --------------------------------------------------
        # C) integrate rope: node 0 from Franka, rest from forces
        # --------------------------------------------------
        f[b, 0] = racc * mass0

        f_tile = wp.tile_load(f[b], P)
        v_new = wp.tile_map(integrate_one, v_tile, f_tile, inv_mass_tile, dt)

        p_tile = wp.tile_load(pos[b], P)
        p_new = p_tile + dt * v_new

        wp.tile_store(vel[b], v_new)
        wp.tile_store(pos[b], p_new)

        # sync loop-carried real ee state from integrated node 0
        rvel = v_new[0]
        rpos = p_new[0]

        # --------------------------------------------------
        # D) update goal state from control action
        # --------------------------------------------------
        a = control[b, idx]
        gvel = gvel + dt * a
        gpos = gpos + dt * gvel

        # --------------------------------------------------
        # E) record trajectory
        # --------------------------------------------------
        if record_interval > 0:
            t = s + 1
            if (t % record_interval) == 0:
                frame = t // record_interval
                if frame <= max_frames:
                    wp.tile_store(traj[b, frame - 1], p_new)
                    goal_traj[b, frame - 1] = gpos
                    real_traj[b, frame - 1] = rpos


class WarpRopeFranka:
    def __init__(
            self,
            franka_runner,
            batch_size=1280,
            L=1.0,
            segment_lengths=None,
            mass=0.005,
            tip_extra_mass=0.0,
            k=500.0,
            damping=2.0,
            bending_k=0.0,
            bending_damping=0.0,
            air_drag=0.0,
            g=9.81,
            dt=0.001,
            max_record_steps=10000,
            record_interval=10,
            ctr_period=25,
            device="cuda"
    ):
        self.device = device
        self.franka = franka_runner
        self.batch_size = int(batch_size)
        self.L = float(L)
        self.base_mass = float(mass)
        self.tip_extra_mass = float(tip_extra_mass)
        self.k = float(k)
        self.damping = float(damping)
        self.bending_k = float(bending_k)
        self.bending_damping = float(bending_damping)
        self.air_drag = float(air_drag)
        self.g = float(g)
        self.dt = float(dt)

        self.ctr_period = int(ctr_period)
        self.record_interval = int(record_interval)
        self.max_record_steps = int(max_record_steps)
        self.max_frames = self.max_record_steps // self.record_interval

        if segment_lengths is None:
            seg_len_np = np.full((S,), self.L / float(N), dtype=np.float32)
        else:
            seg_len_np = np.asarray(segment_lengths, dtype=np.float32)
            if seg_len_np.shape != (S,):
                raise ValueError(f"segment_lengths must have shape ({S},), got {seg_len_np.shape}")

        self.segment_lengths_np = seg_len_np
        self.rest_length_arr = wp.array(seg_len_np, dtype=wp.float32, device=self.device)

        self.dx = self.L / float(N)

        self.traj = wp.zeros((self.batch_size, self.max_frames, P), dtype=wp.vec3, device=self.device)
        self.goal_traj = wp.zeros((self.batch_size, self.max_frames), dtype=wp.vec3, device=self.device)
        self.real_traj = wp.zeros((self.batch_size, self.max_frames), dtype=wp.vec3, device=self.device)
        self.f = wp.zeros((self.batch_size, P), dtype=wp.vec3, device=self.device)

        self.pos = None
        self.vel = None
        self.control = None
        self.goal_pos0 = None
        self.goal_vel0 = None
        self.real_pos0 = None
        self.real_vel0 = None

        node_mass_np = np.full((P,), self.base_mass, dtype=np.float32)
        node_mass_np[-1] += self.tip_extra_mass

        node_mg_np = self.g * node_mass_np
        node_inv_mass_np = 1.0 / node_mass_np

        self.node_mass = wp.array(node_mass_np, dtype=wp.float32, device=self.device)
        self.node_mg = wp.array(node_mg_np, dtype=wp.float32, device=self.device)
        self.node_inv_mass = wp.array(node_inv_mass_np, dtype=wp.float32, device=self.device)

    @staticmethod
    def _expand_state(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.expand(batch_size, -1).contiguous()

    def set_state_and_action(
            self,
            rope_pos,
            rope_vel,
            action,
            goal_pos,
            goal_vel,
    ):
        rope_pos = rope_pos.expand(self.batch_size, -1, -1).contiguous()
        rope_vel = rope_vel.expand(self.batch_size, -1, -1).contiguous()

        self.pos = wp.from_torch(rope_pos, dtype=wp.vec3)
        self.vel = wp.from_torch(rope_vel, dtype=wp.vec3)

        expected_timesteps = self.max_record_steps // self.ctr_period
        if action.shape[-2] != expected_timesteps:
            raise ValueError(
                f"Action timeframe mismatch: got {action.shape[-2]}, expected {expected_timesteps} "
                f"(steps={self.max_record_steps}, ctr_period={self.ctr_period})"
            )

        if action.dim() == 2:
            action = action.expand(self.batch_size, -1, -1).contiguous()
        elif action.dim() != 3:
            raise ValueError(f"Unsupported action shape: {action.shape}")
        self.control = wp.from_torch(action.contiguous(), dtype=wp.vec3)

        goal_pos = self._expand_state(goal_pos, self.batch_size)
        goal_vel = self._expand_state(goal_vel, self.batch_size)
        self.goal_pos0 = wp.from_torch(goal_pos, dtype=wp.vec3)
        self.goal_vel0 = wp.from_torch(goal_vel, dtype=wp.vec3)


    def simulate(self, steps=1000, block_dim=TILE_THREADS):
        wp.launch_tiled(
            simulate_tiled_franka,
            dim=[self.batch_size],
            inputs=[
                self.pos, self.vel, self.f,
                self.control, self.ctr_period,
                self.traj, self.record_interval, self.max_frames,
                self.goal_pos0, self.goal_vel0,
                self.goal_traj, self.real_traj,
                self.node_mass, self.node_mg, self.node_inv_mass,
                self.k, self.damping, self.bending_k, self.bending_damping,
                self.air_drag, self.rest_length_arr,

                self.franka["z0"],

                self.franka["w_ir"],
                self.franka["w_iz"],
                self.franka["w_in"],

                self.franka["w_hr"],
                self.franka["w_hz"],
                self.franka["w_hn"],

                self.franka["b_r"],
                self.franka["b_z"],
                self.franka["b_in"],
                self.franka["b_hn"],

                self.franka["fc1_w_h"],
                self.franka["fc1_w_x"],
                self.franka["fc1_b"],

                self.franka["fc2_w"],
                self.franka["fc2_b"],

                self.franka["fc3_w"],
                self.franka["fc3_b"],

                self.dt,
                float(self.franka["dt"]),
                int(steps)
            ],
            block_dim=int(block_dim),
            device=self.device
        )

        wp.synchronize()
        return {
            "rope_traj": wp.to_torch(self.traj),
            "goal_traj": wp.to_torch(self.goal_traj),
            "real_traj": wp.to_torch(self.real_traj),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    franka_runner = load_runner("best_model.pt", device="cuda")

    ctr_period = 1
    record_interval = 25
    segment_lengths = [0.098, 0.101, 0.099, 0.100, 0.102, 0.097, 0.101, 0.102]

    sim = WarpRopeFranka(
        franka_runner=franka_runner,
        batch_size=8,
        L=0.8,
        segment_lengths=segment_lengths,
        mass=12.8 / 1000 / N,
        tip_extra_mass=15 / 1000,
        k=10000 * 0.12,
        damping=0.2,
        bending_k=0.0006712,
        bending_damping=0.000401,
        air_drag=0.3006 / 1000,
        g=10.27,
        max_record_steps=10000,
        record_interval=record_interval,
        ctr_period=ctr_period,
        dt=0.001,
        device="cuda"
    )

    init_goal_pos = position_data[0, 0]
    init_goal_vel = velocity_data[0, 0]
    sim.set_state_and_action(
        rope_pos=position_data[0:1],
        rope_vel=velocity_data[0:1],
        action=control_sequence[:10000 // ctr_period],
        goal_pos=init_goal_pos,
        goal_vel=init_goal_vel,
    )

    out = sim.simulate(steps=750)
    print("Rope traj shape:", out["rope_traj"].shape)
    print("Goal traj shape:", out["goal_traj"].shape)
    print("Real traj shape:", out["real_traj"].shape)

    Speed_test = 1
    if Speed_test:
        for i in range(10):
            sim.set_state_and_action(
                rope_pos=position_data[0:1],
                rope_vel=velocity_data[0:1],
                action=control_sequence[:10000 // ctr_period],
                goal_pos=init_goal_pos,
                goal_vel=init_goal_vel,
            )
            out = sim.simulate(steps=750)
        wp.synchronize()
        t0 = time.perf_counter()
        n = 100
        for i in range(n):
            sim.set_state_and_action(
                rope_pos=position_data[0:1],
                rope_vel=velocity_data[0:1],
                action=control_sequence[:10000 // ctr_period],
                goal_pos=init_goal_pos,
                goal_vel=init_goal_vel,
            )
            out = sim.simulate(steps=750)
        wp.synchronize()
        t1 = time.perf_counter()
        print(f"Average rollout time: {(t1 - t0) / n * 1000:.8f} ms")
