import time
import numpy as np
import torch
import warp as wp
# from rope_real import *
wp.init()
# ===== Fixed rope size =====
N = 5
P = N + 1  # nodes
S = N      # spring segments
B = N - 1  # bending triplets
TILE_THREADS = 32

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
def bending_spring(
        p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
        bending_k: float
) -> wp.mat33:
    dx1 = p1 - p0
    l1 = wp.length(dx1)
    d1 = dx1 / (l1 + 1.0e-8)
    dx2 = p2 - p1
    l2 = wp.length(dx2)
    d2 = dx2 / (l2 + 1.0e-8)
    c = wp.dot(d1, d2)
    c = wp.clamp(c, -1.0 + 1.0e-8, 1.0 - 1.0e-8)
    beta = wp.acos(c)
    cr = wp.cross(d1, d2)
    sb = wp.sin(beta)
    scale = bending_k * beta / (sb + 1.0e-8)
    common = scale * cr
    pre = -(wp.cross(d1, common)) / (l1 + 1.0e-8)
    aft = -(wp.cross(d2, common)) / (l2 + 1.0e-8)
    F0 = -pre
    F1 = pre + aft
    F2 = -aft
    return wp.matrix_from_cols(F0, F1, F2)

@wp.func
def bending_damper(
        p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
        v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
        bending_damping: float
) -> wp.mat33:
    dx1 = p1 - p0
    l1 = wp.length(dx1)
    d1 = dx1 / (l1 + 1.0e-8)
    dx2 = p2 - p1
    l2 = wp.length(dx2)
    d2 = dx2 / (l2 + 1.0e-8)
    c1 = wp.cross(d1, d2)
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
    F1 = tmp * dbeta_de1
    F2 = -tmp * dbeta_de2
    F0 = F1
    Fmid = -(F1 + F2)
    F2p = F2
    return wp.matrix_from_cols(F0, Fmid, F2p)

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

@wp.kernel
def simulate_tiled(
        pos: wp.array2d(dtype=wp.vec3),
        vel: wp.array2d(dtype=wp.vec3),
        f: wp.array2d(dtype=wp.vec3),
        control: wp.array2d(dtype=wp.vec3),
        ctr_period: int,
        mode: int,  # 0=acc, 1=vel
        traj: wp.array3d(dtype=wp.vec3),  # (B, frames, P)
        record_interval: int,
        max_frames: int,
        node_mass: wp.array(dtype=wp.float32),
        node_mg: wp.array(dtype=wp.float32),
        node_inv_mass: wp.array(dtype=wp.float32),
        k: float,
        damping: float,
        bending_k: float,
        bending_damping: float,
        air_drag: float,
        rest_length_arr: wp.array(dtype=wp.float32),
        dt: float,
        steps: int
):
    b = wp.tid()
    mg_tile = wp.tile_load(node_mg, P)
    inv_mass_tile = wp.tile_load(node_inv_mass, P)
    mass0 = node_mass[0]
    for s in range(steps):
        idx = s // ctr_period
        if mode != 0:  # vel mode
            vel[b, 0] = control[b, idx]
        # 1) gravity + air drag
        v_tile = wp.tile_load(vel[b], P)
        ga_tile = wp.tile_map(gravity_air_force_mass, v_tile, mg_tile, air_drag)
        wp.tile_store(f[b], ga_tile)
        # 2) stretch springs
        p_i = wp.tile_load(pos[b], S, offset=0)
        p_ip = wp.tile_load(pos[b], S, offset=1)
        v_i = wp.tile_load(vel[b], S, offset=0)
        v_ip = wp.tile_load(vel[b], S, offset=1)
        rest_tile = wp.tile_load(rest_length_arr, S)
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
        # 4) control on node 0
        if mode == 0:  # acc mode
            f[b, 0] = control[b, idx] * mass0
        else:  # vel mode
            f[b, 0] = wp.vec3(0.0, 0.0, 0.0)
        # 5) explicit Euler integration with per-node inverse mass
        f_tile = wp.tile_load(f[b], P)
        v_new = wp.tile_map(integrate_one, v_tile, f_tile, inv_mass_tile, dt)
        p_tile = wp.tile_load(pos[b], P)
        p_new = p_tile + dt * v_new
        wp.tile_store(vel[b], v_new)
        wp.tile_store(pos[b], p_new)
        # 6) record trajectory
        if record_interval > 0:
            t = s + 1
            if (t % record_interval) == 0:
                frame = t // record_interval
                if frame <= max_frames:
                    wp.tile_store(traj[b, frame - 1], p_new)

class WarpRope:
    def __init__(
            self,
            batch_size=1280,
            L=1.0,
            segment_lengths=None,   # only this becomes per-segment
            mass=0.005,             # base mass for each node
            tip_extra_mass=0.0,     # extra mass attached to last node
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
            mode='acc',
            device="cuda"
    ):
        self.device = device
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
        self.mode = mode
        self.record_interval = int(record_interval)
        self.max_record_steps = int(max_record_steps)
        self.max_frames = self.max_record_steps // self.record_interval
        # per-segment rest lengths
        if segment_lengths is None:
            seg_len_np = np.full((S,), self.L / float(N), dtype=np.float32)
        else:
            seg_len_np = np.asarray(segment_lengths, dtype=np.float32)
            if seg_len_np.shape != (S,):
                raise ValueError(f"segment_lengths must have shape ({S},), got {seg_len_np.shape}")
        self.segment_lengths_np = seg_len_np
        self.rest_length_arr = wp.array(seg_len_np, dtype=wp.float32, device=self.device)
        # keep this only for compatibility/debug
        self.dx = self.L / float(N)
        self.traj = wp.zeros((self.batch_size, self.max_frames, P), dtype=wp.vec3, device=self.device)
        self.f = wp.zeros((self.batch_size, P), dtype=wp.vec3, device=self.device)
        self.pos = None
        self.vel = None
        self.control = None
        # per-node mass arrays
        node_mass_np = np.full((P,), self.base_mass, dtype=np.float32)
        node_mass_np[-1] += self.tip_extra_mass
        node_mg_np = self.g * node_mass_np
        node_inv_mass_np = 1.0 / node_mass_np
        self.node_mass = wp.array(node_mass_np, dtype=wp.float32, device=self.device)
        self.node_mg = wp.array(node_mg_np, dtype=wp.float32, device=self.device)
        self.node_inv_mass = wp.array(node_inv_mass_np, dtype=wp.float32, device=self.device)
    def set_state_and_action(self, pos, vel, action):
        pos = pos.expand(self.batch_size, -1, -1).contiguous()
        vel = vel.expand(self.batch_size, -1, -1).contiguous()
        self.pos = wp.from_torch(pos, dtype=wp.vec3)
        self.vel = wp.from_torch(vel, dtype=wp.vec3)
        expected_timesteps = self.max_record_steps // self.ctr_period
        if action.shape[-2] != expected_timesteps:
            raise ValueError(
                f"Action timeframe mismatch: got {action.shape[-2]}, expected {expected_timesteps} "
                f"(steps={self.max_record_steps}, ctr_period={self.ctr_period})"
            )
        if action.dim() == 2:  # (T, 3)
            action = action.expand(self.batch_size, -1, -1).contiguous()
            self.control = wp.from_torch(action, dtype=wp.vec3)
        elif action.dim() == 3:  # (B, T, 3)
            self.control = wp.from_torch(action.contiguous(), dtype=wp.vec3)
        else:
            raise ValueError(f"Unsupported action shape: {action.shape}")
    def simulate(self, steps=1000, block_dim=TILE_THREADS):
        mode_int = 0 if self.mode == 'acc' else 1
        wp.launch_tiled(
            simulate_tiled,
            dim=[self.batch_size],
            inputs=[
                self.pos, self.vel, self.f,
                self.control, self.ctr_period, mode_int,
                self.traj, self.record_interval, self.max_frames,
                self.node_mass, self.node_mg, self.node_inv_mass,
                self.k, self.damping, self.bending_k, self.bending_damping,
                self.air_drag, self.rest_length_arr,
                self.dt, int(steps)
            ],
            block_dim=int(block_dim),
            device=self.device
        )
        wp.synchronize()
        return wp.to_torch(self.traj)  # (B, frames, P, 3)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)
    ctr_period = 1
    record_interval = 25
    # example: each segment has slightly different rest length
    segment_lengths = [0.098, 0.101, 0.099, 0.100, 0.102, 0.097, 0.101, 0.102]
    sim = WarpRope(
        batch_size=1,
        L=0.8,
        segment_lengths=segment_lengths,
        mass=12.8 / 1000 / N,   # base node mass
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
        mode='vel',
        dt=0.001,
        device="cuda"
    )
    sim.set_state_and_action(
        position_data[0:1],
        velocity_data[0:1],
        control_sequence[:10000 // ctr_period]
    )
    traj_warp = sim.simulate(steps=10000)
    print("Trajectory shape:", traj_warp.shape)
    dt = 0.001
    L = 1.0
    model_compare = 1
    if model_compare:
        # === Parameters ===
        N = 8
        L = torch.tensor(segment_lengths)
        mass = [12.8 / 8 / 1000] * N + [(12.8 / 8 + 15) / 1000]
        k = 0.12
        damping = [0.2] * N
        k_bend = [0.0006712] * (N - 1)
        damping_bend = [0.000401] * (N - 1)
        twisting = 0.0
        air_drag = 0.3006
        g = 10.27
        dt = 0.001
        T = 10.0
        total_steps = int(T / dt)
        # === Create model instance ===
        model = Rope(
            N=N,
            L=L,
            mass=mass,
            k=k,
            k_bend=k_bend,
            damping=damping,
            damping_bend=damping_bend,
            twisting=twisting,
            air_drag=air_drag,
            g=g,
            dt=dt,
            device=device,
            mode='id'
        )
        print("here")
        show_start = 0
        start = time.time()
        with torch.no_grad():
            traj_ref = model.simulate(
                position_data[show_start:show_start + 1],
                velocity_data[show_start:show_start + 1],
                control_sequence[show_start:].unsqueeze(0),
                steps=int(10000),
                mode='vel',
                record_interval=record_interval
            )
        print("done!")
        error = (traj_ref[0].cpu() - traj_warp[0].cpu()).abs()
        print("Error: ", error.sum())
        plot_animation_two_ropes_3d_2(traj_ref, traj_warp, dt, record_interval, L, repeat=True)
    test_time = 1
    if test_time:
        sim.simulate(steps=10)
        wp.synchronize()
        n = 100
        steps = 750
        t0 = time.perf_counter()
        for _ in range(n):
            sim.set_state_and_action(
                position_data[0:1],
                velocity_data[0:1],
                control_sequence[:10000 // ctr_period]
            )
            traj_warp = sim.simulate(steps=steps)
        wp.synchronize()
        t1 = time.perf_counter()
        print("Average time:", (t1 - t0) / n)
 