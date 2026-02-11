import time
import numpy as np
import torch
import warp as wp
from common.rope import *

wp.init()

# ===== 固定 N（如果你 N 不变，这样最简单也最快）=====
N = 20
P = N + 1      # particles
S = N          # spring segments: i=0..N-1
B = N - 1      # bending triplets: i=0..N-2

TILE_THREADS = 32


@wp.func
def gravity_air_force(v: wp.vec3, mg: float, air_drag: float) -> wp.vec3:
    return -(wp.vec3(0.0, 0.0, mg) + air_drag * v)


@wp.func
def linear_spring_damping(pi: wp.vec3, pj: wp.vec3, vi: wp.vec3, vj: wp.vec3,
                         k: float, damping: float, rest_length: float) -> wp.vec3:

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

    # beta = acos(clamp(dot(d1,d2)))
    c = wp.dot(d1, d2)
    c = wp.clamp(c, -1.0 + 1.0e-8, 1.0 - 1.0e-8)
    beta = wp.acos(c)

    cr = wp.cross(d1, d2)

    # common = (k * beta / (sin(beta)+eps)) * cross
    sb = wp.sin(beta)
    scale = bending_k * beta / (sb + 1.0e-8)
    common = scale * cr

    # pre = -cross(d1, common) / l1
    # aft = -cross(d2, common) / l2
    pre = -(wp.cross(d1, common)) / (l1 + 1.0e-8)
    aft = -(wp.cross(d2, common)) / (l2 + 1.0e-8)

    # rope.py 的散射方式：
    # i:   -pre
    # i+1: pre + aft
    # i+2: -aft
    F0 = -pre
    F1 = pre + aft
    F2 = -aft

    return wp.matrix_from_cols(F0, F1, F2)


@wp.func
def bending_damper(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
                   v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
                   bending_damping: float) -> wp.mat33:
    """
    返回一个 3x3 矩阵，把 (F0,F1,F2) 打包在 mat33 的三列里：
      col0 = F0, col1 = F1, col2 = F2
    这样 tile_map 的返回类型是内建 mat33（支持），避免自定义 struct。
    """
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

    # mat33 的列向量
    return wp.matrix_from_cols(F0, Fmid, F2p)


@wp.func
def bending_spring_damper(
    p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
    bending_k: float, bending_damping: float
) -> wp.mat33:

    # --- shared geometry (exactly same ops as your funcs) ---
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

    # --- spring part (same as bending_spring) ---
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

    # --- damping part (same as bending_damper) ---
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
    # 取第0列
    return wp.vec3(m[0, 0], m[1, 0], m[2, 0])


@wp.func
def mat_col1(m: wp.mat33) -> wp.vec3:
    return wp.vec3(m[0, 1], m[1, 1], m[2, 1])


@wp.func
def mat_col2(m: wp.mat33) -> wp.vec3:
    return wp.vec3(m[0, 2], m[1, 2], m[2, 2])


@wp.kernel
def simulate_tiled(
    pos: wp.array2d(dtype=wp.vec3),
    vel: wp.array2d(dtype=wp.vec3),
    f:   wp.array2d(dtype=wp.vec3),
    
    control: wp.array2d(dtype=wp.vec3),
    ctr_period: int,
    mode: int,   # 0=acc, 1=vel

    traj: wp.array3d(dtype=wp.vec3),   # (B, frames, P)
    record_interval: int,
    max_frames: int,

    mg: float,
    k: float,
    damping: float,
    bending_k: float,
    bending_damping: float,
    air_drag: float,
    rest_length: float,
    dt: float,
    mass: float,
    steps: int
):
    b = wp.tid()  # launch_tiled(dim=[B])

    for s in range(steps):

        idx = s // ctr_period

        if mode != 0:  # vel
            vel[b, 0] = control[b, idx]

        # 1) 重力 + 空阻（这里不用 tile_map，直接用 tile 算）
        v_tile = wp.tile_load(vel[b], P)
        ga_tile = wp.tile_map(gravity_air_force, v_tile, mg, air_drag)
        wp.tile_store(f[b], ga_tile)

        # 2) 弹簧段力（S 段）并 scatter 到端点（atomic）
        p_i  = wp.tile_load(pos[b], S, offset=0)   # i
        p_ip = wp.tile_load(pos[b], S, offset=1)   # i+1
        v_i  = wp.tile_load(vel[b], S, offset=0)
        v_ip = wp.tile_load(vel[b], S, offset=1)

        seg_f = wp.tile_map(linear_spring_damping, p_i, p_ip, v_i, v_ip, k, damping, rest_length)
        wp.tile_atomic_add(f[b],  seg_f,  offset=0)
        wp.tile_atomic_add(f[b], -seg_f,  offset=1)

        # 3) 弯曲弹簧力 + 弯曲阻尼（B 组：i=0..N-2），scatter 到 i/i+1/i+2（atomic）
        if (bending_k != 0.0) or (bending_damping != 0.0):
            bp0 = wp.tile_load(pos[b], B, offset=0)
            bp1 = wp.tile_load(pos[b], B, offset=1)
            bp2 = wp.tile_load(pos[b], B, offset=2)

            bv0 = wp.tile_load(vel[b], B, offset=0)
            bv1 = wp.tile_load(vel[b], B, offset=1)
            bv2 = wp.tile_load(vel[b], B, offset=2)

            bendM = wp.tile_map(bending_spring_damper, bp0, bp1, bp2, bv0, bv1, bv2, bending_k, bending_damping)

            b0 = wp.tile_map(mat_col0, bendM)
            b1 = wp.tile_map(mat_col1, bendM)
            b2 = wp.tile_map(mat_col2, bendM)
            wp.tile_atomic_add(f[b], b0, offset=0)
            wp.tile_atomic_add(f[b], b1, offset=1)
            wp.tile_atomic_add(f[b], b2, offset=2)

        # 4) 控制作用到第 0 粒子
        if mode == 0:  # acc
            f[b, 0] = control[b, idx] * mass
        else:  # vel
            f[b, 0] = wp.vec3(0.0, 0.0, 0.0)

        # 5) 积分 (显式欧拉)
        f_tile = wp.tile_load(f[b], P)
        v_new = v_tile + (dt / mass) * f_tile

        # if mode != 0: # vel (to keep the same with rope.py)
        #     v_new[0] = control[b, idx]

        p_tile = wp.tile_load(pos[b], P)
        p_new = p_tile + dt * v_new

        wp.tile_store(vel[b], v_new)
        wp.tile_store(pos[b], p_new)

        # ---- 记录：每隔 interval 写一次 ----
        if record_interval > 0:
            t = s + 1
            if (t % record_interval) == 0:
                frame = t // record_interval  # 1,2,3...
                if frame <= max_frames:
                    wp.tile_store(traj[b, frame-1], p_new)  # 不计算初始pos


class WarpRope:
    def __init__(self, batch_size=1280, L=1.0, mass=0.005, k=500.0, damping=2.0, bending_k=0.0,
                 bending_damping=0.0, air_drag=0.0, g=9.81, dt=0.001,
                 max_record_steps=10000, record_interval=10, ctr_period=25, mode='acc'):

        self.batch_size = int(batch_size)
        self.L = float(L)
        self.mass = float(mass)
        self.k = float(k)
        self.damping = float(damping)
        self.bending_k = float(bending_k)
        self.bending_damping = float(bending_damping)
        self.air_drag = float(air_drag)
        self.mg = float(g) * self.mass
        self.dt = float(dt)
        self.dx = self.L / float(N)  # N 固定

        self.ctr_period = ctr_period
        self.mode = mode

        self.record_interval = int(record_interval)
        self.max_record_steps = int(max_record_steps)  # make sure: max_record_steps = 运行的steps
        self.max_frames = self.max_record_steps // self.record_interval
        # traj: (B, frames, P)
        self.traj = wp.zeros((self.batch_size, self.max_frames, P), dtype=wp.vec3, device="cuda")

        # ✅ 推荐布局：(B, P) of vec3
        self.pos = None
        self.vel = None
        self.f   = wp.zeros((self.batch_size, P), dtype=wp.vec3, device="cuda")
        self.control = None

    def set_state_and_action(self, pos, vel, action):
        pos = pos.expand(self.batch_size, -1, -1).contiguous()
        self.pos = wp.from_torch(pos, dtype=wp.vec3)
        vel = vel.expand(self.batch_size, -1, -1).contiguous()
        self.vel = wp.from_torch(vel, dtype=wp.vec3)
        
        # Check dimension
        expected_timesteps = self.max_record_steps // self.ctr_period
        if action.shape[-2] != expected_timesteps:
            raise ValueError(
                f"Action timeframe mismatch: got {action.shape[-2]}, expected {expected_timesteps} "
                f"(steps={self.max_record_steps}, ctr_period={self.ctr_period})"
            )
        if action.dim() == 2:  # (T, 3)
            action = action.expand(self.batch_size, -1,-1).contiguous()
            self.control = wp.from_torch(action, dtype=wp.vec3)
        elif action.dim() == 3:  # (B, T, 3)
            self.control = wp.from_torch(action.contiguous(), dtype=wp.vec3)

    def simulate(self, steps=1000, block_dim=TILE_THREADS):

        if self.mode == 'acc':
            mode_int = 0
        else:
            mode_int = 1

        wp.launch_tiled(
            simulate_tiled,
            dim=[self.batch_size],
            inputs=[
                self.pos, self.vel, self.f, self.control, self.ctr_period, mode_int,
                self.traj, self.record_interval, self.max_frames,
                self.mg, self.k, self.damping, self.bending_k, self.bending_damping,
                self.air_drag, self.dx,
                self.dt, self.mass,
                int(steps)
            ],
            block_dim=int(block_dim),
            device="cuda"
        )

        wp.synchronize()
        return wp.to_torch(self.traj)  # (B, frames, P, 3)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # sampled_data = np.load('../data/fixed_tip_pos_high.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_low.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_horizontal_init.npy').astype(np.float32)
    sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/static_init_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    ctr_period = 1
    record_interval = 25
    sim = WarpRope(
        batch_size=1280,
        L=1.0,
        mass=0.0025 * 40 / N,
        k=10000 * 0.46,
        damping=0.2,
        bending_k=0.0006712,
        bending_damping=0.000401,
        air_drag=0.2206 / 1000,
        g=10.07,
        max_record_steps=750,
        record_interval=record_interval,
        ctr_period=ctr_period,
        mode='vel',
        dt=0.001
    )

    sim.set_state_and_action(position_data[0:1], velocity_data[0:1], control_sequence[:750//ctr_period])
    traj_warp = sim.simulate(steps=750)
    print("Trajectory shape:", traj_warp.shape)

    dt = 0.001
    L = 1.0
    # plot_animation_3d(traj_warp, dt, record_interval, L, repeat=True, batch_idx=0)

    model_compare = 0
    if model_compare:
        # === Parameters ===
        N = 20  # Number of segments
        L = 1.0  # Total length (m)
        mass = 0.0025 * 40 / N  # Mass per segment (kg)
        k = 0.46  # Spring stiffness
        damping = [0.2] * N  # Damping coefficient
        k_bend = [0.0006712] * (N - 1)  # Bending stiffness
        damping_bend = [0.000401] * (N - 1)  # Bending damping coefficient
        twisting = 0.00002
        air_drag = 0.2206  # Drag coefficient
        g = 10.07  # Gravitational acceleration
        dt = 0.001  # Time step (s)
        T = 10.0  # Simulation duration (s)
        total_steps = int(T / dt)

        # === Create model instance ===
        model = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, twisting=twisting,
                     air_drag=air_drag,
                     g=g, dt=dt, device=device, mode='id')

        # === show before id ===
        print("here")
        show_start = 0
        # with torch.no_grad():  # Disable gradient computation
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

        error = (traj_ref[0].cpu() - traj_warp[878].cpu()).abs()
        print("Error: ", error.sum())

        plot_animation_two_ropes_3d_2(traj_ref, traj_warp, dt, record_interval, L, repeat=True)

    test_time = 1
    if test_time:
        # warmup
        sim.simulate(steps=10)
        wp.synchronize()

        # benchmark
        n = 100
        steps = 750
        t0 = time.perf_counter()
        for _ in range(n):
            sim.set_state_and_action(position_data[0:1], velocity_data[0:1], control_sequence[:750 // ctr_period])
            traj_warp = sim.simulate(steps=steps)
        wp.synchronize()
        t1 = time.perf_counter()
        print("Average time:", (t1 - t0) / n)
