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
    # 对应 warp2 弹簧段力计算 :contentReference[oaicite:1]{index=1}
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
    # 对齐 rope.py _bending_spring_force 的计算：:contentReference[oaicite:3]{index=3}
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
    control: wp.array(dtype=wp.vec3),

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

        # 0) 清零 f
        f_zero = wp.tile_zeros(shape=(P,), dtype=wp.vec3)
        wp.tile_store(f[b], f_zero)

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

            # --- bending spring ---
            if bending_k != 0.0:
                bendKM = wp.tile_map(bending_spring, bp0, bp1, bp2, bending_k)
                bk0 = wp.tile_map(mat_col0, bendKM)
                bk1 = wp.tile_map(mat_col1, bendKM)
                bk2 = wp.tile_map(mat_col2, bendKM)
                wp.tile_atomic_add(f[b], bk0, offset=0)
                wp.tile_atomic_add(f[b], bk1, offset=1)
                wp.tile_atomic_add(f[b], bk2, offset=2)

            # --- bending damping (你现有的) ---
            if bending_damping != 0.0:
                bv0 = wp.tile_load(vel[b], B, offset=0)
                bv1 = wp.tile_load(vel[b], B, offset=1)
                bv2 = wp.tile_load(vel[b], B, offset=2)

                bendDM = wp.tile_map(bending_damper, bp0, bp1, bp2, bv0, bv1, bv2, bending_damping)
                bf0 = wp.tile_map(mat_col0, bendDM)
                bf1 = wp.tile_map(mat_col1, bendDM)
                bf2 = wp.tile_map(mat_col2, bendDM)
                wp.tile_atomic_add(f[b], bf0, offset=0)
                wp.tile_atomic_add(f[b], bf1, offset=1)
                wp.tile_atomic_add(f[b], bf2, offset=2)

        # 4) 控制作用到第 0 粒子（warp2：if i==0 f[b,0]=control[b]*mass）:contentReference[oaicite:4]{index=4}
        f[b, 0] = control[b] * mass

        # 5) 积分（显式欧拉，warp2 同款）:contentReference[oaicite:5]{index=5}
        f_tile = wp.tile_load(f[b], P)
        v_new = v_tile + (dt / mass) * f_tile
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
                 max_record_steps=10000, record_interval=10, ctr_freq=25):

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

        self.ctr_freq = ctr_freq

        self.record_interval = int(record_interval)
        self.max_record_steps = int(max_record_steps)
        self.max_frames = self.max_record_steps // self.record_interval
        # traj: (B, frames, P)
        self.traj = wp.zeros((self.batch_size, self.max_frames, P), dtype=wp.vec3, device="cuda")

        # ✅ 推荐布局：(B, P) of vec3
        self.pos = wp.zeros((self.batch_size, P), dtype=wp.vec3, device="cuda")
        self.vel = wp.zeros((self.batch_size, P), dtype=wp.vec3, device="cuda")
        self.f   = wp.zeros((self.batch_size, P), dtype=wp.vec3, device="cuda")
        self.control = wp.zeros((self.batch_size,), dtype=wp.vec3, device="cuda")

    def set_state(self, pos, vel):
        pos = pos.expand(self.batch_size, -1, -1).contiguous()
        self.pos = wp.from_torch(pos, dtype=wp.vec3)
        vel = vel.expand(self.batch_size, -1, -1).contiguous()
        self.vel = wp.from_torch(vel, dtype=wp.vec3)

    def simulate(self, steps=1000, block_dim=TILE_THREADS):

        wp.launch_tiled(
            simulate_tiled,
            dim=[self.batch_size],
            inputs=[
                self.pos, self.vel, self.f, self.control,
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
    sampled_data = np.load('../data/fixed_tip_horizontal_init.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    sim = WarpRope(
        batch_size=1280,
        L=1.0,
        mass=0.0025 * 40 / N,
        k=10000 * 0.46,
        damping=0.2,
        bending_k=0.0006712,
        bending_damping=0.000401,   # warp2 的 damping_bend :contentReference[oaicite:6]{index=6}
        air_drag=0.2206 / 1000,
        g=10.07,
        max_record_steps=10000,
        record_interval=10,
        ctr_freq=25,
        dt=0.001
    )

    sim.set_state(position_data[0:1], velocity_data[0:1])

    traj_warp = sim.simulate(steps=10000)
    print("Trajectory shape:", traj_warp.shape)
    dt = 0.001
    L = 1.0
    record_interval = 10
    # plot_animation_3d(traj_warp, dt, record_interval, L, repeat=True, batch_idx=0)

    model_compare = 1
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
        record_interval = 10

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
        n = 10
        steps = 10000
        t0 = time.perf_counter()
        for _ in range(n):
            sim.simulate(steps=steps)
        wp.synchronize()
        t1 = time.perf_counter()
        print("Average time:", (t1 - t0) / n)
