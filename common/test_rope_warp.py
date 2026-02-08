import warp as wp
import numpy as np

wp.init()

@wp.kernel
def rope_step(
    x: wp.array(dtype=wp.vec3f),   # 位置 (batch, N+1)
    v: wp.array(dtype=wp.vec3f),   # 速度 (batch, N+1)
    f: wp.array(dtype=wp.vec3f),   # 力 (batch, N+1)
    control_force: wp.array(dtype=wp.vec3f), # (batch,)
    lengths: wp.array(dtype=float), # (batch, N)
    directions: wp.array(dtype=wp.vec3f), # (batch, N)
    N: int, dx: float, mass: float, k: float, damping: float, damping_bend: float,
    air_drag: float, g: float, dt: float
):
    b, i = wp.tid()  # batch, node

    # 清零力
    if i < N+1:
        f[b, i] = wp.vec3f(0.0, 0.0, -mass * g) - air_drag * v[b, i]

    # 弹簧力
    if i < N:
        delta = x[b, i+1] - x[b, i]
        length = wp.length(delta) + 1e-8
        dir = delta / length
        lengths[b, i] = length
        directions[b, i] = dir

        rel_vel = v[b, i+1] - v[b, i]
        proj = wp.dot(rel_vel, dir)
        spring_force = k * (length - dx) * dir
        damp_force = damping * proj * dir

        wp.atomic_add(f, (b, i), spring_force + damp_force)
        wp.atomic_add(f, (b, i+1), -(spring_force + damp_force))

    # 控制力
    if i == 0:
        f[b, 0] += control_force[b] * mass

    # TODO: 弯曲阻尼力（可仿照 Taichi 代码实现）

    # 积分
    if i < N+1:
        v[b, i] += dt * f[b, i] / mass
        x[b, i] += dt * v[b, i]

def simulate_rope(
    batch_size, N, steps, **params
):
    # 分配数据
    x = wp.zeros((batch_size, N+1), dtype=wp.vec3f, device="cuda")
    v = wp.zeros((batch_size, N+1), dtype=wp.vec3f, device="cuda")
    f = wp.zeros((batch_size, N+1), dtype=wp.vec3f, device="cuda")
    control_force = wp.zeros((batch_size,), dtype=wp.vec3f, device="cuda")
    lengths = wp.zeros((batch_size, N), dtype=float, device="cuda")
    directions = wp.zeros((batch_size, N), dtype=wp.vec3f, device="cuda")

    # 初始化
    for b in range(batch_size):
        for i in range(N+1):
            x_host = np.array([b*0.1, 0.0, -i*params['dx']], dtype=np.float32)
            x[b, i] = wp.vec3f(*x_host)
            v[b, i] = wp.vec3f(0.0, 0.0, 0.0)
        control_force[b] = wp.vec3f(0.0, 0.0, 0.0)

    # 仿真主循环
    for step in range(steps):
        wp.launch(
            kernel=rope_step,
            dim=(batch_size, N+1),
            inputs=[x, v, f, control_force, lengths, directions,
                    N, params['dx'], params['mass'], params['k'], params['damping'],
                    params['damping_bend'], params['air_drag'], params['g'], params['dt']],
            device="cuda"
        )

    # 拷回CPU
    x_host = x.numpy()
    return x_host

# 参数设置
N = 20
batch_size = 1280
steps = 10000
params = dict(
    N=N,
    dx=1.0/N,
    mass=0.0025*40/N,
    k=0.46*10000,
    damping=0.2,
    damping_bend=0.000401,
    air_drag=0.2206/1000,
    g=10.07,
    dt=0.001
)

# 运行仿真
x_result = simulate_rope(batch_size, N, steps, **params)
print("x_result shape:", x_result.shape)