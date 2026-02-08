import warp as wp
from common.rope import *

wp.init()


class WarpRopeSim:
    def __init__(self, batch_size=1280, N=20, L=1.0, mass=0.005, k=500.0, damping=2.0,
                 bending_k=0.0, bending_damping=0.0, air_drag=0.0, g=9.81,
                 dt=0.001, max_record_steps=1000, record_interval=10):

        self.batch_size = batch_size
        self.N = N
        self.L = L
        self.mass = mass
        self.k = k
        self.damping = damping
        self.bending_k = bending_k
        self.bending_damping = bending_damping
        self.air_drag = air_drag
        self.mg = g * mass
        self.dt = dt
        self.dx = L / N
        self.record_interval = record_interval
        self.max_record_steps = max_record_steps
        self.max_saved_frames = max_record_steps // record_interval

        # Particle state as 2D tensors: [batch, N+1, 3]
        shape = (batch_size, N + 1, 3)
        self.pos = wp.array(np.zeros((batch_size, N + 1, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")  # 2d
        self.vel = wp.array(np.zeros((batch_size, N + 1, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")  # 2d
        self.f = wp.array(np.zeros((batch_size, N + 1, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")  # 2d
        self.control = wp.array(np.zeros((batch_size, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")  # 1d

        # Init positions
        pos_np = np.zeros(shape, dtype=np.float32)
        for b in range(batch_size):
            for i in range(N + 1):
                pos_np[b, i] = [b * 0.1, 0.0, -i * self.dx]
        self.q = wp.array(pos_np, dtype=wp.vec3, device="cuda")

        # Trajectory storage (on host)
        self.trajectory = torch.zeros((batch_size, self.max_saved_frames, N + 1, 3), dtype=torch.float32, device="cuda")

    @wp.kernel
    def apply_basic_force(f: wp.array2d(dtype=wp.vec3),
                      pos: wp.array2d(dtype=wp.vec3),
                      vel: wp.array2d(dtype=wp.vec3),
                      mg: float, k: float, damping: float, bending_damping: float,
                      air_drag: float, rest_length: float,
                      N: int):

        b, i = wp.tid()
        n = wp.float32(1.0)
        for _ in range(int(n)):
            f[b, i] -= (wp.vec3(0.0, 0.0, mg) + air_drag * vel[b, i]) / n

            if i < N:
                dx1 = pos[b, i + 1] - pos[b, i]
                l1 = wp.length(dx1)
                d1 = dx1 / (l1 + 1e-8)

                rel_vel = vel[b, i + 1] - vel[b, i]
                proj_vel = wp.dot(rel_vel, d1)

                spring_force = (k * (l1 - rest_length) + damping * proj_vel) * d1

                f[b, i] += spring_force / n
                f[b, i+1] -= spring_force / n

            if i < N-1:
                dx2 = pos[b, i + 2] - pos[b, i + 1]
                l2 = wp.length(dx2)
                d2 = dx2 / (l2 + 1e-8)

                c1 = wp.cross(d1, d2)

                grad_d1 = wp.cross(d1, c1)
                norm_grad_d1 = wp.length(grad_d1)
                dbeta_de1 = grad_d1 / (norm_grad_d1 * l1 + 1e-8)

                grad_d2 = wp.cross(d2, -c1)
                norm_grad_d2 = wp.length(grad_d2)
                dbeta_de2 = grad_d2 / (norm_grad_d2 * l2 + 1e-8)

                vel_diff1 = vel[b, i + 1] - vel[b, i]
                vel_diff2 = vel[b, i + 2] - vel[b, i + 1]
                dbeta_dt = wp.dot(dbeta_de1, vel_diff1) + wp.dot(dbeta_de2, vel_diff2)

                tmp = bending_damping * dbeta_dt
                F1 = tmp * dbeta_de1
                F2 = -tmp * dbeta_de2

                f[b, i] += F1 / n  # inline 形式默认原子操作
                f[b, i + 1] -= (F1 + F2) / n
                f[b, i + 2] += F2 / n

    @wp.kernel
    def control_force_and_integrate(control: wp.array(dtype=wp.vec3), pos: wp.array2d(dtype=wp.vec3),
                                    vel: wp.array2d(dtype=wp.vec3), f: wp.array2d(dtype=wp.vec3),
                                    dt: float, mass: float):
        b, i = wp.tid()
        n = wp.float32(1.0)
        for _ in range((int(n))):
            if i == 0:
                f[b, i] = control[b] * mass

            vel[b, i] += dt * f[b, i] / mass / n
            pos[b, i] += dt * vel[b, i] / n

    def step(self):
        self.f.zero_()

        wp.launch(self.apply_basic_force,
                  dim=(self.batch_size, self.N + 1),
                  inputs=[
                      self.f, self.pos, self.vel,
                      self.mg, self.k, self.damping, self.bending_damping,
                      self.air_drag, self.dx, self.N
                  ])

        wp.launch(self.control_force_and_integrate, dim=(self.batch_size, self.N + 1),
                  inputs=[self.control, self.pos, self.vel, self.f, self.dt, self.mass])

    def simulate(self, steps=1000):
        frame_id = 0
        # start = time.perf_counter()
        self.trajectory[:, frame_id] = wp.to_torch(self.pos)
        # end = time.perf_counter()
        # print('Time to save data: ', end-start)
        frame_id += 1

        for step in range(steps):
            self.step()

            if (step + 1) % self.record_interval == 0 and frame_id < self.max_saved_frames:
                self.trajectory[:, frame_id] = wp.to_torch(self.pos)
                frame_id += 1

    def set_pos(self, pos):
        pos = pos.expand(self.batch_size, -1, -1).contiguous()
        self.pos = wp.from_torch(pos, dtype=wp.vec3)

    def get_trajectory(self):
        return self.trajectory


if __name__ == '__main__':
    # === Parameters ===
    N = 20  # Number of segments
    L = 1.0  # Total length (m)
    mass = 0.0025 * 40 / N  # Mass per segment (kg)
    k = 10000 * 0.46  # Spring stiffness
    damping = 0.2  # Damping coefficient
    k_bend = 0.0006712  # Bending stiffness
    damping_bend = 0.000401  # Bending damping coefficient
    twisting = 0.00002
    air_drag = 0.2206 / 1000  # Drag coefficient
    g = 10.07  # Gravitational acceleration
    dt = 0.001  # Time step (s)
    T = 10.0  # Simulation duration (s)
    total_steps = int(T / dt)
    record_interval = 10

    # === Setup device ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Comment this out
    # device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")

    # === sampled data processing ===
    # sampled_data = np.load('../data/fixed_tip_horizontal_init.npy').astype(np.float32)  # (sample_num, 1+node_num*6+2)
    sampled_data = np.load('../data/fixed_tip_pos_low.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_high.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_x_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_3d.npy').astype(np.float32)
    # sampled_data = np.load('../data/static_init_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    sim = WarpRopeSim(batch_size=1280, N=N, L=L, mass=mass, k=k, damping=damping, bending_k=k_bend,
                      bending_damping=damping_bend, air_drag=air_drag, g=g, dt=dt,
                      max_record_steps=total_steps, record_interval=10)

    sim.set_pos(position_data[0:1])  # be careful about GPU warm up

    start = time.perf_counter()
    n = 10
    for i in range(n):
        sim.simulate(steps=10000)
    end = time.perf_counter()
    print('Average time: ', (end - start) / n)
    # sim.simulate(steps=total_steps)

    # traj = sim.get_trajectory()
    # print("Trajectory shape:", traj.shape)
    # # plot_animation_3d(traj, dt, record_interval, L, repeat=True, batch_idx=0)

# # === Parameters ===
#     N = 20  # Number of segments
#     L = 1.0  # Total length (m)
#     mass = 0.0025 * 40 / N  # Mass per segment (kg)
#     k = 0.46  # Spring stiffness
#     damping = [0.2] * N  # Damping coefficient
#     k_bend = [0.0006712] * (N - 1)  # Bending stiffness
#     damping_bend = [0.000401] * (N - 1)  # Bending damping coefficient
#     twisting = 0.00002
#     air_drag = 0.2206  # Drag coefficient
#     g = 10.07  # Gravitational acceleration
#     dt = 0.001  # Time step (s)
#     T = 10.0  # Simulation duration (s)
#     total_steps = int(T / dt)
#     record_interval = 10
#
#     # === Create model instance ===
#     model = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, twisting=twisting,
#                  air_drag=air_drag,
#                  g=g, dt=dt, device=device, mode='id')
#
#     # === show before id ===
#     print("here")
#     show_start = 0
#     # with torch.no_grad():  # Disable gradient computation
#     start = time.time()
#     with torch.no_grad():
#         positions_history_mine = model.simulate(
#             position_data[show_start:show_start + 1],
#             velocity_data[show_start:show_start + 1],
#             control_sequence[show_start:].unsqueeze(0),
#             steps=int(total_steps - show_start),
#             record_interval=record_interval
#         )
#     print("done!")
#
#     plot_animation_two_ropes_3d(traj, positions_history_mine, dt, record_interval, L, repeat=True,
#                                 name1="Warp Rope", name2="Rope")
