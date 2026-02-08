import taichi as ti
from common.rope import *

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, kernel_profiler=True)
# ti.init(arch=ti.gpu)


@ti.data_oriented
class BatchedRope3D:
    def __init__(self, batch_size, N=20, L=1.0, mass=0.005, k=500.0, damping=2.0, damping_bend=0.0,
                 air_drag=0.0, g=9.81, dt=0.001, max_record_steps=1000, record_interval=10):
        self.batch_size = batch_size
        self.N = N
        self.L = L
        self.dx = L / N
        self.mass = mass
        self.k = k
        self.damping = damping
        self.damping_bend = damping_bend
        self.air_drag = air_drag
        self.g = g
        self.dt = dt
        self.dim = 3

        self.record_interval = record_interval
        self.max_record_steps = max_record_steps
        self.max_saved_frames = max_record_steps // record_interval

        shape = (batch_size, N + 1)
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)
        self.f = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)
        self.control_force = ti.Vector.field(self.dim, dtype=ti.f32, shape=batch_size)

        self.lengths = ti.field(dtype=ti.f32, shape=(self.batch_size, self.N))
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(self.batch_size, self.N))

        self.trajectory = ti.Vector.field(self.dim, dtype=ti.f32,
                                          shape=(batch_size, self.max_saved_frames, N + 1))

        self.reset()

    @ti.kernel
    def reset(self):
        for b, i in ti.ndrange(self.batch_size, self.N + 1):
                self.x[b, i] = ti.Vector([b * 0.1, 0.0, -i * self.dx])  # initial stretched
                self.v[b, i] = ti.Vector([0.0, 0.0, 0.0])

        for b in range(self.batch_size):
            self.control_force[b] = ti.Vector([0.0, 0.0, 0.0])

    # @ti.kernel
    # def set_pos_vel(self, pos: ti.types.ndarray()):
    #     for b, i in ti.ndrange(self.batch_size, self.N + 1):
    #         for d in ti.static(range(3)):
    #             self.x[b, i][d] = pos[i, d]

    def set_pos_vel(self, pos):
        batched_np = np.ascontiguousarray(
            np.tile(pos[None, :, :].cpu().numpy(), (self.batch_size, 1, 1)), dtype=np.float32
        )
        self.x.from_numpy(batched_np)

    @ti.kernel
    def apply_forces(self):
        ccc = 1

        for b, i in ti.ndrange(self.batch_size, self.N + 1):
            # ccc = 1
            self.f[b, i] = ti.Vector([0.0, 0.0, -self.mass * self.g]) - self.air_drag * self.v[b, i]  # gravity + drag

        for b, i in ti.ndrange(self.batch_size, self.N):
            ccc = 1

            delta = self.x[b, i + 1] - self.x[b, i]
            length = delta.norm() + 1e-8
            dir = delta / length

            self.lengths[b, i] = length
            self.directions[b, i] = dir

            rel_vel = self.v[b, i + 1] - self.v[b, i]
            proj = rel_vel.dot(dir)

            spring_force = self.k * (length - self.dx) * dir
            damp_force = self.damping * proj * dir

            self.f[b, i] += spring_force + damp_force
            self.f[b, i + 1] -= spring_force + damp_force

    @ti.kernel
    def apply_bending_damping(self):
        ccc = 1
        for b, i in ti.ndrange(self.batch_size, self.N - 1):
            ccc = 1

            d1 = self.directions[b, i]
            d2 = self.directions[b, i + 1]
            l1 = self.lengths[b, i]
            l2 = self.lengths[b, i + 1]

            c1 = d1.cross(d2)

            grad_d1 = d1.cross(c1)
            norm_grad_d1 = grad_d1.norm()
            dbeta_de1 = grad_d1 / (norm_grad_d1 * l1 + 1e-8)

            grad_d2 = d2.cross(-c1)
            norm_grad_d2 = grad_d2.norm()
            dbeta_de2 = grad_d2 / (norm_grad_d2 * l2 + 1e-8)

            vel_diff1 = self.v[b, i + 1] - self.v[b, i]
            vel_diff2 = self.v[b, i + 2] - self.v[b, i + 1]
            dbeta_dt = dbeta_de1.dot(vel_diff1) + dbeta_de2.dot(vel_diff2)

            tmp = self.damping_bend * dbeta_dt
            F1 = tmp * dbeta_de1
            F2 = -tmp * dbeta_de2

            self.f[b, i] += F1
            self.f[b, i + 1] -= (F1 + F2)
            self.f[b, i + 2] += F2

        for b in range(self.batch_size):
            # ccc = 1
            self.f[b, 0] = self.control_force[b] * self.mass

    @ti.kernel
    def integrate(self):
        ccc = 1
        # for b, i in ti.ndrange(self.batch_size, self.N + 1):
        #     # self.v[b, i] += self.dt * self.f[b, i] / self.mass
        #     # self.x[b, i] += self.dt * self.v[b, i]
        #     ccc = 1

    @ti.kernel
    def record(self, frame_id: ti.i32):
        for b, i in ti.ndrange(self.batch_size, self.N + 1):
            self.trajectory[b, frame_id, i] = self.x[b, i]

    def step(self):
        self.apply_forces()
        self.apply_bending_damping()  # 重新计算 direction 和 length
        self.integrate()
        # ccc = 1

    def simulate(self, steps=1000):
        frame_id = 0
        self.record(frame_id)
        frame_id += 1

        for step in range(steps):
            self.step()
            # if (step + 1) % self.record_interval == 0 and frame_id < self.max_saved_frames:
            #     self.record(frame_id)
            #     frame_id += 1

    def get_trajectory(self):
        return self.trajectory.to_numpy()  # (batch_size, saved_frames, N+1, 3)

    def set_control_force(self, batch_idx, vec3):
        self.control_force[batch_idx] = ti.Vector(vec3)


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
    sampled_data = np.load('../data/fixed_tip_horizontal_init.npy').astype(np.float32)  # (sample_num, 1+node_num*6+2)
    # sampled_data = np.load('../data/fixed_tip_pos_low.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_high.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_x_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_3d.npy').astype(np.float32)
    # sampled_data = np.load('../data/static_init_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    rope_sim = BatchedRope3D(batch_size=1280, N=N, L=L, mass=mass, k=k,
                             damping=damping, damping_bend=damping_bend,
                             air_drag=air_drag, g=g, dt=dt,
                             max_record_steps=total_steps, record_interval=record_interval)

    # start = time.perf_counter()
    # n=10
    # for i in range(n):
    #     rope_sim.set_pos_vel(position_data[0])
    # end = time.perf_counter()
    # print('Average time: ', (end - start) / n)
    rope_sim.set_pos_vel(position_data[0])

    start = time.perf_counter()
    n = 10
    for i in range(n):
        rope_sim.simulate(steps=total_steps)
    end = time.perf_counter()
    print('Average time: ', (end - start)/n)
    # rope_sim.simulate(steps=total_steps)
    #
    # traj = rope_sim.get_trajectory()  # shape: (4, 100, 21, 3)
    # print("Trajectory shape:", traj.shape)
    #
    # # plot_animation_3d(traj[0], dt, record_interval, L, repeat=True, batch_idx=0)
    #
    # # === Parameters ===
    # N = 20  # Number of segments
    # L = 1.0  # Total length (m)
    # mass = 0.0025 * 40 / N  # Mass per segment (kg)
    # k = 0.46  # Spring stiffness
    # damping = [0.2] * N  # Damping coefficient
    # k_bend = [0.0006712] * (N - 1)  # Bending stiffness
    # damping_bend = [0.000401] * (N - 1)  # Bending damping coefficient
    # twisting = 0.00002
    # air_drag = 0.2206  # Drag coefficient
    # g = 10.07  # Gravitational acceleration
    # dt = 0.001  # Time step (s)
    # T = 10.0  # Simulation duration (s)
    # total_steps = int(T / dt)
    # record_interval = 10
    #
    # # === Create model instance ===
    # model = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, twisting=twisting,
    #              air_drag=air_drag,
    #              g=g, dt=dt, device=device, mode='id')
    #
    # # === show before id ===
    # print("here")
    # show_start = 0
    # # with torch.no_grad():  # Disable gradient computation
    # start = time.time()
    # with torch.no_grad():
    #     positions_history_mine = model.simulate(
    #         position_data[show_start:show_start + 1],
    #         velocity_data[show_start:show_start + 1],
    #         control_sequence[show_start:].unsqueeze(0),
    #         steps=int(total_steps - show_start),
    #         record_interval=record_interval
    #     )
    # positions_history_mine = positions_history_mine.numpy()
    # print("done!")
    #
    # plot_animation_two_ropes_3d(traj[0], positions_history_mine[0], dt, record_interval, L, repeat=True,
    #                             name1="taichi Rope", name2="Rope")

    ti.profiler.print_kernel_profiler_info()

