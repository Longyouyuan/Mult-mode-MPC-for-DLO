import warp as wp
import numpy as np
import time

wp.init()


class WarpRopeSim:
    def __init__(self, batch_size=1280, N=20, L=1.0, mass=0.005, k=500.0, damping=2.0, air_drag=0.0,
                 g=9.81, dt=0.001, max_record_steps=1000, record_interval=10):

        self.batch_size = batch_size
        self.N = N
        self.L = L
        self.mass = mass
        self.k = k
        self.damping = damping
        self.air_drag = air_drag
        self.g = g
        self.dt = dt
        self.dx = L / N
        self.record_interval = record_interval
        self.max_record_steps = max_record_steps
        self.max_saved_frames = max_record_steps // record_interval

        self.total_particles = batch_size * (N + 1)
        self.spring_count = batch_size * N

        # Particle data
        # self.test = wp.array(np.zeros((batch_size, N+1, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")
        # print('Test shape: ', self.test.shape)
        self.q = wp.array(np.zeros((self.total_particles, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")
        self.v = wp.array(np.zeros((self.total_particles, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")
        self.f = wp.array(np.zeros((self.total_particles, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")
        self.control = wp.array(np.zeros((self.total_particles, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")

        # Spring connections
        indices = []
        for b in range(batch_size):
            base = b * (N + 1)
            for i in range(N):
                indices.append((base + i, base + i + 1))
        self.spring_i = wp.array([i for i, j in indices], dtype=wp.int32, device="cuda")
        self.spring_j = wp.array([j for i, j in indices], dtype=wp.int32, device="cuda")

        # Init positions
        positions = np.zeros((self.total_particles, 3), dtype=np.float32)
        for b in range(batch_size):
            for i in range(N + 1):
                idx = b * (N + 1) + i
                positions[idx] = [b * 0.1, 0.0, -i * self.dx]
        self.q = wp.array(positions, dtype=wp.vec3, device="cuda")

        # Trajectory storage (on host)
        self.trajectory = np.zeros((batch_size, self.max_saved_frames, N + 1, 3), dtype=np.float32)

    @wp.kernel
    def apply_gravity(f: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                      g: float, mass: float, air_drag: float):
        i = wp.tid()
        # f[i] += wp.vec3(0.0, 0.0, -g * mass) - air_drag * v[i]
        wp.atomic_add(f, i, -wp.vec3(0.0, 0.0, g * mass) - air_drag * v[i])

    @wp.kernel
    def apply_springs(q: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                      f: wp.array(dtype=wp.vec3),
                      spring_i: wp.array(dtype=int), spring_j: wp.array(dtype=int),
                      k: float, damping: float, rest_length: float):
        tid = wp.tid()
        i = spring_i[tid]
        j = spring_j[tid]

        dx = q[j] - q[i]
        dist = wp.length(dx)
        dir = dx / (dist + 1e-6)

        rel_v = v[j] - v[i]
        proj_v = wp.dot(rel_v, dir)

        spring_damping_f = k * (dist - rest_length) * dir + damping * proj_v * dir

        wp.atomic_add(f, i, spring_damping_f)
        wp.atomic_add(f, j, -spring_damping_f)

    @wp.kernel
    def apply_control_force(f: wp.array(dtype=wp.vec3), control: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        f[i] += control[i]

    @wp.kernel
    def integrate(q: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                  dt: float, mass: float):
        # ccc = 1
        i = wp.tid()
        v[i] += dt * f[i] / mass
        q[i] += dt * v[i]

    @wp.kernel
    def rope_basic_force(
            q: wp.array(dtype=wp.vec3),
            v: wp.array(dtype=wp.vec3),
            f: wp.array(dtype=wp.vec3),
            spring_i: wp.array(dtype=int),
            spring_j: wp.array(dtype=int),
            k: float,
            damping: float,
            g: float,
            mass: float,
            dx: float,
            air_drag: float,
            spring_count: int,
    ):
        tid = wp.tid()

        # === Gravity + Air_drag ===
        wp.atomic_add(f, tid, -wp.vec3(0.0, 0.0, g * mass) - air_drag * v[tid])

        # === Spring damping force ===
        if tid < spring_count:
            i = spring_i[tid]
            j = spring_j[tid]

            dx_vec = q[j] - q[i]
            dist = wp.length(dx_vec)
            dir = dx_vec / (dist + 1e-6)

            rel_v = v[j] - v[i]
            proj_v = wp.dot(rel_v, dir)

            spring_force = k * (dist - dx) * dir + damping * proj_v * dir

            wp.atomic_add(f, i, spring_force)
            wp.atomic_add(f, j, -spring_force)

        if tid < spring_count:
            # ccc = 1
            i = spring_i[tid]
            j = spring_j[tid]

            dx_vec = q[j] - q[i]
            dist = wp.length(dx_vec)
            dir = dx_vec / (dist + 1e-6)

            rel_v = v[j] - v[i]
            proj_v = wp.dot(rel_v, dir)

            spring_force = k * (dist - dx) * dir + damping * proj_v * dir

            wp.atomic_add(f, i, spring_force)
            wp.atomic_add(f, j, -spring_force)

    def step(self):
        self.f.zero_()  # inplace 清零，无需分配新内存

        # wp.launch(
        #     self.rope_basic_force,
        #     dim=self.total_particles,  # 取 max 来覆盖所有粒子 + 弹簧
        #     inputs=[
        #         self.q, self.v, self.f,
        #         self.spring_i, self.spring_j,
        #         self.k, self.damping,
        #         self.g, self.mass,
        #         self.dx,
        #         self.air_drag,
        #         self.spring_count
        #     ]
        # )


        wp.launch(self.apply_gravity, dim=self.total_particles, inputs=[self.f, self.v, self.g, self.mass, self.air_drag])
        wp.launch(self.apply_springs, dim=self.spring_count,
                  inputs=[self.q, self.v, self.f, self.spring_i, self.spring_j,
                          self.k, self.damping, self.dx])
        wp.launch(self.apply_control_force, dim=self.total_particles, inputs=[self.f, self.control])
        wp.launch(self.integrate, dim=self.total_particles,
                  inputs=[self.q, self.v, self.f, self.dt, self.mass])

    def simulate(self, steps=10000):
        frame = 0
        for step in range(steps):
            self.step()
            # if step % self.record_interval == 0 and frame < self.max_saved_frames:
            #     pos = self.q.numpy().reshape(self.batch_size, self.N + 1, 3)
            #     self.trajectory[:, frame] = pos
            #     frame += 1

    def set_control_force(self, batch_idx, vec3):
        idx = batch_idx * (self.N + 1)
        control_np = self.control.numpy()
        control_np[idx] = vec3
        self.control = wp.array(control_np, dtype=wp.vec3, device="cuda")

    def get_trajectory(self):
        return self.trajectory


if __name__ == '__main__':
    sim = WarpRopeSim(batch_size=1280, N=20, max_record_steps=1000, record_interval=10)

    for i in range(4):
        sim.set_control_force(i, [2.0, 0.0, 0.0])

    start = time.perf_counter()
    n = 10
    for i in range(n):
        sim.simulate(steps=10000)
    end = time.perf_counter()

    print('Average time: ', (end - start) / n)

    traj = sim.get_trajectory()
    print("Trajectory shape:", traj.shape)
