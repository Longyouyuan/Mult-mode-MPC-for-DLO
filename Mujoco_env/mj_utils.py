import numpy as np
import mujoco
import matplotlib.pyplot as plt


def cam_setting(viewer, fixed=True):
    with viewer.lock():
        if fixed:
            # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_USER
            viewer.cam.lookat = np.array([0, 0, 0.6])
            viewer.cam.elevation = -30.0  # camera rotation around the axis in the plane （仰角）
            viewer.cam.azimuth = 90.0  # camera rotation around the camera's vertical axis （方位角)
            viewer.cam.distance = 4  # distance from focus point (model.stat.extent is the max limits of the area)
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = 0


def draw_line(start, end, width, rgba, viewer):
    viewer.user_scn.ngeom += 1
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
    size = [0.0, 0.0, 0.0]
    pos = [0, 0, 0]
    mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)


def draw_curve(points, viewer, rgba=None):
    points = points.cpu().numpy()
    if rgba is None:
        rgba = [0.5, 0.5, 0.5, 0.5]
    for i in range(points.shape[0]-1):
        start = points[i]
        end = points[i+1]
        draw_line(start, end, 10, rgba, viewer)


class FastMultiTrajDrawer:
    def __init__(self, viewer, K, H, m,
                 cand_alpha=0.25,
                 mode_rgba=(0.2, 0.4, 1.0, 0.90),
                 lut_size=256):
        """
        这里省略你原来的 init（预分配 mjvGeom 的部分）
        你只需要确保：
          - self.viewer = viewer
          - self.K, self.H, self.m
          - self.cand_nseg = K*(H-1)
          - self.mode_nseg = m*(H-1)
          - viewer.user_scn.ngeom >= self.cand_nseg + self.mode_nseg
        """
        self.viewer = viewer
        self.K = int(K)
        self.H = int(H)
        self.m = int(m)
        self.cand_alpha = float(cand_alpha)
        self.mode_rgba = np.array(mode_rgba, dtype=np.float32)

        # LUT：matplotlib RdYlGn（和你之前实现一致）
        cmap = plt.get_cmap("RdYlGn")
        xs = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        lut = np.stack([cmap(float(x)) for x in xs], axis=0).astype(np.float32)  # (lut,4)
        lut[:, 3] = self.cand_alpha
        self.lut = lut
        self.lut_size = lut_size

        self.cand_nseg = self.K * (self.H - 1)
        self.mode_nseg = self.m * (self.H - 1)

    @staticmethod
    def _to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def update(self, cand_tip_traj, cand_cost, m_mode_trajs, offset=None):
        """
        cand_tip_traj: (K,H,3) torch/numpy
        cand_cost:     (K,)    torch/numpy
        m_mode_trajs:  (m,H,3) torch/numpy
        offset:        (3,) or None
        """
        cand_traj = self._to_np(cand_tip_traj)
        cost = self._to_np(cand_cost)
        modes = self._to_np(m_mode_trajs)

        if cand_traj is None or cost is None or modes is None:
            return

        # shape check（不报错也行，但建议你自己保证一致）
        K, H, _ = cand_traj.shape
        assert K == self.K and H == self.H, f"cand_traj shape {cand_traj.shape} != ({self.K},{self.H},3)"
        assert cost.shape[0] == self.K, f"cand_cost shape {cost.shape} != ({self.K},)"
        assert modes.shape[0] == self.m and modes.shape[1] == self.H, f"modes shape {modes.shape} != ({self.m},{self.H},3)"

        if offset is None:
            offset = np.zeros(3, dtype=np.float32)
        else:
            offset = np.asarray(offset, dtype=np.float32)

        scn = self.viewer.user_scn

        # ---------- 颜色（完全对齐你 matplotlib：reward=-cost -> [0,1] -> RdYlGn）----------
        reward = -cost
        rmin = float(reward.min())
        rmax = float(reward.max())
        denom = (rmax - rmin) if (rmax - rmin) > 1e-8 else 1.0
        w = (reward - rmin) / denom  # [0,1]
        idx_lut = np.clip((w * (self.lut_size - 1)).astype(np.int32), 0, self.lut_size - 1)

        # ---------- 更新 geoms：用 fromto[0:3], fromto[3:6] ----------
        idx = 0
        seg = self.H - 1

        # 候选：每条轨迹一个颜色
        for k in range(self.K):
            rgba = self.lut[idx_lut[k]]
            tr = cand_traj[k]

            # 这条轨迹的所有段
            for t in range(seg):
                g = scn.geoms[idx]

                p0 = tr[t] + offset
                p1 = tr[t + 1] + offset

                # ✅ 兼容你当前版本：用 fromto
                g.fromto[0:3] = p0
                g.fromto[3:6] = p1

                # 颜色
                g.rgba[:] = rgba

                idx += 1

        # 模态：固定蓝色（更粗更亮由你 init 时的 size/rgba 决定，这里也再写一次）
        for k in range(self.m):
            tr = modes[k]
            for t in range(seg):
                g = scn.geoms[idx]

                p0 = tr[t] + offset
                p1 = tr[t + 1] + offset

                g.fromto[0:3] = p0
                g.fromto[3:6] = p1

                g.rgba[:] = self.mode_rgba

                idx += 1










