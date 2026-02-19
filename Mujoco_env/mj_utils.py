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


class TrajDrawer:
    def __init__(self, viewer,
                 goal_traj,             # (T,3) torch/numpy
                 K, H, m,
                 draw_candidates=True,
                 draw_modes=True,
                 goal_width_px=5,
                 cand_width_px=2,
                 mode_width_px=8,
                 goal_rgba=(1.0, 1.0, 1.0, 0.2),
                 mode_rgba=(1.0, 0.2, 0.4, 1.0),
                 lut_size=256):
        self.viewer = viewer
        self.scn = viewer.user_scn

        self.K = int(K)
        self.H = int(H)
        self.m = int(m)

        self.draw_candidates = bool(draw_candidates)
        self.draw_modes = bool(draw_modes)

        self.goal_width_px = float(goal_width_px)
        self.cand_width_px = float(cand_width_px)
        self.mode_width_px = float(mode_width_px)

        self.goal_rgba = np.array(goal_rgba, dtype=np.float32)
        self.mode_rgba = np.array(mode_rgba, dtype=np.float32)

        # --- store goal ---
        goal = np.asarray(goal_traj, dtype=np.float32)
        assert goal.ndim == 2 and goal.shape[1] == 3, f"goal_traj must be (T,3), got {goal.shape}"
        self.goal = goal
        self.T = goal.shape[0]
        self.nseg_goal = max(0, self.T - 1)

        # --- segment counts ---
        self.nseg_cand = self.K * (self.H - 1)
        self.nseg_mode = self.m * (self.H - 1)

        # 最坏情况：goal + cand + mode 全开
        self.ngeom_need = self.nseg_goal + self.nseg_cand * draw_candidates + self.nseg_mode * draw_modes

        # --- color LUT (RdYlGn) ---
        cmap = plt.get_cmap("RdYlGn")
        cmap = plt.get_cmap("viridis")
        self.lut = cmap(np.linspace(0.0, 1.0, lut_size)).astype(np.float32)
        self.lut[:, 3] = 0.75  # 所有候选线统一透明度
        self.lut_size = int(lut_size)

        # --- init geoms once ---
        with self.viewer.lock():
            if getattr(self.scn, "maxgeom", self.ngeom_need) < self.ngeom_need:
                raise ValueError(f"user_scn.maxgeom too small: need {self.ngeom_need}, got {self.scn.maxgeom}")

            self.scn.ngeom = self.ngeom_need

            eye = np.eye(3, dtype=np.float32).reshape(-1)
            for i in range(self.ngeom_need):
                mujoco.mjv_initGeom(
                    self.scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([1, 0, 0], np.float32),
                    pos=np.zeros(3, np.float32),
                    mat=eye,
                    rgba=np.array([1, 1, 1, 1], np.float32)
                )

            # 初始先画一次 goal（否则第一帧可能是空的）
            self.goal_id_next = self._write_goal(off=np.zeros(3, np.float32))

            # 默认只显示 goal（候选/模态等 update 再写）
            self.scn.ngeom = self.nseg_goal

    def set_flags(self, draw_candidates=None, draw_modes=None):
        if draw_candidates is not None:
            self.draw_candidates = bool(draw_candidates)
        if draw_modes is not None:
            self.draw_modes = bool(draw_modes)

    def _write_goal(self, off):
        idx = 0
        for t in range(self.nseg_goal):
            g = self.scn.geoms[idx]
            p0 = self.goal[t] + off
            p1 = self.goal[t + 1] + off
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, self.goal_width_px, p0, p1)
            g.rgba[:] = self.goal_rgba
            idx += 1
        return idx  # next free index

    def update(self, cand_tip_traj=None, cand_cost=None, m_mode_trajs=None, offset=None):
        """
        cand_tip_traj: (K,H,3) torch/numpy
        cand_cost:     (K,)    torch/numpy
        m_mode_trajs:  (m,H,3) torch/numpy
        """
        off = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)

        # cand
        cand = None
        if cand_tip_traj is not None and self.draw_candidates:
            cand = np.asarray(cand_tip_traj, dtype=np.float32)
            assert cand.shape == (self.K, self.H, 3), f"cand_tip_traj must be ({self.K},{self.H},3), got {cand.shape}"

        # modes
        modes = None
        if m_mode_trajs is not None and self.draw_modes:
            modes = np.asarray(m_mode_trajs, dtype=np.float32)
            assert modes.ndim == 3 and modes.shape[1:] == (self.H, 3), f"m_mode_trajs must be (m,{self.H},3), got {modes.shape}"

        # color weights
        if cand is not None:
            if cand_cost is not None:
                cost = np.asarray(cand_cost, dtype=np.float32).reshape(-1)
                assert cost.shape[0] == self.K
                reward = -cost
                rmin, rmax = float(reward.min()), float(reward.max())
                denom = (rmax - rmin) if (rmax - rmin) > 1e-8 else 1.0
                w = (reward - rmin) / denom
            else:
                w = np.full((self.K,), 0.5, dtype=np.float32)

            idx_lut = np.clip((w * (self.lut_size - 1)).astype(np.int32), 0, self.lut_size - 1)
        else:
            idx_lut = None

        with self.viewer.lock():
            idx = 0

            # 1) goal 必画（每帧重写一次，保证不会被覆盖/残留）
            # idx = self._write_goal(off)
            idx = self.goal_id_next

            # 2) candidates（可开关）
            if cand is not None:
                for k in range(self.K):
                    rgba = self.lut[idx_lut[k]]
                    tr = cand[k]
                    for t in range(self.H - 1):
                        g = self.scn.geoms[idx]
                        p0 = tr[t] + off
                        p1 = tr[t + 1] + off
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, self.cand_width_px, p0, p1)
                        g.rgba[:] = rgba
                        idx += 1

            # 3) modes（可开关）
            if modes is not None:
                mm = min(self.m, modes.shape[0])
                for k in range(mm):
                    tr = modes[k]
                    for t in range(self.H - 1):
                        g = self.scn.geoms[idx]
                        p0 = tr[t] + off
                        p1 = tr[t + 1] + off
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, self.mode_width_px, p0, p1)
                        g.rgba[:] = self.mode_rgba
                        idx += 1

            self.scn.ngeom = idx

        return idx


class ForceDrawer:
    def __init__(self, viewer, base_geom_id, rgba=(1, 0, 0, 1), text_rgba=(1, 0, 0, 1), width_px=6):
        self.viewer = viewer
        self.scn = viewer.user_scn
        self.base = int(base_geom_id)   # 从 TrajDrawer 的 ngeom_need 之后开始
        self.width = float(width_px)
        self.rgba = np.array(rgba, np.float32)
        self.text_rgba = np.array(text_rgba, np.float32)

        with self.viewer.lock():
            eye = np.eye(3, dtype=np.float32).reshape(-1)

            # ---- 箭头 geom ----
            mujoco.mjv_initGeom(
                self.scn.geoms[self.base],
                type=mujoco.mjtGeom.mjGEOM_ARROW,   # 先占位
                size=np.array([1, 0, 0], np.float32),
                pos=np.zeros(3, np.float32),
                mat=eye,
                rgba=self.rgba.copy()
            )

            # ---- 文字 geom ----
            mujoco.mjv_initGeom(
                self.scn.geoms[self.base + 1],
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=np.array([100.0, 100.0, 100.0]),
                pos=np.zeros(3, np.float32),
                mat=eye.flatten(),
                rgba=self.text_rgba.copy()
            )

    def update(self, start_xyz, force_xyz=None, scale=0.1, idx_start=None):
        """
        start_xyz: (3,) 箭头起点
        force_xyz: (3,) 力向量；None or 0 => 不画（透明）
        idx_start: TrajDrawer.update() 返回的 idx_end，用于把 ngeom 扩展到包含箭头
        """
        p0 = np.asarray(start_xyz, np.float32)
        arrow = self.scn.geoms[self.base]
        label = self.scn.geoms[self.base + 1]

        with self.viewer.lock():

            if force_xyz is None or np.linalg.norm(force_xyz) < 1e-9:
                arrow.rgba[3] = 0.0
                label.rgba[3] = 0.0
                if idx_start is not None:
                    self.scn.ngeom = idx_start
                return

            f = np.asarray(force_xyz, np.float32)
            p1 = p0 + scale * f

            # ---- 画箭头 ----
            mujoco.mjv_connector(
                arrow,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self.width,
                p0, p1
            )
            arrow.rgba[:] = self.rgba

            # ---- 画文字 ----
            label.pos[:] = p0  # 文字放在箭头末端
            label.label = f"Disturbance!"
            label.rgba[:] = self.text_rgba
            label.size[:] = np.array([1.0, 1.0, 1.0])

            if idx_start is not None:
                self.scn.ngeom = max(idx_start, self.base + 2)











