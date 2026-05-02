import mujoco
import mujoco.viewer as viewer
import random
import time
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from Mujoco_env.mj_utils import *
from common.utils import *
from M2PC import Planner, cost_fn
from common.rope_warp_4 import WarpRope, N, P

# ===================== MuJoCo setup =====================
model = mujoco.MjModel.from_xml_path('../Mujoco_env/cable_show.xml')
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 4)  # 导入关节帧

# 修改仿真步长（单位：秒）/积分器
dt = 0.001
model.opt.timestep = dt
model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # 欧拉积分器

# 获取所有 body 的名称
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
# 获取绳子节点的 body 名称（例如，名称以 'B_' 开头）
cable_body_indices = [i for i, name in enumerate(body_names) if name and name.startswith('B_')]

data.ctrl[0] = 0
data.ctrl[1] = 0
data.ctrl[2] = 0

# ===================== Parameters =====================
L = 1.0
mass = 0.0025 * 40 / N
k = 10000 * 0.46
damping = 0.2
bending_k = 0.0006712 * 0.0
bending_damping = 0.000401
air_drag = 0.2206 / 1000
g = 10.07

T_task = 5.0
total_steps = int(T_task / dt)
mode = 'acc'  # 或 'vel'

ctr_period = 25  # 1000/ctr_period Hz
horizon = 35

# ===== 多模态参数 =====
n_sample = 400
m_modes = 1
assert n_sample % m_modes == 0

n_improve = 10
noise_scale = 1.5
action_dim = 3
limits = torch.tensor([-5.0, 5.0])
total_horizon = int(total_steps / ctr_period)

# diversity 超参
top_k_good = 200
beta = 5.0
wJ = 0.0

visualization = False
DISTURBANCE_DATA_DIR = Path(__file__).resolve().parent / "disturbance_data"

# ===================== Video recording =====================
# 只保存 MuJoCo 视角视频；fps=60 且按 data.time 采样，所以回放速度为 1 倍。
RECORD_VIDEO = True
VIDEO_FPS = 60
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_PATH = VIDEO_PATH = Path(__file__).resolve().parent / "vedios" / "disturbance" / "mujoco_rope_robustness_60fps.mp4"



def _copy_mjv_geom(dst, src):
    """Copy one mjvGeom. scene.geoms is a tuple in MuJoCo Python, so item assignment is invalid."""
    if hasattr(dst, "copy_from"):
        dst.copy_from(src)
        return

    # Fallback for older MuJoCo Python bindings: copy public writable fields one by one.
    for name in dir(src):
        if name.startswith("_") or name in ("copy", "copy_from"):
            continue
        try:
            value = getattr(src, name)
            target = getattr(dst, name)
        except Exception:
            continue
        try:
            if isinstance(target, np.ndarray):
                target[...] = value
            else:
                setattr(dst, name, value)
        except Exception:
            pass


def _append_user_scene_geoms(render_scene, user_scene):
    """把 viewer.user_scn 中 TrajDrawer / ForceDrawer 的自定义几何体复制到离屏场景。"""
    if user_scene is None or user_scene.ngeom <= 0:
        return
    n_copy = min(user_scene.ngeom, render_scene.maxgeom - render_scene.ngeom)
    if n_copy <= 0:
        return
    dst_start = render_scene.ngeom
    for k in range(n_copy):
        _copy_mjv_geom(render_scene.geoms[dst_start + k], user_scene.geoms[k])
    render_scene.ngeom += n_copy

def record_frame(record_scene, record_ctx, record_viewport, video_writer, data, viewer_handle):
    """用当前 viewer 相机离屏渲染一帧，并保留轨迹、目标点、扰动箭头等 user_scn 内容。"""
    with viewer_handle.lock():
        mujoco.mjv_updateScene(
            model,
            data,
            viewer_handle.opt,
            None,
            viewer_handle.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            record_scene,
        )
        _append_user_scene_geoms(record_scene, viewer_handle.user_scn)

    mujoco.mjr_render(record_viewport, record_scene, record_ctx)
    rgb = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, record_viewport, record_ctx)
    video_writer.append_data(np.flipud(rgb))

# ===================== Device =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
set_seed(0)

# ===================== Rope model instance =====================
rope = WarpRope(
    batch_size=m_modes + n_sample,
    L=L,
    mass=mass,
    k=k,
    damping=damping,
    bending_k=bending_k * 0.0,
    bending_damping=bending_damping,
    air_drag=air_drag,
    g=g,
    dt=dt,
    max_record_steps=horizon * ctr_period,
    record_interval=ctr_period,
    ctr_period=ctr_period,
    mode=mode
)

# ===================== Goal (Infinite Eight) =====================
warm_step = int(total_horizon * 0.3)
eight_inf = build_infinite_eight(
    device=device,
    scale_x=0.45 * 2.0,
    scale_y=0.65 * 2.0,
    z0=0.2,
    a=-0.10,                 # a=0->完全匀速推进; |a|越大->速度变化越剧烈
    N_warm= warm_step,             # 它决定你“慢多久”
    N_steady=total_horizon,           # tune (points per cycle)
    du_start_ratio=0.15,    # smaller => slower start;  起步第一步的速度是稳定速度的 du_start_ratio%
    warm_power=2.5  # 加速曲线形状
)

# For visualization over this 5s task window (not necessary for control)
Goal_traj = eight_inf.get_range(start=warm_step, length=total_horizon + 1)  # (total_horizon+1,3)
Goal_traj = eight_inf.get_range(start=warm_step, length=total_horizon + 1)  # (total_horizon+1,3)
# ===================== Planner =====================
planner = Planner(
    rope, cost_fn, dt, ctr_period, horizon,
    n_sample, n_improve, noise_scale, action_dim,
    limits=limits, device=device, mode=mode,
    m_modes=m_modes, top_k_good=top_k_good, beta=beta, wJ=wJ, standard_m2pc=True
)

# ===================== Init state =====================
pos = torch.zeros((1, P, 3), device=device)
pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
vel = torch.zeros((1, P, 3), device=device)

# ===================== Warm up seeds =====================
# Use infinite-goal for the initial horizon, and correct time scale dt * ctr_period
Goal_init = eight_inf.get_range(start=0, length=horizon + 1)              # (horizon+1,3)
vel_start = (Goal_init[1:] - Goal_init[:-1]) / dt * ctr_period       # (horizon,3)
vel_start = vel_start.to(device)

planner.seeds[:] = vel_start.unsqueeze(0).repeat(m_modes, 1, 1)
if m_modes > 1:
    planner.seeds[1:] += 0.05 * torch.randn_like(planner.seeds[1:])

goal = eight_inf.get_range(start=1, length=horizon)  # (horizon,3)

planner.n_improve = n_improve * 100
planner.improve_policy(pos, vel, goal)
planner.n_improve = n_improve

# ===================== Viewer / Simulation loop =====================
with viewer.launch_passive(model, data) as viewer:
    cam_setting(viewer, fixed=False)
    viewer.showinfo = True

    video_writer = None
    gl_ctx = None
    record_scene = None
    record_ctx = None
    record_viewport = None
    next_video_frame_t = 0.0
    if RECORD_VIDEO:
        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            VIDEO_PATH,
            fps=VIDEO_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        # 使用低层 GLContext + mjr_readPixels，避免 mujoco.Renderer 受 XML offwidth/offheight 限制。
        gl_ctx = mujoco.GLContext(VIDEO_WIDTH, VIDEO_HEIGHT)
        gl_ctx.make_current()
        record_scene = mujoco.MjvScene(model, maxgeom=20000)
        record_ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        record_viewport = mujoco.MjrRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT)

    goal_his = []
    slider_pos = []
    rope_top = []

    action_history = []
    vel_history = []
    pos_history = []
    time_record = []

    node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + node * 6 + 3))

    pcf = PositionCommandFilter(np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], dt)

    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    B_first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_first")
    k_cand = 100

    # TrajDrawer still uses a finite window for drawing (Goal_traj over 5s)
    traj_drawer = TrajDrawer(
        viewer, goal_traj=Goal_traj.cpu(),
        K=k_cand, H=horizon, m=m_modes,
        draw_candidates=False, draw_modes=True
    )
    idx_end = traj_drawer.goal_id_next
    force_drawer = ForceDrawer(viewer, base_geom_id=traj_drawer.ngeom_need, width_px=0.015)  # 从后面开始占位

    task_t_start = time.perf_counter()

    force = -1.0
    change = False
    force_show = -1

    for i in range(total_horizon*2+40):
        # == rope state ==
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + node * 3] = data.xpos[1:1+node][::-1].reshape(-1)
        mj_state[0, 1 + node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)

        # === tracking goal (infinite) ===
        global_goal_tracking = eight_inf.get(i + 1).detach().cpu().numpy()
        model.site_pos[1][[0, 1, 2]] = global_goal_tracking

        # === MPC goal horizon: always full horizon (infinite) ===
        goal = eight_inf.get_range(start=i + 1, length=horizon)  # (horizon,3)

        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal)
        action = planner.get_action().cpu().numpy()
        t1 = time.perf_counter()
        time_record.append(t1 - t0)

        # visualization
        if visualization:
            cand_tip_traj, cand_cost, m_mode_trajs = planner.get_cand(k_cand=k_cand)
            idx_end = traj_drawer.update(cand_tip_traj.cpu(), cand_cost.cpu(), m_mode_trajs.cpu(), offset=None)

        # ---- Execute chosen action in MuJoCo ----
        if (i - warm_step) % int(total_horizon/2) == 0:
            v = np.random.randn(2)
            v /= np.linalg.norm(v)
            v = np.array([v[0], v[1], 0.0])/2
            # print("V", v)

        for j in range(ctr_period):
            if mode == 'vel':
                pos_target = pcf.input_velocity(action)
            elif mode == 'acc':
                pos_target = pcf.input_acceleration(action)

            data.ctrl[0] = pos_target[0]
            data.ctrl[1] = pos_target[1]
            data.ctrl[2] = pos_target[2]

            global_goal_action_pos = pos_target + np.array([0.0, 0.0, 1.2])
            model.site_pos[0][[0, 1, 2]] = global_goal_action_pos

            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0, 0.5 * 9.81])

            if (i - warm_step) % int(total_horizon/2) <= 1:
                # data.xfrc_applied[B_first, :3] = v
                data.xfrc_applied[B_first, :3] = np.array([force, 0, 0])
                change = True
            elif change is True:
                change = False
                force *= -1

            # 先更新扰动箭头，再 step/render，这样录制帧里也能看到当前扰动方向。
            if (i - warm_step) % int(total_horizon / 2) <= 1 * 10:
                start = data.xpos[B_first].copy() + np.array([-0.05*force, 0, 0])
                force_drawer.update(
                    start,
                    force_xyz=np.array([force_show, 0, 0]),
                    scale=1.0,
                    idx_start=idx_end,
                )
                if (i - warm_step) % int(total_horizon / 2) == 1 * 10:
                    start = data.xpos[B_first].copy() + np.array([0.05, 0, 0])
                    force_drawer.update(
                        start,
                        force_xyz=np.array([0, 0, 0]),
                        scale=1.0,
                        idx_start=idx_end,
                    )
                    force_show *= -1

            mujoco.mj_step(model, data)

            if RECORD_VIDEO:
                while data.time + 1e-12 >= next_video_frame_t:
                    record_frame(record_scene, record_ctx, record_viewport, video_writer, data, viewer)
                    next_video_frame_t += 1.0 / VIDEO_FPS

        viewer.sync()

        goal_his.append(global_goal_action_pos)
        slider_pos.append(data.xpos[slider_id].copy())
        rope_top.append(data.site_xpos[-1].copy())

        action_history.append(action)
        vel_history.append(vel.clone())
        pos_history.append(pos.clone())

        planner.update_policy()

    if RECORD_VIDEO:
        if video_writer is not None:
            video_writer.close()
        if record_ctx is not None:
            record_ctx.free()
        if gl_ctx is not None:
            gl_ctx.free()
        print(f"Saved MuJoCo video to: {VIDEO_PATH}")

task_t_end = time.perf_counter()
print("average time on inference: ", torch.tensor(time_record).mean(), "Desired time:", dt * ctr_period)
print("Task time:", T_task, "   Spent time:", task_t_end - task_t_start)

# No post-processing saves or matplotlib plots: this script only writes the MuJoCo video.
