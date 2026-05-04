import mujoco
import mujoco.viewer as viewer
import time
from pathlib import Path
import imageio.v2 as imageio
from Mujoco_env.mj_utils import *
from common.utils import *
from M2PC import Planner, cost_fn
from common.rope_warp_4 import WarpRope, N, P
import warnings


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
n_sample = 402 * 1
m_modes = 3
assert n_sample % m_modes == 0

n_improve = 1
noise_scale = 2.0
action_dim = 3
limits = torch.tensor([-5.0, 5.0])
total_horizon = int(total_steps / ctr_period)

# diversity 超参
top_k_good = 200 * 3
beta = 1.0
wJ = 1.7

visualization = True


# ===================== Video / data output =====================
# Match the robustness recorder: low-level GLContext + mjr_readPixels, with
# explicit x264 settings so the 1920x1080 video is not blurred by defaults.
RECORD_VIDEO = False
RECORD_DATA = False
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 60
VIDEO_DIR = Path("./vedios/obs_avoidance")
VIDEO_PATH = VIDEO_DIR / "m2pc_rope_obstacle_avoidance.mp4"

DATA_DIR = Path("./obstacle_avoidance_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


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

def _copy_mjv_geom(dst, src):
    """Copy one MjvGeom in a MuJoCo-version-compatible way.

    Older mujoco Python wheels do not provide MjvGeom.copy_from(), and
    scene.geoms is tuple-like, so copy fields into the existing destination
    geom instead of assigning a new geom object.
    """
    for name in dir(src):
        if name.startswith('_'):
            continue
        try:
            value = getattr(src, name)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            target = getattr(dst, name)
        except Exception:
            continue
        try:
            # numpy fields: pos, mat, size, rgba, etc.
            target[...] = value
            continue
        except Exception:
            pass
        try:
            setattr(dst, name, value)
        except Exception:
            pass


def _append_user_scene_geoms(render_scene, user_scene):
    """Copy TrajDrawer goal/prediction geoms from viewer.user_scn into the offscreen scene."""
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
    """Render one offscreen frame using the current viewer camera and user_scn overlays."""
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
# plot_goal_traj removed: this recording script only saves video and npy data.

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

# # ===================== Warm up seeds =====================
# # Use infinite-goal for the initial horizon, and correct time scale dt * ctr_period
# Goal_init = eight_inf.get_range(start=0, length=horizon + 1)              # (horizon+1,3)
# vel_start = (Goal_init[1:] - Goal_init[:-1]) / dt * ctr_period       # (horizon,3)
# vel_start = vel_start.to(device)
#
# planner.seeds[:] = vel_start.unsqueeze(0).repeat(m_modes, 1, 1)
# if m_modes > 1:
#     planner.seeds[1:] += 0.05 * torch.randn_like(planner.seeds[1:])
#
# goal = eight_inf.get_range(start=1, length=horizon)  # (horizon,3)
#
# planner.n_improve = n_improve * 100
# planner.improve_policy(pos, vel, goal)
# planner.n_improve = n_improve

# ===================== Viewer / Simulation loop =====================
with viewer.launch_passive(model, data) as viewer:
    cam_setting(viewer, fixed=True)
    viewer.showinfo = True

    pos_history = []
    time_record = []
    profile_records = []

    node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + node * 6 + 3))

    pcf = PositionCommandFilter(np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], dt)

    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    B_first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_first")

    cyl1_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_1")
    cyl2_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_2")
    model.body_pos[cyl1_body] = np.array([0.433, 0.26, -1.2])
    model.body_pos[cyl2_body] = np.array([0.0, 0.9, 0.2])
    Obs_info = np.array([[0.06, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    cable_body_set = set(cable_body_indices)
    rope_geom_ids = set()
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE and model.geom_bodyid[gid] in cable_body_set:
            rope_geom_ids.add(gid)

    cyl_geom_ids = set()
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            b = model.geom_bodyid[gid]
            if b == cyl1_body or b == cyl2_body:
                cyl_geom_ids.add(gid)

    for gid in cyl_geom_ids:
        radius = model.geom_size[gid][0]
        half_height = model.geom_size[gid][1]
        Obs_info[0, 0] = radius
        Obs_info[0, 1] = half_height
        print("radius:", radius, "half_height:", half_height)

    prev_hit = False
    hit_times = 0

    k_cand = 100
    traj_drawer = TrajDrawer(
        viewer, goal_traj=Goal_traj.cpu(),
        K=k_cand, H=horizon, m=m_modes,
        draw_candidates=False, draw_modes=True
    )

    video_writer = None
    gl_ctx = None
    record_scene = None
    record_ctx = None
    record_viewport = None
    next_frame_time = 0.0
    frame_interval = 1.0 / VIDEO_FPS
    if RECORD_VIDEO:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        # video_writer = imageio.get_writer(
        #     VIDEO_PATH,
        #     fps=VIDEO_FPS,
        #     codec="libx264",
        #     quality=8,
        #     macro_block_size=1,
        # )
        video_writer = imageio.get_writer(
            VIDEO_PATH,
            fps=VIDEO_FPS,
            codec="libx264",
            macro_block_size=1,
            output_params=[
                "-b:v", "12M",
                "-maxrate", "12M",
                "-bufsize", "24M",
                "-pix_fmt", "yuv420p",
            ],
        )

        # Low-level GLContext + mjr_readPixels avoids mujoco.Renderer XML offwidth/offheight limits.
        gl_ctx = mujoco.GLContext(VIDEO_WIDTH, VIDEO_HEIGHT)
        gl_ctx.make_current()
        record_scene = mujoco.MjvScene(model, maxgeom=20000)
        record_ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        record_viewport = mujoco.MjrRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT)

    task_t_start = time.perf_counter()

    try:
        # Record the initial view including the static goal trajectory drawn by TrajDrawer.
        viewer.sync()
        if RECORD_VIDEO:
            record_frame(record_scene, record_ctx, record_viewport, video_writer, data, viewer)
            next_frame_time += frame_interval

        for i in range(total_horizon * 2 + 40):
            # == rope state ==
            mujoco.mj_forward(model, data)
            mj_state[0, 0] = data.time
            mj_state[0, 1:1 + node * 3] = data.xpos[1:1 + node][::-1].reshape(-1)
            mj_state[0, 1 + node * 3:-3] = data.sensordata
            pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)

            # === tracking goal (infinite) ===
            global_goal_tracking = eight_inf.get(i + 1).detach().cpu().numpy()
            model.site_pos[1][[0, 1, 2]] = global_goal_tracking

            # === MPC goal horizon: always full horizon (infinite) ===
            goal = eight_inf.get_range(start=i + 1, length=horizon)
            Obs_info[1, :] = data.xpos[cyl1_body].copy()
            Obs_info[2, :] = data.xpos[cyl2_body].copy()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            planner.improve_policy(pos, vel, goal, Obs=Obs_info)
            if planner.last_profile is not None:
                profile_records.append(planner.last_profile.copy())
            if torch.isinf(planner.J_star).any().item():
                print(f"[WARN] step={i}: planner.J_star contains inf, J_star={planner.J_star.detach().cpu().tolist()}")
            action = planner.get_action(rule='greedy').cpu().numpy()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            time_record.append(t1 - t0)

            # Draw predicted trajectories into viewer.user_scn. The recorder copies these geoms.
            if visualization:
                cand_tip_traj, cand_cost, m_mode_trajs = planner.get_cand(k_cand=k_cand)
                idx_end = traj_drawer.update(cand_tip_traj.cpu(), cand_cost.cpu(), m_mode_trajs.cpu(), offset=None)

            # ---- Execute chosen action in MuJoCo ----
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

                mujoco.mj_step(model, data)

                # Sample frames by simulation time, so the video plays at 1x speed and 60 fps.
                if RECORD_VIDEO:
                    while data.time + 1e-12 >= next_frame_time:
                        record_frame(record_scene, record_ctx, record_viewport, video_writer, data, viewer)
                        next_frame_time += frame_interval

            viewer.sync()

            hit = False
            for ci in range(data.ncon):
                c = data.contact[ci]
                g1, g2 = c.geom1, c.geom2
                if (g1 in rope_geom_ids and g2 in cyl_geom_ids) or (g2 in rope_geom_ids and g1 in cyl_geom_ids):
                    hit = True
                    break

            if hit and (not prev_hit):
                hit_times += 1
                warnings.warn(
                    f"[t={data.time:.4f}] Rope touches cylinder {hit_times} times!",
                    category=UserWarning
                )

            prev_hit = hit
            pos_history.append(pos.clone())
            planner.update_policy()

    finally:
        if video_writer is not None:
            video_writer.close()
        if record_ctx is not None:
            record_ctx.free()
        if gl_ctx is not None:
            gl_ctx.free()


task_t_end = time.perf_counter()
def print_timing_stats(name, values):
    if len(values) == 0:
        print(f"{name}: no samples")
        return
    values_t = torch.as_tensor(values, dtype=torch.float64)
    mean_s = values_t.mean().item()
    var_s2 = values_t.var(unbiased=False).item()
    print(f"{name}: mean={mean_s:.6f} s ({mean_s * 1000.0:.3f} ms), var={var_s2:.6e} s^2")


def save_profile_records(path, records):
    keys = ("improve", "rollout", "cost", "multimodal", "overhead")
    arrays = {
        key: np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
        for key in keys
    }
    np.savez(path, **arrays)
    print(f"Saved profile records to: {path}")


print_timing_stats("time_record / outer improve_policy timing", time_record)
for key, label in [
    ("improve", "planner.improve_policy internal total"),
    ("rollout", "rollout"),
    ("cost", "cost"),
    ("multimodal", "M2PC multimodal selection"),
    ("overhead", "profiled overhead"),
]:
    print_timing_stats(label, [record[key] for record in profile_records if key in record])

print("Task time:", T_task, "   Spent time:", task_t_end - task_t_start)
print("Desired time:", dt * ctr_period)
print(f"Saved video to: {VIDEO_PATH}")
save_profile_records(DATA_DIR / "m2pc_profile_records.npz", profile_records)

# ===================== Save data only =====================
pos_history = torch.cat(pos_history, dim=0)
Goal_run = eight_inf.get_range(start=1, length=pos_history.shape[0] + 1)
if RECORD_DATA:
    np.save(DATA_DIR / "m2pc_rope_traj.npy", pos_history.cpu().numpy())
    np.save(DATA_DIR / "goal_traj.npy", Goal_run.cpu().numpy())
    print(f"Saved rope trajectory to: {DATA_DIR / 'm2pc_rope_traj.npy'}")
    print(f"Saved goal trajectory to: {DATA_DIR / 'goal_traj.npy'}")

