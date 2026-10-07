import mujoco.viewer as viewer
import random
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Mujoco_env.mj_utils import *
from common.utils import *
from M2PC import Planner, cost_fn
from common.rope_warp_4 import WarpRope, N, P

# ===================== MuJoCo setup =====================
model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / 'Mujoco_env' / 'cable_show.xml'))
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
beta = 1000.0
wJ = 1700.0

visualization = False
DISTURBANCE_DATA_DIR = Path(__file__).resolve().parent / "disturbance_data"

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
plot_goal_traj(Goal_traj, T_task)

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

            mujoco.mj_step(model, data)

        if (i - warm_step) % int(total_horizon / 2) <= 1*3:
            # view force
            start = data.xpos[B_first].copy() + np.array([0.05, 0, 0])
            force_drawer.update(start, force_xyz=data.xfrc_applied[B_first, :3].copy(),
                                    scale=1.0, idx_start=idx_end)

        viewer.sync()

        goal_his.append(global_goal_action_pos)
        slider_pos.append(data.xpos[slider_id].copy())
        rope_top.append(data.site_xpos[-1].copy())

        action_history.append(action)
        vel_history.append(vel.clone())
        pos_history.append(pos.clone())

        planner.update_policy()

task_t_end = time.perf_counter()
print("average time on inference: ", torch.tensor(time_record).mean(), "Desired time:", dt * ctr_period)
print("Task time:", T_task, "   Spent time:", task_t_end - task_t_start)

# ===================== Post plots (keep your original logic) =====================
pos_history = torch.cat(pos_history, dim=0)

# Compare tip vs goal: now goal for the whole run is infinite, so create a matching-length goal window
Goal_run = eight_inf.get_range(start=1, length=pos_history.shape[0] + 1)  # align with your previous Goal_traj[1:]
disturbance_indices = np.array([warm_step, warm_step + int(total_horizon / 2)], dtype=np.int64)
disturbance_points = eight_inf.get(disturbance_indices.tolist()).detach().cpu().numpy().astype(np.float32)

goal_trajectory = Goal_run[:pos_history.shape[0]].detach().cpu().numpy().astype(np.float32)
rope_trajectory = pos_history.detach().cpu().numpy().astype(np.float32)

DISTURBANCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
np.save(DISTURBANCE_DATA_DIR / "goal_trajectory.npy", goal_trajectory)
np.save(DISTURBANCE_DATA_DIR / "rope_trajectory.npy", rope_trajectory)
np.save(DISTURBANCE_DATA_DIR / "disturbance_points.npy", disturbance_points)
np.savez(
    DISTURBANCE_DATA_DIR / "robustness_disturbance_data.npz",
    goal_trajectory=goal_trajectory,
    rope_trajectory=rope_trajectory,
    disturbance_points=disturbance_points,
    disturbance_indices=disturbance_indices,
    dt=np.float32(dt),
    ctr_period=np.int64(ctr_period),
)
print(f"Saved disturbance data to: {DISTURBANCE_DATA_DIR}")

plot_tip_vs_goal_and_error(pos_history, Goal_run, dt, ctr_period,
                           dist=disturbance_points)

# === action curves ===
action_history_np = np.array(action_history)
time_axis = np.arange(action_history_np.shape[0]) * ctr_period * dt

plt.figure(figsize=(10, 6))
plt.plot(time_axis, action_history_np[:, 0], label='Action X', color='r')
plt.plot(time_axis, action_history_np[:, 1], label='Action Y', color='g')
plt.plot(time_axis, action_history_np[:, 2], label='Action Z', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Action Value')
plt.title('Action XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === velocity curves ===
vel_history_np = np.stack([
    v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    for v in vel_history
], axis=0)[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(time_axis, vel_history_np[:, 0, 0], label='Velocity X', color='r', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 0, 1], label='Velocity Y', color='g', linestyle='--')
plt.plot(time_axis, vel_history_np[:, 0, 2], label='Velocity Z', color='b', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Value')
plt.title('Top Velocity XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# pos 轨迹
tip_pos_history = pos_history[:, 0, :]
plt.figure(figsize=(10, 6))
plt.plot(time_axis, tip_pos_history[:, 0].cpu().numpy(), label='Tip X', color='m')
plt.plot(time_axis, tip_pos_history[:, 1].cpu().numpy(), label='Tip Y', color='c')
plt.plot(time_axis, tip_pos_history[:, 2].cpu().numpy(), label='Tip Z', color='y')
plt.xlabel('Time (s)')
plt.ylabel('top Position')
plt.title('Top Point Position XYZ over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== Check low-level controller performance ==========================
time_idx = list(range(len(goal_his)))
goal_array = np.array(goal_his)
slider_pos_array = np.array(slider_pos)
rope_top_array = np.array(rope_top)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(time_idx, goal_array[:, 0], label='Goal', color='blue', linestyle='-')
ax1.plot(time_idx, slider_pos_array[:, 0], label='Slider', color='green', linestyle='--')
ax1.plot(time_idx, rope_top_array[:, 0], label='Rope Top', color='red', linestyle=':')
ax1.set_title('X-axis Position Tracking')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('X Position')
ax1.grid(True)
ax1.legend()

ax2.plot(time_idx, goal_array[:, 1], label='Goal', color='blue', linestyle='-')
ax2.plot(time_idx, slider_pos_array[:, 1], label='Slider', color='green', linestyle='--')
ax2.plot(time_idx, rope_top_array[:, 1], label='Rope Top', color='red', linestyle=':')
ax2.set_title('Y-axis Position Tracking')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Y Position')
ax2.grid(True)
ax2.legend()

ax3.plot(time_idx, goal_array[:, 2], label='Goal', color='blue', linestyle='-')
ax3.plot(time_idx, slider_pos_array[:, 2], label='Slider', color='green', linestyle='--')
ax3.plot(time_idx, rope_top_array[:, 2], label='Rope Top', color='red', linestyle=':')
ax3.set_title('Z-axis Position Tracking')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Z Position')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()
