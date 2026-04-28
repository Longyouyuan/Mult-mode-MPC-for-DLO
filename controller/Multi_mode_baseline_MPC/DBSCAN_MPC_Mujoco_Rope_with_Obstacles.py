import mujoco.viewer as viewer
import time
from Mujoco_env.mj_utils import *
from common.utils import *
from DBSCAN_MPC import Planner, cost_fn
from common.rope_warp_4 import WarpRope, N, P
import warnings


@torch.no_grad()
def _fit_traj_count(trajs, costs, target_k):
    """
    TrajDrawer requires cand_tip_traj to have exactly (K,H,3).
    This helper truncates or pads by repeating the lowest-cost available trajectories.
    """
    if trajs is None or costs is None:
        return None, None

    target_k = int(target_k)
    cur_k = int(trajs.shape[0])
    if cur_k == target_k:
        return trajs, costs
    if cur_k > target_k:
        return trajs[:target_k], costs[:target_k]

    if cur_k == 0:
        return trajs, costs

    order = torch.argsort(costs)
    need = target_k - cur_k
    repeat_idx = order[torch.arange(need, device=trajs.device) % cur_k]
    trajs = torch.cat([trajs, trajs[repeat_idx]], dim=0)
    costs = torch.cat([costs, costs[repeat_idx]], dim=0)
    return trajs, costs


@torch.no_grad()
def get_dbscan_cluster_viz(planner, max_draw_clusters=8, members_per_cluster=4, fallback_k=None):
    """
    Build fixed-shape visualization tensors for DBSCAN-MPC.

    Returns:
        cluster_member_trajs: (K,H,3) exactly, K=max_draw_clusters*members_per_cluster
        cluster_member_cost: (K,) exactly
        cluster_rep_trajs: (M,H,3), M<=max_draw_clusters
    """
    target_k = int(max_draw_clusters) * max(1, int(members_per_cluster))
    if fallback_k is None:
        fallback_k = target_k
    else:
        fallback_k = min(int(fallback_k), target_k)

    if planner.last_tip_traj is None or planner.last_cost is None:
        return None, None, None

    labels = getattr(planner, "last_dbscan_labels", None)
    good_idx = getattr(planner, "last_good_idx", None)

    if labels is None or good_idx is None or labels.numel() == 0 or good_idx.numel() == 0:
        cand_tip_traj, cand_cost, rep_trajs = planner.get_cand(k_cand=fallback_k, rule='best')
        cand_tip_traj, cand_cost = _fit_traj_count(cand_tip_traj, cand_cost, target_k)
        return cand_tip_traj, cand_cost, rep_trajs

    labels = labels.to(device=planner.last_cost.device)
    good_idx = good_idx.to(device=planner.last_cost.device)

    cluster_ids = torch.unique(labels[labels >= 0])
    if cluster_ids.numel() == 0:
        cand_tip_traj, cand_cost, rep_trajs = planner.get_cand(k_cand=fallback_k, rule='best')
        cand_tip_traj, cand_cost = _fit_traj_count(cand_tip_traj, cand_cost, target_k)
        return cand_tip_traj, cand_cost, rep_trajs

    cluster_infos = []
    for cid in cluster_ids.tolist():
        local_idxs = torch.where(labels == int(cid))[0]
        full_idxs = good_idx[local_idxs]
        costs = planner.last_cost[full_idxs]
        mean_cost = costs.mean()
        rep_pos = torch.argmin(costs)
        rep_full_idx = full_idxs[rep_pos]

        sorted_local = torch.argsort(costs)
        sorted_full = full_idxs[sorted_local]
        member_full = sorted_full[sorted_full != rep_full_idx]
        member_full = member_full[:members_per_cluster]

        cluster_infos.append((float(mean_cost.item()), rep_full_idx, member_full))

    cluster_infos.sort(key=lambda x: x[0])
    cluster_infos = cluster_infos[:max_draw_clusters]

    rep_full_indices = []
    member_full_indices = []
    for _, rep_full_idx, member_full in cluster_infos:
        rep_full_indices.append(rep_full_idx)
        if member_full.numel() > 0:
            member_full_indices.append(member_full)

    rep_full_indices = torch.stack(rep_full_indices).to(device=planner.last_cost.device, dtype=torch.long)
    cluster_rep_trajs = planner.last_tip_traj[rep_full_indices]

    if len(member_full_indices) > 0:
        member_full_indices = torch.cat(member_full_indices, dim=0).to(device=planner.last_cost.device, dtype=torch.long)
        cluster_member_trajs = planner.last_tip_traj[member_full_indices]
        cluster_member_cost = planner.last_cost[member_full_indices]
    else:
        cluster_member_trajs = cluster_rep_trajs
        cluster_member_cost = planner.last_cost[rep_full_indices]

    cluster_member_trajs, cluster_member_cost = _fit_traj_count(
        cluster_member_trajs, cluster_member_cost, target_k
    )
    return cluster_member_trajs, cluster_member_cost, cluster_rep_trajs


# ===================== MuJoCo setup =====================
model = mujoco.MjModel.from_xml_path('../../Mujoco_env/cable_show.xml')
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

# ===== DBSCAN-MPC 参数 =====
n_sample = 402 * 1   # rollout batch = 1 + n_sample
n_improve = 1
noise_scale = 2.0
action_dim = 3
limits = torch.tensor([-5.0, 5.0])
total_horizon = int(total_steps / ctr_period)

# DBSCAN 超参
top_k_good = 200 * 3     # 保持原来的 good pool 规模；DBSCAN 会先 top-k，再过滤碰撞
dbscan_eps = 5.0         # 若 cluster 太少/全 noise，可增大；若全部合成一类，可减小
dbscan_min_samples = 5
collision_thr = 1e5
normalize_dbscan_feat = True

# DBSCAN cluster 可视化参数
max_draw_clusters = 10       # 类似原来的 m_modes，只用于画 cluster representatives
members_per_cluster = 8     # 每个 cluster 再画几个其它成员，类似 cand_tip_traj
k_cand_fallback = max_draw_clusters * max(1, members_per_cluster)

visualization = True

# ===================== Device =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
set_seed(0)

# ===================== Rope model instance =====================
rope = WarpRope(
    batch_size=1 + n_sample,
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
    N_warm=warm_step,        # 它决定你“慢多久”
    N_steady=total_horizon,  # tune (points per cycle)
    du_start_ratio=0.15,     # smaller => slower start
    warm_power=2.5
)

# For visualization over this 5s task window (not necessary for control)
Goal_traj = eight_inf.get_range(start=warm_step, length=total_horizon + 1)  # (total_horizon+1,3)
plot_goal_traj(Goal_traj, T_task)

# ===================== Planner =====================
planner = Planner(
    rope, cost_fn, dt, ctr_period, horizon,
    n_sample, n_improve, noise_scale, action_dim,
    limits=limits, device=device, mode=mode,
    top_k_good=top_k_good,
    dbscan_eps=dbscan_eps,
    dbscan_min_samples=dbscan_min_samples,
    collision_thr=collision_thr,
    normalize_dbscan_feat=normalize_dbscan_feat,
    standard_m2pc=True
)

# ===================== Init state =====================
pos = torch.zeros((1, P, 3), device=device)
pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
vel = torch.zeros((1, P, 3), device=device)

# # ===================== Warm up seeds =====================
# # Use infinite-goal for the initial horizon, and correct time scale dt * ctr_period
# Goal_init = eight_inf.get_range(start=0, length=horizon + 1)              # (horizon+1,3)
# vel_start = (Goal_init[1:] - Goal_init[:-1]) / dt * ctr_period           # (horizon,3)
# vel_start = vel_start.to(device)
#
# planner.seeds[:] = vel_start.unsqueeze(0)
# goal = eight_inf.get_range(start=1, length=horizon)  # (horizon,3)
#
# planner.n_improve = n_improve * 100
# planner.improve_policy(pos, vel, goal)
# planner.n_improve = n_improve

# ===================== Viewer / Simulation loop =====================
with viewer.launch_passive(model, data) as viewer:
    cam_setting(viewer, fixed=True)
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

    cyl1_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_1")
    cyl2_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_2")
    model.body_pos[cyl1_body] = np.array([0.433, 0.26, -1.2])  # world position 相对于 parent（world）
    model.body_pos[cyl2_body] = np.array([0.0, 0.9, 0.2])
    Obs_info = np.array([[0.06, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # [[radius, half_height, None], [Obs1_pos], [Obs2_pos]]

    # ===== 用你的 cable_body_indices 收集 rope 的 geom（rope 是 capsule）=====
    cable_body_set = set(cable_body_indices)
    rope_geom_ids = set()
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE and model.geom_bodyid[gid] in cable_body_set:
            rope_geom_ids.add(gid)

    # ===== 用你的 cyl1_body/cyl2_body 收集 cylinder 的 geom（cylinder 是 CYLINDER）=====
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

    prev_hit = False  # 防刷屏：只在刚碰到时打印
    hit_times = 0

    k_cand = max_draw_clusters * max(1, members_per_cluster)
    # TrajDrawer still uses a finite window for drawing (Goal_traj over 5s).
    # cand_tip_traj 画每个 cluster 的若干成员；m_mode_trajs 位置画 cluster representatives。
    traj_drawer = TrajDrawer(
        viewer, goal_traj=Goal_traj.cpu(),
        K=k_cand, H=horizon, m=max_draw_clusters,
        draw_candidates=True, draw_modes=True
    )

    task_t_start = time.perf_counter()

    for i in range(total_horizon * 2):
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
        goal = eight_inf.get_range(start=i + 1, length=horizon)  # (horizon,3)
        Obs_info[1, :] = data.xpos[cyl1_body].copy()  # (3,) numpy array
        Obs_info[2, :] = data.xpos[cyl2_body].copy()

        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal, Obs=Obs_info)

        # labels = planner.last_dbscan_labels
        # num_clusters = 0 if labels is None else int(torch.unique(labels[labels >= 0]).numel())
        # print(f"step={i}, num_clusters={num_clusters}")

        if torch.isinf(planner.J_star).any().item():
            print(f"[WARN] step={i}: planner.J_star contains inf, J_star={planner.J_star.detach().cpu().tolist()}")
        action = planner.get_action(rule='greedy').cpu().numpy()  # DBSCAN-MPC has one selected seed
        t1 = time.perf_counter()
        time_record.append(t1 - t0)

        # visualization
        if visualization:
            cand_tip_traj, cand_cost, cluster_rep_trajs = get_dbscan_cluster_viz(
                planner,
                max_draw_clusters=max_draw_clusters,
                members_per_cluster=members_per_cluster,
                fallback_k=k_cand_fallback
            )
            idx_end = traj_drawer.update(
                None if cand_tip_traj is None else cand_tip_traj.cpu(),
                None if cand_cost is None else cand_cost.cpu(),
                None if cluster_rep_trajs is None else cluster_rep_trajs.cpu(),
                offset=None
            )

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
plot_tip_vs_goal_and_error(pos_history, Goal_run, dt, ctr_period)

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
