import mujoco.viewer as viewer
from Mujoco_env.mj_utils import *
import time
from common.utils import *
import struct
import zmq

STATE_HDR = struct.Struct("<iqII")
ACT_HDR   = struct.Struct("<iqf")

def now_ns():
    return time.perf_counter_ns()

# === mujoco ===
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

# === Parameters ===
N = 20
P = N + 1
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
horizon = 30  # 20 doesn't work; 25 can work
H = horizon

# ===== 多模态参数 =====
n_sample = 400     # 总采样数
m_modes = 1        # 模态数（单模态=1）
assert n_sample % m_modes == 0

n_improve = 10  # 10 will be good. 1 also fine?
noise_scale = 1.5  # 0.25 looks good for naive case, 1.5更好for naive case？
action_dim = 3
limits = torch.tensor([-5.0, 5.0])  # 15 太大了，不要太大
last_point_repeat = 100 * 0
total_horizon = int(total_steps / ctr_period) + last_point_repeat

# diversity 超参
top_k_good = 200
beta = 5.0
wJ = 0.0

visualization = True

# === 目标轨迹（保留原逻辑）===

# === 1. Generate sin/egg/eight Trajs ===
points = half_dense_then_uniform(
    N=total_horizon + 1 - last_point_repeat, ratio=0.3, sharpness=2.0,
    mode='exp', interval=(0.0, 2.0), plot=True
)
device = "cpu"
Goal_traj = eight_traj(points, scale_x=0.45*2.0, scale_y=0.65*2.0, z0=0.2, loops=1, plot=False, device=device)  # 5s
# Goal_traj = egg_traj(points, scale_x=0.38*2.0, scale_y=0.52*2.0, plot=False, device=device)
# Goal_traj = sin_traj(points, width=0.45*2.0, plot=False, device=device)
# Goal_traj = torch.vstack((torch.sin(4*points)*0.25, points,
#                           torch.ones(total_horizon + 1 - last_point_repeat) * 0.2)).T.to(device)
plot_goal_traj(Goal_traj, T_task)

# # === 2. Draw Traj ===
#
# Goal_traj = build_goal_traj_from_drawn(
#     drawn_path="../my_trajs/SpongeBob.npy",  # SpongeBob flower PatrickStar
#     total_horizon=total_horizon-last_point_repeat,
#     device=device,
#     z0=0.2,
#     scale_x=3.0,
#     scale_y=3.0,
#     keep_aspect=False,  # 允许非等比缩放（你说可能不是方形）
#     sigma=0.0,  # 高斯顺滑
#     uniform_M=1000,  # 越大越均匀/越平滑（但太大也没必要）
#     ratio=0.3,
#     sharpness=2.0,
#     interval=(0.0, 2.0)
# )
# plot_goal_traj(Goal_traj, T_task)

Goal_traj = extend_last_point(Goal_traj, N=last_point_repeat)

# === Init state（保留原逻辑）===
pos = np.zeros((1, P, 3), dtype=np.float32)
pos[:, :, 2] = np.linspace(1.2, 0.2, num=P)
vel = np.zeros((1, P, 3), dtype=np.float32)

# ==================== connect with rocco ================
ctx = zmq.Context.instance()
server_ip = "10.240.21.20"
# PUB: state -> server:6000
pub = ctx.socket(zmq.PUB)
pub.connect(f"tcp://{server_ip}:4532")

# SUB: action <- server:6001
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{server_ip}:6001")

print(f"[client] PUB connect tcp://{server_ip}:4532 (state out)")
print(f"[client] SUB connect tcp://{server_ip}:6001 (action in)")

# 重要：PUB/SUB 需要一点建链时间，否则第一条消息可能丢
time.sleep(0.2)

# ==================== warm up ========================
goal = Goal_traj[1:1 + horizon, :].numpy()
t_state_ns = now_ns()
step = -1
frames = [
    STATE_HDR.pack(step, t_state_ns, P, H),
    pos.reshape(-1).tobytes(),
    vel.reshape(-1).tobytes(),
    goal.reshape(-1).tobytes(),
]
pub.send_multipart(frames)
time.sleep(5.0)


with viewer.launch_passive(model, data) as viewer:
    # launch_passive means all the simulation should be done by the user

    cam_setting(viewer, fixed=True)
    viewer.showinfo = True  # 启动时就显示 info（等同于自动 F2）

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
    Goal_np = Goal_traj.detach().cpu().numpy()
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    k_cand = 100
    traj_drawer = TrajDrawer(viewer, goal_traj=Goal_traj.cpu(),
                             K=k_cand, H=horizon, m=m_modes,
                             draw_candidates=True, draw_modes=True)

    task_t_start = time.perf_counter()

    for i in range(total_horizon):
        # print('i: ', i)

        # == rope state ==
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + node * 3] = data.xpos[1:1+node][::-1].reshape(-1)
        mj_state[0, 1 + node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)

        # === tracking goal ===
        global_goal_tracking = Goal_np[i+1]
        model.site_pos[1][[0, 1, 2]] = global_goal_tracking

        if i < (total_horizon - horizon):
            goal = Goal_traj[i + 1:i + 1 + horizon, :]
        else:
            goal = Goal_traj[i + 1:, :]  # 尾段短的也没事：内部会 padding

        t0 = time.perf_counter()
        t_state_ns = now_ns()
        frames = [
            STATE_HDR.pack(i, t_state_ns, P, H),
            pos.numpy().reshape(-1).tobytes(),
            vel.numpy().reshape(-1).tobytes(),
            goal.numpy().reshape(-1).tobytes(),
        ]
        pub.send_multipart(frames)
        rep = sub.recv_multipart()
        t1 = time.perf_counter()
        time_record.append(t1 - t0)
        if len(rep) != 2:
            print("[client] bad reply")
            continue

        step_ref, t_state_ns, compute_ms = ACT_HDR.unpack(rep[0])
        action = np.frombuffer(rep[1], dtype=np.float32)
        total_time = (now_ns() - t_state_ns) / 1e6
        if total_time > 25.0:
            print(f"Total time: {total_time:.3f}ms, Compute={compute_ms:.3f}ms, Network delay={total_time-compute_ms:.3f}")
        # print("action: ", action)
        # print("Time spent:", t1 - t0, "Desired time:", dt * ctr_period)

        # ---- 执行 chosen action in MuJoCo ----
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

            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0, 0.5 * 9.81])  # 施加力

            mujoco.mj_step(model, data)

        viewer.sync()  # let viewer show updated info

        goal_his.append(global_goal_action_pos)
        slider_pos.append(data.xpos[slider_id].copy())
        rope_top.append(data.site_xpos[-1].copy())

        action_history.append(action)
        vel_history.append(vel.clone())
        pos_history.append(pos.clone())

task_t_end = time.perf_counter()
print("average time on inference: ", torch.tensor(time_record).mean(), "Desired time:", dt * ctr_period)
print("Task time:", T_task, "   Spent time:", task_t_end-task_t_start)

# === 动画/轨迹（保留）===
pos_history = torch.cat(pos_history, dim=0)
plot_tip_vs_goal_and_error(pos_history, Goal_traj[1:], dt, ctr_period)

# === action 曲线（保留）===
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

# === velocity 曲线（保留）===
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
# 创建时间轴
time = list(range(len(goal_his)))

# 将数据转换为numpy数组以便于操作
goal_array = np.array(goal_his)
slider_pos_array = np.array(slider_pos)
rope_top_array = np.array(rope_top)
ctr_history = np.array(action_history)

# 创建带有三个子图的图形
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# 绘制x轴误差
ax1.plot(time, goal_array[:, 0], label='Goal', color='blue', linestyle='-')
ax1.plot(time, slider_pos_array[:, 0], label='Slider', color='green', linestyle='--')
ax1.plot(time, rope_top_array[:, 0], label='Rope Top', color='red', linestyle=':')
ax1.set_title('X-axis Position Tracking')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('X Position')
ax1.grid(True)
ax1.legend()

# 绘制y轴误差
ax2.plot(time, goal_array[:, 1], label='Goal', color='blue', linestyle='-')
ax2.plot(time, slider_pos_array[:, 1], label='Slider', color='green', linestyle='--')
ax2.plot(time, rope_top_array[:, 1], label='Rope Top', color='red', linestyle=':')
ax2.set_title('Y-axis Position Tracking')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Y Position')
ax2.grid(True)
ax2.legend()

# 绘制z轴误差
ax3.plot(time, goal_array[:, 2], label='Goal', color='blue', linestyle='-')
ax3.plot(time, slider_pos_array[:, 2], label='Slider', color='green', linestyle='--')
ax3.plot(time, rope_top_array[:, 2], label='Rope Top', color='red', linestyle=':')
ax3.set_title('Z-axis Position Tracking')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Z Position')
ax3.grid(True)
ax3.legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
