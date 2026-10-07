from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
from Mujoco_env.mj_utils import *
from common.utils import *

from M2PC import Planner, cost_fn
from common.rope_warp_4 import WarpRope, N, P


# === Parameters ===
L = 1.0
mass = 0.0025 * 40 / N
k = 10000 * 0.46
damping = 0.2
bending_k = 0.0006712 * 0.0
bending_damping = 0.000401
air_drag = 0.2206 / 1000
g = 10.07
dt = 0.001
mode = 'acc'  # 或 'vel'

ctr_period = 25  # 1000/ctr_period Hz
horizon = 30  # 20 doesn't work; 25 can work

# ===== 多模态参数 =====
n_sample = 400     # 总采样数 400
m_modes = 1        # 模态数（单模态=1）
assert n_sample % m_modes == 0

n_improve = 10  # 10 will be good. 1 also fine?
noise_scale = 1.5 * 1.0  # 1.5
action_dim = 3
limits = torch.tensor([-5.0 * 1.0, 5.0 * 1.0])  # 5.0
last_point_repeat = 100 * 0

# diversity 超参
top_k_good = 200
beta = 5000.0
wJ = 1000.0

# 只做 MuJoCo 后台测试，不开 viewer/动画
visualization = False

# === Setup device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
set_seed(0)

TRACKING_DIR = Path(__file__).resolve().parent / "tracking_data"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)


def build_trajectory_suite(device, ctr_period):
    traj_specs = []

    for total_time in [4.0, 5.0, 6.0]:
        total_horizon_t = int(1000 / ctr_period * total_time)
        points = half_dense_then_uniform(
            N=total_horizon_t + 1,
            ratio=0.3,
            sharpness=2.0,
            mode='exp',
            interval=(0.0, 2.0),
            plot=False,
        )
        traj_specs.append((f"sin_{total_time:.1f}s", sin_traj(points, width=0.45 * 2.0, plot=False, device=device), total_time))
        traj_specs.append((f"egg_{total_time:.1f}s", egg_traj(points, scale_x=0.38 * 2.0, scale_y=0.52 * 2.0, plot=False, device=device), total_time))
        traj_specs.append((f"eight_{total_time:.1f}s", eight_traj(points, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2, loops=1, plot=False, device=device), total_time))

    for total_time in [12.0, 15.0, 18.0]:
        total_horizon_t = int(1000 / ctr_period * total_time)
        for drawn_name in ['SpongeBob', 'flower', 'PatrickStar']:
            traj_specs.append((
                f"{drawn_name}_{total_time:.1f}s",
                build_goal_traj_from_drawn(
                    drawn_path=str(REPO_ROOT / 'my_trajs' / f'{drawn_name}.npy'),
                    total_horizon=total_horizon_t,
                    device=device,
                    z0=0.2,
                    scale_x=3.0,
                    scale_y=3.0,
                    keep_aspect=False,
                    sigma=0.0,
                    uniform_M=1000,
                    ratio=0.3,
                    sharpness=2.0,
                    interval=(0.0, 2.0),
                ),
                total_time,
            ))

    return traj_specs


def make_save_stem(traj_name):
    # egg_5.0s -> m2pc_egg_5s
    stem = traj_name.replace('.0s', 's')
    return f"m2pc_{stem}"


def create_mujoco_model_data():
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / 'Mujoco_env' / 'cable_show.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 4)
    model.opt.timestep = dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    data.ctrl[0] = 0
    data.ctrl[1] = 0
    data.ctrl[2] = 0
    return model, data


def create_planner():
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
        mode=mode,
    )
    planner = Planner(
        rope, cost_fn, dt, ctr_period, horizon,
        n_sample, n_improve, noise_scale, action_dim,
        limits=limits, device=device, mode=mode,
        m_modes=m_modes, top_k_good=top_k_good, beta=beta, wJ=wJ, standard_m2pc=True,
    )
    return planner


def run_single_trajectory(traj_name, goal_traj, total_time):
    model, data = create_mujoco_model_data()
    planner = create_planner()

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    cable_body_indices = [i for i, name in enumerate(body_names) if name and name.startswith('B_')]
    node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + node * 6 + 3))

    total_horizon = int(total_time / dt / ctr_period) + last_point_repeat
    goal_traj = extend_last_point(goal_traj, N=last_point_repeat)
    goal_np = goal_traj.detach().cpu().numpy()

    pcf = PositionCommandFilter(np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], dt)
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")

    action_history = []
    pos_history = []
    rope_endpoint_history = []
    rope_site_endpoint_history = []
    slider_pos_history = []
    goal_tracking_history = []
    time_record = []

    task_t_start = time.perf_counter()

    for i in range(total_horizon):
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + node * 3] = data.xpos[1:1 + node][::-1].reshape(-1)
        mj_state[0, 1 + node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)

        global_goal_tracking = goal_np[i + 1]
        model.site_pos[1][[0, 1, 2]] = global_goal_tracking

        if i < (total_horizon - horizon):
            goal = goal_traj[i + 1:i + 1 + horizon, :]
        else:
            goal = goal_traj[i + 1:, :]

        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal)
        action = planner.get_action().cpu().numpy()
        t1 = time.perf_counter()
        time_record.append(t1 - t0)

        for _ in range(ctr_period):
            if mode == 'vel':
                pos_target = pcf.input_velocity(action)
            elif mode == 'acc':
                pos_target = pcf.input_acceleration(action)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            data.ctrl[0] = pos_target[0]
            data.ctrl[1] = pos_target[1]
            data.ctrl[2] = pos_target[2]
            model.site_pos[0][[0, 1, 2]] = pos_target + np.array([0.0, 0.0, 1.2])
            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0.0, 0.5 * 9.81])
            mujoco.mj_step(model, data)

        # rope endpoint used for tracking result. This matches the SPiD-style tip index -1.
        rope_endpoint = pos[0, -1, :].detach().cpu().numpy().copy()
        rope_site_endpoint = data.site_xpos[-1].copy()

        action_history.append(action.copy())
        pos_history.append(pos.detach().cpu())
        rope_endpoint_history.append(rope_endpoint)
        rope_site_endpoint_history.append(rope_site_endpoint)
        slider_pos_history.append(data.xpos[slider_id].copy())
        goal_tracking_history.append(global_goal_tracking.copy())

        planner.update_policy()

    task_t_end = time.perf_counter()

    rope_endpoint_array = np.asarray(rope_endpoint_history, dtype=np.float32)
    goal_array = np.asarray(goal_tracking_history, dtype=np.float32)
    error = rope_endpoint_array - goal_array[:rope_endpoint_array.shape[0]]
    err_norm = np.linalg.norm(error, axis=1)
    rmse_cm = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))) * 100.0)
    mean_err_cm = float(np.mean(err_norm) * 100.0)
    max_err_cm = float(np.max(err_norm) * 100.0)
    avg_infer_ms = float(np.mean(time_record) * 1000.0) if len(time_record) > 0 else float('nan')

    save_stem = make_save_stem(traj_name)
    endpoint_path = TRACKING_DIR / f"{save_stem}.npy"
    np.save(endpoint_path, rope_endpoint_array)

    # # Extra metadata file for later analysis/plotting. The endpoint .npy above is the requested rope endpoint trajectory.
    # np.savez(
    #     TRACKING_DIR / f"{save_stem}_data.npz",
    #     rope_endpoint=rope_endpoint_array,
    #     goal=goal_array,
    #     action=np.asarray(action_history, dtype=np.float32),
    #     rope_site_endpoint=np.asarray(rope_site_endpoint_history, dtype=np.float32),
    #     slider_pos=np.asarray(slider_pos_history, dtype=np.float32),
    #     dt=dt,
    #     ctr_period=ctr_period,
    #     total_time=total_time,
    # )

    return {
        'traj_name': traj_name,
        'save_stem': save_stem,
        'endpoint_path': endpoint_path,
        'total_time': total_time,
        'frames': rope_endpoint_array.shape[0],
        'rope_endpoint': rope_endpoint_array,
        'goal': goal_array,
        'rmse_cm': rmse_cm,
        'mean_err_cm': mean_err_cm,
        'max_err_cm': max_err_cm,
        'avg_infer_ms': avg_infer_ms,
        'task_time_s': task_t_end - task_t_start,
    }


def print_result(result):
    print(
        f"[Eval] {result['traj_name']:<20} frames={result['frames']:>4} "
        f"rmse={result['rmse_cm']:>7.2f}cm mean={result['mean_err_cm']:>7.2f}cm "
        f"max={result['max_err_cm']:>7.2f}cm infer={result['avg_infer_ms']:>7.2f}ms "
        f"saved={result['endpoint_path'].name}"
    )


def save_summary_csv(results):
    csv_path = TRACKING_DIR / "m2pc_18traj_summary.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('traj_name,total_time,frames,rmse_cm,mean_err_cm,max_err_cm,avg_infer_ms,task_time_s,endpoint_file\n')
        for r in results:
            f.write(
                f"{r['traj_name']},{r['total_time']},{r['frames']},"
                f"{r['rmse_cm']:.6f},{r['mean_err_cm']:.6f},{r['max_err_cm']:.6f},"
                f"{r['avg_infer_ms']:.6f},{r['task_time_s']:.6f},{r['endpoint_path'].name}\n"
            )
    return csv_path


def plot_all_results(results):
    fig = plt.figure(figsize=(24, 12))
    all_points = np.concatenate(
        [
            np.concatenate((result['goal'][:result['rope_endpoint'].shape[0]], result['rope_endpoint']), axis=0)
            for result in results
        ],
        axis=0,
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius == 0:
        radius = 1e-6

    for idx, result in enumerate(results):
        ax = fig.add_subplot(3, 6, idx + 1, projection='3d')
        rope_endpoint = result['rope_endpoint']
        goal = result['goal'][:rope_endpoint.shape[0]]

        ax.plot(goal[:, 0], goal[:, 1], goal[:, 2], label='Goal', linewidth=1.5)
        ax.plot(rope_endpoint[:, 0], rope_endpoint[:, 1], rope_endpoint[:, 2], label='Rope endpoint', linewidth=1.2)
        ax.set_title(f"{result['traj_name']}\nRMSE {result['rmse_cm']:.1f} cm", fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=90, azim=-90)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('M2PC MuJoCo Tracking: Rope Endpoint vs Goal Trajectory', fontsize=16)
    plt.tight_layout()
    fig_path = TRACKING_DIR / "m2pc_18traj_tracking_summary.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()
    return fig_path


def main():
    traj_specs = build_trajectory_suite(device, ctr_period)
    print(f"Evaluating {len(traj_specs)} trajectories in MuJoCo without viewer")
    print(f"Saving rope endpoint trajectories to: {TRACKING_DIR}")

    results = []
    for traj_idx, (traj_name, goal_traj, total_time) in enumerate(traj_specs, start=1):
        print("=" * 100)
        print(f"[{traj_idx}/{len(traj_specs)}] Running {traj_name}, total_time={total_time:.1f}s")
        result = run_single_trajectory(traj_name, goal_traj, total_time)
        print_result(result)
        results.append(result)

    print("=" * 100)
    mean_rmse = float(np.mean([r['rmse_cm'] for r in results]))
    worst = max(results, key=lambda r: r['rmse_cm'])
    print(f"[Summary] trajectories={len(results)} mean_rmse={mean_rmse:.2f}cm")
    print(f"[Summary] worst={worst['traj_name']} rmse={worst['rmse_cm']:.2f}cm max={worst['max_err_cm']:.2f}cm")

    csv_path = save_summary_csv(results)
    fig_path = plot_all_results(results)
    print(f"[Save] summary csv: {csv_path}")
    print(f"[Save] summary figure: {fig_path}")


if __name__ == '__main__':
    main()
