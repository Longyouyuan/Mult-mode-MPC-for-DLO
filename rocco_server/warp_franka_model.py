from __future__ import annotations
import time

import numpy as np
import torch
import warp as wp

from franka_model import SparseGRUFrankaModel

wp.init()
wp.set_module_options({"enable_backward": False})

# fixed model sizes
H = wp.constant(64)
R = wp.constant(64)
X_DIM = wp.constant(12)
FEAT_DIM = wp.constant(76)
OUT_DIM = wp.constant(3)
ONE = wp.constant(1)
TILE_THREADS = 32 * 2 # or 64


# ------------------------------
# checkpoint / weight packing
# ------------------------------

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _col(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1).astype(np.float32)


def load_runner(best_model_path: str, device: str = "cuda") -> dict:
    ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = SparseGRUFrankaModel(
        hist_len=int(args["hist_len"]),
        dt=float(ckpt.get("dt", args.get("dt_ms", 1.0) * 1.0e-3)),
        encoder_hidden=int(args.get("encoder_hidden", 64)),
        rollout_hidden=64,
        residual_hidden=64,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    state = model.state_dict()

    w_ih = _to_np(state["rollout_cell.weight_ih"])   # (192, 12)
    w_hh = _to_np(state["rollout_cell.weight_hh"])   # (192, 64)
    b_ih = _to_np(state["rollout_cell.bias_ih"])     # (192,)
    b_hh = _to_np(state["rollout_cell.bias_hh"])     # (192,)

    def split3(w: np.ndarray):
        return w[0:64], w[64:128], w[128:192]

    w_ir, w_iz, w_in = split3(w_ih)
    w_hr, w_hz, w_hn = split3(w_hh)
    b_ir, b_iz, b_in = split3(b_ih)
    b_hr, b_hz, b_hn = split3(b_hh)

    fc1_w = _to_np(state["acc_head.net.0.weight"])   # (64, 76)
    fc1_w_h = fc1_w[:, :64].copy()                    # (64, 64)
    fc1_w_x = fc1_w[:, 64:].copy()                    # (64, 12)
    fc1_b = _to_np(state["acc_head.net.0.bias"])     # (64,)
    fc2_w = _to_np(state["acc_head.net.2.weight"])   # (64, 64)
    fc2_b = _to_np(state["acc_head.net.2.bias"])     # (64,)
    fc3_w = _to_np(state["acc_head.net.4.weight"])   # (3, 64)
    fc3_b = _to_np(state["acc_head.net.4.bias"])     # (3,)
    z0 = _to_np(state["learned_z0"]).reshape(-1)     # (64,)

    return {
        "device": device,
        "dt": float(model.dt),

        "z0": wp.array(_col(z0), dtype=wp.float32, device=device),       # (64,1)

        "w_ir": wp.array(w_ir, dtype=wp.float32, device=device),         # (64,12)
        "w_iz": wp.array(w_iz, dtype=wp.float32, device=device),
        "w_in": wp.array(w_in, dtype=wp.float32, device=device),

        "w_hr": wp.array(w_hr, dtype=wp.float32, device=device),         # (64,64)
        "w_hz": wp.array(w_hz, dtype=wp.float32, device=device),
        "w_hn": wp.array(w_hn, dtype=wp.float32, device=device),

        "b_r":  wp.array(_col(b_ir + b_hr), dtype=wp.float32, device=device),  # (64,1) pre-fused
        "b_z":  wp.array(_col(b_iz + b_hz), dtype=wp.float32, device=device),  # (64,1) pre-fused
        "b_in": wp.array(_col(b_in), dtype=wp.float32, device=device),
        "b_hn": wp.array(_col(b_hn), dtype=wp.float32, device=device),

        "fc1_w_h": wp.array(fc1_w_h, dtype=wp.float32, device=device),   # (64,64)
        "fc1_w_x": wp.array(fc1_w_x, dtype=wp.float32, device=device),   # (64,12)
        "fc1_b": wp.array(_col(fc1_b), dtype=wp.float32, device=device), # (64,1)

        "fc2_w": wp.array(fc2_w, dtype=wp.float32, device=device),       # (64,64)
        "fc2_b": wp.array(_col(fc2_b), dtype=wp.float32, device=device),

        "fc3_w": wp.array(fc3_w, dtype=wp.float32, device=device),       # (3,64)
        "fc3_b": wp.array(_col(fc3_b), dtype=wp.float32, device=device), # (3,1)
    }


# ------------------------------
# scalar math
# ------------------------------

@wp.func
def sigmoid_sum3(a: float, b: float, c: float) -> float:
    s = a + b + c
    return 1.0 / (1.0 + wp.exp(-s))


@wp.func
def tanh_fma(a: float, b: float, c: float) -> float:
    """tanh(a + b + c)"""
    s = a + b + c
    ex = wp.exp(s)
    enx = wp.exp(-s)
    return (ex - enx) / (ex + enx)


@wp.func
def relu_sum3(a: float, b: float, c: float) -> float:
    return wp.max(a + b + c, 0.0)


@wp.func
def relu_sum(a: float, b: float) -> float:
    return wp.max(a + b, 0.0)


@wp.func
def gru_mix(z_val: float, n_val: float, h_val: float) -> float:
    """(1-z)*n + z*h  ==  n + z*(h-n)"""
    return n_val + z_val * (h_val - n_val)


# ------------------------------
# kernel
# ------------------------------

@wp.kernel(enable_backward=False)
def rollout_kernel(
    p0: wp.array(dtype=wp.vec3, ndim=1),
    v0: wp.array(dtype=wp.vec3, ndim=1),
    gp: wp.array(dtype=wp.vec3, ndim=2),
    gv: wp.array(dtype=wp.vec3, ndim=2),

    z0: wp.array(dtype=wp.float32, ndim=2),

    w_ir: wp.array(dtype=wp.float32, ndim=2),
    w_iz: wp.array(dtype=wp.float32, ndim=2),
    w_in: wp.array(dtype=wp.float32, ndim=2),

    w_hr: wp.array(dtype=wp.float32, ndim=2),
    w_hz: wp.array(dtype=wp.float32, ndim=2),
    w_hn: wp.array(dtype=wp.float32, ndim=2),

    b_r:  wp.array(dtype=wp.float32, ndim=2),
    b_z:  wp.array(dtype=wp.float32, ndim=2),
    b_in: wp.array(dtype=wp.float32, ndim=2),
    b_hn: wp.array(dtype=wp.float32, ndim=2),

    fc1_w_h: wp.array(dtype=wp.float32, ndim=2),
    fc1_w_x: wp.array(dtype=wp.float32, ndim=2),
    fc1_b: wp.array(dtype=wp.float32, ndim=2),

    fc2_w: wp.array(dtype=wp.float32, ndim=2),
    fc2_b: wp.array(dtype=wp.float32, ndim=2),

    fc3_w: wp.array(dtype=wp.float32, ndim=2),
    fc3_b: wp.array(dtype=wp.float32, ndim=2),

    dt: float,
    horizon: int,

    pred_p: wp.array(dtype=wp.vec3, ndim=2),
    pred_v: wp.array(dtype=wp.vec3, ndim=2),
):
    b = wp.tid()

    p = p0[b]
    v = v0[b]

    # initial latent
    h = wp.tile_load(z0, shape=(H, ONE), offset=(0, 0))

    # time-invariant weights/biases: load once
    w_ir_t = wp.tile_load(w_ir, shape=(H, X_DIM), offset=(0, 0))
    w_iz_t = wp.tile_load(w_iz, shape=(H, X_DIM), offset=(0, 0))
    w_in_t = wp.tile_load(w_in, shape=(H, X_DIM), offset=(0, 0))

    w_hr_t = wp.tile_load(w_hr, shape=(H, H), offset=(0, 0))
    w_hz_t = wp.tile_load(w_hz, shape=(H, H), offset=(0, 0))
    w_hn_t = wp.tile_load(w_hn, shape=(H, H), offset=(0, 0))

    b_r_t  = wp.tile_load(b_r,  shape=(H, ONE), offset=(0, 0))
    b_z_t  = wp.tile_load(b_z,  shape=(H, ONE), offset=(0, 0))
    b_in_t = wp.tile_load(b_in, shape=(H, ONE), offset=(0, 0))
    b_hn_t = wp.tile_load(b_hn, shape=(H, ONE), offset=(0, 0))

    fc1_wh_t = wp.tile_load(fc1_w_h, shape=(R, H), offset=(0, 0))
    fc1_wx_t = wp.tile_load(fc1_w_x, shape=(R, X_DIM), offset=(0, 0))
    fc1_b_t = wp.tile_load(fc1_b, shape=(R, ONE), offset=(0, 0))

    fc2_w_t = wp.tile_load(fc2_w, shape=(R, R), offset=(0, 0))
    fc2_b_t = wp.tile_load(fc2_b, shape=(R, ONE), offset=(0, 0))

    fc3_w_t = wp.tile_load(fc3_w, shape=(OUT_DIM, R), offset=(0, 0))
    fc3_b_t = wp.tile_load(fc3_b, shape=(OUT_DIM, ONE), offset=(0, 0))

    for t in range(horizon):
        pg_t = gp[b, t]
        vg_t = gv[b, t]

        # x = [p, v, p_goal, v_goal]  -> (12, 1)
        x = wp.tile_zeros(shape=(X_DIM, ONE), dtype=wp.float32)
        x[0, 0] = p[0]
        x[1, 0] = p[1]
        x[2, 0] = p[2]
        x[3, 0] = v[0]
        x[4, 0] = v[1]
        x[5, 0] = v[2]
        x[6, 0] = pg_t[0]
        x[7, 0] = pg_t[1]
        x[8, 0] = pg_t[2]
        x[9, 0] = vg_t[0]
        x[10, 0] = vg_t[1]
        x[11, 0] = vg_t[2]

        # ---- GRU gate matmuls
        ir = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        iz = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        inn = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hr = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hz = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)
        hn = wp.tile_zeros(shape=(H, ONE), dtype=wp.float32)

        wp.tile_matmul(w_ir_t, x, ir)
        wp.tile_matmul(w_iz_t, x, iz)
        wp.tile_matmul(w_in_t, x, inn)
        wp.tile_matmul(w_hr_t, h, hr)
        wp.tile_matmul(w_hz_t, h, hz)
        wp.tile_matmul(w_hn_t, h, hn)

        # ---- GRU gates (fused element-wise)
        r = wp.tile_map(sigmoid_sum3, ir, hr, b_r_t)
        z = wp.tile_map(sigmoid_sum3, iz, hz, b_z_t)
        r_hn_b = wp.tile_map(wp.mul, r, hn + b_hn_t)
        n = wp.tile_map(tanh_fma, inn, b_in_t, r_hn_b)
        h = wp.tile_map(gru_mix, z, n, h)

        # ---- MLP: W1 @ [h; x] = W1_h @ h + W1_x @ x
        h1 = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        h1x = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        h2 = wp.tile_zeros(shape=(R, ONE), dtype=wp.float32)
        acc = wp.tile_zeros(shape=(OUT_DIM, ONE), dtype=wp.float32)

        wp.tile_matmul(fc1_wh_t, h, h1)
        wp.tile_matmul(fc1_wx_t, x, h1x)
        h1 = wp.tile_map(relu_sum3, h1, h1x, fc1_b_t)

        wp.tile_matmul(fc2_w_t, h1, h2)
        h2 = wp.tile_map(relu_sum, h2, fc2_b_t)

        wp.tile_matmul(fc3_w_t, h2, acc)
        acc = acc + fc3_b_t

        a = wp.vec3(acc[0, 0], acc[1, 0], acc[2, 0])

        v = v + dt * a
        p = p + dt * v

        pred_p[b, t] = p
        pred_v[b, t] = v


# ------------------------------
# public API
# ------------------------------

def rollout(
    runner: dict,
    p0: torch.Tensor,            # (B, 3)
    v0: torch.Tensor,            # (B, 3)
    goal_p_future: torch.Tensor, # (B, T, 3)
    goal_v_future: torch.Tensor, # (B, T, 3)
):
    device = runner["device"]
    batch = int(p0.shape[0])
    horizon = int(goal_p_future.shape[1])

    p0_wp = wp.from_torch(p0.contiguous(), dtype=wp.vec3)
    v0_wp = wp.from_torch(v0.contiguous(), dtype=wp.vec3)
    gp_wp = wp.from_torch(goal_p_future.contiguous(), dtype=wp.vec3)
    gv_wp = wp.from_torch(goal_v_future.contiguous(), dtype=wp.vec3)

    pred_p = wp.empty((batch, horizon), dtype=wp.vec3, device=device)
    pred_v = wp.empty((batch, horizon), dtype=wp.vec3, device=device)

    wp.launch_tiled(
        rollout_kernel,
        dim=[batch],
        inputs=[
            p0_wp,
            v0_wp,
            gp_wp,
            gv_wp,

            runner["z0"],

            runner["w_ir"],
            runner["w_iz"],
            runner["w_in"],

            runner["w_hr"],
            runner["w_hz"],
            runner["w_hn"],

            runner["b_r"],
            runner["b_z"],
            runner["b_in"],
            runner["b_hn"],

            runner["fc1_w_h"],
            runner["fc1_w_x"],
            runner["fc1_b"],

            runner["fc2_w"],
            runner["fc2_b"],

            runner["fc3_w"],
            runner["fc3_b"],

            runner["dt"],
            horizon,
        ],
        outputs=[pred_p, pred_v],
        block_dim=TILE_THREADS,
        device=device,
    )

    wp.synchronize_device(device)
    return wp.to_torch(pred_p), wp.to_torch(pred_v)


__all__ = ["load_runner", "rollout"]


if __name__ == "__main__":
    from franka_model import load_and_resample, FrankaSparseWindowDataset

    runner = load_runner("best_model.pt", device="cuda")

    csv_path = "franka_data/ee_collection_run1.csv"
    tdata = load_and_resample(csv_path, dt_ms=1.0)
    tds = FrankaSparseWindowDataset(tdata, (0, len(tdata["t_s"])), 31, 10, 750, 20)

    batch = 64
    sample_indices = np.random.choice(len(tds), batch, replace=False)

    p0_list, v0_list, goal_p_list, goal_v_list = [], [], [], []
    ee_p_hist_list, ee_v_hist_list, goal_p_hist_list, goal_v_hist_list = [], [], [], []
    for idx in sample_indices:
        sample = tds[int(idx)]
        p0_list.append(sample["ee_p_hist"][-1])
        v0_list.append(sample["ee_v_hist"][-1])
        goal_p_list.append(sample["goal_p_future"])
        goal_v_list.append(sample["goal_v_future"])
        ee_p_hist_list.append(sample["ee_p_hist"])
        ee_v_hist_list.append(sample["ee_v_hist"])
        goal_p_hist_list.append(sample["goal_p_hist"])
        goal_v_hist_list.append(sample["goal_v_hist"])

    p0 = torch.stack(p0_list).to("cuda")
    v0 = torch.stack(v0_list).to("cuda")
    goal_p_future = torch.stack(goal_p_list).to("cuda")
    goal_v_future = torch.stack(goal_v_list).to("cuda")
    ee_p_hist = torch.stack(ee_p_hist_list).to("cuda")      # (32, hist_len, 3)
    ee_v_hist = torch.stack(ee_v_hist_list).to("cuda")      # (32, hist_len, 3)
    goal_p_hist = torch.stack(goal_p_hist_list).to("cuda")  # (32, hist_len, 3)
    goal_v_hist = torch.stack(goal_v_hist_list).to("cuda")  # (32, hist_len, 3)

    Compare_test = 1
    if Compare_test:
        pred_p_warp, pred_v_warp = rollout(runner, p0, v0, goal_p_future, goal_v_future)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("best_model.pt", map_location=device, weights_only=False)
        args = ckpt["args"]
        model = SparseGRUFrankaModel(
            hist_len=int(args["hist_len"]),
            dt=float(ckpt.get("dt", args.get("dt_ms", 1.0) * 1.0e-3)),
            encoder_hidden=int(args.get("encoder_hidden", 64)),
            rollout_hidden=64,
            residual_hidden=64,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            pred_p_torch, pred_v_torch, _ = model.rollout(
            ee_p_hist,
            ee_v_hist,
            goal_p_hist,
            goal_v_hist,
            goal_p_future,
            goal_v_future,
        )

        err_p = pred_p_warp - pred_p_torch
        err_v = pred_v_warp - pred_v_torch
        print("Position error (m): ", torch.norm(err_p, dim=-1).max())
        print("Velocity error (m/s): ", torch.norm(err_v, dim=-1).max())

    Speed_test = 1
    if Speed_test:
        for i in range(10):
            pred_p, pred_v = rollout(runner, p0, v0, goal_p_future, goal_v_future)
        wp.synchronize()
        t0 = time.perf_counter()
        for i in range(10):
            pred_p, pred_v = rollout(runner, p0, v0, goal_p_future, goal_v_future)
        wp.synchronize()
        t1 = time.perf_counter()
        print(f"Average rollout time: {(t1 - t0) / 10 * 1000:.8f} ms")

        print(pred_p.shape, pred_v.shape)