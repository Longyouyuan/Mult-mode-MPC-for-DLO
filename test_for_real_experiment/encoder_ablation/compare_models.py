# Accuracy + timing comparison: deployed encoder-free model vs. history-encoder model.
# Usage (from anywhere):
#   python compare_models.py <repo_root> <npz_dir> <no_enc_ckpt> <enc_ckpt> <out_json>
import json, os, sys, time, types

repo, npz_dir, ckpt_noenc, ckpt_enc, out_json = sys.argv[1:6]
exp_dir = os.path.join(repo, "test_for_real_experiment")
rocco_dir = os.path.join(repo, "rocco_server")

sys.modules["pandas"] = types.ModuleType("pandas")  # franka_model imports pandas only for CSV reading
sys.path.insert(0, rocco_dir)  # rocco_server/franka_model.py is what the Warp modules import
import franka_model  # noqa: F401  (load under the stub)
import warp as wp
import warp_franka_model as wfm
from warp_rope_franka_model import WarpRopeFranka, P
del sys.modules["pandas"]

import numpy as np
import torch
from torch.utils.data import DataLoader

from franka_model import (FrankaSparseWindowDataset, SparseGRUFrankaModel, compute_loss, split_train_val)

HIST, HIST_STRIDE, HORIZON, STRIDE = 31, 10, 750, 20
B_REAL = 1001  # m_modes + n_sample in rocco_server/zmq_topic_m2pc_lastrope_with_franka.py
dev = torch.device("cuda")


class EncoderGRUFrankaModel(SparseGRUFrankaModel):
    def rollout(self, ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist, goal_p_future, goal_v_future):
        z = self._history_to_latent(ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist)
        return self._roll(z, ee_p_hist, ee_v_hist, goal_p_future, goal_v_future)

    def _roll(self, z, ee_p_hist, ee_v_hist, goal_p_future, goal_v_future):
        p_now, v_now = ee_p_hist[:, -1, :], ee_v_hist[:, -1, :]
        pred_p, pred_v = [], []
        for k in range(goal_p_future.shape[1]):
            x = torch.cat([p_now, v_now, goal_p_future[:, k, :], goal_v_future[:, k, :]], dim=-1)
            z = self.rollout_cell(x, z)
            a = self.acc_head(torch.cat([z, x], dim=-1))
            v_now = v_now + self.dt * a
            p_now = p_now + self.dt * v_now
            pred_p.append(p_now); pred_v.append(v_now)
        return torch.stack(pred_p, 1), torch.stack(pred_v, 1), {}


def load_model(cls, path):
    c = torch.load(path, map_location=dev, weights_only=False)
    m = cls(HIST, float(c["dt"])).to(dev).eval()
    m.load_state_dict(c["model_state"])
    return m


@torch.no_grad()
def accuracy(model, dataset):
    """Window-weighted metrics over a dataset (cm, cm/s)."""
    sums = {"max_err": 0.0, "sq_p": 0.0, "sq_v": 0.0, "final_err": 0.0}
    n = 0
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        batch = {k: v.to(dev) for k, v in batch.items()}
        pp, pv, _ = model.rollout(batch["ee_p_hist"], batch["ee_v_hist"], batch["goal_p_hist"],
                                  batch["goal_v_hist"], batch["goal_p_future"], batch["goal_v_future"])
        ep = torch.linalg.norm(pp - batch["target_p"], dim=-1)  # (B, T)
        ev = torch.linalg.norm(pv - batch["target_v"], dim=-1)
        b = ep.shape[0]
        sums["max_err"] += ep.max(1).values.sum().item()
        sums["final_err"] += ep[:, -1].sum().item()
        sums["sq_p"] += (ep ** 2).mean(1).sum().item()
        sums["sq_v"] += (ev ** 2).mean(1).sum().item()
        n += b
    return {"avg_max_pos_err_cm": sums["max_err"] / n * 100, "final_pos_err_cm": sums["final_err"] / n * 100,
            "pos_rmse_cm": (sums["sq_p"] / n) ** 0.5 * 100, "vel_rmse_cm_s": (sums["sq_v"] / n) ** 0.5 * 100,
            "windows": n}


def timeit(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); wp.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); wp.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); wp.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "std_ms": float(ts.std()), "median_ms": float(np.median(ts)), "n": n}


def main():
    res = {"gpu": torch.cuda.get_device_name(0), "ckpt_no_encoder": ckpt_noenc, "ckpt_encoder": ckpt_enc}
    m_noenc = load_model(SparseGRUFrankaModel, ckpt_noenc)
    m_enc = load_model(EncoderGRUFrankaModel, ckpt_enc)

    # ---------------- accuracy ----------------
    d2 = dict(np.load(os.path.join(npz_dir, "ee_collection_run2.npz")))
    sp = split_train_val(len(d2["t_s"]), 0.15)
    sets = {"val (run2 last 15%)": FrankaSparseWindowDataset(d2, sp["val"], HIST, HIST_STRIDE, HORIZON, STRIDE)}
    for n in ["ee_collection_run1", "ee_collection_run3"]:
        d = dict(np.load(os.path.join(npz_dir, n + ".npz")))
        sets[f"test {n[-4:]}"] = FrankaSparseWindowDataset(d, (0, len(d["t_s"])), HIST, HIST_STRIDE, HORIZON, STRIDE)
    res["accuracy"] = {}
    for name, ds in sets.items():
        res["accuracy"][name] = {"no_encoder": accuracy(m_noenc, ds), "encoder": accuracy(m_enc, ds)}
        print(name, json.dumps(res["accuracy"][name]), flush=True)

    # ---------------- timing: PyTorch model rollout ----------------
    torch.manual_seed(0)
    res["timing_torch"] = {}
    for bsz in [1, B_REAL]:
        hist = [torch.randn(bsz, HIST, 3, device=dev) * 0.1 for _ in range(4)]
        fut = [torch.randn(bsz, HORIZON, 3, device=dev) * 0.1 for _ in range(2)]
        with torch.no_grad():
            r = {
                "no_encoder_rollout": timeit(lambda: m_noenc.rollout(*hist, *fut), 10),
                "encoder_rollout(per-sample encoder)": timeit(lambda: m_enc.rollout(*hist, *fut), 10),
                "encoder_only(shared, B=1)": timeit(lambda: m_enc._history_to_latent(*[h[:1] for h in hist]), 200),
                "encoder_only(per-sample)": timeit(lambda: m_enc._history_to_latent(*hist), 200),
            }
        res["timing_torch"][f"B={bsz}, {HORIZON} steps"] = r
        print("torch", bsz, json.dumps(r), flush=True)

    # ---------------- timing: deployed Warp planning step (rope + Franka) ----------------
    # The Warp kernel takes one latent h0 shared by all candidates (runner["z0"]).
    # No encoder: h0 = learned constant. Encoder: h0 = encoder(history) computed once per MPC step.
    runners = {"no_encoder": wfm.load_runner(ckpt_noenc, device="cuda"), "encoder": wfm.load_runner(ckpt_enc, device="cuda")}
    pos = torch.zeros(1, P, 3, device=dev); pos[0, :, 2] = -torch.linspace(0, 0.5, P, device=dev)
    vel = torch.zeros(1, P, 3, device=dev)
    act = torch.randn(B_REAL, HORIZON // 25, 3, device=dev)
    g0 = torch.zeros(3, device=dev)
    hist1 = [torch.randn(1, HIST, 3, device=dev) * 0.1 for _ in range(4)]
    res["timing_warp"] = {}
    for name, runner in runners.items():
        rope = WarpRopeFranka(franka_runner=runner, batch_size=B_REAL, L=0.5,
                              segment_lengths=[0.0973, 0.0996, 0.0993, 0.0986, 0.0990][: P - 1],
                              mass=12.8e-3 / (P - 1), tip_extra_mass=15e-3, k=1200.0, damping=0.2,
                              bending_k=0.0006712, bending_damping=0.000401, air_drag=0.3006e-3, g=10.27,
                              dt=1e-3, max_record_steps=HORIZON, record_interval=25, ctr_period=25, device="cuda")
        z0_t = wp.to_torch(runner["z0"])  # (64, 1) view onto the kernel's latent buffer

        if name == "encoder":
            def step():
                with torch.no_grad():
                    z0_t.copy_(m_enc._history_to_latent(*hist1).reshape(64, 1))
                rope.set_state_and_action(pos, vel, act, g0, g0)
                rope.simulate(steps=HORIZON)
        else:
            def step():
                rope.set_state_and_action(pos, vel, act, g0, g0)
                rope.simulate(steps=HORIZON)
        res["timing_warp"][name] = timeit(step, 50)
        print("warp", name, json.dumps(res["timing_warp"][name]), flush=True)

    json.dump(res, open(out_json, "w"), indent=2)
    print("saved", out_json)


if __name__ == "__main__":
    main()
