# Standalone inference speed of the execution dynamics model, with vs. without history encoder.
# One call = predict 750 future EE steps from history (+ current state) and the commanded trajectory.
# Usage (run from rocco_server/): python bench_standalone.py <repo_root> <no_enc_ckpt> <enc_ckpt> <out_json>
import json, os, sys, time, types

repo, ckpt_noenc, ckpt_enc, out_json = sys.argv[1:5]
sys.modules["pandas"] = types.ModuleType("pandas")  # franka_model imports pandas only for CSV reading
sys.path.insert(0, os.path.join(repo, "rocco_server"))
import franka_model  # noqa: F401
import warp as wp
import warp_franka_model as wfm
del sys.modules["pandas"]

import numpy as np
import torch
from franka_model import SparseGRUFrankaModel

HIST, HORIZON = 31, 750
dev = torch.device("cuda")


class EncoderGRUFrankaModel(SparseGRUFrankaModel):
    def rollout(self, ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist, goal_p_future, goal_v_future):
        z = self._history_to_latent(ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist)
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


def sync():
    torch.cuda.synchronize(); wp.synchronize()


def interleaved(fns, n, warmup=5):
    """Time several callables in alternating order so drift affects all equally."""
    for _ in range(warmup):
        for f in fns.values():
            f()
    ts = {k: [] for k in fns}
    for _ in range(n):
        for k, f in fns.items():
            sync(); t0 = time.perf_counter(); f(); sync()
            ts[k].append((time.perf_counter() - t0) * 1e3)
    return {k: {"median_ms": float(np.median(v)), "mean_ms": float(np.mean(v)), "std_ms": float(np.std(v)),
                "p10_ms": float(np.percentile(v, 10)), "p90_ms": float(np.percentile(v, 90)), "n": n}
            for k, v in ts.items()}


def main():
    torch.manual_seed(0)
    res = {"gpu": torch.cuda.get_device_name(0), "horizon_steps": HORIZON, "hist_len": HIST,
           "ckpt_no_encoder": ckpt_noenc, "ckpt_encoder": ckpt_enc}
    m_noenc = load_model(SparseGRUFrankaModel, ckpt_noenc)
    m_enc = load_model(EncoderGRUFrankaModel, ckpt_enc)
    r_noenc = wfm.load_runner(ckpt_noenc, device="cuda")
    r_enc = wfm.load_runner(ckpt_enc, device="cuda")
    z0_enc = wp.to_torch(r_enc["z0"])  # (64,1) latent buffer read by the Warp kernel

    for bsz, n_torch, n_warp in [(1, 30, 300), (1001, 30, 100)]:
        hist = [torch.randn(bsz, HIST, 3, device=dev) * 0.1 for _ in range(4)]
        fut_p, fut_v = [torch.randn(bsz, HORIZON, 3, device=dev) * 0.1 for _ in range(2)]
        p0, v0 = hist[0][:, -1].contiguous(), hist[1][:, -1].contiguous()
        h1 = [h[:1] for h in hist]  # Warp kernel uses one shared h0 -> encoder on one (shared) history

        def warp_enc():
            with torch.no_grad():
                z0_enc.copy_(m_enc._history_to_latent(*h1).reshape(64, 1))
            wfm.rollout(r_enc, p0, v0, fut_p, fut_v)

        with torch.no_grad():
            t = interleaved({
                "pytorch_no_encoder": lambda: m_noenc.rollout(*hist, fut_p, fut_v),
                "pytorch_encoder": lambda: m_enc.rollout(*hist, fut_p, fut_v),
            }, n_torch)
            t.update(interleaved({
                "warp_no_encoder": lambda: wfm.rollout(r_noenc, p0, v0, fut_p, fut_v),
                "warp_encoder": warp_enc,
            }, n_warp))
            t.update(interleaved({
                "encoder_only_pytorch": lambda: m_enc._history_to_latent(*hist),
            }, 300))
        res[f"B={bsz}"] = t
        for k, v in t.items():
            print(f"B={bsz:5d} {k:22s} median {v['median_ms']:8.3f} ms  mean {v['mean_ms']:8.3f}  std {v['std_ms']:7.3f}  p10-p90 {v['p10_ms']:.3f}-{v['p90_ms']:.3f}", flush=True)

    json.dump(res, open(out_json, "w"), indent=2)
    print("saved", out_json)


if __name__ == "__main__":
    main()
