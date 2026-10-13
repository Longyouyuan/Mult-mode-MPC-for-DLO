# History-encoder ablation for the GRU execution dynamics model (franka_model.SparseGRUFrankaModel).
# Usage: python train_encoder_ablation.py <repo>/test_for_real_experiment <npz_dir> <out_dir>
#            --lr LR --epochs E [--init CKPT] [--no-encoder] [--warm-equal] [--resume]
# The deployed model was trained in stages (lr 1e-3, then 2e-4, each resumed from the previous best).
# --init reproduces the original script's auto-resume: load weights, fresh Adam, best_val = val error of loaded model.
# --no-encoder trains the deployed architecture (learned constant h0) instead, as the control.
# --warm-equal (encoder only): after --init from an encoder-free ckpt, set init_rollout W=0, b=atanh(z0) so the
#   encoder model starts functionally identical to the loaded encoder-free model.
# --test-each-epoch: also evaluate run1/run3 after every epoch (logged only; checkpoint selection stays on val)
#   and keep a checkpoint per epoch (epoch_XXX.pt).
import json, os, sys, time, types
sys.modules.setdefault("pandas", types.ModuleType("pandas"))  # franka_model imports pandas only for CSV reading

import numpy as np
import torch
from torch.utils.data import DataLoader

exp_dir, npz_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
resume = "--resume" in sys.argv
opt_arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
sys.path.insert(0, exp_dir)
from franka_model import (FrankaSparseWindowDataset, SparseGRUFrankaModel, evaluate,
                          split_train_val, train_one_epoch)
del sys.modules["pandas"]  # drop the stub so torch._dynamo does not trip on it

# Same as deployed checkpoint args (franka_dynamic/best_model.pt)
ARGS = dict(dt_ms=1.0, hist_len=31, hist_stride=10, horizon=750, stride=20, val_ratio=0.15,
            batch_size=256, epochs=int(opt_arg("--epochs")), lr=float(opt_arg("--lr")), rollout_weight=1.0,
            encoder_hidden=64, rollout_hidden=64, residual_hidden=64, seed=0, use_encoder="--no-encoder" not in sys.argv,
            warm_equal="--warm-equal" in sys.argv,
            init_ckpt=opt_arg("--init"), test_each_epoch="--test-each-epoch" in sys.argv)


class EncoderGRUFrankaModel(SparseGRUFrankaModel):
    """Identical to SparseGRUFrankaModel but h0 comes from the history encoder."""

    def rollout(self, ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist, goal_p_future, goal_v_future):
        z = self._history_to_latent(ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist)
        p_now, v_now = ee_p_hist[:, -1, :], ee_v_hist[:, -1, :]
        pred_p, pred_v, acc_seq = [], [], []
        for k in range(goal_p_future.shape[1]):
            x = torch.cat([p_now, v_now, goal_p_future[:, k, :], goal_v_future[:, k, :]], dim=-1)
            z = self.rollout_cell(x, z)
            a = self.acc_head(torch.cat([z, x], dim=-1))
            v_now = v_now + self.dt * a
            p_now = p_now + self.dt * v_now
            pred_p.append(p_now); pred_v.append(v_now); acc_seq.append(a)
        return torch.stack(pred_p, 1), torch.stack(pred_v, 1), {"acc": torch.stack(acc_seq, 1)}


def load(name):
    return dict(np.load(os.path.join(npz_dir, name + ".npz")))


def ds(data, rng):
    return FrankaSparseWindowDataset(data, rng, ARGS["hist_len"], ARGS["hist_stride"], ARGS["horizon"], ARGS["stride"])


def main():
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(ARGS["seed"]); np.random.seed(ARGS["seed"])
    dev = torch.device("cuda")

    train_data = load("ee_collection_run2")
    sp = split_train_val(len(train_data["t_s"]), ARGS["val_ratio"])
    train_ds, val_ds = ds(train_data, sp["train"]), ds(train_data, sp["val"])
    test_sets = {}
    for n in ["ee_collection_run1", "ee_collection_run3"]:
        td = load(n)
        test_sets[n] = ds(td, (0, len(td["t_s"])))
    sizes = {"train": len(train_ds), "val": len(val_ds), **{f"test_{k}": len(v) for k, v in test_sets.items()}}
    print("splits", sp, "sizes", sizes, flush=True)

    train_loader = DataLoader(train_ds, batch_size=ARGS["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=ARGS["batch_size"], shuffle=False, num_workers=0)

    model_cls = EncoderGRUFrankaModel if ARGS["use_encoder"] else SparseGRUFrankaModel
    model = model_cls(ARGS["hist_len"], float(train_data["dt"]), ARGS["encoder_hidden"],
                      ARGS["rollout_hidden"], ARGS["residual_hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ARGS["lr"])
    best_ckpt, last_ckpt = os.path.join(out_dir, "best_model.pt"), os.path.join(out_dir, "last.pt")
    history, best_val, start = [], float("inf"), 1
    if ARGS["init_ckpt"] and not (resume and os.path.exists(last_ckpt)):
        model.load_state_dict(torch.load(ARGS["init_ckpt"], map_location=dev, weights_only=False)["model_state"])
        if ARGS["use_encoder"] and ARGS["warm_equal"]:
            with torch.no_grad():
                model.init_rollout.weight.zero_()
                model.init_rollout.bias.copy_(torch.atanh(model.learned_z0[0].clamp(-0.999, 0.999)))
        best_val = evaluate(model, val_loader, dev)["avg_max_pos_err"]
        print(f"init from {ARGS['init_ckpt']}: val {best_val * 100:.3f} cm", flush=True)
        torch.save({"model_state": model.state_dict(), "args": ARGS, "dt": float(train_data["dt"]), "epoch": 0}, best_ckpt)
    if resume and os.path.exists(last_ckpt):
        c = torch.load(last_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(c["model_state"]); opt.load_state_dict(c["opt_state"])
        history, best_val, start = c["history"], c["best_val"], c["epoch"] + 1
        torch.set_rng_state(c["rng"].cpu()); print("resumed at epoch", start, flush=True)

    for ep in range(start, ARGS["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, dev, ARGS["rollout_weight"])
        va = evaluate(model, val_loader, dev, ARGS["rollout_weight"])
        rec = {"epoch": ep, "train": tr, "val": va, "sec": time.time() - t0}
        msg = ""
        if ARGS["test_each_epoch"]:
            rec["test"] = {n: evaluate(model, DataLoader(d, batch_size=ARGS["batch_size"]), dev) for n, d in test_sets.items()}
            errs = {n: m["avg_max_pos_err"] for n, m in rec["test"].items()}
            n_win = {n: len(d) for n, d in test_sets.items()}
            pooled = sum(errs[n] * n_win[n] for n in errs) / sum(n_win.values())
            msg = " | test " + " ".join(f"{n[-4:]} {e * 100:.3f}" for n, e in errs.items()) + f" pooled {pooled * 100:.3f} cm"
            torch.save({"model_state": model.state_dict(), "args": ARGS, "dt": float(train_data["dt"]), "epoch": ep},
                       os.path.join(out_dir, f"epoch_{ep:03d}.pt"))
        history.append(rec)
        print(f"epoch {ep:03d} | {time.time() - t0:.0f}s | train {tr['avg_max_pos_err'] * 100:.3f} cm "
              f"| val {va['avg_max_pos_err'] * 100:.3f} cm | val loss {va['loss']:.3e}{msg}", flush=True)
        if va["avg_max_pos_err"] < best_val:
            best_val = va["avg_max_pos_err"]
            torch.save({"model_state": model.state_dict(), "args": ARGS, "dt": float(train_data["dt"]), "epoch": ep}, best_ckpt)
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "history": history,
                    "best_val": best_val, "epoch": ep, "rng": torch.get_rng_state()}, last_ckpt)

    model.load_state_dict(torch.load(best_ckpt, map_location=dev, weights_only=False)["model_state"])
    res = {"args": ARGS, "splits": sp, "dataset_sizes": sizes, "history": history,
           "best_epoch": torch.load(best_ckpt, weights_only=False)["epoch"],
           "val_metrics": evaluate(model, val_loader, dev), "test_metrics": {}}
    for n, d in test_sets.items():
        res["test_metrics"][n] = evaluate(model, DataLoader(d, batch_size=ARGS["batch_size"]), dev)
        print("test", n, f"{res['test_metrics'][n]['avg_max_pos_err'] * 100:.3f} cm", flush=True)
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
