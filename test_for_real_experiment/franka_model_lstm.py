import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

XYZ = ["x", "y", "z"]


def load_and_resample(csv_path: str, dt_ms: float = 1.0) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = [
        "t_ns",
        "pos_goal_x", "pos_goal_y", "pos_goal_z",
        "vel_goal_x", "vel_goal_y", "vel_goal_z",
        "ee_pos_x", "ee_pos_y", "ee_pos_z",
        "ee_vel_x", "ee_vel_y", "ee_vel_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    t_ns = df["t_ns"].to_numpy(dtype=np.int64)
    t_rel_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    dt_s = dt_ms * 1e-3
    t_uniform = np.arange(0.0, t_rel_s[-1], dt_s, dtype=np.float64)

    def interp_cols(cols: List[str]) -> np.ndarray:
        arr = df[cols].to_numpy(dtype=np.float64)
        out = np.empty((len(t_uniform), arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(t_uniform, t_rel_s, arr[:, j]).astype(np.float32)
        return out

    return {
        "t_s": t_uniform.astype(np.float32),
        "goal_p": interp_cols([f"pos_goal_{a}" for a in XYZ]),
        "goal_v": interp_cols([f"vel_goal_{a}" for a in XYZ]),
        "ee_p": interp_cols([f"ee_pos_{a}" for a in XYZ]),
        "ee_v": interp_cols([f"ee_vel_{a}" for a in XYZ]),
        "dt": np.float32(dt_s),
        "raw_len": np.int64(len(df)),
        "resampled_len": np.int64(len(t_uniform)),
    }


def split_indices(total_len: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1")
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    n_test = total_len - n_train - n_val
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }


def split_train_val(total_len: int, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("Require 0 < val_ratio < 1")
    n_val = int(total_len * val_ratio)
    n_train = total_len - n_val
    return {
        "train": (0, n_train),
        "val": (n_train, total_len),
    }


def make_sparse_offsets(hist_len: int, hist_stride: int) -> np.ndarray:
    if hist_len <= 0 or hist_stride <= 0:
        raise ValueError("hist_len and hist_stride must be positive")
    offsets = np.arange(hist_len - 1, -1, -1, dtype=np.int64) * hist_stride
    return offsets


class FrankaSparseWindowDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        split_range: Tuple[int, int],
        hist_len: int,
        hist_stride: int,
        horizon: int,
        stride: int,
    ):
        self.hist_len = int(hist_len)
        self.hist_stride = int(hist_stride)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.hist_offsets = make_sparse_offsets(self.hist_len, self.hist_stride)
        self.max_back = int(self.hist_offsets[0])

        s0, s1 = split_range
        start_min = max(s0, self.max_back)
        start_max = s1 - self.horizon - 1
        if start_max <= start_min:
            raise ValueError("Split too short for chosen sparse history/horizon.")

        self.start_indices = np.arange(start_min, start_max, self.stride, dtype=np.int64)
        self.data = {
            "goal_p": data["goal_p"],
            "goal_v": data["goal_v"],
            "ee_p": data["ee_p"],
            "ee_v": data["ee_v"],
        }
        self.dt = float(data["dt"])

    def __len__(self):
        return len(self.start_indices)

    def _take_sparse_hist(self, arr: np.ndarray, t: int) -> np.ndarray:
        idx = t - self.hist_offsets
        return arr[idx]

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        ee_p_hist = self._take_sparse_hist(self.data["ee_p"], t)
        ee_v_hist = self._take_sparse_hist(self.data["ee_v"], t)
        goal_p_hist = self._take_sparse_hist(self.data["goal_p"], t)
        goal_v_hist = self._take_sparse_hist(self.data["goal_v"], t)

        goal_p_future = self.data["goal_p"][t: t + self.horizon]
        goal_v_future = self.data["goal_v"][t: t + self.horizon]
        target_p = self.data["ee_p"][t + 1: t + self.horizon + 1]
        target_v = self.data["ee_v"][t + 1: t + self.horizon + 1]

        return {
            "ee_p_hist": torch.from_numpy(ee_p_hist),
            "ee_v_hist": torch.from_numpy(ee_v_hist),
            "goal_p_hist": torch.from_numpy(goal_p_hist),
            "goal_v_hist": torch.from_numpy(goal_v_hist),
            "goal_p_future": torch.from_numpy(goal_p_future),
            "goal_v_future": torch.from_numpy(goal_v_future),
            "target_p": torch.from_numpy(target_p),
            "target_v": torch.from_numpy(target_v),
        }


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseLSTMFrankaModel(nn.Module):
    def __init__(
        self,
        hist_len: int,
        dt: float,
        encoder_hidden: int = 64,
        rollout_hidden: int = 64,
        residual_hidden: int = 64,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.dt = float(dt)
        self.encoder_hidden = int(encoder_hidden)
        self.rollout_hidden = int(rollout_hidden)

        self.history_encoder = nn.LSTM(
            input_size=12,
            hidden_size=self.encoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.init_rollout_h = nn.Linear(self.encoder_hidden, self.rollout_hidden)
        self.init_rollout_c = nn.Linear(self.encoder_hidden, self.rollout_hidden)
        self.rollout_cell = nn.LSTMCell(input_size=12, hidden_size=self.rollout_hidden)

        self.acc_head = ResidualMLP(
            in_dim=self.rollout_hidden + 12,
            hidden_dim=residual_hidden,
            out_dim=3,
        )

    def _history_to_latent(
        self,
        ee_p_hist: torch.Tensor,
        ee_v_hist: torch.Tensor,
        goal_p_hist: torch.Tensor,
        goal_v_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_seq = torch.cat([ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist], dim=-1)
        _, (h_n, c_n) = self.history_encoder(hist_seq)
        h0 = torch.tanh(self.init_rollout_h(h_n[-1]))
        c0 = torch.tanh(self.init_rollout_c(c_n[-1]))
        return h0, c0

    def rollout(
        self,
        ee_p_hist: torch.Tensor,
        ee_v_hist: torch.Tensor,
        goal_p_hist: torch.Tensor,
        goal_v_hist: torch.Tensor,
        goal_p_future: torch.Tensor,
        goal_v_future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h, c = self._history_to_latent(ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist)

        p_now = ee_p_hist[:, -1, :]
        v_now = ee_v_hist[:, -1, :]

        pred_p = []
        pred_v = []
        acc_seq = []

        horizon = goal_p_future.shape[1]
        for k in range(horizon):
            p_goal = goal_p_future[:, k, :]
            v_goal = goal_v_future[:, k, :]

            rollout_in = torch.cat([p_now, v_now, p_goal, v_goal], dim=-1)
            h, c = self.rollout_cell(rollout_in, (h, c))

            a = self.acc_head(torch.cat([h, p_now, v_now, p_goal, v_goal], dim=-1))

            v_next = v_now + self.dt * a
            p_next = p_now + self.dt * v_next

            pred_p.append(p_next)
            pred_v.append(v_next)
            acc_seq.append(a)

            p_now = p_next
            v_now = v_next

        aux = {
            "acc": torch.stack(acc_seq, dim=1),
        }
        return torch.stack(pred_p, dim=1), torch.stack(pred_v, dim=1), aux



def compute_loss(pred_p, pred_v, target_p, target_v, rollout_weight: float = 1.0):
    one_step_p = torch.mean((pred_p[:, 0, :] - target_p[:, 0, :]) ** 2)
    one_step_v = torch.mean((pred_v[:, 0, :] - target_v[:, 0, :]) ** 2)
    roll_p = torch.mean((pred_p - target_p) ** 2)
    roll_v = torch.mean((pred_v - target_v) ** 2)
    pos_err_norm = torch.linalg.norm(pred_p - target_p, dim=-1)
    avg_max_pos_err = pos_err_norm.max(dim=1).values.mean()
    loss = 0.5 * one_step_p + 1.0 * one_step_v + rollout_weight * (5.0 * roll_p + 1.0 * roll_v)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "one_step_p": float(one_step_p.detach().cpu()),
        "one_step_v": float(one_step_v.detach().cpu()),
        "roll_p": float(roll_p.detach().cpu()),
        "roll_v": float(roll_v.detach().cpu()),
        "avg_max_pos_err": float(avg_max_pos_err.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, device, rollout_weight: float = 1.0):
    model.eval()
    agg = None
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_p, pred_v, _ = model.rollout(
            batch["ee_p_hist"],
            batch["ee_v_hist"],
            batch["goal_p_hist"],
            batch["goal_v_hist"],
            batch["goal_p_future"],
            batch["goal_v_future"],
        )
        _, metrics = compute_loss(pred_p, pred_v, batch["target_p"], batch["target_v"], rollout_weight)
        if agg is None:
            agg = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            agg[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}



def train_one_epoch(model, loader, optimizer, device, rollout_weight: float = 1.0, grad_clip: float = 1.0):
    model.train()
    agg = None
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_p, pred_v, _ = model.rollout(
            batch["ee_p_hist"],
            batch["ee_v_hist"],
            batch["goal_p_hist"],
            batch["goal_v_hist"],
            batch["goal_p_future"],
            batch["goal_v_future"],
        )
        loss, metrics = compute_loss(pred_p, pred_v, batch["target_p"], batch["target_v"], rollout_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if agg is None:
            agg = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            agg[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}



def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_output_dir(path: str, keep: Tuple[str, ...] = ("best_model.pt",)):
    for name in os.listdir(path):
        if name in keep:
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)


def sample_rollout_indices(dataset_len: int, num_samples: int, seed: int) -> np.ndarray:
    if dataset_len <= 0:
        raise ValueError("Dataset is empty, cannot visualize rollouts.")

    picked = min(int(num_samples), int(dataset_len))
    if picked == 1:
        return np.array([0], dtype=np.int64)

    rng = np.random.default_rng(seed)
    remaining = np.arange(1, dataset_len, dtype=np.int64)
    sampled = rng.choice(remaining, size=picked - 1, replace=False)
    return np.concatenate([np.array([0], dtype=np.int64), np.sort(sampled)])


@torch.no_grad()
def save_rollout_plot(model, dataset, device, out_dir: str, title: str, num_samples: int = 20, seed: int = 0):
    model.eval()
    make_output_dir(out_dir)
    sample_indices = sample_rollout_indices(len(dataset), num_samples, seed)
    records = []

    for plot_idx, sample_idx in enumerate(sample_indices):
        sample = dataset[int(sample_idx)]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        pred_p, pred_v, _ = model.rollout(
            batch["ee_p_hist"],
            batch["ee_v_hist"],
            batch["goal_p_hist"],
            batch["goal_v_hist"],
            batch["goal_p_future"],
            batch["goal_v_future"],
        )
        _, sample_metrics = compute_loss(
            pred_p,
            pred_v,
            batch["target_p"],
            batch["target_v"],
        )

        pred_p = pred_p[0].cpu().numpy()
        pred_v = pred_v[0].cpu().numpy()
        tgt_p = batch["target_p"][0].cpu().numpy()
        tgt_v = batch["target_v"][0].cpu().numpy()
        goal_p = batch["goal_p_future"][0].cpu().numpy()
        goal_v = batch["goal_v_future"][0].cpu().numpy()

        dt = dataset.dt
        integrated_p = np.zeros_like(tgt_p)
        integrated_p[0] = tgt_p[0]
        for k in range(1, len(tgt_v)):
            integrated_p[k] = integrated_p[k - 1] + dt * tgt_v[k]

        pos_err_norm = np.linalg.norm(pred_p - tgt_p, axis=1)
        vel_err_norm = np.linalg.norm(pred_v - tgt_v, axis=1)
        max_pos_err = float(np.max(pos_err_norm))
        max_vel_err = float(np.max(vel_err_norm))

        t = np.arange(dataset.horizon) * dataset.dt
        fig, axs = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
        labels = ["x", "y", "z"]

        for i, label in enumerate(labels):
            ax_pos = axs[i, 0]
            ax_vel = axs[i, 1]

            ax_pos.plot(t, tgt_p[:, i], label="ee_pos_true")
            ax_pos.plot(t, pred_p[:, i], label="ee_pos_pred")
            ax_pos.plot(t, integrated_p[:, i], label="ee_pos_integrated", linestyle="--", alpha=0.7)
            ax_pos.plot(t, goal_p[:, i], label="goal_pos", alpha=0.7)
            ax_pos.set_title(f"Position {label}")
            ax_pos.set_xlabel("time [s]")
            ax_pos.set_ylabel("m")
            ax_pos.grid(True)
            ax_pos.legend()

            ax_vel.plot(t, tgt_v[:, i], label="ee_vel_true")
            ax_vel.plot(t, pred_v[:, i], label="ee_vel_pred")
            ax_vel.plot(t, goal_v[:, i], label="goal_vel", alpha=0.7)
            ax_vel.set_title(f"Velocity {label}")
            ax_vel.set_xlabel("time [s]")
            ax_vel.set_ylabel("m/s")
            ax_vel.grid(True)
            ax_vel.legend()

        axs[3, 0].plot(t, pos_err_norm, label="|pos_error|_2", color="tab:red")
        axs[3, 0].set_title(f"Position 2-norm error | max={max_pos_err:.6f} m")
        axs[3, 0].set_xlabel("time [s]")
        axs[3, 0].set_ylabel("m")
        axs[3, 0].grid(True)
        axs[3, 0].legend()

        axs[3, 1].plot(t, vel_err_norm, label="|vel_error|_2", color="tab:orange")
        axs[3, 1].set_title(f"Velocity 2-norm error | max={max_vel_err:.6f} m/s")
        axs[3, 1].set_xlabel("time [s]")
        axs[3, 1].set_ylabel("m/s")
        axs[3, 1].grid(True)
        axs[3, 1].legend()

        fig.suptitle(
            f"{title} | sample_idx={int(sample_idx)} | loss={sample_metrics['loss']:.6f} | "
        )
        fig.tight_layout()

        file_name = f"sample_{plot_idx:02d}_idx_{int(sample_idx):05d}.png"
        fig.savefig(os.path.join(out_dir, file_name), dpi=160)
        plt.close(fig)

        records.append({
            "sample_idx": int(sample_idx),
            "file": file_name,
            "loss": float(sample_metrics["loss"]),
            "avg_max_pos_err": float(sample_metrics["avg_max_pos_err"]),
            "max_pos_err": max_pos_err,
            "max_vel_err": max_vel_err,
        })

    return {
        "out_dir": out_dir,
        "samples": records,
    }



def save_training_curve(history, out_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, history["train_loss"], label="train_loss")
    ax.plot(epochs, history["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_train", type=str, default="./franka_data/ee_collection_run2.csv",
                     help="训练用CSV（大数据集），按 val_ratio 切分出 train/val")
    ap.add_argument("--csv_test", type=str, nargs="+",
                     default=["./franka_data/ee_collection_run1.csv",
                              "./franka_data/ee_collection_run3.csv"],
                     help="测试用CSV（可多个，跨session泛化评估）")
    ap.add_argument("--out_dir", type=str, default="./franka_dynamic_lstm", help="输出目录（模型、曲线、rollout结果）")
    ap.add_argument("--dt_ms", type=float, default=1.0, help="重采样时间间隔（ms），建议1.0")
    ap.add_argument("--hist_len", type=int, default=31, help="稀疏历史点个数，实际和目标共用同一个长度")
    ap.add_argument("--hist_stride", type=int, default=10, help="历史采样间隔（step）；5表示每隔5ms取一个点")
    ap.add_argument("--horizon", type=int, default=750, help="rollout长度（step），100=100ms")
    ap.add_argument("--stride", type=int, default=20, help="滑窗步长，越小样本越多（推荐10~20）")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="训练CSV中用作验证集的比例（尾部切出）")
    ap.add_argument("--batch_size", type=int, default=256, help="batch大小（窗口数）")
    ap.add_argument("--epochs", type=int, default=10, help="训练轮数")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument("--rollout_weight", type=float, default=1.0, help="rollout loss权重")
    ap.add_argument("--encoder_hidden", type=int, default=64, help="history encoder的LSTM隐藏维度")
    ap.add_argument("--rollout_hidden", type=int, default=64, help="rollout LSTMCell隐藏维度")
    ap.add_argument("--residual_hidden", type=int, default=64, help="residual MLP隐藏维度")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    return ap.parse_args()



def summarize_model(model: SparseLSTMFrankaModel, batch: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
    model.eval()
    with torch.no_grad():
        _, _, aux = model.rollout(
            batch["ee_p_hist"],
            batch["ee_v_hist"],
            batch["goal_p_hist"],
            batch["goal_v_hist"],
            batch["goal_p_future"],
            batch["goal_v_future"],
        )
    acc_mean = aux["acc"].mean(dim=(0, 1)).cpu().tolist()
    return {
        "acc_mean": acc_mean,
    }



@torch.no_grad()
def benchmark_rollout_speed(model, dataset, device, horizon: int = 750, n_repeats: int = 10):
    """Benchmark single-batch rollout (no grad) for real-time use.
    Prints average time in ms over n_repeats runs.
    """
    model.eval()
    sample = dataset[0]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    # truncate future to requested horizon
    batch["goal_p_future"] = batch["goal_p_future"][:, :horizon, :]
    batch["goal_v_future"] = batch["goal_v_future"][:, :horizon, :]

    # warmup
    for _ in range(3):
        model.rollout(
            batch["ee_p_hist"], batch["ee_v_hist"],
            batch["goal_p_hist"], batch["goal_v_hist"],
            batch["goal_p_future"], batch["goal_v_future"],
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    import time
    times = []
    for _ in range(n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.rollout(
            batch["ee_p_hist"], batch["ee_v_hist"],
            batch["goal_p_hist"], batch["goal_v_hist"],
            batch["goal_p_future"], batch["goal_v_future"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    print(f"[Benchmark] batch=1, horizon={horizon} (0.75s), repeats={n_repeats}")
    print(f"  times (ms): {[f'{t:.2f}' for t in times]}")
    print(f"  average: {avg_ms:.2f} ms")
    return avg_ms


def main():
    args = parse_args()
    make_output_dir(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load training CSV (run2) and split into train/val ---
    print(f"Loading train CSV: {args.csv_train}")
    train_data = load_and_resample(args.csv_train, dt_ms=args.dt_ms)
    tv_splits = split_train_val(len(train_data["t_s"]), args.val_ratio)

    train_ds = FrankaSparseWindowDataset(train_data, tv_splits["train"], args.hist_len, args.hist_stride, args.horizon, args.stride)
    val_ds = FrankaSparseWindowDataset(train_data, tv_splits["val"], args.hist_len, args.hist_stride, args.horizon, args.stride)

    print(f"  train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # --- Load test CSVs (run1, run3, ...) ---
    test_datasets = {}  # name -> (data_dict, dataset)
    for csv_path in args.csv_test:
        name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. "ee_collection_run1"
        print(f"Loading test CSV: {csv_path} -> {name}")
        tdata = load_and_resample(csv_path, dt_ms=args.dt_ms)
        tds = FrankaSparseWindowDataset(tdata, (0, len(tdata["t_s"])), args.hist_len, args.hist_stride, args.horizon, args.stride)
        test_datasets[name] = (tdata, tds)
        print(f"  test samples ({name}): {len(tds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseLSTMFrankaModel(
        hist_len=args.hist_len,
        dt=float(train_data["dt"]),
        encoder_hidden=args.encoder_hidden,
        rollout_hidden=args.rollout_hidden,
        residual_hidden=args.residual_hidden,
    ).to(device)

    best_ckpt = os.path.join(args.out_dir, "best_model.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.rollout_weight)
        val_metrics = evaluate(model, val_loader, device, args.rollout_weight)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.6f} | val loss {val_metrics['loss']:.6f} "
            f"| roll_p(train/val) {train_metrics['roll_p']:.6f}/{val_metrics['roll_p']:.6f} "
            f"| roll_v(train/val) {train_metrics['roll_v']:.6f}/{val_metrics['roll_v']:.6f} "
            f"| avg_max_pos_err(train/val) {train_metrics['avg_max_pos_err']:.6f}/{val_metrics['avg_max_pos_err']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "dt": train_data["dt"],
            }, best_ckpt)

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    clean_output_dir(args.out_dir)

    # --- Final evaluation ---
    train_metrics = evaluate(model, train_loader, device, args.rollout_weight)
    val_metrics = evaluate(model, val_loader, device, args.rollout_weight)

    save_training_curve(history, os.path.join(args.out_dir, "training_curve.png"))
    val_plot_summary = save_rollout_plot(
        model,
        val_ds,
        device,
        os.path.join(args.out_dir, "val_rollout"),
        "Validation rollout",
        num_samples=20,
        seed=args.seed,
    )

    # --- Evaluate each test CSV separately ---
    all_test_metrics = {}
    all_test_plots = {}
    for name, (tdata, tds) in test_datasets.items():
        tloader = DataLoader(tds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_metrics = evaluate(model, tloader, device, args.rollout_weight)
        all_test_metrics[name] = t_metrics
        print(f"Test [{name}]: avg_max_pos_err={t_metrics['avg_max_pos_err']:.6f}")

        plot_dir = os.path.join(args.out_dir, f"test_rollout_{name}")
        t_plot = save_rollout_plot(
            model, tds, device, plot_dir,
            f"Test rollout ({name})",
            num_samples=20,
            seed=args.seed + 1,
        )
        all_test_plots[name] = t_plot

    # --- Probe model summary ---
    first_test_name = next(iter(test_datasets))
    probe_ds = test_datasets[first_test_name][1]
    probe_sample = probe_ds[len(probe_ds) // 2]
    probe_batch = {k: v.unsqueeze(0).to(device) for k, v in probe_sample.items()}
    model_summary = summarize_model(model, probe_batch)

    # --- Build summary ---
    summary = {
        "csv_train": args.csv_train,
        "csv_test": args.csv_test,
        "train_data": {
            "raw_len": int(train_data["raw_len"]),
            "resampled_len": int(train_data["resampled_len"]),
        },
        "dt": float(train_data["dt"]),
        "splits": {k: [int(v[0]), int(v[1])] for k, v in tv_splits.items()},
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            **{f"test_{name}": len(tds) for name, (_, tds) in test_datasets.items()},
        },
        "history": {
            "hist_len": args.hist_len,
            "hist_stride": args.hist_stride,
            "hist_offsets_steps": train_ds.hist_offsets.tolist(),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": all_test_metrics,
        "visualization": {
            "val": val_plot_summary,
            **{f"test_{name}": tp for name, tp in all_test_plots.items()},
        },
        "model_summary": model_summary,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    test_csv_str = ", ".join(args.csv_test)
    test_dirs_str = "\n".join(
        f"- test_rollout_{name}/: 20 rollout plots" for name in test_datasets
    )
    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Franka sparse-history LSTM lag model training outputs\n"
            "===============================================\n\n"
            f"csv_train: {args.csv_train}\n"
            f"csv_test: {test_csv_str}\n"
            f"dt_ms: {args.dt_ms}\n"
            f"hist_len: {args.hist_len}\n"
            f"hist_stride: {args.hist_stride}\n"
            f"horizon: {args.horizon}\n"
            f"stride: {args.stride}\n"
            f"val_ratio: {args.val_ratio}\n"
            f"epochs: {args.epochs}\n"
            f"encoder_hidden: {args.encoder_hidden}\n"
            f"rollout_hidden: {args.rollout_hidden}\n"
            f"residual_hidden: {args.residual_hidden}\n\n"
            "Files:\n"
            "- best_model.pt: best checkpoint on validation set\n"
            "- training_curve.png\n"
            "- val_rollout/: 20 rollout plots (from train CSV val split)\n"
            f"{test_dirs_str}\n"
            "- summary.json\n"
        )

    print("Done.")
    print(json.dumps(summary, indent=2))

    # Speed benchmark: batch=1, 0.75s rollout
    benchmark_rollout_speed(model, val_ds, device, horizon=750, n_repeats=10)


if __name__ == "__main__":
    main()
