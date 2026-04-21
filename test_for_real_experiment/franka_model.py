import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

XYZ = ["x", "y", "z"]
DEFAULT_REAL_DATA_EXTRA_GLOB = "./franka_data/real_data*.csv"
DEFAULT_BASELINE_CKPT = "./franka_dynamic/best_model.pt"
DEFAULT_FINETUNE_OUT_DIR = "./franka_dynamic_real_data_finetune"
DEFAULT_RUN2_NAME = "ee_collection_run2.csv"


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


class FrankaConcatDataset(Dataset):
    def __init__(self, datasets: List[FrankaSparseWindowDataset]):
        if not datasets:
            raise ValueError("datasets must not be empty")
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets], dtype=np.int64)

        first = datasets[0]
        self.dt = float(first.dt)
        self.horizon = int(first.horizon)
        self.hist_len = int(first.hist_len)
        self.hist_stride = int(first.hist_stride)
        self.hist_offsets = first.hist_offsets.copy()

        for ds in datasets[1:]:
            if not np.isclose(float(ds.dt), self.dt):
                raise ValueError("All datasets must share the same dt")
            if ds.horizon != self.horizon or ds.hist_len != self.hist_len or ds.hist_stride != self.hist_stride:
                raise ValueError("All datasets must share the same history/horizon settings")

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))
        sample_idx = idx if dataset_idx == 0 else idx - int(self.cumulative_sizes[dataset_idx - 1])
        return self.datasets[dataset_idx][sample_idx]


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


class SparseGRUFrankaModel(nn.Module):
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

        self.history_encoder = nn.GRU(
            input_size=12,
            hidden_size=self.encoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.init_rollout = nn.Linear(self.encoder_hidden, self.rollout_hidden)
        self.rollout_cell = nn.GRUCell(input_size=12, hidden_size=self.rollout_hidden)
        self.learned_z0 = nn.Parameter(torch.zeros(1, self.rollout_hidden))

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
    ) -> torch.Tensor:
        hist_seq = torch.cat([ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist], dim=-1)
        _, h_n = self.history_encoder(hist_seq)
        z0 = torch.tanh(self.init_rollout(h_n[-1]))
        return z0

    def rollout(
        self,
        ee_p_hist: torch.Tensor,
        ee_v_hist: torch.Tensor,
        goal_p_hist: torch.Tensor,
        goal_v_hist: torch.Tensor,
        goal_p_future: torch.Tensor,
        goal_v_future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # z = self._history_to_latent(ee_p_hist, ee_v_hist, goal_p_hist, goal_v_hist)

        # z = torch.zeros(
        #     ee_p_hist.shape[0],
        #     self.rollout_hidden,
        #     device=ee_p_hist.device,
        #     dtype=ee_p_hist.dtype,
        # )
        
        z = self.learned_z0.expand(ee_p_hist.shape[0], -1)

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
            z = self.rollout_cell(rollout_in, z)

            a = self.acc_head(torch.cat([z, p_now, v_now, p_goal, v_goal], dim=-1))

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


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def is_run2_csv(path: str) -> bool:
    return os.path.basename(path) == DEFAULT_RUN2_NAME


def expand_csv_paths(entries: Optional[List[str]]) -> List[str]:
    if entries is None:
        return []

    expanded = []
    seen = set()
    for entry in entries:
        matches = sorted(glob.glob(entry))
        if not matches:
            matches = [entry]

        for path in matches:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")

            key = normalize_path(path)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(path)
    return expanded


def collect_train_csvs(
    csv_train: List[str],
    csv_train_extra: Optional[List[str]],
    csv_train_extra_glob: Optional[str],
) -> Tuple[List[str], List[str]]:
    base_csvs = expand_csv_paths(csv_train)

    extra_entries = [] if csv_train_extra is None else list(csv_train_extra)
    if csv_train_extra_glob:
        extra_entries.append(csv_train_extra_glob)
    extra_csvs = expand_csv_paths(extra_entries)

    base_keys = {normalize_path(path) for path in base_csvs}
    extra_csvs = [path for path in extra_csvs if normalize_path(path) not in base_keys]
    return base_csvs, extra_csvs


def trim_resampled_data(data: Dict[str, np.ndarray], keep_len: int) -> Dict[str, np.ndarray]:
    trimmed = {}
    for key, value in data.items():
        if key in {"t_s", "goal_p", "goal_v", "ee_p", "ee_v"}:
            trimmed[key] = value[:keep_len]
        else:
            trimmed[key] = value
    trimmed["resampled_len"] = np.int64(keep_len)
    return trimmed


def build_train_val_datasets(
    csv_paths: List[str],
    dt_ms: float,
    hist_len: int,
    hist_stride: int,
    horizon: int,
    stride: int,
    val_ratio: float,
    train_repeat_by_csv: Dict[str, int],
    source_role_by_csv: Dict[str, str],
    train_only_by_csv: Dict[str, bool],
    keep_ratio_by_csv: Dict[str, float],
) -> Tuple[Dataset, Dataset, List[Dict[str, object]], float]:
    train_parts = []
    val_parts = []
    source_summaries = []
    reference_dt = None
    min_total_len = int(make_sparse_offsets(hist_len, hist_stride)[0]) + int(horizon) + 2

    for csv_path in csv_paths:
        print(f"Loading train CSV: {csv_path}")
        data = load_and_resample(csv_path, dt_ms=dt_ms)
        original_resampled_len = int(data["resampled_len"])
        keep_ratio = float(keep_ratio_by_csv.get(normalize_path(csv_path), 1.0))
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"keep ratio must be in (0, 1], got {keep_ratio} for {csv_path}")

        if keep_ratio < 1.0:
            keep_len = max(min_total_len, int(original_resampled_len * keep_ratio))
            keep_len = min(keep_len, original_resampled_len)
            if keep_len < original_resampled_len:
                data = trim_resampled_data(data, keep_len)
                print(
                    f"  keeping first {keep_len}/{original_resampled_len} resampled steps "
                    f"({keep_len / original_resampled_len:.2%})"
                )

        data_dt = float(data["dt"])
        if reference_dt is None:
            reference_dt = data_dt
        elif not np.isclose(data_dt, reference_dt):
            raise ValueError("All training CSVs must share the same resampled dt")

        train_only = bool(train_only_by_csv.get(normalize_path(csv_path), False))
        total_len = len(data["t_s"])

        if train_only:
            splits = {
                "train": (0, total_len),
                "val": (total_len, total_len),
            }
            train_ds = FrankaSparseWindowDataset(
                data,
                splits["train"],
                hist_len,
                hist_stride,
                horizon,
                stride,
            )
            val_ds = None
            val_count = 0
            print(
                f"  using full sequence for training only: {csv_path}"
            )
        else:
            splits = split_train_val(total_len, val_ratio)
            train_ds = FrankaSparseWindowDataset(
                data,
                splits["train"],
                hist_len,
                hist_stride,
                horizon,
                stride,
            )

            val_ds = None
            try:
                val_ds = FrankaSparseWindowDataset(
                    data,
                    splits["val"],
                    hist_len,
                    hist_stride,
                    horizon,
                    stride,
                )
                val_count = len(val_ds)
            except ValueError:
                val_count = 0
                print(
                    f"  validation split skipped for {csv_path} because it is too short for the selected history/horizon."
                )

        repeat = max(1, int(train_repeat_by_csv.get(normalize_path(csv_path), 1)))
        for _ in range(repeat):
            train_parts.append(train_ds)
        if val_ds is not None:
            val_parts.append(val_ds)

        source_summary = {
            "name": os.path.splitext(os.path.basename(csv_path))[0],
            "path": csv_path,
            "role": source_role_by_csv.get(normalize_path(csv_path), "base"),
            "raw_len": int(data["raw_len"]),
            "resampled_len": int(data["resampled_len"]),
            "original_resampled_len": original_resampled_len,
            "keep_ratio": keep_ratio,
            "train_repeat": repeat,
            "train_only": train_only,
            "train_samples": len(train_ds),
            "val_samples": val_count,
            "val_skipped": val_ds is None,
            "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        }
        source_summaries.append(source_summary)
        print(
            f"  train samples: {len(train_ds)}, val samples: {val_count}, repeat: {repeat}, role: {source_summary['role']}, train_only: {train_only}"
        )

    if not train_parts:
        raise ValueError("No training datasets were built")
    if not val_parts:
        raise ValueError("No validation datasets were built. Reduce horizon/hist_len or provide longer CSVs.")

    train_ds = train_parts[0] if len(train_parts) == 1 else FrankaConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else FrankaConcatDataset(val_parts)
    return train_ds, val_ds, source_summaries, float(reference_dt)


def filter_test_csvs(csv_test: Optional[List[str]], train_csvs: List[str]) -> List[str]:
    train_keys = {normalize_path(path) for path in train_csvs}
    filtered = []
    for csv_path in expand_csv_paths(csv_test):
        if normalize_path(csv_path) in train_keys:
            print(f"Skipping test CSV also used for training: {csv_path}")
            continue
        filtered.append(csv_path)
    return filtered


def get_reference_dataset(dataset: Dataset) -> FrankaSparseWindowDataset:
    if isinstance(dataset, FrankaConcatDataset):
        return dataset.datasets[0]
    return dataset


def save_checkpoint(path: str, model: SparseGRUFrankaModel, args, dt: float, train_csvs: List[str], test_csvs: List[str]):
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "dt": dt,
            "train_csvs": train_csvs,
            "test_csvs": test_csvs,
        },
        path,
    )


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
    ap.add_argument("--csv_train", type=str, nargs="+", default=["./franka_data/ee_collection_run2.csv"],
                     help="基础训练CSV，可多个；每个CSV都会各自切分 train/val")
    ap.add_argument("--base_train_keep_ratio", type=float, default=0.5,
                     help="ee_collection_run2.csv 保留比例，默认0.5；1.0表示全部使用")
    ap.add_argument("--csv_train_extra", type=str, nargs="+", default=None,
                     help="增量/DAgger训练CSV，可多个；未显式指定额外数据时会自动加载 real_data*.csv，且默认整段只用于训练")
    ap.add_argument("--csv_train_extra_glob", type=str, default=None,
                     help="增量/DAgger训练CSV通配符，例如 ./franka_data/real_data*.csv；若不显式指定，脚本会自动尝试加载该模式")
    ap.add_argument("--csv_test", type=str, nargs="+",
                     default=["./franka_data/ee_collection_run1.csv",
                              "./franka_data/ee_collection_run3.csv"],
                     help="测试用CSV（可多个或glob）；会自动跳过已经用于训练的CSV")
    ap.add_argument("--out_dir", type=str, default="./franka_dynamic/finetune_real_data",
                     help="输出目录（模型、曲线、rollout结果）；默认写到单独的real_data微调目录")
    ap.add_argument("--dt_ms", type=float, default=1.0, help="重采样时间间隔（ms），建议1.0")
    ap.add_argument("--hist_len", type=int, default=1, help="稀疏历史点个数，实际和目标共用同一个长度")
    ap.add_argument("--hist_stride", type=int, default=10, help="历史采样间隔（step）；5表示每隔5ms取一个点")
    ap.add_argument("--horizon", type=int, default=750, help="rollout长度（step），100=100ms")
    ap.add_argument("--stride", type=int, default=20, help="滑窗步长，越小样本越多（推荐10~20）")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="训练CSV中用作验证集的比例（尾部切出）")
    ap.add_argument("--batch_size", type=int, default=128, help="batch大小（窗口数）")
    ap.add_argument("--epochs", type=int, default=15, help="训练轮数")
    ap.add_argument("--lr", type=float, default=5e-4, help="学习率")
    ap.add_argument("--rollout_weight", type=float, default=1.0, help="rollout loss权重")
    ap.add_argument("--encoder_hidden", type=int, default=64, help="history encoder的GRU隐藏维度")
    ap.add_argument("--rollout_hidden", type=int, default=64, help="rollout GRUCell隐藏维度")
    ap.add_argument("--residual_hidden", type=int, default=64, help="residual MLP隐藏维度")
    ap.add_argument("--init_ckpt", type=str, default=None,
                     help="从已有checkpoint继续训练；默认先尝试 out_dir/best_model.pt，再尝试 ./franka_dynamic/best_model.pt")
    ap.add_argument("--real_data_repeat", type=int, default=32, help="增量真实数据训练集重复次数，>1 可提高其训练权重")
    ap.add_argument("--split_extra_val", action="store_true", help="对增量真实数据也执行 train/val 切分；默认不切，全部用于训练")
    ap.add_argument("--reset_best", action="store_true", help="加载checkpoint后重新开始统计best模型")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    return ap.parse_args()



def summarize_model(model: SparseGRUFrankaModel, batch: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
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

    if args.csv_train_extra is None and args.csv_train_extra_glob is None:
        auto_extra_matches = sorted(glob.glob(DEFAULT_REAL_DATA_EXTRA_GLOB))
        if auto_extra_matches:
            args.csv_train_extra_glob = DEFAULT_REAL_DATA_EXTRA_GLOB
            print(f"Auto-loading extra training CSVs from: {args.csv_train_extra_glob}")

    base_train_csvs, extra_train_csvs = collect_train_csvs(
        args.csv_train,
        args.csv_train_extra,
        args.csv_train_extra_glob,
    )
    all_train_csvs = base_train_csvs + extra_train_csvs
    if not all_train_csvs:
        raise ValueError("No training CSVs provided")

    print("Training CSVs:")
    for csv_path in base_train_csvs:
        print(f"  [base] {csv_path}")
    for csv_path in extra_train_csvs:
        print(f"  [extra] {csv_path}")

    train_repeat_by_csv = {normalize_path(path): 1 for path in all_train_csvs}
    source_role_by_csv = {normalize_path(path): "base" for path in base_train_csvs}
    train_only_by_csv = {normalize_path(path): False for path in all_train_csvs}
    keep_ratio_by_csv = {normalize_path(path): 1.0 for path in all_train_csvs}
    for path in base_train_csvs:
        if is_run2_csv(path):
            keep_ratio_by_csv[normalize_path(path)] = args.base_train_keep_ratio
    for path in extra_train_csvs:
        train_repeat_by_csv[normalize_path(path)] = max(1, args.real_data_repeat)
        source_role_by_csv[normalize_path(path)] = "extra"
        train_only_by_csv[normalize_path(path)] = not args.split_extra_val

    train_ds, val_ds, train_source_summaries, train_dt = build_train_val_datasets(
        all_train_csvs,
        dt_ms=args.dt_ms,
        hist_len=args.hist_len,
        hist_stride=args.hist_stride,
        horizon=args.horizon,
        stride=args.stride,
        val_ratio=args.val_ratio,
        train_repeat_by_csv=train_repeat_by_csv,
        source_role_by_csv=source_role_by_csv,
        train_only_by_csv=train_only_by_csv,
        keep_ratio_by_csv=keep_ratio_by_csv,
    )
    print(f"Combined train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # --- Load test CSVs (run1, run3, ...) ---
    test_datasets = {}  # name -> (data_dict, dataset)
    args.csv_test = filter_test_csvs(args.csv_test, all_train_csvs)
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
    model = SparseGRUFrankaModel(
        hist_len=args.hist_len,
        dt=train_dt,
        encoder_hidden=args.encoder_hidden,
        rollout_hidden=args.rollout_hidden,
        residual_hidden=args.residual_hidden,
    ).to(device)

    best_ckpt = os.path.join(args.out_dir, "best_model.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    load_ckpt_path = None
    if args.init_ckpt is not None:
        load_ckpt_path = args.init_ckpt
    elif os.path.exists(best_ckpt):
        load_ckpt_path = best_ckpt
    elif os.path.exists(DEFAULT_BASELINE_CKPT):
        load_ckpt_path = DEFAULT_BASELINE_CKPT

    if load_ckpt_path is not None:
        print(f"Loading checkpoint from: {load_ckpt_path}")
        ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if not os.path.exists(best_ckpt) or normalize_path(load_ckpt_path) != normalize_path(best_ckpt):
            save_checkpoint(best_ckpt, model, args, train_dt, all_train_csvs, args.csv_test)

        # if not args.reset_best:
        #     val_metrics = evaluate(model, val_loader, device, args.rollout_weight)
        #     best_val = val_metrics["avg_max_pos_err"]
        #     print("Best val avg_max_pos_err after loading ckpt:", best_val)

    if load_ckpt_path is None and args.epochs <= 0:
        raise ValueError("epochs must be > 0 when no init_ckpt or existing out_dir/best_model.pt is available")

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

        if train_metrics["avg_max_pos_err"] < best_val:
            best_val = train_metrics["avg_max_pos_err"]
            save_checkpoint(best_ckpt, model, args, train_dt, all_train_csvs, args.csv_test)

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
    probe_ds = val_ds
    probe_sample = probe_ds[len(probe_ds) // 2]
    probe_batch = {k: v.unsqueeze(0).to(device) for k, v in probe_sample.items()}
    model_summary = summarize_model(model, probe_batch)

    ref_ds = get_reference_dataset(train_ds)
    train_raw_len_total = sum(item["raw_len"] for item in train_source_summaries)
    train_resampled_len_total = sum(item["resampled_len"] for item in train_source_summaries)

    # --- Build summary ---
    summary = {
        "csv_train_base": base_train_csvs,
        "csv_train_extra": extra_train_csvs,
        "csv_train_all": all_train_csvs,
        "csv_test": args.csv_test,
        "train_data": {
            "num_sources": len(train_source_summaries),
            "raw_len_total": int(train_raw_len_total),
            "resampled_len_total": int(train_resampled_len_total),
        },
        "train_sources": train_source_summaries,
        "dt": float(train_dt),
        "splits": {item["name"]: item["splits"] for item in train_source_summaries},
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            **{f"test_{name}": len(tds) for name, (_, tds) in test_datasets.items()},
        },
        "history": {
            "hist_len": args.hist_len,
            "hist_stride": args.hist_stride,
            "hist_offsets_steps": ref_ds.hist_offsets.tolist(),
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

    base_csv_str = ", ".join(base_train_csvs)
    extra_csv_str = ", ".join(extra_train_csvs) if extra_train_csvs else "none"
    test_csv_str = ", ".join(args.csv_test) if args.csv_test else "none"
    test_dirs_str = "\n".join(
        f"- test_rollout_{name}/: 20 rollout plots" for name in test_datasets
    ) if test_datasets else "- no test rollout generated"
    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Franka sparse-history GRU lag model training outputs\n"
            "===============================================\n\n"
            f"csv_train_base: {base_csv_str}\n"
            f"csv_train_extra: {extra_csv_str}\n"
            f"csv_test: {test_csv_str}\n"
            f"dt_ms: {args.dt_ms}\n"
            f"hist_len: {args.hist_len}\n"
            f"hist_stride: {args.hist_stride}\n"
            f"horizon: {args.horizon}\n"
            f"stride: {args.stride}\n"
            f"val_ratio: {args.val_ratio}\n"
            f"base_train_keep_ratio: {args.base_train_keep_ratio}\n"
            f"epochs: {args.epochs}\n"
            f"init_ckpt: {args.init_ckpt}\n"
            f"real_data_repeat: {args.real_data_repeat}\n"
            f"split_extra_val: {args.split_extra_val}\n"
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
    benchmark_rollout_speed(model, probe_ds, device, horizon=min(args.horizon, 750), n_repeats=10)


if __name__ == "__main__":
    main()
