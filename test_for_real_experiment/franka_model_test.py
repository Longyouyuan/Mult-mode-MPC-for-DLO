import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from franka_model import (
	FrankaSparseWindowDataset,
	SparseGRUFrankaModel,
	benchmark_rollout_speed,
	clean_output_dir,
	evaluate,
	load_and_resample,
	make_output_dir,
	save_rollout_plot,
	summarize_model,
)


def list_real_data_test_csvs(csv_dir: str) -> List[str]:
	real_data_csvs = sorted(glob.glob(os.path.join(csv_dir, "real_data*.csv")))
	if not real_data_csvs:
		raise ValueError(f"No real_data*.csv files found in {csv_dir}")
	return real_data_csvs


def parse_args():
	ap = argparse.ArgumentParser()
	# ap.add_argument("--ckpt", type=str, default="./franka_dynamic/best_model.pt", help="已训练好的GRU模型checkpoint")
	ap.add_argument("--ckpt", type=str, default="./franka_dynamic/finetune_real_data/best_model.pt", help="已训练好的GRU模型checkpoint")
	ap.add_argument("--csv_test", type=str, nargs="+", default=None,
					help="测试用CSV；默认只测试 franka_data 下的 real_data*.csv 数据")
	ap.add_argument("--out_dir", type=str, default="./franka_dynamic/test_only", help="测试输出目录")
	ap.add_argument("--dt_ms", type=float, default=None, help="重采样时间间隔（ms）；默认读取checkpoint训练参数")
	ap.add_argument("--horizon", type=int, default=None, help="rollout长度（step）；默认读取checkpoint训练参数")
	ap.add_argument("--stride", type=int, default=None, help="滑窗步长；默认读取checkpoint训练参数")
	ap.add_argument("--batch_size", type=int, default=None, help="batch大小；默认读取checkpoint训练参数")
	ap.add_argument("--num_plots", type=int, default=20, help="每个测试集保存的rollout图数量")
	ap.add_argument("--seed", type=int, default=0, help="随机种子")
	return ap.parse_args()


def build_model_from_checkpoint(ckpt: Dict, device: torch.device) -> tuple[SparseGRUFrankaModel, Dict]:
	ckpt_args = ckpt.get("args", {})
	model = SparseGRUFrankaModel(
		hist_len=int(ckpt_args["hist_len"]),
		dt=float(ckpt.get("dt", 1.0e-3)),
		encoder_hidden=int(ckpt_args.get("encoder_hidden", 64)),
		rollout_hidden=int(ckpt_args.get("rollout_hidden", 64)),
		residual_hidden=int(ckpt_args.get("residual_hidden", 64)),
	).to(device)
	model.load_state_dict(ckpt["model_state"])
	return model, ckpt_args


def summarize_goal_vs_ee_metrics(dataset: FrankaSparseWindowDataset, batch_size: int) -> Dict[str, float]:
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	window_max_pos_errs = []
	for batch in loader:
		pos_err_norm = torch.linalg.norm(batch["goal_p_future"] - batch["target_p"], dim=-1)
		window_max_pos_errs.append(pos_err_norm.max(dim=1).values)
	if not window_max_pos_errs:
		raise ValueError("Dataset is empty, cannot compute goal-vs-ee metrics.")
	avg_max_pos_err = torch.cat(window_max_pos_errs).mean()
	return {
		"avg_max_pos_err": float(avg_max_pos_err.detach().cpu()),
	}


def main():
	args = parse_args()
	make_output_dir(args.out_dir)
	clean_output_dir(args.out_dir, keep=())

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
	model, ckpt_args = build_model_from_checkpoint(ckpt, device)

	if args.csv_test is None:
		args.csv_test = list_real_data_test_csvs("./franka_data")

	dt_ms = args.dt_ms if args.dt_ms is not None else float(ckpt_args.get("dt_ms", 1.0))
	hist_len = int(ckpt_args["hist_len"])
	hist_len = int(1)
	hist_stride = int(ckpt_args["hist_stride"])
	horizon = args.horizon if args.horizon is not None else int(ckpt_args["horizon"])
	stride = args.stride if args.stride is not None else int(ckpt_args["stride"])
	batch_size = args.batch_size if args.batch_size is not None else int(ckpt_args.get("batch_size", 256))

	test_datasets = {}
	for csv_path in args.csv_test:
		name = os.path.splitext(os.path.basename(csv_path))[0]
		print(f"Loading test CSV: {csv_path} -> {name}")
		tdata = load_and_resample(csv_path, dt_ms=dt_ms)
		tds = FrankaSparseWindowDataset(
			tdata,
			(0, len(tdata["t_s"])),
			hist_len,
			hist_stride,
			horizon,
			stride,
		)
		test_datasets[name] = (tdata, tds)
		print(f"  test samples ({name}): {len(tds)}")

	all_test_metrics = {}
	all_goal_vs_ee_metrics = {}
	all_test_plots = {}
	for name, (_, tds) in test_datasets.items():
		tloader = DataLoader(tds, batch_size=batch_size, shuffle=False, num_workers=0)
		t_metrics = evaluate(model, tloader, device)
		goal_vs_ee_metrics = summarize_goal_vs_ee_metrics(tds, batch_size=batch_size)
		all_test_metrics[name] = t_metrics
		all_goal_vs_ee_metrics[name] = goal_vs_ee_metrics
		print(f"Test [{name}]: avg_max_pos_err={t_metrics['avg_max_pos_err']:.6f}")
		print(f"Goal-vs-EE [{name}]: avg_max_pos_err={goal_vs_ee_metrics['avg_max_pos_err']:.6f}")

		plot_dir = os.path.join(args.out_dir, f"test_rollout_{name}")
		t_plot = save_rollout_plot(
			model,
			tds,
			device,
			plot_dir,
			f"Test rollout ({name})",
			num_samples=args.num_plots,
			seed=args.seed + 1,
		)
		all_test_plots[name] = t_plot

	first_test_name = next(iter(test_datasets))
	probe_ds = test_datasets[first_test_name][1]
	probe_sample = probe_ds[len(probe_ds) // 2]
	probe_batch = {k: v.unsqueeze(0).to(device) for k, v in probe_sample.items()}
	model_summary = summarize_model(model, probe_batch)
	benchmark_ms = benchmark_rollout_speed(model, probe_ds, device, horizon=min(horizon, 750), n_repeats=10)

	summary = {
		"ckpt": args.ckpt,
		"csv_test": args.csv_test,
		"eval_settings": {
			"dt_ms": dt_ms,
			"hist_len": hist_len,
			"hist_stride": hist_stride,
			"horizon": horizon,
			"stride": stride,
			"batch_size": batch_size,
		},
		"dataset_sizes": {
			f"test_{name}": len(tds) for name, (_, tds) in test_datasets.items()
		},
		"test_metrics": all_test_metrics,
		"goal_vs_ee_metrics": all_goal_vs_ee_metrics,
		"visualization": {
			f"test_{name}": tp for name, tp in all_test_plots.items()
		},
		"model_summary": model_summary,
		"benchmark_ms": benchmark_ms,
	}

	with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	test_csv_str = ", ".join(args.csv_test)
	test_dirs_str = "\n".join(
		f"- test_rollout_{name}/: {args.num_plots} rollout plots" for name in test_datasets
	)
	with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
		f.write(
			"Franka sparse-history GRU model test outputs\n"
			"======================================\n\n"
			f"ckpt: {args.ckpt}\n"
			f"csv_test: {test_csv_str}\n"
			f"dt_ms: {dt_ms}\n"
			f"hist_len: {hist_len}\n"
			f"hist_stride: {hist_stride}\n"
			f"horizon: {horizon}\n"
			f"stride: {stride}\n"
			f"batch_size: {batch_size}\n\n"
			"Files:\n"
			f"{test_dirs_str}\n"
			"- summary.json\n"
		)

	print("Done.")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
