Franka sparse-history GRU model test outputs
======================================

ckpt: ./franka_dynamic/finetune_real_data/best_model.pt
csv_test: ./franka_data\real_data_eight_20260416_222916.csv, ./franka_data\real_data_eight_20260416_223121.csv
dt_ms: 1.0
hist_len: 1
hist_stride: 10
horizon: 750
stride: 20
batch_size: 128

Files:
- test_rollout_real_data_eight_20260416_222916/: 20 rollout plots
- test_rollout_real_data_eight_20260416_223121/: 20 rollout plots
- summary.json
