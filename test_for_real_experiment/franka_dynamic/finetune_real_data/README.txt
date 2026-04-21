Franka sparse-history GRU lag model training outputs
===============================================

csv_train_base: ./franka_data/ee_collection_run2.csv
csv_train_extra: ./franka_data\real_data_eight_20260416_222916.csv, ./franka_data\real_data_eight_20260416_223121.csv
csv_test: ./franka_data/ee_collection_run1.csv, ./franka_data/ee_collection_run3.csv
dt_ms: 1.0
hist_len: 1
hist_stride: 10
horizon: 750
stride: 20
val_ratio: 0.15
base_train_keep_ratio: 0.5
epochs: 15
init_ckpt: None
real_data_repeat: 32
split_extra_val: False
encoder_hidden: 64
rollout_hidden: 64
residual_hidden: 64

Files:
- best_model.pt: best checkpoint on validation set
- training_curve.png
- val_rollout/: 20 rollout plots (from train CSV val split)
- test_rollout_ee_collection_run1/: 20 rollout plots
- test_rollout_ee_collection_run3/: 20 rollout plots
- summary.json
