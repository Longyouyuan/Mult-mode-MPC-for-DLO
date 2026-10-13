# Resample run1-3 CSVs exactly as franka_model.load_and_resample does and save as npz,
# so the training script can run in an env without pandas.
# Usage (any env with pandas): python convert_npz.py <franka_data_dir> <out_dir>
import os
import sys

import numpy as np
import pandas as pd

XYZ = ["x", "y", "z"]
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name in ["ee_collection_run1", "ee_collection_run2", "ee_collection_run3"]:
    df = pd.read_csv(os.path.join(src, name + ".csv"))
    t_ns = df["t_ns"].to_numpy(dtype=np.int64)
    t_rel = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    dt = 1e-3
    tu = np.arange(0.0, t_rel[-1], dt, dtype=np.float64)

    def ic(cols):
        arr = df[cols].to_numpy(dtype=np.float64)
        o = np.empty((len(tu), arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            o[:, j] = np.interp(tu, t_rel, arr[:, j]).astype(np.float32)
        return o

    np.savez(os.path.join(out, name + ".npz"), t_s=tu.astype(np.float32),
             goal_p=ic([f"pos_goal_{a}" for a in XYZ]), goal_v=ic([f"vel_goal_{a}" for a in XYZ]),
             ee_p=ic([f"ee_pos_{a}" for a in XYZ]), ee_v=ic([f"ee_vel_{a}" for a in XYZ]),
             dt=np.float32(dt), raw_len=np.int64(len(df)), resampled_len=np.int64(len(tu)))
    print(name, len(df), len(tu))
