import time
import numpy as np
import torch
import warp as wp

TILE_SIZE = wp.constant(256)
TILE_THREADS = 64

@wp.kernel
def compute(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):

    # obtain our block index
    i = wp.tid()

    # load a row from global memory
    t = wp.tile_load(a[i], TILE_SIZE)

    # cooperatively compute the sum of the tile elements; s is a single element tile
    s = wp.tile_sum(t)

    # store s in global memory
    wp.tile_store(b[i], s)

N = 10

a_np = np.arange(N).reshape(-1, 1) * np.ones((1, 256), dtype=float)
a = wp.array(a_np, dtype=float)
b = wp.zeros((N,1), dtype=float)

wp.launch_tiled(compute, dim=[a.shape[0]], inputs=[a, b], block_dim=TILE_THREADS)

print(f"b = {b[:,0]}")