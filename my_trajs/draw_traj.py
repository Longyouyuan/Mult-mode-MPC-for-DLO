from common.utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt


file_path = draw_traj_xy()

# file_path = 'SpongeBob.npy'

my = np.load(file_path)

plt.plot(my[:,0], my[:,1], marker='o')
plt.title("My Drawn Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_horizon = 150

Goal_traj = build_goal_traj_from_drawn(
    drawn_path=file_path,
    total_horizon=total_horizon,
    device=device,
    z0=0.2,
    scale_x=1.0,
    scale_y=1.0,
    keep_aspect=False,   # 允许非等比缩放（你说可能不是方形）
    sigma=0.0,
    uniform_M=1000,      # 越大越均匀/越平滑（但太大也没必要）
    ratio=0.3,
    sharpness=2.0,
    interval=(0.0, 2.0)
)

plt.plot(Goal_traj[:,0].cpu().numpy(), Goal_traj[:,1].cpu().numpy(), marker='o')
plt.title("Goal Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


ccc=1