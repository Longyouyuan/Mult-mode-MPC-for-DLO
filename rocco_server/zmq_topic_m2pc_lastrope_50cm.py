#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import struct
import time
import zmq
import numpy as np
import torch
 
from M2PC_lastrope import Planner, cost_fn
from rope_warp_4_lastrope import WarpRope, N
 
STATE_HDR = struct.Struct("<iqII")  # step(int32), t_state_ns(int64), P(uint32), H(uint32)
ACT_HDR   = struct.Struct("<iqf")   # step_ref(int32), t_state_ns(int64), compute_ms(float32)
 
 
def now_ns():
    return time.perf_counter_ns()
 
 
class PlannerServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[server] device = {self.device}")
 
        # ----- MPC settings -----
        self.dt = 0.001
        self.ctr_period = 25  # 25
        self.horizon = 30
        self.n_sample = 1000
        self.m_modes = 1
        self.n_improve = 1
        self.noise_scale = 1.8
        self.action_dim = 3
        self.limits = torch.tensor([-5.0/5.0, 5.0/5.0], device=self.device, dtype=torch.float32)  # careful about here
        self.top_k_good = 200
        self.beta = 5.0
        self.wJ = 0.0
        self.warmup_improve = 200
 
        # ----- Rope model params -----
        L = 0.5
        segment_lengths = [0.0973, 0.0996, 0.0993, 0.0986, 0.0990]
        mass = 12.8 / 1000 / N
        tip_extra_mass = 15 / 1000
        k = 10000 * 0.12
        damping = 0.2
        bending_k = 0.0006712
        bending_damping = 0.000401
        air_drag = 0.3006 / 1000
        g = 10.27
        mode = "acc"
 
        self.rope = WarpRope(
            batch_size=self.m_modes + self.n_sample,
            L=L,
            segment_lengths=segment_lengths,
            mass=mass,
            tip_extra_mass=tip_extra_mass,
            k=k,
            damping=damping,
            bending_k=bending_k,
            bending_damping=bending_damping,
            air_drag=air_drag,
            g=g,
            dt=self.dt,
            max_record_steps=self.horizon * self.ctr_period,
            record_interval=self.ctr_period,
            ctr_period=self.ctr_period,
            mode=mode
        )
 
        self.planner = Planner(
            rope=self.rope,
            cost_fn=cost_fn,
            dt=self.dt,
            ctr_period=self.ctr_period,
            horizon=self.horizon,
            n_sample=self.n_sample,
            n_improve=self.n_improve,
            noise_scale=self.noise_scale,
            action_dim=self.action_dim,
            limits=self.limits,
            device=self.device,
            mode=mode,
            m_modes=self.m_modes,
            top_k_good=self.top_k_good,
            beta=self.beta,
            wJ=self.wJ,
            standard_m2pc=True
        )
 
        # warm up GPU context
        if self.device.type == "cuda":
            torch.zeros(1, device=self.device)
            torch.cuda.synchronize()
 
    @torch.no_grad()
    def infer(self, frames):
        step, t_state_ns, P, H = STATE_HDR.unpack(frames[0])
 
        # 这里用 np.frombuffer + torch.from_numpy，开销小且稳定
        pos_np  = np.frombuffer(frames[1], dtype=np.float32)
        vel_np  = np.frombuffer(frames[2], dtype=np.float32)
        goal_np = np.frombuffer(frames[3], dtype=np.float32)
 
        if pos_np.size != P * 3 or vel_np.size != P * 3:  #  or goal_np.size != H * 3
            raise ValueError(
                f"size mismatch: pos={pos_np.size}, vel={vel_np.size}, goal={goal_np.size}, "
                f"expect {P*3}, {P*3}, {H*3}"
            )
 
        pos  = torch.from_numpy(pos_np).view(1, P, 3).to(self.device)
        vel  = torch.from_numpy(vel_np).view(1, P, 3).to(self.device)
        goal = torch.from_numpy(goal_np).view(-1, 3).to(self.device)
 
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = now_ns()
 
        if step < 0:  # warm up
            old = self.planner.n_improve
            self.planner.n_improve = self.warmup_improve
            self.planner.improve_policy(pos, vel, goal)
            self.planner.n_improve = old
            action = self.planner.get_action(rule="greedy")
        else:
            self.planner.improve_policy(pos, vel, goal)
            action = self.planner.get_action(rule="greedy")
            self.planner.update_policy()
            # print("pos_0: ", pos)  # careful about the expesive comment
            # print("vel: ", vel)
            # print("goal:", goal)
            # print("Planned goal:", self.planner.traj_seeds)
            # print("Action: ", self.planner.seeds)
 
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        compute_ms = (now_ns() - t0) / 1e6
 
        act_np = action.detach().to("cpu").float().numpy().reshape(3).astype(np.float32, copy=False)
        return step, t_state_ns, act_np, compute_ms
 
 
def main(state_sub_bind="tcp://0.0.0.0:4532", action_pub_bind="tcp://0.0.0.0:6001"):
    ctx = zmq.Context.instance()
 
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.bind(state_sub_bind)
 
    pub = ctx.socket(zmq.PUB)
    pub.bind(action_pub_bind)
 
    print(f"[server] SUB bind {state_sub_bind}  (state in)")
    print(f"[server] PUB bind {action_pub_bind} (action out)")
    print("[server] waiting...")
 
    planner_server = PlannerServer()
 
    while True:
        # 阻塞拿到至少一条完整消息
        frames = sub.recv_multipart()
 
        # 清空队列，只保留最新一条
        while True:
            try:
                old_step, _, _, _ = STATE_HDR.unpack(frames[0])
                frames = sub.recv_multipart(flags=zmq.NOBLOCK)
                print(f"[server] step {old_step} discarded")
            except zmq.Again:
                break
            except Exception:
                break
 
        if len(frames) != 4:
            print(f"[server] bad frame count: {len(frames)}")
            continue
 
        try:
            step, t_state_ns, action, compute_ms = planner_server.infer(frames)
 
            if compute_ms >= planner_server.ctr_period/2 and step >= 0:
                print(f"[server] step {step} overtime: {compute_ms:.3f} ms")
 
            if step >= 0:
                pub.send_multipart([
                    ACT_HDR.pack(int(step), int(t_state_ns), float(compute_ms)),
                    action.tobytes()
                ])
                # print("step:", step, " action:", action)
            else:
                print(f"Warmup: step {step} done, compute_ms {compute_ms}ms...")

 
        except Exception as e:
            print(f"[server] infer error: {e}")
 
 
if __name__ == "__main__":
    main()