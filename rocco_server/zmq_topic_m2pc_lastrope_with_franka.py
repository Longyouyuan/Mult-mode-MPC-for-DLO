#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import time
import zmq
import numpy as np
import torch
import warp as wp

from M2PC_lastrope_with_franka import Planner, cost_fn
from warp_rope_franka_model import WarpRopeFranka, N, P, load_runner


# state message:
#   frame0: header
#   frame1: pos      float32, shape=(P,3)
#   frame2: vel      float32, shape=(P,3)
#   frame3: goal     float32, shape=(H,3)   [H can be <= planner.horizon]
#   frame4: goal_pos float32, shape=(3,)
#   frame5: goal_vel float32, shape=(3,)
STATE_HDR = struct.Struct("<iqII")   # step(int32), t_state_ns(int64), P(uint32), H(uint32)

# action message:
#   frame0: header
#   frame1: action      float32, shape=(3,)
#   frame2: real_pos_1  float32, shape=(3,)
ACT_HDR = struct.Struct("<iqf")      # step_ref(int32), t_state_ns(int64), compute_ms(float32)


def now_ns():
    return time.perf_counter_ns()


class PlannerServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[server] device = {self.device}")

        # ----- MPC settings -----
        self.dt = 0.001
        self.ctr_period = 25
        self.horizon = 30
        self.n_sample = 1000  # tune here
        self.m_modes = 1
        self.n_improve = 1
        self.noise_scale = 1.8
        self.action_dim = 3
        self.limits = torch.tensor([-5.0 / 5.0, 5.0 / 5.0], device=self.device, dtype=torch.float32)  # careful about here
        self.top_k_good = 200
        self.beta = 5.0
        self.wJ = 0.0
        self.warmup_improve = 200

        # ----- Rope / Franka model params -----
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

        model_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.franka_runner = load_runner("best_model.pt", device=model_device)

        # planner rollout model
        self.rope = WarpRopeFranka(
            franka_runner=self.franka_runner,
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
            device=model_device,
        )

        # make sure standard_m2pc=True 且 m=1
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
            standard_m2pc=True,
        )

        # warm up GPU context
        if self.device.type == "cuda":
            torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, frames):
        step, t_state_ns, p_msg, h_msg = STATE_HDR.unpack(frames[0])

        pos_np = np.frombuffer(frames[1], dtype=np.float32)
        vel_np = np.frombuffer(frames[2], dtype=np.float32)
        goal_np = np.frombuffer(frames[3], dtype=np.float32)
        goal_pos_np = np.frombuffer(frames[4], dtype=np.float32)
        goal_vel_np = np.frombuffer(frames[5], dtype=np.float32)

        if pos_np.size != p_msg * 3:
            raise ValueError(f"pos size mismatch: got {pos_np.size}, expect {p_msg * 3}")
        if vel_np.size != p_msg * 3:
            raise ValueError(f"vel size mismatch: got {vel_np.size}, expect {p_msg * 3}")
        # if goal_np.size != h_msg * 3:
        #     raise ValueError(f"goal size mismatch: got {goal_np.size}, expect {h_msg * 3}")
        if goal_pos_np.size != 3:
            raise ValueError(f"goal_pos size mismatch: got {goal_pos_np.size}, expect 3")
        if goal_vel_np.size != 3:
            raise ValueError(f"goal_vel size mismatch: got {goal_vel_np.size}, expect 3")

        pos = torch.from_numpy(pos_np).view(1, p_msg, 3).to(self.device)
        vel = torch.from_numpy(vel_np).view(1, p_msg, 3).to(self.device)
        goal = torch.from_numpy(goal_np).view(h_msg, 3).to(self.device)
        goal_pos = torch.from_numpy(goal_pos_np).view(1, 3).to(self.device)
        goal_vel = torch.from_numpy(goal_vel_np).view(1, 3).to(self.device)

        if p_msg != P:
            raise ValueError(f"P mismatch: msg P={p_msg}, model P={P}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = now_ns()

        # warmup or normal planning
        if step < 0:
            old = self.planner.n_improve
            self.planner.n_improve = self.warmup_improve
            self.planner.improve_policy(pos, vel, goal, goal_pos, goal_vel)
            self.planner.n_improve = old
            action = self.planner.get_action(rule="greedy")
        else:
            self.planner.improve_policy(pos, vel, goal, goal_pos, goal_vel)
            action = self.planner.get_action(rule="greedy")
            self.planner.update_policy()

        real_pos_1 = self.planner.real_pos_1

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        compute_ms = (now_ns() - t0) / 1e6

        act_np = (
            action.detach().to("cpu").float().numpy().reshape(3).astype(np.float32, copy=False)
        )
        real_pos_np = (
            real_pos_1.detach().to("cpu").float().numpy().reshape(3).astype(np.float32, copy=False)
        )

        return step, t_state_ns, act_np, real_pos_np, compute_ms


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
        frames = sub.recv_multipart()

        # drain queue, keep newest only
        while True:
            try:
                old_step, _, _, _ = STATE_HDR.unpack(frames[0])
                frames = sub.recv_multipart(flags=zmq.NOBLOCK)
                print(f"[server] step {old_step} discarded")
            except zmq.Again:
                break
            except Exception:
                break

        if len(frames) != 6:
            print(f"[server] bad frame count: {len(frames)} (expect 6)")
            continue

        try:
            step, t_state_ns, action, real_pos_1, compute_ms = planner_server.infer(frames)

            if compute_ms >= planner_server.ctr_period / 2 and step >= 0:
                print(f"[server] step {step} overtime: {compute_ms:.3f} ms")

            if step >= 0:
                pub.send_multipart([
                    ACT_HDR.pack(int(step), int(t_state_ns), float(compute_ms)),
                    action.tobytes(),
                    real_pos_1.tobytes(),
                ])
            else:
                print(f"[server] warmup step {step} done, compute_ms {compute_ms:.3f} ms")

        except Exception as e:
            print(f"[server] infer error: {e}")


if __name__ == "__main__":
    main()