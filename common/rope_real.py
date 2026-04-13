import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from common.utils import *
import math


class Rope(nn.Module):
    """
    A PyTorch implementation of a mass-spring-damper system for simulating rope/chain dynamics.
    
    This model represents a chain of point masses connected by springs and dampers, with optional:
    - Linear springs and damping between adjacent masses
    - Bending springs and damping between segments
    - Air drag forces
    - Gravity
    - Fixed/driven boundary conditions
    
    The system is integrated using explicit Euler integration.
    
    Key features:
    - Batched computation for parallel simulation of multiple ropes
    - GPU support via PyTorch tensors
    - Efficient vectorized implementation
    """
    def __init__(self, N=10, L=0.5, mass=0.02, k=5000, k_bend=0.0, damping=2.0,
                 damping_bend=0.0, twisting=0.0, air_drag=0.001, g=9.81, dt=0.001, device='cpu', mode='id', y_standard=None):
        super().__init__()

        self.N = N
        self.L = L
        self.mass_val = mass
        self.k_val = k
        self.damping_val = damping
        self.k_bend_val = k_bend
        self.damping_bend_val = damping_bend
        self.twisting = twisting
        self.air_drag_val = air_drag
        self.g_val = g
        self.dt_val = dt
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if y_standard is not None:
            self.y_standard = y_standard.to(device)
        self.EPS = 1e-7

        if mode == 'id':
            print('rope: identification mode')
            # Learnable Parameters (using original _tensor names for compatibility)
            # self.mass_tensor = nn.Parameter(torch.tensor(mass, dtype=torch.float32))
            self.k_tensor = nn.Parameter(torch.tensor(self.k_val, dtype=torch.float32))
            self.damping_tensor = nn.Parameter(torch.tensor(self.damping_val, dtype=torch.float32).unsqueeze(1))
            self.k_bend_tensor = nn.Parameter(torch.tensor(self.k_bend_val, dtype=torch.float32))
            self.damping_bend_tensor = nn.Parameter(torch.tensor(self.damping_bend_val, dtype=torch.float32).unsqueeze(1))
            self.twisting_tensor = nn.Parameter(torch.tensor(twisting, dtype=torch.float32))
            self.air_drag_tensor = nn.Parameter(torch.tensor(self.air_drag_val, dtype=torch.float32))
            self.g_tensor = nn.Parameter(torch.tensor(self.g_val, dtype=torch.float32))

            # Register buffers for constants, ensuring float32 and correct device
            self.register_buffer('mass_tensor', torch.tensor(self.mass_val, dtype=torch.float32))
            # self.register_buffer('k_tensor', torch.tensor(self.k_val, dtype=torch.float32))
            # self.register_buffer('k_bend_tensor', torch.tensor(self.k_bend_val, dtype=torch.float32))
            # self.register_buffer('damping_tensor', torch.tensor(self.damping_val, dtype=torch.float32).unsqueeze(1))
            # self.register_buffer('damping_bend_tensor', torch.tensor(self.damping_bend_val, dtype=torch.float32).unsqueeze(1))
            # self.register_buffer('air_drag_tensor', torch.tensor(self.air_drag_val, dtype=torch.float32))
            # self.register_buffer('g_tensor', torch.tensor(self.g_val, dtype=torch.float32))
            self.register_buffer('dt_tensor', torch.tensor(self.dt_val, dtype=torch.float32))
            self.register_buffer('dx_tensor', L.clone().detach().view(1, self.N, 1))
            self.register_buffer('zero_vec_2d', torch.tensor([0.0, 0.0], dtype=torch.float32))
        elif mode == 'ctr':
            print('rope: control mode')
            # Register buffers for constants, ensuring float32 and correct device
            self.register_buffer('mass_tensor', torch.tensor(self.mass_val, dtype=torch.float32))
            self.register_buffer('k_tensor', torch.tensor(self.k_val, dtype=torch.float32))
            self.register_buffer('k_bend_tensor', torch.tensor(self.k_bend_val, dtype=torch.float32))
            self.register_buffer('damping_tensor', torch.tensor(self.damping_val, dtype=torch.float32).unsqueeze(1))
            self.register_buffer('damping_bend_tensor', torch.tensor(self.damping_bend_val, dtype=torch.float32).unsqueeze(1))
            self.register_buffer('air_drag_tensor', torch.tensor(self.air_drag_val, dtype=torch.float32))
            self.register_buffer('g_tensor', torch.tensor(self.g_val, dtype=torch.float32))
            self.register_buffer('dt_tensor', torch.tensor(self.dt_val, dtype=torch.float32))
            self.register_buffer('dx_tensor', L.clone().detach().view(1, self.N, 1))
            self.register_buffer('zero_vec_2d', torch.tensor([0.0, 0.0], dtype=torch.float32))

        self.to(self.device)

    def clamp_parameters(self):
        with torch.no_grad():
            # self.mass_tensor.data.clamp_(0.0, 1e6)
            self.k_tensor.data.clamp_(0.12, 0.12)  # 对mujoco越大越好
            self.damping_tensor.data.clamp_(0.0, 0.2)  # k很大时不重要
            self.k_bend_tensor.data.clamp_(0.0, 1)  # 设为0，偏小更好
            self.damping_bend_tensor.data.clamp_(0.0, 0.0015)  # 玄学，要调
            self.air_drag_tensor.data.clamp_(0.0, 100.0)  # air_drag和g配合调
            self.g_tensor.data.clamp_(9.0, 11.0)

    def _bending_spring_force(self, positions, direction, lengths):
        dot_products = torch.sum(direction[:, :-1] * direction[:, 1:], dim=2)
        dot_products = torch.clamp(dot_products, -1.0 + 1e-8, 1.0 - 1e-7)
        betas = torch.acos(dot_products)
        # betas = torch.clamp(betas, self.EPS, math.pi - self.EPS)  # 关键改动

        cross = torch.linalg.cross(direction[:, :-1], direction[:, 1:], dim=2)

        # s = torch.sin(betas).clamp_min(self.EPS)
        s = torch.sin(betas) + 1e-8
        common = (self.k_bend_tensor * betas / s).unsqueeze(2) * cross

        pre_bending_force = -torch.linalg.cross(direction[:, :-1], common, dim=2) / lengths[:, :-1]
        after_bending_force = -torch.linalg.cross(direction[:, 1:], common, dim=2) / lengths[:, 1:]

        bending_force = torch.zeros_like(positions, device=self.device, dtype=torch.float32)
        bending_force[:, 1:-1] += pre_bending_force + after_bending_force
        bending_force[:, :-2] += -pre_bending_force
        bending_force[:, 2:] += -after_bending_force
        
        return bending_force
    
    def _bending_spring_force_simple(self, positions):
        rest_bend_length = 2 * self.dx_tensor

        # Vectors between particle i and particle i+2
        bend_delta = positions[:, 2:] - positions[:, :-2]  # Shape: (batch_size, N-1, 2)
        
        bend_length = torch.linalg.norm(bend_delta, dim=2, keepdim=True)
        bend_direction = bend_delta / (bend_length + 1e-8)

        # Force along the i <-> i+2 spring
        bend_force_on_segments = self.k_bend_tensor.view(1, self.k_bend_tensor.shape[0], 1) * (bend_length - rest_bend_length) * bend_direction

        bending_force = torch.zeros_like(positions, device=self.device, dtype=torch.float32)
        
        bending_force[:, :-2] += bend_force_on_segments # Apply to particle i
        bending_force[:, 2:] -= bend_force_on_segments  # Apply to particle i+2
        
        return bending_force
    
    def _bending_damping_force(self, velocities, direction, lengths):
        cross1 = torch.linalg.cross(direction[:, :-1], direction[:, 1:], dim=2)
        grad_direction_e1 = torch.linalg.cross(direction[:, :-1], cross1, dim=2)
        dbeta_de1 = grad_direction_e1 / (torch.linalg.norm(grad_direction_e1, dim=2, keepdim=True) * lengths[:, :-1] + 1e-8)

        # cross2 = torch.linalg.cross(direction[:, 1:], direction[:, :-1], dim=2)
        cross2 = -cross1
        grad_direction_e2 = torch.linalg.cross(direction[:, 1:], cross2, dim=2)
        dbeta_de2 = grad_direction_e2 / (torch.linalg.norm(grad_direction_e2, dim=2, keepdim=True) * lengths[:, 1:] + 1e-8)

        vel_diff1 = velocities[:, 1:-1] - velocities[:, :-2]
        vel_diff2 = velocities[:, 2:] - velocities[:, 1:-1]
        dbeta_dt = torch.sum(dbeta_de1 * vel_diff1 + dbeta_de2 * vel_diff2, dim=2, keepdim=True)

        tmp = self.damping_bend_tensor * dbeta_dt
        F_i_minus1 = tmp * dbeta_de1
        F_i_plus1 = -tmp * dbeta_de2

        bending_damping_force = torch.zeros_like(velocities, device=self.device, dtype=torch.float32)
        bending_damping_force[:, :-2] += F_i_minus1
        bending_damping_force[:, 1:-1] -= (F_i_minus1 + F_i_plus1)  # Corrected logic
        bending_damping_force[:, 2:] += F_i_plus1

        return bending_damping_force
    
    def _bending_damping_force_simple(self, positions, velocities):
        # Vector and direction for i <-> i+2 connections
        bend_vec = positions[:, 2:] - positions[:, :-2]  # Shape: (batch_size, N-1, 3)
        bend_len = torch.linalg.norm(bend_vec, dim=2, keepdim=True)
        bend_dir = bend_vec / (bend_len + 1e-8) # Shape: (batch_size, N-1, 3)

        # Relative velocity between particle i and particle i+2
        rel_vel = velocities[:, 2:] - velocities[:, :-2]  # Shape: (batch_size, N-1, 3)
        
        # Project relative velocity onto the i <-> i+2 direction
        proj_vel = torch.sum(rel_vel * bend_dir, dim=2, keepdim=True)  # Shape: (batch_size, N-1, 1)
        
        # Bending damping force along the i <-> i+2 direction
        bending_damping_force_on_segments = self.damping_bend_tensor * proj_vel * bend_dir  # Shape:(batch_size, N-1, 3)

        bending_damping_force = torch.zeros_like(velocities, device=self.device, dtype=torch.float32)
        
        bending_damping_force[:, :-2] += bending_damping_force_on_segments  # Apply to particle i
        bending_damping_force[:, 2:] -= bending_damping_force_on_segments  # Apply to particle i+2

        return bending_damping_force

    def _twisting_force(self, positions, direction, lengths):
        """
            Twisting force implementation of paper:
                Physically based real-time interactive assembly simulation of cable harness
        """
        delta_cross = torch.linalg.cross(direction[:, 1:], direction[:, :-1], dim=2)
        ai = torch.linalg.cross(direction[:, 1:-1], delta_cross[:, :-1], dim=2)
        bi = torch.linalg.cross(direction[:, 1:-1], delta_cross[:, 1:], dim=2)
        denom = torch.sum(ai * bi, dim=2)
        phi_i = torch.atan(torch.norm(torch.cross(ai, bi, dim=2), dim=2) /
                           (denom + 0.9e-8 + 1e-8 * denom.sign())).unsqueeze(2)
        # phi_i = torch.where(phi_i < 0, phi_i + math.pi, phi_i)

        # dot_products_phi = (torch.sum(ai * bi, dim=2) + 1e-40) / (torch.norm(ai, dim=2) * torch.norm(bi, dim=2) + 1e-40)
        # dot_products_phi = torch.clamp(dot_products_phi, -1.0 + 1e-8, 1.0 - 1e-8)
        # phi_i2 = torch.acos(dot_products_phi).unsqueeze(2)
        # phi_gpt = self.compute_geometric_torsion(positions)

        dot_products = torch.sum(direction[:, :-1] * direction[:, 1:], dim=2)
        dot_products = torch.clamp(dot_products, -1.0 + 1e-8, 1.0 - 1e-8)
        betas = torch.acos(dot_products).unsqueeze(2)
        cross = torch.linalg.cross(direction[:, :-1], direction[:, 1:], dim=2)
        common = cross / (torch.sin(betas) + 1e-8)

        twisting_force = torch.zeros_like(positions, device=self.device, dtype=torch.float32)

        force_1 = self.twisting_tensor * phi_i / (lengths[:, :-2] * torch.sin(betas[:, :-1])) * common[:, :-1]
        twisting_force[:, :-3] += force_1
        twisting_force[:, 1:-2] -= force_1

        force_2 = self.twisting_tensor * phi_i / (lengths[:, 1:-1] * torch.tan(betas[:, 1:])) * common[:, 1:]
        twisting_force[:, 2:-1] += force_2
        twisting_force[:, 1:-2] -= force_2

        force_3 = self.twisting_tensor * phi_i / (lengths[:, 1:-1] * torch.tan(betas[:, :-1])) * common[:, :-1]
        twisting_force[:, 2:-1] += force_3
        twisting_force[:, 1:-2] -= force_3

        force_4 = self.twisting_tensor * phi_i / (lengths[:, 2:] * torch.sin(betas[:, 1:])) * common[:, 1:]
        twisting_force[:, 2:-1] += force_4
        twisting_force[:, 3:] -= force_4

        if torch.isnan(twisting_force).any():
            print('wrong here')

        return twisting_force

    def _compute_geometric_torsion_angle(self, positions):
        """
        positions: (B, N, 3)
        returns: torsion angles at positions[:, 1:-2], shape (B, N-3)
        """
        p0 = positions[:, :-3]  # (B, N-3, 3)
        p1 = positions[:, 1:-2]
        p2 = positions[:, 2:-1]
        p3 = positions[:, 3:]

        # Compute directions
        b1 = torch.nn.functional.normalize(p1 - p0, dim=-1)  # shape (B, N-3, 3)
        b2 = torch.nn.functional.normalize(p2 - p1, dim=-1)
        b3 = torch.nn.functional.normalize(p3 - p2, dim=-1)

        # Compute normals
        n1 = torch.cross(b1, b2, dim=2)
        n2 = torch.cross(b2, b3, dim=2)

        # Check if n1 n2 is too small (Points p0 p1 p2 are collinear)
        mask = torch.zeros(n1.shape[0], n1.shape[1], dtype=torch.bool)
        mask |= (torch.norm(n1, dim=2) < 1e-8) | (torch.norm(n2, dim=2) < 1e-8)

        n1_norm = torch.nn.functional.normalize(n1, dim=-1)
        n2_norm = torch.nn.functional.normalize(n2, dim=-1)

        # Dot and triple product for atan2
        dot = (n1_norm * n2_norm).sum(dim=-1)  # (B, N-3)
        cross = torch.cross(n1_norm, n2_norm, dim=2)  # (B, N-3, 3)
        sin = (cross * b2).sum(dim=-1)

        # Final torsion angle (signed)
        angle = torch.atan2(sin, dot)  # (B, N-3)
        angle = torch.where(mask, torch.zeros_like(angle), angle)

        return angle  # in radians, range [-pi, pi]

    def _twisting_force_by_autodiff(self, positions):
        positions_copy = positions.clone().detach().requires_grad_(True)
        phi = self._compute_geometric_torsion_angle(positions_copy)
        energy = 0.5 * self.twisting_tensor * (phi ** 2).sum()  # sum over batch

        # another way to calculate gradient (for higher diff)
        # twisting_force_by_gradient = -torch.autograd.grad(energy, positions_copy, create_graph=True)[0]

        energy.backward()
        twisting_force_by_gradient = -positions_copy.grad

        return twisting_force_by_gradient

    def dynamics(self, positions, velocities, control_input):
        """
        Calculates the forces acting on the system based on current state,
        and then computes the resulting accelerations.
        Does NOT apply boundary conditions like fixed points or driving forces here.
        Args:
            positions (torch.Tensor): Current positions (batch_size, N+1, 2).
            velocities (torch.Tensor): Current velocities (batch_size, N+1, 2).
        Returns:
            torch.Tensor: Calculated accelerations (batch_size, N+1, 2).
        """
        forces = torch.zeros_like(positions, device=self.device, dtype=torch.float32)

        # === Gravity ===
        forces[:, :, 2] -= self.mass_tensor * self.g_tensor  # Broadcasting

        # === Air damping ===
        forces -= (self.air_drag_tensor / 1000) * velocities  # scaler up k with 10000

        # === Spring + Damping ===
        delta = positions[:, 1:] - positions[:, :-1]  # point from i to i+1
        lengths = torch.linalg.norm(delta, dim=2, keepdim=True)
        direction = delta / (lengths + 1e-8)

        rel_vel = velocities[:, 1:] - velocities[:, :-1]  # point from i to i+1
        proj_vel = torch.sum(rel_vel * direction, dim=2, keepdim=True)

        # f_spring = (10000 * self.k_tensor) * (lengths - self.dx_tensor) * direction  # scaler down k with 10000
        # f_damp = self.damping_tensor * proj_vel * direction
        # f_total = f_spring + f_damp

        # 合并计算
        f_total = (
                          (10000 * self.k_tensor) * (lengths - self.dx_tensor) +
                          self.damping_tensor * proj_vel
                  ) * direction

        forces[:, :-1] += f_total
        forces[:, 1:]  -= f_total

        # === bending spring force ===
        forces += self._bending_spring_force(positions, direction, lengths)
        # forces += self._bending_spring_force_simple(positions)

        # === bending damping force ===
        forces += self._bending_damping_force(velocities, direction, lengths)
        # forces += self._bending_damping_force_simple(positions, velocities)

        # === twisting force (geometric torsion) ===
        # forces += self._twisting_force(positions, direction, lengths)
        # forces += self._twisting_force_by_autodiff(positions)

        # === Apply top point force constraint (top point is driven/fixed) ===
        forces[:, 0, :] = control_input * self.mass_tensor[0]  # Set force on the first particle
        # forces[:, 0, 2] = 0

        # === Newton Law to get accelerations ===
        accelerations = forces / self.mass_tensor.view(1, self.N+1, 1)

        return accelerations

    def symplectic_euler_step(self, positions, velocities, control_input, mode='acc'):

        if mode == 'acc':
            current_accelerations = self.dynamics(positions, velocities, control_input)

            next_velocities = velocities + self.dt_tensor * current_accelerations
            next_positions = positions + self.dt_tensor * next_velocities

            return next_positions, next_velocities
        elif mode == 'vel':
            velocities[:, 0, :] = control_input
            current_accelerations = self.dynamics(positions, velocities, torch.zeros_like(control_input))

            next_velocities = velocities + self.dt_tensor * current_accelerations
            next_positions = positions + self.dt_tensor * next_velocities

            return next_positions, next_velocities
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def RK4_step(self, positions, velocities, control_input, mode='acc'):
        """
        Runge-Kutta 4th order integration for rope dynamics.
        """
        dt = self.dt_tensor

        if mode == 'acc':
            # k1
            a1 = self.dynamics(positions, velocities, control_input)
            v1 = velocities
            # k2
            v2 = velocities + 0.5 * dt * a1
            p2 = positions + 0.5 * dt * v1
            a2 = self.dynamics(p2, v2, control_input)
            # k3
            v3 = velocities + 0.5 * dt * a2
            p3 = positions + 0.5 * dt * v2
            a3 = self.dynamics(p3, v3, control_input)
            # k4
            v4 = velocities + dt * a3
            p4 = positions + dt * v3
            a4 = self.dynamics(p4, v4, control_input)

            next_positions = positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            next_velocities = velocities + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
            return next_positions, next_velocities

        elif mode == 'vel':
            # For 'vel' mode, the control_input is velocity for the first particle
            # At each substep, we need to set velocities[:, 0] = control_input
            # k1
            v1 = velocities.clone()
            v1[:, 0] = control_input
            a1 = self.dynamics(positions, v1, torch.zeros_like(control_input))
            # k2
            v2 = velocities + 0.5 * dt * a1
            v2[:, 0] = control_input
            p2 = positions + 0.5 * dt * v1
            a2 = self.dynamics(p2, v2, torch.zeros_like(control_input))
            # k3
            v3 = velocities + 0.5 * dt * a2
            v3[:, 0] = control_input
            p3 = positions + 0.5 * dt * v2
            a3 = self.dynamics(p3, v3, torch.zeros_like(control_input))
            # k4
            v4 = velocities + dt * a3
            v4[:, 0] = control_input
            p4 = positions + dt * v3
            a4 = self.dynamics(p4, v4, torch.zeros_like(control_input))

            next_positions = positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            next_velocities = velocities + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
            return next_positions, next_velocities
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def energy(self, batch_pos, batch_vel):

        vel_squared_without_head = (batch_vel ** 2).sum(dim=-1)  # (batch, N)
        k_energy = 0.5 * self.mass_tensor * vel_squared_without_head

        y_cur = batch_pos[:, :, 2]
        p_energy = self.mass_tensor * self.g_tensor * (y_cur - self.y_standard)

        # # 增加底部权重
        # weight = torch.linspace(0.5, 1.5, steps=self.N, device=self.device).unsqueeze(0)
        # k_energy *= weight
        # p_energy[:, 1:] *= weight

        return k_energy.sum(-1) + p_energy.sum(-1)

    def simulation_for_ctr(self, batch_pos, batch_vel, control_sequence, mode='acc', ctr_period=10):

        for i in range(ctr_period):
            # Update state
            batch_pos, batch_vel = self.symplectic_euler_step(batch_pos, batch_vel, control_sequence, mode=mode)

        return batch_pos, batch_vel
    
    def simulation_for_training(self, initial_positions, initial_velocities, control_sequence, steps=10):
        """
        Runs simulation for training with gradient information preserved for identification.
        
        Args:
            initial_positions (torch.Tensor): Initial positions (batch_size, N+1, 3)
            initial_velocities (torch.Tensor): Initial velocities (batch_size, N+1, 3)
            steps (int): Number of simulation steps
            
        Returns:
            tuple: (positions_traj_pred, velocities_traj_pred) containing full trajectory
                  positions_traj_pred shape: (batch_size, steps, N+1, 3)
                  velocities_traj_pred shape: (batch_size, steps, N+1, 3)
        """
        positions = initial_positions.clone()
        velocities = initial_velocities.clone()
        control_sequence = control_sequence.clone()
        
        positions_traj_pred = []
        velocities_traj_pred = []
        
        for i in range(steps):
            # Update state
            positions, velocities = self.symplectic_euler_step(positions, velocities, control_sequence[:, i])
            
            positions_traj_pred.append(positions)
            velocities_traj_pred.append(velocities)
            
        return (torch.stack(positions_traj_pred, dim=1), 
                torch.stack(velocities_traj_pred, dim=1))

    def simulate_with_controller(self, initial_positions, initial_velocities, controller, steps=1000, record_interval=10, mode='acc'):
        positions = initial_positions.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 2)
        velocities = initial_velocities.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 2)

        positions_history = []
        # 1. Unconditionally record initial state (t=0)
        positions_history.append(positions.detach().clone().cpu().unsqueeze(1))

        for i in range(steps):

            # === symplectic_euler_step now handles calling dynamics and applying control input ===
            action = controller(positions, velocities)
            positions, velocities = self.symplectic_euler_step(positions, velocities, action, mode=mode)

            # === Record trajectory ===
            if (i + 1) % record_interval == 0:
                positions_history.append(positions.detach().clone().cpu().unsqueeze(1))

        if not positions_history:
            return initial_positions.clone().cpu().unsqueeze(1)
        else:
            return torch.cat(positions_history, dim=1)

    def simulate(self, initial_positions, initial_velocities, control_sequence, steps=1000, record_interval=10,
                 mode='acc'):
        """
        Runs the full simulation loop, using euler_step (which calls dynamics).
        """
        positions = initial_positions.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 3)
        velocities = initial_velocities.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 3)
        control_sequence = control_sequence.clone().to(self.device, dtype=torch.float32)  # (batch_size, steps, 3)

        positions_history = []
        # # 1. Unconditionally record initial state (t=0)
        # positions_history.append(positions.detach().clone().cpu().unsqueeze(1))

        for i in range(steps):

            # === symplectic_euler_step now handles calling dynamics and applying control input ===
            positions, velocities = self.symplectic_euler_step(positions, velocities, control_sequence[:, i], mode=mode)

            # === Record trajectory ===
            if (i + 1) % record_interval == 0:
                positions_history.append(positions.detach().clone().cpu().unsqueeze(1))

        if not positions_history:
            return initial_positions.clone().cpu().unsqueeze(1)
        else:
            return torch.cat(positions_history, dim=1)

    def sampling_forward(self, initial_positions, initial_velocities, control_sequence, horizion=100,
                         ctr_period=10, mode='acc'):
        """
        Runs the full simulation loop, using euler_step (which calls dynamics).
        """
        positions = initial_positions.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 2)
        velocities = initial_velocities.clone().to(self.device, dtype=torch.float32)  # (batch_size, N+1, 2)
        control_sequence = control_sequence.clone().to(self.device, dtype=torch.float32)  # (batch_size, steps, 2)

        positions_history = []

        for i in range(horizion):
            positions, velocities = self.simulation_for_ctr(positions, velocities, control_sequence[:, i],
                                                            ctr_period=ctr_period, mode=mode)
            positions_history.append(positions.detach().clone().unsqueeze(1))

        if not positions_history:
            return initial_positions.clone().cpu().unsqueeze(1)
        else:
            return torch.cat(positions_history, dim=1)


# Main execution block
if __name__ == "__main__":
    # === Parameters ===
    N = 20  # Number of segments
    L = 1.0  # Total length (m)
    mass = 0.0025 * 40 / N  # Mass per segment (kg)
    k = 0.46  # Spring stiffness
    damping = [0.2] * N  # Damping coefficient
    k_bend = [0.0006712] * (N - 1)  # Bending stiffness
    damping_bend = [0.000401] * (N - 1)  # Bending damping coefficient
    twisting = 0.00002
    air_drag = 0.2206  # Drag coefficient
    g = 10.07  # Gravitational acceleration
    dt = 0.001  # Time step (s)
    T = 10.0  # Simulation duration (s)
    total_steps = int(T / dt)
    record_interval = 10

    # === Setup device ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Comment this out
    # device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")

    # === Create model instance ===
    model = Rope(N=N, L=L, mass=mass, k=k, k_bend=k_bend, damping=damping, damping_bend=damping_bend, twisting=twisting, air_drag=air_drag,
                 g=g, dt=dt, device=device, mode='id')

    # # === compile model ===
    # compile_model(model)

    # === sampled data processing ===
    # sampled_data = np.load('../data/fixed_tip_horizontal_init.npy').astype(np.float32)  # (sample_num, 1+node_num*6+2)
    sampled_data = np.load('../data/fixed_tip_pos_low.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_high.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_x_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/pos_low_with_xyz_drive.npy').astype(np.float32)
    # sampled_data = np.load('../data/fixed_tip_pos_3d.npy').astype(np.float32)
    # sampled_data = np.load('../data/static_init_with_xyz_drive.npy').astype(np.float32)
    position_data, velocity_data, control_sequence = mj_data_to_my_data(N, sampled_data, device)

    # plot_animation_3d(position_data[::10], 0.001, 10, 1.0)

    # === show before id ===
    print("here")
    show_start = 0
    # with torch.no_grad():  # Disable gradient computation
    start = time.perf_counter()
    with torch.no_grad():
        positions_history_mine = model.simulate(
            position_data[show_start:show_start + 1],
            velocity_data[show_start:show_start + 1],
            control_sequence[show_start:].unsqueeze(0),
            steps=int(total_steps-show_start),
            record_interval=record_interval
        )
    print("done!")
    print('time: ', time.perf_counter()-start)
    position_data_for_show = position_data[show_start:][::record_interval].unsqueeze(0)
    print('Err: ', (position_data_for_show.cpu() - positions_history_mine.cpu()).abs().sum())
    plot_animation_two_ropes_3d(position_data_for_show, positions_history_mine, dt, record_interval, L, repeat=True)
    # plot_animation_two_ropes_3d(position_data_for_show, positions_history_mine, dt, record_interval, L,
    #                             save_path="../ani/fixed_tip_3d_twist_0_00005_gradient.mp4")
