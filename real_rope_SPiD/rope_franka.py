from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rope_real import Rope
from test_for_real_experiment.franka_model import SparseGRUFrankaModel


def _to_device_tensor(value, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _as_python_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _repeat_scalar_if_needed(value, length: int):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return [_as_python_float(value)] * length
        values = value.detach().cpu().reshape(-1).tolist()
        if len(values) != length:
            raise ValueError(f"Expected {length} values, got {len(values)}")
        return values

    try:
        tensor_value = torch.as_tensor(value)
    except (TypeError, ValueError):
        return [float(value)] * length

    if tensor_value.numel() == 1:
        return [float(tensor_value.reshape(-1)[0])] * length
    values = tensor_value.reshape(-1).tolist()
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return values


def _segment_lengths(N: int, L, segment_lengths=None) -> torch.Tensor:
    if segment_lengths is None:
        lengths = torch.as_tensor(L, dtype=torch.float32)
        if lengths.numel() == 1:
            lengths = torch.full((N,), float(lengths.reshape(-1)[0]) / float(N), dtype=torch.float32)
        else:
            lengths = lengths.reshape(-1).to(dtype=torch.float32)
    else:
        lengths = torch.as_tensor(segment_lengths, dtype=torch.float32).reshape(-1)

    if lengths.numel() != N:
        raise ValueError(f"segment_lengths/L must provide {N} segment lengths, got {lengths.numel()}")
    return lengths


def _node_masses(N: int, mass, tip_extra_mass: float = 0.0) -> torch.Tensor:
    masses = torch.as_tensor(mass, dtype=torch.float32).reshape(-1)
    if masses.numel() == 1:
        masses = torch.full((N + 1,), float(masses[0]), dtype=torch.float32)
    elif masses.numel() != N + 1:
        raise ValueError(f"mass must be scalar or have {N + 1} node masses, got {masses.numel()}")

    masses = masses.clone()
    masses[-1] += float(tip_extra_mass)
    return masses


def _checkpoint_arg(args: Dict, name: str, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _infer_franka_dims(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    rollout_hidden = 64
    residual_hidden = 64
    encoder_hidden = 64

    if "learned_z0" in state:
        rollout_hidden = int(state["learned_z0"].shape[-1])
    elif "rollout_cell.weight_hh" in state:
        rollout_hidden = int(state["rollout_cell.weight_hh"].shape[1])

    if "acc_head.net.0.weight" in state:
        residual_hidden = int(state["acc_head.net.0.weight"].shape[0])

    if "history_encoder.weight_hh_l0" in state:
        encoder_hidden = int(state["history_encoder.weight_hh_l0"].shape[1])

    return encoder_hidden, rollout_hidden, residual_hidden


def load_franka_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
) -> SparseGRUFrankaModel:
    """Load the SparseGRUFrankaModel checkpoint used by franka_model.py."""
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        args = ckpt.get("args", {})
        encoder_hidden, rollout_hidden, residual_hidden = _infer_franka_dims(state)
        model = SparseGRUFrankaModel(
            hist_len=int(_checkpoint_arg(args, "hist_len", 1)),
            dt=float(ckpt.get("dt", float(_checkpoint_arg(args, "dt_ms", 1.0)) * 1.0e-3)),
            encoder_hidden=int(_checkpoint_arg(args, "encoder_hidden", encoder_hidden)),
            rollout_hidden=int(_checkpoint_arg(args, "rollout_hidden", rollout_hidden)),
            residual_hidden=int(_checkpoint_arg(args, "residual_hidden", residual_hidden)),
        )
    else:
        state = ckpt
        encoder_hidden, rollout_hidden, residual_hidden = _infer_franka_dims(state)
        model = SparseGRUFrankaModel(
            hist_len=1,
            dt=1.0e-3,
            encoder_hidden=encoder_hidden,
            rollout_hidden=rollout_hidden,
            residual_hidden=residual_hidden,
        )

    model.load_state_dict(state)
    model.to(device)
    if eval_mode:
        model.eval()
    return model


class RopeFranka(nn.Module):
    """
    Torch version of the rope + Franka composition in warp_rope_franka_model.py.

    The control sequence can be commanded goal acceleration or commanded goal velocity.
    At every rope time step:
    1. Franka predicts the real end-effector acceleration from real state and goal state.
    2. That acceleration drives rope node 0 through rope_real.Rope dynamics.
    3. The commanded goal state is integrated from the control sequence.
    """

    def __init__(
        self,
        franka_model: Optional[SparseGRUFrankaModel] = None,
        franka_ckpt_path: Optional[Union[str, Path]] = None,
        rope_model: Optional[Rope] = None,
        batch_size: Optional[int] = None,
        N: int = 8,
        L: float = 1.0,
        segment_lengths=None,
        mass: float = 0.005,
        tip_extra_mass: float = 0.0,
        k: float = 0.12,
        damping=0.2,
        k_bend=0.0,
        damping_bend=0.0,
        twisting: float = 0.0,
        air_drag: float = 0.0,
        g: float = 9.81,
        dt: float = 0.001,
        rope_mode: str = "ctr",
        max_record_steps: int = 10000,
        record_interval: int = 10,
        ctr_period: int = 25,
        control_mode: str = "acc",
        device: Union[str, torch.device] = "cpu",
        freeze_franka: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.ctr_period = int(ctr_period)
        self.record_interval = int(record_interval)
        self.max_record_steps = int(max_record_steps)
        self.control_mode = self._normalize_control_mode(control_mode)

        if franka_model is None:
            if franka_ckpt_path is None:
                default_ckpt = Path(__file__).with_name("best_model.pt")
                if default_ckpt.exists():
                    franka_ckpt_path = default_ckpt
                else:
                    raise ValueError("Provide either franka_model or franka_ckpt_path")
            franka_model = load_franka_model(franka_ckpt_path, device=self.device)
        self.franka = franka_model.to(self.device)

        if freeze_franka:
            self.franka.eval()
            for param in self.franka.parameters():
                param.requires_grad_(False)

        if rope_model is None:
            segment_lengths_tensor = _segment_lengths(int(N), L, segment_lengths)
            node_mass_tensor = _node_masses(int(N), mass, tip_extra_mass)
            damping_values = _repeat_scalar_if_needed(damping, int(N))
            damping_bend_values = _repeat_scalar_if_needed(damping_bend, max(int(N) - 1, 1))

            rope_model = Rope(
                N=int(N),
                L=segment_lengths_tensor.cpu(),
                mass=node_mass_tensor.cpu().tolist(),
                k=k,
                k_bend=k_bend,
                damping=damping_values,
                damping_bend=damping_bend_values,
                twisting=twisting,
                air_drag=air_drag,
                g=g,
                dt=dt,
                device=self.device,
                mode=rope_mode,
            )
        else:
            rope_model = rope_model.to(self.device)

        rope_model.device = self.device
        self.rope = rope_model
        self.N = int(self.rope.N)
        self.P = self.N + 1

        self._stored_rope_pos = None
        self._stored_rope_vel = None
        self._stored_action = None
        self._stored_goal_pos = None
        self._stored_goal_vel = None
        self._stored_control_mode = self.control_mode

    @property
    def dt(self) -> torch.Tensor:
        return self.rope.dt_tensor

    def _target_batch_size(self, rope_pos: torch.Tensor, action: Optional[torch.Tensor] = None) -> int:
        if self.batch_size is not None:
            return self.batch_size
        if action is not None and action.dim() == 3:
            return int(action.shape[0])
        if rope_pos.dim() == 3:
            return int(rope_pos.shape[0])
        return 1

    @staticmethod
    def _normalize_control_mode(control_mode: str) -> str:
        mode = str(control_mode).lower()
        aliases = {
            "acc": "acc",
            "acceleration": "acc",
            "goal_acc": "acc",
            "goal_acceleration": "acc",
            "vel": "vel",
            "velocity": "vel",
            "goal_vel": "vel",
            "goal_velocity": "vel",
        }
        if mode not in aliases:
            raise ValueError(f"control_mode must be 'acc' or 'vel', got {control_mode!r}")
        return aliases[mode]

    def _prepare_rope_state(self, value, batch_size: int, name: str) -> torch.Tensor:
        tensor = _to_device_tensor(value, self.device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3 or tensor.shape[1:] != (self.P, 3):
            raise ValueError(f"{name} must have shape ({self.P}, 3) or (B, {self.P}, 3), got {tuple(tensor.shape)}")
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, -1, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {tensor.shape[0]} does not match target batch size {batch_size}")
        return tensor.contiguous()

    def _prepare_vec3(self, value, batch_size: int, name: str, default: torch.Tensor) -> torch.Tensor:
        if value is None:
            tensor = default
        else:
            tensor = _to_device_tensor(value, self.device, dtype=default.dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2 or tensor.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (3,) or (B, 3), got {tuple(tensor.shape)}")
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {tensor.shape[0]} does not match target batch size {batch_size}")
        return tensor.contiguous()

    def _prepare_action(self, action, batch_size: int, steps: Optional[int], ctr_period: int) -> torch.Tensor:
        tensor = _to_device_tensor(action, self.device)
        if tensor.dim() == 1:
            if tensor.shape[0] != 3:
                raise ValueError(f"action must end with dim 3, got {tuple(tensor.shape)}")
            tensor = tensor.view(1, 1, 3).expand(batch_size, -1, -1)
        elif tensor.dim() == 2:
            if tensor.shape[-1] != 3:
                raise ValueError(f"action must end with dim 3, got {tuple(tensor.shape)}")
            tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
        elif tensor.dim() == 3:
            if tensor.shape[-1] != 3:
                raise ValueError(f"action must end with dim 3, got {tuple(tensor.shape)}")
            if tensor.shape[0] == 1 and batch_size != 1:
                tensor = tensor.expand(batch_size, -1, -1)
            elif tensor.shape[0] != batch_size:
                raise ValueError(f"action batch size {tensor.shape[0]} does not match target batch size {batch_size}")
        else:
            raise ValueError(f"Unsupported action shape: {tuple(tensor.shape)}")

        if steps is not None:
            required = (int(steps) + int(ctr_period) - 1) // int(ctr_period)
            if tensor.shape[1] < required:
                raise ValueError(
                    f"action has {tensor.shape[1]} control steps, but {required} are required "
                    f"for steps={steps}, ctr_period={ctr_period}"
                )
        return tensor.contiguous()

    def _initial_hidden(self, batch_size: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden_state is not None:
            hidden = _to_device_tensor(hidden_state, self.device)
            if hidden.dim() == 1:
                hidden = hidden.unsqueeze(0)
            if hidden.shape[0] == 1 and batch_size != 1:
                hidden = hidden.expand(batch_size, -1)
            elif hidden.shape[0] != batch_size:
                raise ValueError(f"hidden_state batch size {hidden.shape[0]} does not match {batch_size}")
            return hidden.contiguous()
        return self.franka.learned_z0.expand(batch_size, -1).contiguous()

    def _franka_acc_step(
        self,
        real_pos: torch.Tensor,
        real_vel: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_vel: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([real_pos, real_vel, goal_pos, goal_vel], dim=-1)
        hidden = self.franka.rollout_cell(x, hidden)
        acc = self.franka.acc_head(torch.cat([hidden, x], dim=-1))
        return acc, hidden

    def set_state_and_action(
        self,
        rope_pos,
        rope_vel,
        action,
        goal_pos=None,
        goal_vel=None,
        control_mode: Optional[str] = None,
    ) -> None:
        action_tensor = _to_device_tensor(action, self.device)
        rope_pos_tensor = _to_device_tensor(rope_pos, self.device)
        batch_size = self._target_batch_size(rope_pos_tensor, action_tensor)

        self._stored_rope_pos = self._prepare_rope_state(rope_pos_tensor, batch_size, "rope_pos")
        self._stored_rope_vel = self._prepare_rope_state(rope_vel, batch_size, "rope_vel")
        self._stored_action = self._prepare_action(action_tensor, batch_size, None, self.ctr_period)
        self._stored_goal_pos = self._prepare_vec3(
            goal_pos,
            batch_size,
            "goal_pos",
            default=self._stored_rope_pos[:, 0, :],
        )
        self._stored_goal_vel = self._prepare_vec3(
            goal_vel,
            batch_size,
            "goal_vel",
            default=self._stored_rope_vel[:, 0, :],
        )
        self._stored_control_mode = (
            self.control_mode if control_mode is None else self._normalize_control_mode(control_mode)
        )

    def rollout(
        self,
        initial_positions,
        initial_velocities,
        action_sequence,
        goal_pos=None,
        goal_vel=None,
        steps: Optional[int] = None,
        record_interval: Optional[int] = None,
        ctr_period: Optional[int] = None,
        hidden_state: Optional[torch.Tensor] = None,
        detach_history: bool = False,
        control_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        ctr_period = self.ctr_period if ctr_period is None else int(ctr_period)
        record_interval = self.record_interval if record_interval is None else int(record_interval)
        control_mode = self.control_mode if control_mode is None else self._normalize_control_mode(control_mode)

        initial_positions_tensor = _to_device_tensor(initial_positions, self.device)
        action_tensor = _to_device_tensor(action_sequence, self.device)
        batch_size = self._target_batch_size(initial_positions_tensor, action_tensor)

        positions = self._prepare_rope_state(initial_positions_tensor, batch_size, "initial_positions").clone()
        velocities = self._prepare_rope_state(initial_velocities, batch_size, "initial_velocities").clone()
        if steps is None:
            if action_tensor.dim() == 1:
                control_steps = 1
            elif action_tensor.dim() == 2:
                control_steps = int(action_tensor.shape[0])
            else:
                control_steps = int(action_tensor.shape[1])
            steps = control_steps * ctr_period
        steps = int(steps)
        actions = self._prepare_action(action_tensor, batch_size, steps, ctr_period)

        goal_pos_now = self._prepare_vec3(goal_pos, batch_size, "goal_pos", positions[:, 0, :])
        goal_vel_now = self._prepare_vec3(goal_vel, batch_size, "goal_vel", velocities[:, 0, :])
        hidden = self._initial_hidden(batch_size, hidden_state)

        rope_history = []
        rope_vel_history = []
        goal_history = []
        goal_vel_history = []
        real_history = []
        real_vel_history = []
        acc_history = []

        dt = self.dt
        for step_idx in range(steps):
            action_idx = step_idx // ctr_period
            action = actions[:, action_idx, :]

            franka_acc, hidden = self._franka_acc_step(
                positions[:, 0, :],
                velocities[:, 0, :],
                goal_pos_now,
                goal_vel_now,
                hidden,
            )
            positions, velocities = self.rope.symplectic_euler_step(
                positions,
                velocities,
                franka_acc,
                mode="acc",
            )

            if control_mode == "acc":
                goal_vel_now = goal_vel_now + dt * action
            elif control_mode == "vel":
                goal_vel_now = action
            else:
                raise RuntimeError(f"Unsupported normalized control_mode: {control_mode}")
            goal_pos_now = goal_pos_now + dt * goal_vel_now

            if record_interval > 0 and (step_idx + 1) % record_interval == 0:
                record_values = (
                    positions,
                    velocities,
                    goal_pos_now,
                    goal_vel_now,
                    positions[:, 0, :],
                    velocities[:, 0, :],
                    franka_acc,
                )
                if detach_history:
                    record_values = tuple(value.detach().clone() for value in record_values)
                rope_history.append(record_values[0])
                rope_vel_history.append(record_values[1])
                goal_history.append(record_values[2])
                goal_vel_history.append(record_values[3])
                real_history.append(record_values[4])
                real_vel_history.append(record_values[5])
                acc_history.append(record_values[6])

        def stack_or_empty(history, trailing_shape):
            if history:
                return torch.stack(history, dim=1)
            return torch.empty((batch_size, 0, *trailing_shape), device=self.device, dtype=positions.dtype)

        return {
            "rope_traj": stack_or_empty(rope_history, (self.P, 3)),
            "rope_vel_traj": stack_or_empty(rope_vel_history, (self.P, 3)),
            "goal_traj": stack_or_empty(goal_history, (3,)),
            "goal_vel_traj": stack_or_empty(goal_vel_history, (3,)),
            "real_traj": stack_or_empty(real_history, (3,)),
            "real_vel_traj": stack_or_empty(real_vel_history, (3,)),
            "franka_acc_traj": stack_or_empty(acc_history, (3,)),
            "final_rope_pos": positions,
            "final_rope_vel": velocities,
            "final_goal_pos": goal_pos_now,
            "final_goal_vel": goal_vel_now,
            "final_hidden": hidden,
        }

    def simulate(
        self,
        steps: Optional[int] = None,
        record_interval: Optional[int] = None,
        ctr_period: Optional[int] = None,
        detach_history: bool = False,
        control_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if self._stored_rope_pos is None:
            raise RuntimeError("Call set_state_and_action() before simulate(), or call rollout() directly.")
        if steps is None:
            steps = self.max_record_steps
        return self.rollout(
            self._stored_rope_pos,
            self._stored_rope_vel,
            self._stored_action,
            self._stored_goal_pos,
            self._stored_goal_vel,
            steps=steps,
            record_interval=record_interval,
            ctr_period=ctr_period,
            detach_history=detach_history,
            control_mode=self._stored_control_mode if control_mode is None else control_mode,
        )

    def simulation_for_ctr(
        self,
        batch_pos,
        batch_vel,
        control_sequence,
        goal_pos=None,
        goal_vel=None,
        hidden_state: Optional[torch.Tensor] = None,
        ctr_period: int = 10,
        return_goal_state: bool = False,
        control_mode: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        if control_mode is None and mode is not None:
            control_mode = mode

        batch_pos_tensor = _to_device_tensor(batch_pos, self.device)
        control_tensor = _to_device_tensor(control_sequence, self.device)
        if control_tensor.dim() == 2:
            inferred_batch = batch_pos_tensor.shape[0] if batch_pos_tensor.dim() == 3 else 1
            if control_tensor.shape == (inferred_batch, 3):
                control_tensor = control_tensor.unsqueeze(1)

        out = self.rollout(
            batch_pos_tensor,
            batch_vel,
            control_tensor,
            goal_pos=goal_pos,
            goal_vel=goal_vel,
            steps=int(ctr_period),
            record_interval=int(ctr_period),
            ctr_period=int(ctr_period),
            hidden_state=hidden_state,
            control_mode=control_mode,
        )
        if return_goal_state:
            return (
                out["final_rope_pos"],
                out["final_rope_vel"],
                out["final_goal_pos"],
                out["final_goal_vel"],
                out["final_hidden"],
            )
        return out["final_rope_pos"], out["final_rope_vel"]

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.rollout(*args, **kwargs)


__all__ = ["RopeFranka", "load_franka_model"]
