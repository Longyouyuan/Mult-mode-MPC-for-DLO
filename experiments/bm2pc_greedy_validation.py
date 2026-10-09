"""BM2PC selection-rule validation without modifying the production controller.

The script provides three selectors:
  dynamic: the production Dynamic-Norm Greedy selector;
  fixed-greedy: exact marginal-gain greedy for the paper reference objective;
  fixed-exhaustive: vectorized exhaustive search for the same objective.

All experimental variants live in this file.  The production controller is
imported and its dynamic selector is called directly.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTROLLER = ROOT / "controller"
if str(CONTROLLER) not in sys.path:
    sys.path.insert(0, str(CONTROLLER))

import mujoco  # noqa: E402
import torch  # noqa: E402

from common.rope_warp_4 import N, P, WarpRope  # noqa: E402
from common.utils import (  # noqa: E402
    build_infinite_eight,
    mj_data_to_my_data,
    PositionCommandFilter,
    set_seed,
)
from controller.M2PC import Planner as ProductionPlanner  # noqa: E402
from controller.M2PC import cost_fn  # noqa: E402


EPS = 1.0e-8
RANGE_TOL = 1.0e-12
FILTER_THRESHOLD = 1.0e5
REFERENCE_BETA = 600.0
REFERENCE_WJ = 850.0
REFERENCE_PRIOR_WEIGHT = 1.0
N_MODES = 3


def minmax_norm_np(x: np.ndarray) -> Tuple[np.ndarray, float, float, float, bool]:
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo
    if span < RANGE_TOL:
        return np.zeros_like(x), lo, hi, span, True
    return (x - lo) / (span + EPS), lo, hi, span, False


def pairwise_sq(features: np.ndarray) -> np.ndarray:
    z = np.asarray(features, dtype=np.float64)
    norms = np.sum(z * z, axis=1, keepdims=True)
    d = norms + norms.T - 2.0 * (z @ z.T)
    return np.maximum(d, 0.0)


def fixed_objective(features: np.ndarray, scores: np.ndarray,
                    beta: float, indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0.0
    d = pairwise_sq(features)
    value = float(np.sum(np.asarray(scores, dtype=np.float64)[idx]))
    if idx.size > 1:
        value += float(beta * np.sum(d[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)]))
    return value


def fixed_greedy(features: np.ndarray, scores: np.ndarray, beta: float,
                 n: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    z = np.asarray(features, dtype=np.float64)
    w = np.asarray(scores, dtype=np.float64)
    m = z.shape[0]
    if n < 1 or n > m:
        raise ValueError(f"invalid n={n} for M={m}")
    d = pairwise_sq(z)
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    for step in range(n):
        marginal = w.copy()
        if selected:
            marginal += beta * np.sum(d[:, selected], axis=1)
        if selected:
            marginal[np.asarray(selected, dtype=np.int64)] = -np.inf
        # np.argmax returns the first index, giving deterministic lowest-index ties.
        nxt = int(np.argmax(marginal))
        trace.append({
            "step": step + 1,
            "marginal": marginal.copy(),
            "selected": nxt,
            "raw_div_sum": np.sum(d[:, selected], axis=1) if selected else np.zeros(m),
        })
        selected.append(nxt)
    return np.asarray(selected, dtype=np.int64), trace


_COMBINATION_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, ...]] = {}


def combination_indices(m: int, n: int) -> Tuple[np.ndarray, ...]:
    key = (m, n)
    if key in _COMBINATION_CACHE:
        return _COMBINATION_CACHE[key]
    if n == 3:
        ii: List[np.ndarray] = []
        jj: List[np.ndarray] = []
        kk: List[np.ndarray] = []
        for i in range(m - 2):
            for j in range(i + 1, m - 1):
                k = np.arange(j + 1, m, dtype=np.int32)
                ii.append(np.full(k.size, i, dtype=np.int32))
                jj.append(np.full(k.size, j, dtype=np.int32))
                kk.append(k)
        result = (np.concatenate(ii), np.concatenate(jj), np.concatenate(kk))
    else:
        rows = np.asarray(list(itertools.combinations(range(m), n)), dtype=np.int32)
        result = tuple(rows[:, col] for col in range(n))
    _COMBINATION_CACHE[key] = result
    return result


def fixed_exhaustive(features: np.ndarray, scores: np.ndarray, beta: float,
                     n: int = N_MODES) -> Tuple[np.ndarray, float, float]:
    z = np.asarray(features, dtype=np.float64)
    w = np.asarray(scores, dtype=np.float64)
    m = z.shape[0]
    if n < 1 or n > m:
        raise ValueError(f"invalid n={n} for M={m}")
    combos = combination_indices(m, n)
    d = pairwise_sq(z)
    if n == 1:
        values = w[combos[0]]
    elif n == 2:
        i, j = combos
        values = w[i] + w[j] + beta * d[i, j]
    elif n == 3:
        i, j, k = combos
        values = w[i] + w[j] + w[k] + beta * (d[i, j] + d[i, k] + d[j, k])
    else:
        values = np.empty(len(combos[0]), dtype=np.float64)
        for row in range(values.size):
            idx = np.asarray([c[row] for c in combos], dtype=np.int64)
            values[row] = fixed_objective(z, w, beta, idx)
    best_pos = int(np.argmax(values))
    min_pos = int(np.argmin(values))
    best = np.asarray([c[best_pos] for c in combos], dtype=np.int64)
    worst = float(values[min_pos])
    return best, float(values[best_pos]), worst


def naive_exhaustive(features: np.ndarray, scores: np.ndarray, beta: float,
                     n: int) -> Tuple[Tuple[int, ...], float, Tuple[int, ...], float]:
    best_idx: Tuple[int, ...] | None = None
    worst_idx: Tuple[int, ...] | None = None
    best_value = -np.inf
    worst_value = np.inf
    for idx in itertools.combinations(range(len(scores)), n):
        value = fixed_objective(features, scores, beta, idx)
        if value > best_value:
            best_value, best_idx = value, idx
        if value < worst_value:
            worst_value, worst_idx = value, idx
    assert best_idx is not None and worst_idx is not None
    return best_idx, best_value, worst_idx, worst_value


def certificate(features: np.ndarray, scores: np.ndarray, beta: float,
                selected: Sequence[int]) -> Dict[str, Any]:
    z = np.asarray(features, dtype=np.float64)
    w = np.asarray(scores, dtype=np.float64)
    g = np.asarray(selected, dtype=np.int64)
    n = int(g.size)
    if n < 1 or len(np.unique(g)) != n or np.any(g < 0) or np.any(g >= len(w)):
        raise ValueError("selected indices must be unique and in range")
    mu = np.sum(z[g], axis=0)
    h = w + beta * (n * np.sum(z * z, axis=1) - 2.0 * (z @ mu))
    order = np.lexsort((np.arange(len(h)), -h))
    top = np.sort(order[:n])
    f_direct = fixed_objective(z, w, beta, g)
    centered = z - np.mean(z[g], axis=0, keepdims=True)
    a = w + beta * n * np.sum(centered * centered, axis=1)
    f_centered = float(np.sum(a[g]))
    delta = float(np.sum(h[top]) - np.sum(h[g]))
    delta_centered = float(np.sum(a[np.sort(np.lexsort((np.arange(len(a)), -a))[:n])]) - np.sum(a[g]))
    return {
        "F_selected": f_direct,
        "F_selected_centered": f_centered,
        "certificate": delta,
        "certificate_centered": delta_centered,
        "H_indices": top.tolist(),
        "auxiliary_scores": h.tolist(),
        "residual_F": f_direct - f_centered,
        "residual_delta": delta - delta_centered,
    }


def dynamic_trace(cost: np.ndarray, features: np.ndarray, log_prior: np.ndarray,
                  beta: float, wj: float, m: int, selected: Sequence[int]) -> Dict[str, Any]:
    valid_n = len(cost)
    c_norm, cmin, cmax, crange, cdeg = minmax_norm_np(cost)
    lp = np.asarray(log_prior, dtype=np.float64)
    feat = np.asarray(features, dtype=np.float64).reshape(valid_n, -1)
    trace: List[Dict[str, Any]] = []
    first_score = -wj * c_norm + lp
    first = int(selected[0])
    trace.append({
        "step": 1, "selected": first, "raw_div_sum": np.zeros(valid_n),
        "div_norm": np.zeros(valid_n), "div_min": 0.0, "div_max": 0.0,
        "div_range": 0.0, "cost_norm": c_norm.copy(),
        "score": first_score.copy(), "cost_component": (-wj * c_norm).copy(),
        "div_component": np.zeros(valid_n), "prior_component": lp.copy(),
        "cost_min": cmin, "cost_max": cmax, "cost_range": crange,
        "cost_degenerate": cdeg,
    })
    div_sum = np.sum((feat - feat[first]) ** 2, axis=1)
    chosen = [first]
    for step in range(2, m + 1):
        d_norm, dmin, dmax, drange, ddeg = minmax_norm_np(div_sum)
        score = -wj * c_norm + beta * d_norm + lp
        score[np.asarray(chosen, dtype=np.int64)] = -np.inf
        nxt = int(selected[step - 1])
        trace.append({
            "step": step, "selected": nxt, "raw_div_sum": div_sum.copy(),
            "div_norm": d_norm.copy(), "div_min": dmin, "div_max": dmax,
            "div_range": drange, "score": score.copy(),
            "cost_norm": c_norm.copy(), "cost_component": (-wj * c_norm).copy(),
            "div_component": (beta * d_norm).copy(), "prior_component": lp.copy(),
            "cost_min": cmin, "cost_max": cmax, "cost_range": crange,
            "cost_degenerate": cdeg, "div_degenerate": ddeg,
            "alpha_cost": None if cdeg else wj / (crange + EPS),
            "alpha_div": None if ddeg else beta / (drange + EPS),
            "relative_alpha": None if cdeg or ddeg else (beta * (crange + EPS)) / (wj * (drange + EPS)),
        })
        chosen.append(nxt)
        div_sum += np.sum((feat - feat[nxt]) ** 2, axis=1)
    return {
        "valid_n": valid_n, "cost": np.asarray(cost), "log_prior": lp,
        "features": np.asarray(features), "selected": list(map(int, selected)),
        "trace": trace,
    }


def _filter_pool(cost_g_org: torch.Tensor, tip_traj_g_org: torch.Tensor,
                 log_prior_g_org: torch.Tensor | None, thr: float, m: int):
    valid_n = int((cost_g_org < thr).sum().item())
    if valid_n == 0 or valid_n <= m:
        return valid_n, cost_g_org, tip_traj_g_org, log_prior_g_org
    lp = log_prior_g_org[:valid_n] if log_prior_g_org is not None else None
    return valid_n, cost_g_org[:valid_n], tip_traj_g_org[:valid_n], lp


class DynamicTracePlanner(ProductionPlanner):
    """Production selector with detached score tracing."""

    def __init__(self, *args, trace_sink=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=FILTER_THRESHOLD):
        valid_n, cost_g, tip_g, lp_g = _filter_pool(cost_g_org, tip_traj_g_org, log_prior_g_org, thr, self.m)
        result = super()._greedy_select(cost_g_org, tip_traj_g_org, log_prior_g_org, thr)
        if valid_n > self.m:
            selected = result.detach().cpu().numpy().astype(np.int64)
            lp = np.zeros(valid_n, dtype=np.float64) if lp_g is None else lp_g.detach().cpu().numpy().astype(np.float64)
            record = dynamic_trace(
                cost_g.detach().cpu().numpy().astype(np.float64),
                tip_g.detach().cpu().numpy().astype(np.float64), lp,
                float(self.beta), float(self.wJ), self.m, selected,
            )
            if self.trace_sink is not None:
                self.trace_sink(record)
        return result


class FixedGreedyPlanner(ProductionPlanner):
    """Planner whose selection stage directly uses fixed-F marginal gains."""

    def __init__(self, *args, trace_sink=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=FILTER_THRESHOLD):
        valid_n, cost_g, tip_g, lp_g = _filter_pool(cost_g_org, tip_traj_g_org, log_prior_g_org, thr, self.m)
        if valid_n == 0:
            warnings.warn("Fixed-F Greedy fallback: no valid candidates", UserWarning)
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        if valid_n <= self.m:
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        cost = cost_g.detach().cpu().numpy().astype(np.float64)
        feat = tip_g.detach().cpu().numpy().astype(np.float64).reshape(valid_n, -1)
        lp = np.zeros(valid_n, dtype=np.float64) if lp_g is None else lp_g.detach().cpu().numpy().astype(np.float64)
        scores = -REFERENCE_WJ * cost + REFERENCE_PRIOR_WEIGHT * lp
        selected, trace = fixed_greedy(feat, scores, REFERENCE_BETA, self.m)
        if self.trace_sink is not None:
            self.trace_sink({"valid_n": valid_n, "cost": cost, "features": feat,
                             "log_prior": lp, "scores": scores,
                             "selected": selected.tolist(), "fixed_trace": trace})
        return torch.as_tensor(selected, device=self.device, dtype=torch.long)


class FixedExhaustivePlanner(ProductionPlanner):
    """Planner whose selection stage directly uses fixed-F exhaustive search."""

    def __init__(self, *args, trace_sink=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=FILTER_THRESHOLD):
        valid_n, cost_g, tip_g, lp_g = _filter_pool(cost_g_org, tip_traj_g_org, log_prior_g_org, thr, self.m)
        if valid_n == 0:
            warnings.warn("Fixed-F Exhaustive fallback: no valid candidates", UserWarning)
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        if valid_n <= self.m:
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        cost = cost_g.detach().cpu().numpy().astype(np.float64)
        feat = tip_g.detach().cpu().numpy().astype(np.float64).reshape(valid_n, -1)
        lp = np.zeros(valid_n, dtype=np.float64) if lp_g is None else lp_g.detach().cpu().numpy().astype(np.float64)
        scores = -REFERENCE_WJ * cost + REFERENCE_PRIOR_WEIGHT * lp
        selected, optimum, minimum = fixed_exhaustive(feat, scores, REFERENCE_BETA, self.m)
        selected = np.sort(selected.astype(np.int64))
        if self.trace_sink is not None:
            self.trace_sink({"valid_n": valid_n, "cost": cost, "features": feat,
                             "log_prior": lp, "scores": scores,
                             "selected": selected.tolist(), "F_opt": optimum,
                             "F_min": minimum})
        return torch.as_tensor(selected, device=self.device, dtype=torch.long)


@dataclass
class ExperimentConfig:
    dt: float = 0.001
    ctr_period: int = 25
    horizon: int = 35
    n_sample: int = 402
    m_modes: int = 3
    n_improve: int = 1
    noise_scale: float = 1.5
    top_k_good: int = 200
    beta: float = 600.0
    wj: float = 850.0
    prior_weight: float = 1.0
    task_duration: float = 5.0
    filtering_threshold: float = FILTER_THRESHOLD
    action_low: float = -5.0
    action_high: float = 5.0


def make_planner(kind: str, rope: WarpRope, cfg: ExperimentConfig,
                 device: torch.device, trace_sink=None):
    cls = {"dynamic": DynamicTracePlanner,
           "fixed-greedy": FixedGreedyPlanner,
           "fixed-exhaustive": FixedExhaustivePlanner}[kind]
    return cls(
        rope, cost_fn, cfg.dt, cfg.ctr_period, cfg.horizon, cfg.n_sample,
        cfg.n_improve, cfg.noise_scale, 3,
        limits=torch.tensor([cfg.action_low, cfg.action_high]),
        device=device, mode="acc", m_modes=cfg.m_modes,
        top_k_good=cfg.top_k_good, beta=cfg.beta, wJ=cfg.wj,
        standard_m2pc=True, prior_weight=cfg.prior_weight,
        trace_sink=trace_sink,
    )


def prepare_simulation(cfg: ExperimentConfig, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "Mujoco_env" / "cable_show.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 4)
    model.opt.timestep = cfg.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    cable_body_indices = [i for i, name in enumerate(body_names) if name and name.startswith("B_")]
    cyl1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_1")
    cyl2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_2")
    model.body_pos[cyl1] = np.array([0.433, 0.26, -1.2])
    model.body_pos[cyl2] = np.array([0.0, 0.9, 0.2])
    mujoco.mj_forward(model, data)
    geom_ids = [gid for gid in range(model.ngeom)
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
                and model.geom_bodyid[gid] in (cyl1, cyl2)]
    radius = float(model.geom_size[geom_ids[0]][0])
    half_h = float(model.geom_size[geom_ids[0]][1])
    obs = np.array([[radius, half_h, 0.0], data.xpos[cyl1].copy(), data.xpos[cyl2].copy()])
    rope = WarpRope(
        batch_size=cfg.m_modes + cfg.n_sample, L=1.0,
        mass=0.0025 * 40 / N, k=10000 * 0.46, damping=0.2,
        bending_k=0.0, bending_damping=0.000401, air_drag=0.2206 / 1000,
        g=10.07, dt=cfg.dt, max_record_steps=cfg.horizon * cfg.ctr_period,
        record_interval=cfg.ctr_period, ctr_period=cfg.ctr_period, mode="acc",
    )
    total_steps = int(cfg.task_duration / cfg.dt)
    total_horizon = int(total_steps / cfg.ctr_period)
    goal_builder = build_infinite_eight(
        device=device, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2,
        a=-0.10, N_warm=int(total_horizon * 0.3), N_steady=total_horizon,
        du_start_ratio=0.15, warm_power=2.5,
    )
    return model, data, rope, obs, goal_builder, device, cable_body_indices, cyl1, cyl2


def run_episode(kind: str, seed: int, cfg: ExperimentConfig,
                output_dir: Path | None = None) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    prepared = prepare_simulation(cfg, seed)
    model, data, rope, obs, goal_builder, device, cable_body_indices, cyl1, cyl2 = prepared
    planner = make_planner(kind, rope, cfg, device, trace_sink=traces.append)
    pos = torch.zeros((1, P, 3), device=device)
    pos[:, :, 2] = torch.linspace(1.2, 0.2, steps=P, device=device)
    vel = torch.zeros((1, P, 3), device=device)
    cable_body_set = set(cable_body_indices)
    rope_geom_ids = {gid for gid in range(model.ngeom)
                     if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE
                     and model.geom_bodyid[gid] in cable_body_set}
    cyl_geom_ids = {gid for gid in range(model.ngeom)
                    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and model.geom_bodyid[gid] in (cyl1, cyl2)}
    total_steps = int(cfg.task_duration / cfg.dt)
    total_horizon = int(total_steps / cfg.ctr_period)
    mj_node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + mj_node * 6 + 3))
    actions: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    goals: List[np.ndarray] = []
    profile: List[Dict[str, float]] = []
    collisions = 0
    prev_hit = False
    wall_times: List[float] = []
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    pcf = PositionCommandFilter(
        np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], cfg.dt
    )
    for step in range(total_horizon * 2 + 40):
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + mj_node * 3] = data.xpos[1:1 + mj_node][::-1].reshape(-1)
        mj_state[0, 1 + mj_node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)
        goal = goal_builder.get_range(start=step + 1, length=cfg.horizon)
        obs[1, :] = data.xpos[cyl1].copy()
        obs[2, :] = data.xpos[cyl2].copy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal, Obs=obs)
        action = planner.get_action(rule="greedy").detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        if planner.last_profile is not None:
            profile.append(dict(planner.last_profile))
        for _ in range(cfg.ctr_period):
            target = pcf.input_acceleration(np.asarray(action, dtype=np.float64))
            data.ctrl[:3] = target
            model.site_pos[0][[0, 1, 2]] = target + np.array([0.0, 0.0, 1.2])
            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0.0, 0.5 * 9.81])
            mujoco.mj_step(model, data)
        hit = False
        for ci in range(data.ncon):
            g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
            if ((g1 in rope_geom_ids and g2 in cyl_geom_ids) or
                    (g2 in rope_geom_ids and g1 in cyl_geom_ids)):
                hit = True
                break
        if hit and not prev_hit:
            collisions += 1
        prev_hit = hit
        actions.append(action.copy())
        positions.append(data.site_xpos[-1].copy())
        goals.append((target + np.array([0.0, 0.0, 1.2])).copy())
        planner.update_policy()
    pos_arr = np.asarray(positions)
    goal_arr = np.asarray(goals)
    errors = np.linalg.norm(pos_arr - goal_arr, axis=1)
    metrics = {
        "controller": kind, "seed": seed, "steps": len(actions),
        "tracking_rmse": float(np.sqrt(np.mean(errors * errors))),
        "tracking_mean": float(np.mean(errors)),
        "tracking_max": float(np.max(errors)),
        "collision_count": collisions,
        "wall_mean_ms": float(np.mean(wall_times) * 1000.0),
        "wall_median_ms": float(np.median(wall_times) * 1000.0),
        "wall_p95_ms": float(np.percentile(wall_times, 95) * 1000.0),
        "deadline_miss_rate": float(np.mean(np.asarray(wall_times) > 0.025)),
        "fallback_count": int(len(actions) - len(traces)),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / f"trajectory_{kind}_seed{seed}.npz",
                            actions=np.asarray(actions), positions=pos_arr, goals=goal_arr,
                            wall_times=np.asarray(wall_times))
        with (output_dir / f"traces_{kind}_seed{seed}.pkl").open("wb") as fh:
            pickle.dump(traces, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with (output_dir / f"profile_{kind}_seed{seed}.json").open("w", encoding="utf-8") as fh:
            json.dump(profile, fh)
        (output_dir / f"metrics_{kind}_seed{seed}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return {"metrics": metrics, "traces": traces}


def test_math(seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    failures: List[str] = []
    for case in range(100):
        m = int(rng.integers(3, 13))
        n = int(rng.integers(1, min(3, m) + 1))
        z = rng.normal(size=(m, 5))
        if case % 10 == 0:
            z[1] = z[0]
        w = rng.normal(size=m)
        beta = float(rng.uniform(0.0, 10.0))
        exact, best, worst = fixed_exhaustive(z, w, beta, n)
        naive_best, naive_best_v, naive_worst, naive_worst_v = naive_exhaustive(z, w, beta, n)
        if not np.allclose(best, naive_best_v, rtol=1e-10, atol=1e-10) or not np.allclose(worst, naive_worst_v, rtol=1e-10, atol=1e-10):
            failures.append(f"exhaustive case {case}")
        greedy, trace = fixed_greedy(z, w, beta, n)
        for step, row in enumerate(trace):
            selected = greedy[:step]
            for candidate in range(m):
                if candidate in selected:
                    continue
                lhs = fixed_objective(z, w, beta, list(selected) + [candidate])
                base = fixed_objective(z, w, beta, selected)
                if not np.isclose(row["marginal"][candidate], lhs - base, rtol=1e-9, atol=1e-9):
                    failures.append(f"marginal case {case} step {step}")
                    break
        for selected in (greedy, exact, np.asarray(naive_best)):
            cert = certificate(z, w, beta, selected)
            gap = best - cert["F_selected"]
            if gap < -1e-8 or gap > cert["certificate"] + 1e-7:
                failures.append(f"certificate case {case}")
    # Translation invariance and constant score shifts.
    z = rng.normal(size=(8, 4)); w = rng.normal(size=8); beta = 2.0
    idx = np.array([0, 2, 5])
    if not np.isclose(fixed_objective(z, w, beta, idx), fixed_objective(z + 17.0, w, beta, idx), atol=1e-7):
        failures.append("feature translation")
    if not np.isclose((fixed_objective(z, w + 3.0, beta, idx) - fixed_objective(z, w, beta, idx)), 9.0, atol=1e-7):
        failures.append("score shift")
    result = {"passed": not failures, "failures": failures, "cases": 100}
    if failures:
        raise AssertionError(json.dumps(result))
    return result


def read_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rb") as fh:
        yield from pickle.load(fh)


def analyze_traces(run_dir: Path, seed: int, controller: str) -> List[Dict[str, Any]]:
    source = run_dir / f"traces_{controller}_seed{seed}.pkl"
    rows: List[Dict[str, Any]] = []
    if not source.exists():
        return rows
    for step, record in enumerate(read_traces(source)):
        if record.get("valid_n", 0) <= N_MODES:
            continue
        features = np.asarray(record["features"], dtype=np.float64).reshape(record["valid_n"], -1)
        scores = -REFERENCE_WJ * np.asarray(record["cost"], dtype=np.float64) + REFERENCE_PRIOR_WEIGHT * np.asarray(record["log_prior"], dtype=np.float64)
        gd = np.asarray(record["selected"], dtype=np.int64)
        gf, _ = fixed_greedy(features, scores, REFERENCE_BETA, N_MODES)
        exact, fopt, fmin = fixed_exhaustive(features, scores, REFERENCE_BETA, N_MODES)
        cd = certificate(features, scores, REFERENCE_BETA, gd)
        cf = certificate(features, scores, REFERENCE_BETA, gf)
        fd = cd["F_selected"]
        ff = cf["F_selected"]
        gd_gap = fopt - fd
        gf_gap = fopt - ff
        rows.append({
            "seed": seed, "step": step, "M": record["valid_n"],
            "F_dynamic": fd, "F_fixed_greedy": ff, "F_exact": fopt, "F_min": fmin,
            "gap_dynamic": gd_gap, "gap_fixed_greedy": gf_gap,
            "normalization_effect": ff - fd,
            "bound_dynamic": cd["certificate"], "bound_fixed_greedy": cf["certificate"],
            "slack_dynamic": cd["certificate"] - gd_gap,
            "slack_fixed_greedy": cf["certificate"] - gf_gap,
            "dynamic_bound_ok": int(gd_gap >= -1e-7 and gd_gap <= cd["certificate"] + 1e-7),
            "fixed_greedy_bound_ok": int(gf_gap >= -1e-7 and gf_gap <= cf["certificate"] + 1e-7),
            "exact_dynamic": int(np.isclose(fd, fopt, rtol=1e-8, atol=1e-7)),
            "exact_fixed_greedy": int(np.isclose(ff, fopt, rtol=1e-8, atol=1e-7)),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(run_dir: Path, metrics: List[Dict[str, Any]], offline: List[Dict[str, Any]], smoke: bool) -> None:
    controller_summary: Dict[str, Dict[str, float]] = {}
    for controller in sorted({str(row.get("controller")) for row in metrics}):
        subset = [row for row in metrics if str(row.get("controller")) == controller]
        def vals(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in subset], dtype=np.float64)
        controller_summary[controller] = {
            "episodes": float(len(subset)),
            "rmse_mean": float(np.mean(vals("tracking_rmse"))),
            "rmse_std": float(np.std(vals("tracking_rmse"))),
            "collision_free_rate": float(np.mean(vals("collision_count") == 0)),
            "planning_mean_ms": float(np.mean(vals("wall_mean_ms"))),
            "planning_p95_mean_ms": float(np.mean(vals("wall_p95_ms"))),
            "deadline_miss_rate_mean": float(np.mean(vals("deadline_miss_rate"))),
            "fallback_steps_total": float(np.sum(vals("fallback_count"))),
        }
    offline_summary: Dict[str, float] = {}
    if offline:
        def ovals(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in offline], dtype=np.float64)
        offline_summary = {
            "snapshots": float(len(offline)),
            "dynamic_gap_mean": float(np.mean(ovals("gap_dynamic"))),
            "dynamic_gap_p95": float(np.percentile(ovals("gap_dynamic"), 95)),
            "dynamic_gap_max": float(np.max(ovals("gap_dynamic"))),
            "fixed_greedy_gap_mean": float(np.mean(ovals("gap_fixed_greedy"))),
            "fixed_greedy_gap_p95": float(np.percentile(ovals("gap_fixed_greedy"), 95)),
            "fixed_greedy_gap_max": float(np.max(ovals("gap_fixed_greedy"))),
            "normalization_effect_mean": float(np.mean(ovals("normalization_effect"))),
            "normalization_effect_min": float(np.min(ovals("normalization_effect"))),
            "normalization_effect_max": float(np.max(ovals("normalization_effect"))),
            "dynamic_bound_violations": float(np.sum(ovals("dynamic_bound_ok") < 0.5)),
            "fixed_greedy_bound_violations": float(np.sum(ovals("fixed_greedy_bound_ok") < 0.5)),
            "fixed_greedy_exact_frequency": float(np.mean(ovals("exact_fixed_greedy"))),
        }
    lines = [
        "# BM2PC Greedy Validation Report", "",
        f"Smoke run: `{smoke}`", "",
        "## Objective fidelity", "",
        "Dynamic-Norm Greedy uses the production cost min-max normalization and per-step dynamic diversity normalization. Fixed-F Greedy and Fixed-F Exhaustive use the fixed reference objective with raw rollout costs, log prior, raw squared Euclidean features, and beta=600.", "",
        "## Offline comparison", "",
        "`gap_fixed_greedy` is the fixed-F greedy approximation gap. `normalization_effect = F_fixed_greedy - F_dynamic` combines cost normalization and dynamic diversity normalization effects.", "",
        "## Closed-loop comparison", "",
        "Closed-loop values compare three selection rules under matched initial seeds. Later candidate pools can diverge because selected warm starts and states diverge.", "",
        "## Certificate", "",
        "The certificate is a finite filtered-pool bound for each returned set. It is not an approximation-ratio or stability guarantee.", "",
        "## Aggregate results", "", "```json", json.dumps({"controllers": controller_summary, "offline": offline_summary}, indent=2), "```", "",
    ]
    if metrics:
        lines += ["## Closed-loop metrics", "", "```json", json.dumps(metrics, indent=2), "```", ""]
    if offline:
        lines += ["## Offline summary", "", "```json", json.dumps(offline[:10], indent=2), "```", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(run_dir: Path, metrics: List[Dict[str, Any]], offline: List[Dict[str, Any]]) -> None:
    """Generate compact, reproducible paper-facing figures and proposition text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        dyn = np.sort(np.asarray([float(row["gap_dynamic"]) for row in offline]))
        fix = np.sort(np.asarray([float(row["gap_fixed_greedy"]) for row in offline]))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for values, label, color in ((dyn, "Dynamic-Norm", "tab:blue"), (fix, "Fixed-F Greedy", "tab:orange")):
            y = np.linspace(1.0 / len(values), 1.0, len(values))
            ax.plot(values, y, label=label, color=color)
        ax.set_xlabel("Fixed-F objective gap")
        ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "objective_gap_cdf.pdf")
        fig.savefig(fig_dir / "objective_gap_cdf.png", dpi=180)
        plt.close(fig)
    if metrics:
        methods = sorted({str(row["controller"]) for row in metrics})
        mean_rmse = [np.mean([float(row["tracking_rmse"]) for row in metrics if row["controller"] == method]) for method in methods]
        mean_time = [np.mean([float(row["wall_mean_ms"]) for row in metrics if row["controller"] == method]) for method in methods]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        axes[0].bar(methods, mean_rmse, color=["tab:blue", "tab:orange", "tab:green"][:len(methods)])
        axes[0].set_ylabel("Tracking RMSE")
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].bar(methods, mean_time, color=["tab:blue", "tab:orange", "tab:green"][:len(methods)])
        axes[1].axhline(25.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Mean planning time (ms)")
        axes[1].tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / "closed_loop_summary.pdf")
        fig.savefig(fig_dir / "closed_loop_summary.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for method, color in zip(methods, ["tab:blue", "tab:orange", "tab:green"]):
            path = run_dir / f"trajectory_{method}_seed0.npz"
            if path.exists():
                traj = np.load(path)["positions"]
                ax.plot(traj[:, 0], traj[:, 1], label=method, color=color)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "seed0_xy_trajectories.pdf")
        fig.savefig(fig_dir / "seed0_xy_trajectories.png", dpi=180)
        plt.close(fig)
    tex = r"""% Auto-generated from bm2pc_greedy_validation.py
\begin{proposition}[A posteriori fixed-pool certificate]
Let $G$ be any selected set of size $n$ and let
$F(S)=\sum_{i\in S}w_i+\beta\sum_{i<j}\|z_i-z_j\|_2^2$ with $\beta\ge 0$.
Define $\mu_G=\sum_{i\in G}z_i$ and
$h_i=w_i+\beta(n\|z_i\|_2^2-2z_i^\top\mu_G)$.
If $H$ contains the $n$ largest $h_i$, then
$0\le F(S^\star)-F(G)\le\Delta_G$ where
$\Delta_G=\sum_{i\in H}h_i-\sum_{i\in G}h_i$.
\end{proposition}

The proposition evaluates the returned set relative to the fixed reference
objective. It does not assert that the production Dynamic-Norm selector
maximizes this objective at each marginal step.
"""
    (run_dir / "theory_and_paper_text.tex").write_text(tex, encoding="utf-8")


def write_audit_and_timing(run_dir: Path, metrics: List[Dict[str, Any]]) -> None:
    audit_rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("traces_dynamic_seed*.pkl")):
        for record in read_traces(path):
            for row in record.get("trace", []):
                if row.get("alpha_cost") is not None and row.get("alpha_div") is not None:
                    audit_rows.append(row)
    def stat(key: str) -> Dict[str, float]:
        values = np.asarray([float(row[key]) for row in audit_rows if row.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            return {"count": 0.0}
        return {"count": float(values.size), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)), "max": float(np.max(values))}
    audit_text = "\n".join([
        "# Objective audit",
        "",
        "Production Dynamic-Norm score:",
        "`-850 * norm(cost) + 600 * norm(cumulative_squared_distance) + log_prior`.",
        "",
        f"Normalization uses epsilon={EPS:g} in the denominator and returns zeros when range < {RANGE_TOL:g}.",
        "The fixed reference objective uses raw rollout cost and raw squared Euclidean features.",
        "The difference between Fixed-F Greedy and Dynamic-Norm is therefore a combined cost-and-diversity normalization effect.",
        "",
        "## Effective raw slopes",
        "",
        "```json",
        json.dumps({"alpha_cost": stat("alpha_cost"), "alpha_diversity": stat("alpha_div"), "relative_alpha": stat("relative_alpha")}, indent=2),
        "```",
        "",
        "The relative slope is `600 * (cost_range + epsilon) / (850 * (diversity_range + epsilon))` for nondegenerate ranges.",
    ])
    (run_dir / "objective_audit.md").write_text(audit_text, encoding="utf-8")
    timing_rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("profile_*_seed*.json")):
        stem = path.stem.replace("profile_", "")
        controller, seed_text = stem.rsplit("_seed", 1)
        profile = json.loads(path.read_text(encoding="utf-8"))
        for stage in ("improve", "rollout", "cost", "multimodal", "overhead"):
            values = np.asarray([float(row[stage]) * 1000.0 for row in profile if stage in row], dtype=np.float64)
            if values.size:
                timing_rows.append({"controller": controller, "seed": int(seed_text), "stage": stage,
                                    "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
                                    "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values))})
    write_csv(run_dir / "timing_metrics.csv", timing_rows)


def run_experiment(run_dir: Path, seeds: Sequence[int], smoke: bool = False) -> None:
    cfg = ExperimentConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg.__dict__, "seeds": list(seeds)}, indent=2), encoding="utf-8")
    all_metrics: List[Dict[str, Any]] = []
    offline: List[Dict[str, Any]] = []
    controllers = ["dynamic", "fixed-greedy", "fixed-exhaustive"]
    for seed in seeds:
        for controller in controllers:
            print(f"RUN controller={controller} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "episode",
                 "--controller", controller, "--seed", str(seed),
                 "--run-dir", str(run_dir)],
                check=True,
            )
            metrics_path = run_dir / f"metrics_{controller}_seed{seed}.json"
            all_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        rows = analyze_traces(run_dir, seed, "dynamic")
        offline.extend(rows)
        write_csv(run_dir / "per_snapshot_metrics.csv", offline)
        write_csv(run_dir / "closed_loop_metrics.csv", all_metrics)
        print(f"DONE seed={seed} offline_snapshots={len(rows)}", flush=True)
    write_report(run_dir, all_metrics, offline, smoke)
    manifest = {"seeds": list(seeds), "controllers": controllers,
                "production_files_unchanged": True, "smoke": smoke,
                "offline_rows": len(offline), "metrics_rows": len(all_metrics)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "smoke", "full", "episode", "collect", "analyze", "benchmark", "report"])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--controller", choices=["dynamic", "fixed-greedy", "fixed-exhaustive"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run_dir or (ROOT / "results" / "bm2pc_greedy_validation" / time.strftime("%Y%m%d_%H%M%S"))
    if args.command == "episode":
        if args.controller is None:
            raise ValueError("--controller is required for episode")
        run_episode(args.controller, args.seed, ExperimentConfig(), run_dir)
        return
    if args.command == "test":
        print(json.dumps(test_math(), indent=2))
        return
    if args.command == "smoke":
        test_math()
        run_experiment(run_dir, [0], smoke=True)
        return
    if args.command == "full":
        test_math()
        run_experiment(run_dir, [0], smoke=True)
        run_experiment(run_dir, list(range(10)), smoke=False)
        return
    if args.command == "collect":
        run_experiment(run_dir, args.seeds or [0], smoke=False)
        return
    if args.command == "analyze":
        rows = []
        for seed in args.seeds or list(range(10)):
            rows.extend(analyze_traces(run_dir, seed, "dynamic"))
        write_csv(run_dir / "per_snapshot_metrics.csv", rows)
        return
    if args.command == "report":
        metrics = []
        path = run_dir / "closed_loop_metrics.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                metrics = list(csv.DictReader(fh))
        else:
            metrics = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted(run_dir.glob("metrics_*_seed*.json"))]
            if metrics:
                write_csv(run_dir / "closed_loop_metrics.csv", metrics)
        offline = []
        path = run_dir / "per_snapshot_metrics.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                offline = list(csv.DictReader(fh))
        write_report(run_dir, metrics, offline, False)
        generate_artifacts(run_dir, metrics, offline)
        write_audit_and_timing(run_dir, metrics)
        manifest = {
            "git_commit": None,
            "controllers": ["dynamic", "fixed-greedy", "fixed-exhaustive"],
            "seeds": sorted({int(row["seed"]) for row in metrics}) if metrics else [],
            "episodes": len(metrics),
            "offline_snapshots": len(offline),
            "production_sources_modified": False,
            "python": sys.executable,
        }
        try:
            import subprocess as _sp
            manifest["git_commit"] = _sp.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            pass
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return
    if args.command == "benchmark":
        print("Benchmark data are produced during each controller episode in closed_loop_metrics.csv.")
        return


if __name__ == "__main__":
    main()
