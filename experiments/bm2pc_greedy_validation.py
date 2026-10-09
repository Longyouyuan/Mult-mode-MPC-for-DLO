"""BM2PC selection-rule validation without modifying the production controller.

The script compares only two selectors:
  dynamic: the production Dynamic-Norm Greedy selector;
  pool-norm-exhaustive: a fixed pool-normalized exhaustive reference.

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


def pool_norm_terms(cost: np.ndarray, features: np.ndarray,
                    log_prior: np.ndarray, wj: float = REFERENCE_WJ,
                    beta: float = REFERENCE_BETA,
                    prior_weight: float = REFERENCE_PRIOR_WEIGHT) -> Dict[str, Any]:
    """Construct one fixed, pool-wise normalized objective."""
    j = np.asarray(cost, dtype=np.float64).reshape(-1)
    z = np.asarray(features, dtype=np.float64).reshape(j.size, -1)
    lp = np.asarray(log_prior, dtype=np.float64).reshape(-1)
    if lp.size != j.size or z.shape[0] != j.size:
        raise ValueError("cost, features and log_prior must share the pool size")
    jmin = float(np.min(j))
    jmax = float(np.max(j))
    jrange = jmax - jmin
    cost_degenerate = bool(jrange < RANGE_TOL)
    jnorm = np.zeros_like(j) if cost_degenerate else (j - jmin) / (jrange + EPS)
    d = pairwise_sq(z)
    if j.size > 1:
        tri = np.triu_indices(j.size, 1)
        diversity_scale = float(np.max(d[tri]))
    else:
        diversity_scale = 0.0
    diversity_degenerate = bool(diversity_scale < RANGE_TOL)
    beta_eff = 0.0 if diversity_degenerate else float(beta / (diversity_scale + EPS))
    w = -float(wj) * jnorm + float(prior_weight) * lp
    return {
        "cost": j, "features": z, "log_prior": lp, "cost_norm": jnorm,
        "cost_min": jmin, "cost_max": jmax, "cost_range": jrange,
        "cost_degenerate": cost_degenerate, "pairwise_sq": d,
        "diversity_scale": diversity_scale,
        "diversity_degenerate": diversity_degenerate, "beta_eff": beta_eff,
        "weights": w,
    }


def pool_norm_objective(terms: Dict[str, Any], indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0.0
    d = np.asarray(terms["pairwise_sq"], dtype=np.float64)
    value = float(np.sum(np.asarray(terms["weights"], dtype=np.float64)[idx]))
    if idx.size > 1:
        value += float(terms["beta_eff"] * np.sum(d[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)]))
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


def pool_norm_exhaustive(cost: np.ndarray, features: np.ndarray,
                         log_prior: np.ndarray, n: int = N_MODES,
                         wj: float = REFERENCE_WJ,
                         beta: float = REFERENCE_BETA,
                         prior_weight: float = REFERENCE_PRIOR_WEIGHT,
                         return_terms: bool = False):
    terms = pool_norm_terms(cost, features, log_prior, wj, beta, prior_weight)
    m = len(terms["weights"])
    if n < 1 or n > m:
        raise ValueError(f"invalid n={n} for M={m}")
    combos = combination_indices(m, n)
    if n == 1:
        values = terms["weights"][combos[0]]
    elif n == 2:
        i, j = combos
        values = terms["weights"][i] + terms["weights"][j] + terms["beta_eff"] * terms["pairwise_sq"][i, j]
    elif n == 3:
        i, j, k = combos
        values = (terms["weights"][i] + terms["weights"][j] + terms["weights"][k]
                  + terms["beta_eff"] * (terms["pairwise_sq"][i, j]
                  + terms["pairwise_sq"][i, k] + terms["pairwise_sq"][j, k]))
    else:
        values = np.asarray([pool_norm_objective(terms, [c[row] for c in combos])
                             for row in range(len(combos[0]))], dtype=np.float64)
    best_pos = int(np.argmax(values))
    min_pos = int(np.argmin(values))
    best = np.sort(np.asarray([c[best_pos] for c in combos], dtype=np.int64))
    worst = np.sort(np.asarray([c[min_pos] for c in combos], dtype=np.int64))
    result = (best, float(values[best_pos]), float(values[min_pos]))
    return result + (terms,) if return_terms else result


def naive_pool_norm_exhaustive(cost: np.ndarray, features: np.ndarray,
                               log_prior: np.ndarray, n: int = N_MODES):
    terms = pool_norm_terms(cost, features, log_prior)
    best_idx, worst_idx = None, None
    best_value, worst_value = -np.inf, np.inf
    for idx in itertools.combinations(range(len(cost)), n):
        value = pool_norm_objective(terms, idx)
        if value > best_value:
            best_value, best_idx = value, idx
        if value < worst_value:
            worst_value, worst_idx = value, idx
    return np.asarray(best_idx, dtype=np.int64), float(best_value), np.asarray(worst_idx, dtype=np.int64), float(worst_value)


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
        self._pending_trace = None

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=FILTER_THRESHOLD):
        valid_n, cost_g, tip_g, lp_g = _filter_pool(cost_g_org, tip_traj_g_org, log_prior_g_org, thr, self.m)
        result = super()._greedy_select(cost_g_org, tip_traj_g_org, log_prior_g_org, thr)
        if valid_n > self.m:
            # Defer all CPU conversion and diagnostic replay until after the
            # production improve_policy timing window.
            self._pending_trace = (cost_g, tip_g, lp_g, result.detach(), valid_n)
        return result

    def flush_trace(self) -> None:
        pending = self._pending_trace
        self._pending_trace = None
        if pending is None or self.trace_sink is None:
            return
        cost_g, tip_g, lp_g, result, valid_n = pending
        selected = result.cpu().numpy().astype(np.int64)
        lp = np.zeros(valid_n, dtype=np.float64) if lp_g is None else lp_g.cpu().numpy().astype(np.float64)
        self.trace_sink(dynamic_trace(
            cost_g.cpu().numpy().astype(np.float64),
            tip_g.cpu().numpy().astype(np.float64), lp,
            float(self.beta), float(self.wJ), self.m, selected,
        ))


class PoolNormExhaustivePlanner(ProductionPlanner):
    """Direct pool-normalized exhaustive selection baseline."""

    def __init__(self, *args, trace_sink=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink

    @torch.no_grad()
    def _greedy_select(self, cost_g_org, tip_traj_g_org, log_prior_g_org=None, thr=FILTER_THRESHOLD):
        valid_n, cost_g, tip_g, lp_g = _filter_pool(cost_g_org, tip_traj_g_org, log_prior_g_org, thr, self.m)
        if valid_n == 0:
            warnings.warn("Pool-Norm Exhaustive fallback: no valid candidates", UserWarning)
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        if valid_n <= self.m:
            return torch.arange(self.m, device=self.device, dtype=torch.long)
        cost = cost_g.detach().cpu().numpy().astype(np.float64)
        feat = tip_g.detach().cpu().numpy().astype(np.float64).reshape(valid_n, -1)
        lp = np.zeros(valid_n, dtype=np.float64) if lp_g is None else lp_g.detach().cpu().numpy().astype(np.float64)
        selected, optimum, minimum, terms = pool_norm_exhaustive(
            cost, feat, lp, self.m, self.wJ, self.beta, self.prior_weight, return_terms=True)
        if self.trace_sink is not None:
            self.trace_sink({"valid_n": valid_n, "diversity_scale": terms["diversity_scale"],
                             "beta_eff": terms["beta_eff"],
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
           "pool-norm-exhaustive": PoolNormExhaustivePlanner}[kind]
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
    attached_positions: List[np.ndarray] = []
    command_goals: List[np.ndarray] = []
    profile: List[Dict[str, float]] = []
    collisions = 0
    collision_substeps = 0
    collision_any = False
    prev_hit = False
    wall_times: List[float] = []
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    rope_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_first")
    attached_end_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_last")
    if rope_tip_site_id < 0 or attached_end_site_id < 0:
        raise RuntimeError("MuJoCo cable endpoint sites S_first/S_last were not found")
    if rope_tip_site_id == attached_end_site_id:
        raise RuntimeError("free rope tip and attached endpoint resolved to the same site")
    pcf = PositionCommandFilter(
        np.array([0.0, 0.0, 0.0]), data.sensordata[[0, 1, 2]], cfg.dt
    )
    for step in range(total_horizon * 2 + 40):
        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + mj_node * 3] = data.xpos[1:1 + mj_node][::-1].reshape(-1)
        mj_state[0, 1 + mj_node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)
        if step == 0:
            state_tip = pos[0, -1].detach().cpu().numpy()
            if not np.allclose(state_tip, data.site_xpos[rope_tip_site_id], rtol=0.0, atol=1.0e-5):
                raise AssertionError("S_first does not match pos[..., -1, :] free-tip state")
        goal = goal_builder.get_range(start=step + 1, length=cfg.horizon)
        tracking_reference = goal[0].detach().cpu().numpy().astype(np.float64, copy=True)
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
        if isinstance(planner, DynamicTracePlanner):
            planner.flush_trace()
        if planner.last_profile is not None:
            profile.append(dict(planner.last_profile))
        for _ in range(cfg.ctr_period):
            target = pcf.input_acceleration(np.asarray(action, dtype=np.float64))
            data.ctrl[:3] = target
            model.site_pos[0][[0, 1, 2]] = target + np.array([0.0, 0.0, 1.2])
            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0.0, 0.5 * 9.81])
            mujoco.mj_step(model, data)
            hit = any(((data.contact[ci].geom1 in rope_geom_ids and data.contact[ci].geom2 in cyl_geom_ids) or
                       (data.contact[ci].geom2 in rope_geom_ids and data.contact[ci].geom1 in cyl_geom_ids))
                      for ci in range(data.ncon))
            collision_substeps += int(hit)
            collision_any = collision_any or hit
            if hit and not prev_hit:
                collisions += 1
            prev_hit = hit
        actions.append(action.copy())
        # S_first is the free rope tip. S_last is constrained to the slider and
        # therefore measures actuator-end following rather than rope-tip tracking.
        positions.append(data.site_xpos[rope_tip_site_id].copy())
        goals.append(tracking_reference)
        attached_positions.append(data.site_xpos[attached_end_site_id].copy())
        command_goals.append((target + np.array([0.0, 0.0, 1.2])).copy())
        planner.update_policy()
    pos_arr = np.asarray(positions)
    goal_arr = np.asarray(goals)
    attached_arr = np.asarray(attached_positions)
    command_arr = np.asarray(command_goals)
    errors = np.linalg.norm(pos_arr - goal_arr, axis=1)
    actuator_errors = np.linalg.norm(attached_arr - command_arr, axis=1)
    metrics = {
        "controller": kind, "seed": seed, "steps": len(actions),
        "tracking_rmse": float(np.sqrt(np.mean(errors * errors))),
        "tracking_mean": float(np.mean(errors)),
        "tracking_max": float(np.max(errors)),
        "actuator_endpoint_rmse": float(np.sqrt(np.mean(actuator_errors * actuator_errors))),
        "actuator_endpoint_mean": float(np.mean(actuator_errors)),
        "actuator_endpoint_max": float(np.max(actuator_errors)),
        "collision_count": collisions,
        "collision_substeps": int(collision_substeps),
        "collision_any": bool(collision_any),
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
                            rope_tip_positions=pos_arr, reference_goals=goal_arr,
                            attached_positions=attached_arr, command_goals=command_arr,
                            wall_times=np.asarray(wall_times))
        with (output_dir / f"traces_{kind}_seed{seed}.pkl").open("wb") as fh:
            pickle.dump(traces, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with (output_dir / f"profile_{kind}_seed{seed}.json").open("w", encoding="utf-8") as fh:
            json.dump(profile, fh)
        (output_dir / f"metrics_{kind}_seed{seed}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return {"metrics": metrics, "traces": traces}


def validate_saved_episode(run_dir: Path, controller: str, seed: int,
                           metrics: Dict[str, Any]) -> None:
    path = run_dir / f"trajectory_{controller}_seed{seed}.npz"
    with np.load(path) as saved:
        tip = saved["rope_tip_positions"]
        reference = saved["reference_goals"]
        attached = saved["attached_positions"]
        command = saved["command_goals"]
        if not np.array_equal(saved["positions"], tip):
            raise AssertionError(f"positions alias mismatch: {path}")
        if not np.array_equal(saved["goals"], reference):
            raise AssertionError(f"goals alias mismatch: {path}")
        tracking_error = np.linalg.norm(tip - reference, axis=1)
        actuator_error = np.linalg.norm(attached - command, axis=1)
        tracking_rmse = float(np.sqrt(np.mean(tracking_error * tracking_error)))
        actuator_rmse = float(np.sqrt(np.mean(actuator_error * actuator_error)))
        if not np.isclose(tracking_rmse, float(metrics["tracking_rmse"]), rtol=1e-12, atol=1e-12):
            raise AssertionError(f"tracking RMSE mismatch: {path}")
        if not np.isclose(actuator_rmse, float(metrics["actuator_endpoint_rmse"]), rtol=1e-12, atol=1e-12):
            raise AssertionError(f"actuator endpoint RMSE mismatch: {path}")
        if np.allclose(tip, attached, rtol=0.0, atol=1e-6):
            raise AssertionError(f"free tip unexpectedly equals attached endpoint: {path}")


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
        cost = rng.normal(size=m)
        lp = w.copy()
        best, fmax, fmin = pool_norm_exhaustive(cost, z, lp, n)
        naive_best, naive_max, naive_worst, naive_min = naive_pool_norm_exhaustive(cost, z, lp, n)
        if not np.array_equal(best, naive_best) or not np.isclose(fmax, naive_max) or not np.isclose(fmin, naive_min):
            failures.append(f"pool exhaustive case {case}")
        terms = pool_norm_terms(cost, z, lp)
        if not np.isclose(fmax, pool_norm_objective(terms, best)):
            failures.append(f"objective case {case}")
        selected = np.asarray(rng.choice(m, size=n, replace=False))
        cert = certificate(z, terms["weights"], terms["beta_eff"], selected)
        gap = fmax - cert["F_selected"]
        if gap < -1e-7 or gap > cert["certificate"] + 1e-6:
            failures.append(f"certificate case {case}")
    # Degenerate scales must remain finite and deterministic.
    for cost, z in ((np.ones(5), rng.normal(size=(5, 4))),
                    (rng.normal(size=5), np.zeros((5, 4))),
                    (np.ones(5), np.zeros((5, 4)))):
        terms = pool_norm_terms(cost, z, np.zeros(5))
        pool_norm_exhaustive(cost, z, np.zeros(5), 3)
        if terms["cost_norm"].shape != (5,) or not np.all(np.isfinite(terms["weights"])):
            failures.append("degenerate scales")
    # Translation invariance and constant score shifts.
    z = rng.normal(size=(8, 4)); w = rng.normal(size=8); beta = 2.0
    idx = np.array([0, 2, 5])
    if not np.isclose(fixed_objective(z, w, beta, idx), fixed_objective(z + 17.0, w, beta, idx), atol=1e-7):
        failures.append("feature translation")
    if not np.isclose((fixed_objective(z, w + 3.0, beta, idx) - fixed_objective(z, w, beta, idx)), 9.0, atol=1e-7):
        failures.append("score shift")
    result = {"passed": not failures, "failures": failures, "cases": 103}
    if failures:
        raise AssertionError(json.dumps(result))
    return result


def read_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rb") as fh:
        yield from pickle.load(fh)


def analyze_traces(run_dir: Path, seed: int, controller: str = "dynamic") -> List[Dict[str, Any]]:
    source = run_dir / f"traces_{controller}_seed{seed}.pkl"
    rows: List[Dict[str, Any]] = []
    if not source.exists():
        return rows
    for step, record in enumerate(read_traces(source)):
        if record.get("valid_n", 0) <= N_MODES:
            continue
        features = np.asarray(record["features"], dtype=np.float64).reshape(record["valid_n"], -1)
        cost = np.asarray(record["cost"], dtype=np.float64)
        log_prior = np.asarray(record["log_prior"], dtype=np.float64)
        terms = pool_norm_terms(cost, features, log_prior)
        gd = np.asarray(record["selected"], dtype=np.int64)
        exact, fopt, fmin = pool_norm_exhaustive(cost, features, log_prior, N_MODES)
        cd = certificate(features, terms["weights"], terms["beta_eff"], gd)
        fd = cd["F_selected"]
        raw_gap = float(fopt - fd)
        scale = float(fopt - fmin)
        tol = 1e-7 * max(1.0, abs(fopt), abs(fd), abs(cd["certificate"]))
        gap = 0.0 if abs(raw_gap) <= tol and raw_gap < 0 else raw_gap
        bound_ok = int(gap >= -tol and gap <= cd["certificate"] + tol)
        normalized_gap = None if scale < RANGE_TOL else gap / scale
        normalized_bound = None if scale < RANGE_TOL else cd["certificate"] / scale
        rows.append({
            "seed": seed, "step": step, "M": record["valid_n"],
            "F_dynamic": fd, "F_exact": fopt, "F_min": fmin, "objective_range": scale,
            "gap": gap, "raw_gap": raw_gap, "Delta_G": cd["certificate"],
            "slack": cd["certificate"] - gap, "normalized_gap": normalized_gap,
            "normalized_bound": normalized_bound,
            "exact_optimal": int(abs(raw_gap) <= tol), "bound_ok": bound_ok,
            "cost_range": terms["cost_range"], "R_D": terms["diversity_scale"],
            "beta_eff": terms["beta_eff"], "cost_degenerate": int(terms["cost_degenerate"]),
            "diversity_degenerate": int(terms["diversity_degenerate"]),
            "objective_range_degenerate": int(scale < RANGE_TOL),
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
            "actuator_endpoint_rmse_mean": float(np.mean(vals("actuator_endpoint_rmse"))),
            "collision_free_rate": float(np.mean(np.asarray([not bool(row.get("collision_any", row.get("collision_count", 0))) for row in subset]))),
            "collision_event_mean": float(np.mean(vals("collision_count"))),
            "collision_substeps_mean": float(np.mean(vals("collision_substeps"))),
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
            "gap_mean": float(np.mean(ovals("gap"))),
            "gap_median": float(np.median(ovals("gap"))),
            "gap_p95": float(np.percentile(ovals("gap"), 95)),
            "normalized_gap_mean": float(np.mean([float(x) for x in (row["normalized_gap"] for row in offline) if x is not None])),
            "normalized_gap_median": float(np.median([float(x) for x in (row["normalized_gap"] for row in offline) if x is not None])),
            "normalized_gap_p95": float(np.percentile([float(x) for x in (row["normalized_gap"] for row in offline) if x is not None], 95)),
            "bound_mean": float(np.mean(ovals("Delta_G"))),
            "bound_median": float(np.median(ovals("Delta_G"))),
            "bound_p95": float(np.percentile(ovals("Delta_G"), 95)),
            "normalized_bound_mean": float(np.mean([float(x) for x in (row["normalized_bound"] for row in offline) if x is not None])),
            "normalized_bound_median": float(np.median([float(x) for x in (row["normalized_bound"] for row in offline) if x is not None])),
            "normalized_bound_p95": float(np.percentile([float(x) for x in (row["normalized_bound"] for row in offline) if x is not None], 95)),
            "exact_optimal_frequency": float(np.mean(ovals("exact_optimal"))),
            "bound_violations": float(np.sum(ovals("bound_ok") < 0.5)),
            "objective_range_degenerate_count": float(np.sum(ovals("objective_range_degenerate") > 0.5)),
        }
    lines = [
        "# BM2PC Greedy Validation Report", "",
        f"Smoke run: `{smoke}`", "",
        "## Objective fidelity", "",
        "Dynamic-Norm Greedy calls production Planner._greedy_select() unchanged. Pool-Norm Exhaustive fixes one cost normalization and one diversity scale per filtered pool, then globally maximizes F_norm over all triples. It is not the exact optimum of the production stepwise score.", "",
        "## Offline comparison", "",
        "The reported gap is F_norm(S_exact)-F_norm(G_dynamic); Delta_G is the a-posteriori certificate using exactly the same pool-normalized weights and beta_eff. Normalized bounds are not clipped and can exceed one; zero objective ranges are excluded from normalized statistics.", "",
        "## Closed-loop comparison", "",
        "`tracking_rmse` is the free rope tip (`S_first`) error relative to the time-aligned Lemniscate reference `goal_builder.get(step + 1)`, measured after executing that control interval. `actuator_endpoint_rmse` separately reports the attached endpoint (`S_last`) error relative to the filtered actuator command.", "",
        "Closed-loop values compare two selection rules under matched initial seeds. Tracking is free-tip S_first versus goal_builder.get(step+1) after the control interval. Collision events and substeps scan every MuJoCo substep.", "",
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
        dyn = np.sort(np.asarray([float(row["gap"]) for row in offline]))
        bound = np.sort(np.asarray([float(row["Delta_G"]) for row in offline]))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for values, label, color in ((dyn, "Dynamic-Norm gap", "tab:blue"), (bound, "Delta_G bound", "tab:orange")):
            y = np.linspace(1.0 / len(values), 1.0, len(values))
            ax.plot(values, y, label=label, color=color)
        ax.set_xlabel("Pool-normalized objective gap / certificate")
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
        axes[0].set_ylabel("Rope-tip tracking RMSE (m)")
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
        reference_plotted = False
        for method, color in zip(methods, ["tab:blue", "tab:orange", "tab:green"]):
            path = run_dir / f"trajectory_{method}_seed0.npz"
            if path.exists():
                saved = np.load(path)
                traj = saved["rope_tip_positions"]
                ax.plot(traj[:, 0], traj[:, 1], label=method, color=color)
                if not reference_plotted:
                    reference = saved["reference_goals"]
                    ax.plot(reference[:, 0], reference[:, 1], "k--", label="Lemniscate reference")
                    reference_plotted = True
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

The proposition evaluates the returned set relative to the fixed pool-normalized
reference objective. It does not assert that the production Dynamic-Norm selector
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
        "Pool-Norm Exhaustive fixes J_norm and R_D once per filtered pool; required conversion and enumeration are included in its multimodal selection timing.",
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
    controllers = ["dynamic", "pool-norm-exhaustive"]
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
            episode_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validate_saved_episode(run_dir, controller, seed, episode_metrics)
            all_metrics.append(episode_metrics)
        rows = analyze_traces(run_dir, seed, "dynamic")
        offline.extend(rows)
        write_csv(run_dir / "per_snapshot_metrics.csv", offline)
        write_csv(run_dir / "closed_loop_metrics.csv", all_metrics)
        print(f"DONE seed={seed} offline_snapshots={len(rows)}", flush=True)
    write_report(run_dir, all_metrics, offline, smoke)
    generate_artifacts(run_dir, all_metrics, offline)
    write_audit_and_timing(run_dir, all_metrics)
    manifest = {"seeds": list(seeds), "controllers": controllers,
                "production_files_unchanged": True, "smoke": smoke,
                "objective": "sum(-850*J_norm + log_prior) + beta_eff*sum_pairwise(D)",
                "normalization": {"epsilon": EPS, "range_tolerance": RANGE_TOL,
                                  "beta_eff": "600/(R_D+1e-8), zero when R_D<1e-12"},
                "pool_norm_is_global_fixed_objective": True,
                "dynamic_is_production_greedy": True,
                "selection_timing": "dynamic trace CPU diagnostics deferred; pool CPU conversion/pairwise/exhaustive included",
                "tracking_metric": "S_first free rope tip vs goal_builder.get(step + 1), post-execution",
                "actuator_endpoint_metric": "S_last attached endpoint vs filtered command target",
                "collision_metric": "rope-cylinder contact rising-edge events and substeps scanned at every MuJoCo substep",
                "closed_loop_metric_consistency": True,
                "offline_rows": len(offline), "metrics_rows": len(all_metrics)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "smoke", "full", "episode", "collect", "analyze", "benchmark", "report"])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--controller", choices=["dynamic", "pool-norm-exhaustive"], default=None)
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
        write_report(run_dir, metrics, offline, "smoke" in run_dir.name)
        generate_artifacts(run_dir, metrics, offline)
        write_audit_and_timing(run_dir, metrics)
        manifest = {
            "git_commit": None,
            "controllers": ["dynamic", "pool-norm-exhaustive"],
            "seeds": sorted({int(row["seed"]) for row in metrics}) if metrics else [],
            "smoke": "smoke" in run_dir.name,
            "episodes": len(metrics),
            "offline_snapshots": len(offline),
            "production_sources_modified": False,
            "objective": "sum(-850*J_norm + log_prior) + beta_eff*sum_pairwise(D)",
            "normalization": {"epsilon": EPS, "range_tolerance": RANGE_TOL,
                              "beta_eff": "600/(R_D+1e-8), zero when R_D<1e-12"},
            "selection_timing": "dynamic trace CPU diagnostics deferred; pool CPU conversion/pairwise/exhaustive included",
            "tracking_metric": "S_first free rope tip vs goal_builder.get(step + 1), post-execution",
            "actuator_endpoint_metric": "S_last attached endpoint vs filtered command target",
            "closed_loop_metric_consistency": True,
            "python": sys.executable,
        }
        try:
            import subprocess as _sp
            manifest["git_commit"] = _sp.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            pass
        cfg = ExperimentConfig()
        (run_dir / "config.json").write_text(
            json.dumps({**cfg.__dict__, "seeds": manifest["seeds"]}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return
    if args.command == "benchmark":
        print("Benchmark data are produced during each controller episode in closed_loop_metrics.csv.")
        return


if __name__ == "__main__":
    main()
