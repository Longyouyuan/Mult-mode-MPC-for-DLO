"""Scenario B: a moving obstacle blocks the currently preferred channel.

Pillar A stands at the Lemniscate crossing point.  It is tall enough that the
rope cannot pass over it, so only the two lateral channels remain.  Before
each crossing, at a lead time ``tau`` ahead of the reference, the side chosen
by the controller's executed mode is read from its predicted tip trajectory,
and pillar B slides in next to A on exactly that side.  Every controller
therefore has its own preferred channel blocked.

Controllers (identical hyperparameters except the ablated component):
  full  : BM2PC, n=3, beta=600
  n1    : BM2PC, n=1, all N samples around one centre
  beta0 : BM2PC, n=3, beta=0 (greedy == cost/prior Top-3)

The production controller is imported unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "controller"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mujoco  # noqa: E402
import torch  # noqa: E402

from common.rope_warp_4 import N, P, WarpRope  # noqa: E402
from common.utils import (  # noqa: E402
    build_infinite_eight,
    mj_data_to_my_data,
    PositionCommandFilter,
    set_seed,
)
from controller.M2PC import Planner, cost_fn  # noqa: E402

FILTER_THRESHOLD = 1.0e5
PARK_XY = np.array([3.0, 3.0])

CONTROLLERS = {
    "full": dict(m_modes=3, beta=600.0),
    "n1": dict(m_modes=1, beta=0.0),
    "beta0": dict(m_modes=3, beta=0.0),
}


@dataclass
class ScenarioConfig:
    # planner (Table I, obstacle avoidance)
    dt: float = 0.001
    ctr_period: int = 25
    horizon: int = 35
    n_sample: int = 402
    n_improve: int = 1
    noise_scale: float = 1.5
    top_k_good: int = 200
    wj: float = 850.0
    prior_weight: float = 1.0
    action_low: float = -5.0
    action_high: float = 5.0
    beta_full: float = 600.0
    # task
    task: str = "line"             # "line": symmetric back-and-forth pass; "lemniscate": Fig. 5 path
    task_duration: float = 5.0     # lemniscate period
    n_steps: int = 440
    line_amp: float = 0.9          # line: y(t) = amp * (1 - cos(w t)), pillar at y = amp
    line_vmax: float = 1.5         # line: peak speed (reached at the pillar) [m/s]
    min_event_step: int = 60       # skip crossings during the start-up transient
    # scenario
    pillar_xy: tuple = (0.0, 0.9)
    pillar_radius: float = 0.05
    pillar_half_h: float = 0.5
    pillar_center_z: float = 0.55
    block: bool = True
    tau: float = 0.4               # lead time of the block before the reference reaches A [s]
    block_offset: float = 0.10     # B centre distance from A centre along the side normal [m]
    start_offset: float = 0.45     # B enters from this distance [m]
    move_time: float = 0.15        # B travel time [s]
    hold_after: float = 0.6        # B removed this long after the reference passes A [s]
    event_window_after: float = 1.0


class PooledSingleModePlanner(Planner):
    """n=1 with the same Top-K pool + threshold filter as the multimodal path.

    The production ``m == 1`` branch is the tracking shortcut: it min-max
    normalizes over *all* samples, so with obstacles the 1e8 collision cost
    squeezes the feasible samples to ~0 and the choice degenerates to the
    log-prior.  Here the normalization range is the filtered Top-K pool, which
    is exactly the first pick of ``_greedy_select`` (the n=3 path).
    """

    def _minmax_norm(self, v, eps=1e-8):
        if self.m != 1 or v.ndim != 1 or v.shape[0] != self.batch:
            return Planner._minmax_norm(v, eps)
        order = torch.argsort(v)
        pool = order[:self.top_k_good]
        pool = pool[v[pool] < FILTER_THRESHOLD]
        if pool.numel() <= 1:
            return Planner._minmax_norm(v, eps)
        vmin, vmax = v[pool].min(), v[pool].max()
        out = torch.full_like(v, 1.0e6)
        out[pool] = (v[pool] - vmin) / (vmax - vmin + eps)
        return out


def make_planner(name: str, cfg: ScenarioConfig, device):
    spec = CONTROLLERS[name]
    m = spec["m_modes"]
    beta = cfg.beta_full if name == "full" else spec["beta"]
    rope = WarpRope(
        batch_size=m + cfg.n_sample, L=1.0, mass=0.0025 * 40 / N, k=10000 * 0.46,
        damping=0.2, bending_k=0.0, bending_damping=0.000401, air_drag=0.2206 / 1000,
        g=10.07, dt=cfg.dt, max_record_steps=cfg.horizon * cfg.ctr_period,
        record_interval=cfg.ctr_period, ctr_period=cfg.ctr_period, mode="acc",
    )
    cls = PooledSingleModePlanner if m == 1 else Planner
    return cls(
        rope, cost_fn, cfg.dt, cfg.ctr_period, cfg.horizon, cfg.n_sample, cfg.n_improve,
        cfg.noise_scale, 3, limits=torch.tensor([cfg.action_low, cfg.action_high]),
        device=device, mode="acc", m_modes=m, top_k_good=cfg.top_k_good, beta=beta,
        wJ=cfg.wj, standard_m2pc=True, prior_weight=cfg.prior_weight,
    )


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def side_of_path(path_xy: np.ndarray, center: np.ndarray, d: np.ndarray):
    """Side (+1/-1) of the pillar on which a path passes, at closest approach."""
    rel = path_xy - center
    k = int(np.argmin(np.linalg.norm(rel, axis=1)))
    return (1 if cross2(d, rel[k]) > 0 else -1), float(np.linalg.norm(rel[k]))


class LineGoal:
    """Back-and-forth straight pass along y at x=0, z=0.2: y = amp (1 - cos(w t))."""

    def __init__(self, amp, vmax, ctrl_dt, device, z0=0.2):
        self.amp, self.w, self.ctrl_dt, self.device, self.z0 = amp, vmax / amp, ctrl_dt, device, z0

    def get_range(self, start, length):
        t = (start + torch.arange(length, device=self.device, dtype=torch.float32)) * self.ctrl_dt
        y = self.amp * (1.0 - torch.cos(self.w * t))
        return torch.stack([torch.zeros_like(y), y, torch.full_like(y, self.z0)], dim=1)

    def get(self, i):
        return self.get_range(i, 1)[0]


def find_crossings(goal_builder, n_steps, center):
    g = np.stack([goal_builder.get(s + 1).detach().cpu().numpy() for s in range(n_steps + 2)])
    dist = np.linalg.norm(g[:, :2] - center, axis=1)
    out = []
    for k in range(1, n_steps):
        if dist[k] < 0.05 and dist[k] <= dist[k - 1] and dist[k] < dist[k + 1]:
            d = g[k + 1, :2] - g[k - 1, :2]
            out.append((k, d / np.linalg.norm(d)))
    return out, g


MODE_RGBA = {"exec": [0.9, 0.1, 0.1, 1.0], "other": [1.0, 0.85, 0.1, 0.8]}


def draw_overlay(viewer, goal_np, mode_trajs, best):
    """Reference horizon (white dots) and predicted tip path of every mode."""
    scn = viewer.user_scn
    eye = np.eye(3).flatten()
    with viewer.lock():
        scn.ngeom = 0
        for p in goal_np:
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.008, 0, 0]), p, eye, np.array([1, 1, 1, 0.8]))
            scn.ngeom += 1
        order = [k for k in range(len(mode_trajs)) if k != best] + [best]
        for k in order:
            rgba = np.array(MODE_RGBA["exec" if k == best else "other"])
            width = 0.008 if k == best else 0.005
            tr = mode_trajs[k]
            for i in range(len(tr) - 1):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), eye, rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, tr[i], tr[i + 1])
                scn.ngeom += 1


def run_episode(name: str, seed: int, cfg: ScenarioConfig, out_dir: Path | None = None,
                viewer: bool = False, slow: float = 1.0):
    """``viewer=True`` opens the MuJoCo passive viewer and paces the loop to
    ``slow`` x real time (planning results are unaffected)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "Mujoco_env" / "cable_show.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 4)
    model.opt.timestep = cfg.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    cable_body_indices = [i for i, nm in enumerate(body_names) if nm and nm.startswith("B_")]
    cyl_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_1")   # moving pillar B
    cyl_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cyl_2")   # fixed pillar A
    for b in (cyl_a, cyl_b):
        model.body_pos[b] = np.array([0.0, 0.0, cfg.pillar_center_z])
    cyl_geoms = [g for g in range(model.ngeom)
                 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
                 and model.geom_bodyid[g] in (cyl_a, cyl_b)]
    for g in cyl_geoms:
        model.geom_size[g][0] = cfg.pillar_radius
        model.geom_size[g][1] = cfg.pillar_half_h
    qadr = {nm: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]
            for nm in ("cyl_1_x", "cyl_1_y", "cyl_2_x", "cyl_2_y")}
    vadr = {nm: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]
            for nm in qadr}
    A = np.asarray(cfg.pillar_xy, dtype=np.float64)

    def place(xy_b, v_b=(0.0, 0.0)):
        data.qpos[qadr["cyl_2_x"]], data.qpos[qadr["cyl_2_y"]] = A
        data.qvel[vadr["cyl_2_x"]] = data.qvel[vadr["cyl_2_y"]] = 0.0
        data.qpos[qadr["cyl_1_x"]], data.qpos[qadr["cyl_1_y"]] = xy_b
        data.qvel[vadr["cyl_1_x"]], data.qvel[vadr["cyl_1_y"]] = v_b

    b_xy = PARK_XY.copy()
    place(b_xy)
    mujoco.mj_forward(model, data)

    rope_geoms = {g for g in range(model.ngeom)
                  if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                  and model.geom_bodyid[g] in set(cable_body_indices)}
    pillar_geoms = set(cyl_geoms)

    planner = make_planner(name, cfg, device)
    total_horizon = int(cfg.task_duration / cfg.dt / cfg.ctr_period)
    ctrl_dt = cfg.dt * cfg.ctr_period
    if cfg.task == "line":
        goal_builder = LineGoal(cfg.line_amp, cfg.line_vmax, ctrl_dt, device)
    else:
        goal_builder = build_infinite_eight(
            device=device, scale_x=0.45 * 2.0, scale_y=0.65 * 2.0, z0=0.2, a=-0.10,
            N_warm=int(total_horizon * 0.3), N_steady=total_horizon,
            du_start_ratio=0.15, warm_power=2.5,
        )
    crossings, _ = find_crossings(goal_builder, cfg.n_steps, A)
    lead = int(round(cfg.tau / ctrl_dt))
    events = []
    for kc, d in crossings:
        if kc - lead < 5 or kc < cfg.min_event_step:
            continue
        events.append(dict(cross_step=kc, d=d, trig_step=kc - lead,
                           off_step=kc + int(round(cfg.hold_after / ctrl_dt)),
                           side=None, start=None, target=None))

    obs = np.zeros((3, 3))
    obs[0, :2] = [cfg.pillar_radius, cfg.pillar_half_h]
    mj_node = len(cable_body_indices) + 1
    mj_state = np.zeros((1, 1 + mj_node * 6 + 3))
    pcf = PositionCommandFilter(np.zeros(3), data.sensordata[[0, 1, 2]].copy(), cfg.dt)
    slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_first")

    log = {k: [] for k in ("tip", "goal", "b_xy", "hit", "infeasible", "exec_mode",
                           "mode_sides", "mode_dists", "wall", "side_counts", "mode_trajs")}
    prev_hit = False
    active = None  # index of the active event

    vw = None
    if viewer:
        import importlib
        vw = importlib.import_module("mujoco.viewer").launch_passive(model, data)
        with vw.lock():
            vw.cam.lookat[:] = [A[0], A[1], 0.45]
            vw.cam.distance = 2.6
            vw.cam.azimuth = 180.0
            vw.cam.elevation = -45.0
        print(f"[viewer] {name} seed={seed} tau={cfg.tau:.2f}  "
              f"white=reference, red=executed mode, yellow=other modes")

    for step in range(cfg.n_steps):
        if vw is not None and not vw.is_running():
            break
        t_step = time.perf_counter()
        # --- pillar B schedule (decided at the start of the control interval) ---
        for ei, ev in enumerate(events):
            if step == ev["off_step"]:
                active = None
                b_xy = PARK_XY.copy()

        mujoco.mj_forward(model, data)
        mj_state[0, 0] = data.time
        mj_state[0, 1:1 + mj_node * 3] = data.xpos[1:1 + mj_node][::-1].reshape(-1)
        mj_state[0, 1 + mj_node * 3:-3] = data.sensordata
        pos, vel, _ = mj_data_to_my_data(N, mj_state.astype(np.float32), device=device)
        goal = goal_builder.get_range(start=step + 1, length=cfg.horizon)
        obs[1, :] = data.xpos[cyl_b]
        obs[2, :] = data.xpos[cyl_a]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        planner.improve_policy(pos, vel, goal, Obs=obs)
        action = planner.get_action(rule="greedy").detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        log["wall"].append(time.perf_counter() - t0)

        m = planner.m
        best = 0 if m == 1 else int(torch.argmin(planner.J_star).item())
        seeds_xy = planner.traj_seeds[:, :, :2].detach().cpu().numpy().astype(np.float64)
        d_now = events[0]["d"] if events else np.array([1.0, 0.0])
        for ev in events:
            if ev["trig_step"] - 40 <= step <= ev["off_step"]:
                d_now = ev["d"]
                break
        sides, dists = [], []
        for k in range(m):
            s, dist = side_of_path(seeds_xy[k], A, d_now)
            sides.append(s)
            dists.append(dist)
        log["mode_sides"].append(sides + [0] * (3 - m))
        log["mode_dists"].append(dists + [np.nan] * (3 - m))
        log["exec_mode"].append(best)
        mt = np.full((3, cfg.horizon, 3), np.nan, dtype=np.float32)
        mt[:m] = planner.traj_seeds.detach().cpu().numpy()
        log["mode_trajs"].append(mt)
        log["infeasible"].append(bool(planner.last_cost.min().item() >= FILTER_THRESHOLD))
        # diagnostics: feasible samples / Top-K pool members per side of pillar A
        cost_np = planner.last_cost.detach().cpu().numpy()
        rel = planner.last_tip_traj[:, :, :2].detach().cpu().numpy() - A
        kmin = np.argmin(np.linalg.norm(rel, axis=2), axis=1)
        relk = rel[np.arange(rel.shape[0]), kmin]
        sgn = np.sign(d_now[0] * relk[:, 1] - d_now[1] * relk[:, 0])
        feas = cost_np < FILTER_THRESHOLD
        topk = np.zeros_like(feas)
        topk[np.argsort(cost_np)[:cfg.top_k_good]] = True
        log["side_counts"].append([int((feas & (sgn > 0)).sum()), int((feas & (sgn < 0)).sum()),
                                   int((feas & topk & (sgn > 0)).sum()), int((feas & topk & (sgn < 0)).sum())])

        # --- trigger: block the executed mode's channel ---
        if cfg.block:
            for ei, ev in enumerate(events):
                if step == ev["trig_step"]:
                    s, _ = side_of_path(seeds_xy[best], A, ev["d"])
                    nrm = s * np.array([-ev["d"][1], ev["d"][0]])
                    ev["side"] = s
                    ev["start"] = A + nrm * cfg.start_offset
                    ev["target"] = A + nrm * cfg.block_offset
                    ev["t0"] = data.time
                    active = ei
                    if vw is not None:
                        print(f"[viewer] t={data.time:5.2f}s  pillar B appears on side {s:+d} "
                              f"(executed mode's side), event {ei}")

        # --- execute ---
        hit_any = False
        for _ in range(cfg.ctr_period):
            v_b = (0.0, 0.0)
            if active is not None:
                ev = events[active]
                a = min(1.0, (data.time - ev["t0"]) / cfg.move_time)
                b_xy = ev["start"] + a * (ev["target"] - ev["start"])
                if a < 1.0:
                    v_b = tuple((ev["target"] - ev["start"]) / cfg.move_time)
            place(b_xy, v_b)
            target = pcf.input_acceleration(np.asarray(action, dtype=np.float64))
            data.ctrl[:3] = target
            data.xfrc_applied[slider_id, :3] = np.array([0.0, 0.0, 0.5 * 9.81])
            mujoco.mj_step(model, data)
            hit = any((c.geom1 in rope_geoms and c.geom2 in pillar_geoms) or
                      (c.geom2 in rope_geoms and c.geom1 in pillar_geoms)
                      for c in data.contact[:data.ncon])
            if vw is not None and hit and not prev_hit:
                print(f"[viewer] t={data.time:5.2f}s  CONTACT rope-pillar")
            hit_any = hit_any or hit
            prev_hit = hit
        log["tip"].append(data.site_xpos[tip_site].copy())
        log["goal"].append(goal[0].detach().cpu().numpy().astype(np.float64))
        log["b_xy"].append(b_xy.copy())
        log["hit"].append(hit_any)
        if vw is not None:
            draw_overlay(vw, goal.detach().cpu().numpy(),
                         planner.traj_seeds.detach().cpu().numpy(), best)
            vw.sync()
            lag = slow * ctrl_dt - (time.perf_counter() - t_step)
            if lag > 0:
                time.sleep(lag)
        planner.update_policy()

    if vw is not None:
        vw.close()
    tip = np.asarray(log["tip"])
    gl = np.asarray(log["goal"])
    err = np.linalg.norm(tip - gl, axis=1)
    hit = np.asarray(log["hit"])
    infeasible = np.asarray(log["infeasible"])
    mode_sides = np.asarray(log["mode_sides"])

    ev_rows = []
    win_after = int(round(cfg.event_window_after / ctrl_dt))
    for ei, ev in enumerate(events):
        if ev["cross_step"] >= len(tip):  # viewer closed early
            continue
        lo, hi = ev["trig_step"], min(len(tip), ev["cross_step"] + win_after)
        seg = slice(lo, hi)
        s_act, clr = side_of_path(tip[seg, :2], A, ev["d"])
        pre = slice(max(0, lo - 10), lo + 1)
        both_sides_pre = bool(np.any(np.ptp(mode_sides[pre, :planner.m], axis=1) > 0)) if planner.m > 1 else False
        collided = bool(hit[seg].any())
        switched = (ev["side"] is not None) and (s_act != ev["side"])
        if collided:
            outcome = "collision"
        elif ev["side"] is None:
            outcome = "no_block"
        elif switched:
            outcome = "switch"
        else:
            outcome = "detour"
        ev_rows.append(dict(
            controller=name, seed=seed, event=ei, tau=cfg.tau, block=cfg.block,
            cross_step=ev["cross_step"], blocked_side=ev["side"], taken_side=s_act,
            outcome=outcome, collision=collided, switched=bool(switched),
            collision_steps=int(hit[seg].sum()), infeasible_steps=int(infeasible[seg].sum()),
            event_max_err=float(err[seg].max()),
            event_rmse=float(np.sqrt(np.mean(err[seg] ** 2))),
            tip_clearance_A=clr, alt_mode_available=both_sides_pre,
        ))
    ep = dict(controller=name, seed=seed, tau=cfg.tau, block=cfg.block,
              rmse=float(np.sqrt(np.mean(err ** 2))), max_err=float(err.max()),
              collision_steps=int(hit.sum()), infeasible_steps=int(infeasible.sum()),
              wall_mean_ms=float(np.mean(log["wall"]) * 1e3))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"traj_{name}_seed{seed}.npz", tip=tip, goal=gl,
            b_xy=np.asarray(log["b_xy"]), hit=hit, infeasible=infeasible,
            exec_mode=np.asarray(log["exec_mode"]), mode_sides=mode_sides,
            mode_dists=np.asarray(log["mode_dists"]), pillar_a=A,
            side_counts=np.asarray(log["side_counts"]),
            mode_trajs=np.asarray(log["mode_trajs"], dtype=np.float32),
            events=json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in e.items()} for e in events]))
    return ep, ev_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    # Windows may briefly lock a freshly written file (indexer / editor preview).
    for attempt in range(20):
        try:
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            return
        except PermissionError:
            time.sleep(0.5)
    print(f"[warn] could not write {path}")


def summarize(ev_rows, ep_rows):
    out = {}
    keys = sorted({(r["tau"], r["controller"]) for r in ep_rows})
    for tau, c in keys:
        ev = [r for r in ev_rows if r["tau"] == tau and r["controller"] == c]
        ep = [r for r in ep_rows if r["tau"] == tau and r["controller"] == c]
        n = max(len(ev), 1)
        out[f"tau={tau:.2f}/{c}"] = dict(
            events=len(ev),
            collision_rate=sum(r["collision"] for r in ev) / n,
            switch_rate=sum(r["outcome"] == "switch" for r in ev) / n,
            detour_rate=sum(r["outcome"] == "detour" for r in ev) / n,
            alt_mode_available=sum(r["alt_mode_available"] for r in ev) / n,
            event_max_err_mean=float(np.mean([r["event_max_err"] for r in ev])) if ev else np.nan,
            infeasible_steps_mean=float(np.mean([r["infeasible_steps"] for r in ev])) if ev else np.nan,
            rmse_mean=float(np.mean([r["rmse"] for r in ep])),
            rmse_std=float(np.std([r["rmse"] for r in ep])),
            wall_ms=float(np.mean([r["wall_mean_ms"] for r in ep])),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "scenario_b" / "run")
    ap.add_argument("--controllers", nargs="*", default=list(CONTROLLERS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(5)))
    ap.add_argument("--taus", nargs="*", type=float, default=[0.4])
    ap.add_argument("--no-block", action="store_true")
    ap.add_argument("--save-traj", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override ScenarioConfig fields, key=value")
    args = ap.parse_args()

    base = ScenarioConfig(block=not args.no_block)
    for kv in args.set:
        k, v = kv.split("=")
        if k == "pillar_xy":
            base = replace(base, pillar_xy=tuple(float(x) for x in v.split(",")))
        else:
            base = replace(base, **{k: type(getattr(base, k))(v)})
    if base.task == "line" and "pillar_xy" not in " ".join(args.set):
        base = replace(base, pillar_xy=(0.0, base.line_amp))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        dict(asdict(base), controllers=args.controllers, seeds=args.seeds, taus=args.taus),
        indent=2), encoding="utf-8")

    ep_rows, ev_rows = [], []
    for tau in args.taus:
        cfg = replace(base, tau=tau)
        for c in args.controllers:
            for s in args.seeds:
                t0 = time.time()
                ep, evs = run_episode(c, s, cfg, args.out / f"tau{tau:.2f}" if args.save_traj else None)
                ep_rows.append(ep)
                ev_rows.extend(evs)
                print(f"tau={tau:.2f} {c:6s} seed={s} rmse={ep['rmse']*100:.2f}cm "
                      f"coll_steps={ep['collision_steps']} outcomes="
                      f"{[e['outcome'] for e in evs]} alt={[int(e['alt_mode_available']) for e in evs]} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                write_csv(args.out / "episodes.csv", ep_rows)
                write_csv(args.out / "events.csv", ev_rows)
    summ = summarize(ev_rows, ep_rows)
    (args.out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for k, v in summ.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
