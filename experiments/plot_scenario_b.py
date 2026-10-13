"""Aggregate and plot the Scenario B sweep (see run_scenario_b_sweep.sh)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"full": "BM2PC (n=3)", "n1": "BM2PC, n=1", "beta0": r"BM2PC, $\beta$=0 (Top-3)"}
COLORS = {"full": "#d62728", "n1": "#1f77b4", "beta0": "#2ca02c"}
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "results" / "scenario_b" / "main_20261012"


def read_csv(path: Path):
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def reclassify(run: Path, rows, win_after_steps=40, pass_margin=0.10):
    """Recompute event outcomes from the saved trajectories.

    pass   : the tip crosses the line through A (normal to the pass direction)
             by ``pass_margin`` without any rope-pillar contact in the window;
             ``switch`` if it crosses on the unblocked side, ``detour`` otherwise
    stall  : no contact, but the tip never gets past A in the window
    collision : any rope-pillar contact in the window
    """
    cache = {}
    for r in rows:
        tau = r["tau"]
        base = Path(r.get("_run", run))
        sub = base / (f"block_tau{tau:.2f}" if r["block"] else "noblock") / f"tau{tau:.2f}"
        key = (sub, r["controller"], int(r["seed"]))
        if key not in cache:
            z = np.load(sub / f"traj_{r['controller']}_seed{int(r['seed'])}.npz")
            cache[key] = (z["tip"], z["hit"], z["pillar_a"], json.loads(str(z["events"])))
        tip, hit, A, events = cache[key]
        e = events[int(r["event"])]
        d = np.asarray(e["d"])
        lo, hi = e["trig_step"], min(len(tip), e["cross_step"] + win_after_steps)
        rel = tip[lo:hi, :2] - A
        prog = rel @ d
        collided = bool(hit[lo:hi].any())
        passed = np.where(prog > pass_margin)[0]
        if collided:
            out = "collision"
        elif len(passed) == 0:
            out = "stall"
        else:
            k = passed[0]
            k0 = k
            while k0 > 0 and prog[k0 - 1] > 0:
                k0 -= 1
            lat = d[0] * rel[k0, 1] - d[1] * rel[k0, 0]
            side = 1 if lat > 0 else -1
            if e["side"] is None:
                out = "pass"
            else:
                out = "switch" if side != e["side"] else "detour"
        r["outcome"] = out
        r["collision"] = collided
        r["passed"] = out in ("pass", "switch", "detour")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN,
                    help="primary run dir (outputs are written here)")
    ap.add_argument("--extra", type=Path, nargs="*", default=None,
                    help="more run dirs to pool (default: <run>_s10_19 if it exists)")
    ap.add_argument("--example-tau", type=float, default=0.70)
    ap.add_argument("--paper", action="store_true", help="also write a single-column figure")
    args = ap.parse_args()
    run = args.run
    if args.extra is None:
        extra = run.parent / f"{run.name}_s10_19"
        args.extra = [extra] if extra.exists() else []
    print("run:", run, "| extra:", [str(e) for e in args.extra])
    ctrls = ["full", "n1", "beta0"]

    ev_all, ep_all = [], []
    for rd in [run] + list(args.extra):
        for d in sorted(rd.glob("block_tau*")):
            if (d / "events.csv").exists():
                ev = read_csv(d / "events.csv")
                for r in ev:
                    r["_run"] = str(rd)
                ev_all += ev
                ep_all += read_csv(d / "episodes.csv")
    ev_all = reclassify(run, ev_all)
    taus = sorted({r["tau"] for r in ev_all})

    table = []
    for tau in taus:
        for c in ctrls:
            ev = [r for r in ev_all if r["tau"] == tau and r["controller"] == c]
            ep = [r for r in ep_all if r["tau"] == tau and r["controller"] == c]
            n = len(ev)
            ncol = sum(r["collision"] for r in ev)
            nsw = sum(r["outcome"] == "switch" for r in ev)
            p_ok, lo, hi = wilson(sum(r["passed"] for r in ev), n)
            table.append(dict(
                tau=tau, controller=c, events=n,
                success_rate=p_ok, success_ci_lo=lo, success_ci_hi=hi,
                collision_rate=ncol / max(n, 1), switch_rate=nsw / max(n, 1),
                detour_rate=sum(r["outcome"] == "detour" for r in ev) / max(n, 1),
                stall_rate=sum(r["outcome"] == "stall" for r in ev) / max(n, 1),
                event_max_err_cm=100 * float(np.mean([r["event_max_err"] for r in ev])),
                infeasible_steps=float(np.mean([r["infeasible_steps"] for r in ev])),
                episode_rmse_cm=100 * float(np.mean([r["rmse"] for r in ep])),
                episode_rmse_std_cm=100 * float(np.std([r["rmse"] for r in ep])),
            ))
    if (run / "noblock" / "events.csv").exists():
        ev = reclassify(run, read_csv(run / "noblock" / "events.csv"))
        ep = read_csv(run / "noblock" / "episodes.csv")
        for c in ctrls:
            e = [r for r in ev if r["controller"] == c]
            p = [r for r in ep if r["controller"] == c]
            n = len(e)
            ncol = sum(r["collision"] for r in e)
            p_ok, lo, hi = wilson(sum(r["passed"] for r in e), n)
            table.append(dict(
                tau="static", controller=c, events=n, success_rate=p_ok,
                success_ci_lo=lo, success_ci_hi=hi, collision_rate=ncol / max(n, 1),
                switch_rate=np.nan, detour_rate=np.nan,
                stall_rate=sum(r["outcome"] == "stall" for r in e) / max(n, 1),
                event_max_err_cm=100 * float(np.mean([r["event_max_err"] for r in e])),
                infeasible_steps=float(np.mean([r["infeasible_steps"] for r in e])),
                episode_rmse_cm=100 * float(np.mean([r["rmse"] for r in p])),
                episode_rmse_std_cm=100 * float(np.std([r["rmse"] for r in p])),
            ))

    with (run / "summary_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(f"tau={r['tau']!s:>6} {r['controller']:6s} n={r['events']:3d} "
              f"success={r['success_rate']*100:5.1f}% [{r['success_ci_lo']*100:4.1f},{r['success_ci_hi']*100:4.1f}] "
              f"coll={r['collision_rate']*100:5.1f}% stall={r['stall_rate']*100:5.1f}% "
              f"switch={r['switch_rate']*100:5.1f}% maxerr={r['event_max_err_cm']:5.1f}cm "
              f"infeas={r['infeasible_steps']:4.1f} rmse={r['episode_rmse_cm']:5.1f}±{r['episode_rmse_std_cm']:4.1f}cm")

    # --- success / switch rate vs tau ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for c in ctrls:
        rows = [r for r in table if r["controller"] == c and r["tau"] != "static"]
        t = np.array([r["tau"] for r in rows])
        s = np.array([r["success_rate"] for r in rows])
        lo = np.array([r["success_ci_lo"] for r in rows])
        hi = np.array([r["success_ci_hi"] for r in rows])
        axes[0].plot(t, 100 * s, "o-", color=COLORS[c], label=LABELS[c])
        axes[0].fill_between(t, 100 * lo, 100 * hi, color=COLORS[c], alpha=0.12)
        axes[1].plot(t, [100 * r["collision_rate"] for r in rows], "o-", color=COLORS[c])
        axes[2].plot(t, [r["event_max_err_cm"] for r in rows], "o-", color=COLORS[c])
    axes[0].set_ylabel("successful passes (%)")
    axes[1].set_ylabel("collision events (%)")
    axes[2].set_ylabel("max tracking error (cm)")
    for ax in axes:
        ax.set_xlabel(r"obstacle lead time $\tau$ (s)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "scenario_b_vs_tau.png", dpi=200)
    fig.savefig(run / "scenario_b_vs_tau.pdf")
    print("saved", run / "scenario_b_vs_tau.png")
    if args.example_tau is not None:
        plot_example(run, args.example_tau, ev_all, ctrls)
    if args.paper:
        plot_paper(run, table, ctrls)


def plot_paper(run: Path, table, ctrls):
    """IEEE single-column (3.5 in) version: success rate with 95% CI, collision rate."""
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7})
    labels = {"full": "BM2PC", "n1": r"$n{=}1$", "beta0": r"$\beta{=}0$"}
    markers = {"full": "o", "n1": "s", "beta0": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55))
    for c in ctrls:
        rows = [r for r in table if r["controller"] == c and r["tau"] != "static"]
        t = np.array([r["tau"] for r in rows])
        axes[0].fill_between(t, [100 * r["success_ci_lo"] for r in rows],
                             [100 * r["success_ci_hi"] for r in rows], color=COLORS[c], alpha=0.12, lw=0)
        axes[0].plot(t, [100 * r["success_rate"] for r in rows], marker=markers[c], ms=3, lw=1.0,
                     color=COLORS[c], label=labels[c])
        axes[1].plot(t, [100 * r["collision_rate"] for r in rows], marker=markers[c], ms=3, lw=1.0,
                     color=COLORS[c])
    axes[0].set_ylabel("success (%)")
    axes[1].set_ylabel("collision (%)")
    for ax in axes:
        ax.set_xlabel(r"lead time $\tau$ (s)")
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=6, width=0.5, length=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.set_xticks([0.6, 0.7, 0.8])
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 70)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.5)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(run / "scenario_b_vs_tau_paper.pdf")
    fig.savefig(run / "scenario_b_vs_tau_paper.png", dpi=300)
    print("saved", run / "scenario_b_vs_tau_paper.png")


def plot_example(run: Path, tau: float, ev_all, ctrls):
    """Top view of one event: full switched while both ablations did not."""
    ev = [r for r in ev_all if abs(r["tau"] - tau) < 1e-6]
    key = None
    for r in ev:
        if r["controller"] != "full" or r["outcome"] not in ("switch", "detour"):
            continue
        others = [o for o in ev if o["seed"] == r["seed"] and o["event"] == r["event"]
                  and o["controller"] != "full"]
        if len(others) == 2 and all(not o["passed"] for o in others):
            key = (int(r["seed"]), int(r["event"]))
            break
    if key is None:
        print("no contrasting example found")
        return
    seed, ei = key
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    src = Path([r for r in ev if int(r["seed"]) == seed][0]["_run"])
    d = src / f"block_tau{tau:.2f}" / f"tau{tau:.2f}"
    for ax, c in zip(axes, ctrls):
        z = np.load(d / f"traj_{c}_seed{seed}.npz")
        events = json.loads(str(z["events"]))
        e = events[ei]
        lo, hi = e["trig_step"] - 15, min(len(z["tip"]), e["cross_step"] + 40)
        tip, goal, hit = z["tip"][lo:hi], z["goal"][lo:hi], z["hit"][lo:hi]
        A = z["pillar_a"]
        r_cyl = 0.05
        ax.plot(goal[:, 1], goal[:, 0], "k--", lw=1, label="reference")
        ax.plot(tip[:, 1], tip[:, 0], color=COLORS[c], lw=2, label="rope tip")
        if hit.any():
            ax.plot(tip[hit, 1], tip[hit, 0], "x", color="k", ms=5, label="contact")
        ax.add_patch(plt.Circle((A[1], A[0]), r_cyl, color="#8c564b"))
        ax.add_patch(plt.Circle((e["target"][1], e["target"][0]), r_cyl, color="#ff7f0e"))
        k0 = e["trig_step"] - lo
        ax.plot(tip[k0, 1], tip[k0, 0], "o", color="k", ms=4)
        r = [o for o in ev if o["controller"] == c and o["seed"] == seed and o["event"] == ei][0]
        ax.set_title(f"{LABELS[c]}\n{r['outcome']}", fontsize=9)
        ax.set_aspect("equal")
        yc = A[1]
        ax.set_xlim(yc - 0.6, yc + 0.6)
        ax.set_ylim(-0.35, 0.35)
        ax.set_xlabel("y (m)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("x (m)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = run / f"scenario_b_example_tau{tau:.2f}_seed{seed}_ev{ei}"
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    print("saved", str(out) + ".png")


if __name__ == "__main__":
    main()
