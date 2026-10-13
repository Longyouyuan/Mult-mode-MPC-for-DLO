"""Snapshot figure of mode predictions around one Scenario B event.

Rows: BM2PC (n=3), n=1, beta=0.  Columns: the plan at the step pillar B
appears (computed without B), the same plan with B overlaid, the first replan
that sees B, and the resulting tip path.
Needs trajectories saved with mode predictions (``mode_trajs``), e.g.

    python experiments/bm2pc_scenario_b.py --out results/scenario_b/example_modes \
        --seeds 0 --taus 0.70 --save-traj --set pillar_radius=0.05 line_vmax=1.2 \
        min_event_step=100 n_steps=480 move_time=0.001 start_offset=0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
CTRLS = ["full", "n1", "beta0"]
ROW_LABELS = {"full": "BM2PC\n(n=3)", "n1": "BM2PC\n(n=1)", "beta0": "BM2PC\n" + r"($\beta$=0)"}
C_EXEC, C_OTHER, C_TIP = "#d62728", "#ff9f1c", "#555555"
C_A, C_B = "#8c564b", "#1f77b4"
CTRL_DT = 0.025
R_CYL, MARGIN = 0.05, 0.02
S = 1.0  # line / marker scale (``--paper`` shrinks it)
F = 1.0  # font scale


class HandlerArrowLine(HandlerBase):
    """Legend entry drawn like a mode prediction: a line ending in an arrowhead."""

    def __init__(self, ms):
        super().__init__()
        self.ms = ms

    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        y = 0.5 * height - ydescent
        x0, x1 = -xdescent, width - xdescent
        line = Line2D([x0, x1 - 0.12 * width], [y, y], color=orig.get_color(),
                      lw=orig.get_linewidth())
        head = Line2D([x1 - 0.08 * width], [y], marker=">", ms=self.ms, ls="",
                      color=orig.get_color())
        for a in (line, head):
            a.set_transform(trans)
        return [line, head]


def draw_pillars(ax, A, B, b_present, b_ghost=False):
    ax.add_patch(Circle((A[1], A[0]), R_CYL, color=C_A, zorder=3))
    ax.add_patch(Circle((A[1], A[0]), R_CYL + MARGIN, fill=False, ls=":", lw=0.6 * S, color=C_A, zorder=3))
    if b_present:
        ax.add_patch(Circle((B[1], B[0]), R_CYL, color=C_B, zorder=3))
        ax.add_patch(Circle((B[1], B[0]), R_CYL + MARGIN, fill=False, ls=":", lw=0.6 * S, color=C_B, zorder=3))
    elif b_ghost:
        ax.add_patch(Circle((B[1], B[0]), R_CYL, fill=False, ls="--", lw=0.8 * S, color=C_B, zorder=3))


def draw_modes(ax, mode_trajs, best):
    ks = [k for k in range(mode_trajs.shape[0]) if not np.isnan(mode_trajs[k]).all()]
    for k in ks:
        if k == best:
            continue
        tr = mode_trajs[k]
        ax.plot(tr[:, 1], tr[:, 0], color=C_OTHER, lw=0.75 * S, alpha=0.95, zorder=6)
        ax.plot(tr[-1, 1], tr[-1, 0], ">", color=C_OTHER, ms=2.4 * S, zorder=6)
    tr = mode_trajs[best]
    ax.plot(tr[:, 1], tr[:, 0], color=C_EXEC, lw=2.0 * S, zorder=5)
    ax.plot(tr[-1, 1], tr[-1, 0], ">", color=C_EXEC, ms=4.5 * S, zorder=5)


def side(traj, A, d):
    """Side of pillar A (+1/-1) on which a predicted tip path passes."""
    rel = traj[:, :2] - A
    k = int(np.argmin(np.linalg.norm(rel, axis=1)))
    return 1 if d[0] * rel[k, 1] - d[1] * rel[k, 0] > 0 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "results" / "scenario_b" / "example_modes")
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", type=int, default=3)
    ap.add_argument("--replan", type=int, default=1,
                    help="steps after B appears for the 'mode switch' snapshot")
    ap.add_argument("--passing", type=int, default=9,
                    help="steps after B appears for the 'bypassing' snapshot")
    ap.add_argument("--paper", action="store_true",
                    help="IEEE double-column size (7.16 in wide) for the manuscript")
    args = ap.parse_args()
    d = args.run / f"tau{args.tau:.2f}"
    global S, F
    if args.paper:
        S, F = 0.7, 0.9
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                             "font.serif": ["Times New Roman", "DejaVu Serif"]})
    figsize = (7.16, 2.5) if args.paper else (15.5, 4.2)
    txt_switch = "mode switch" if args.paper else "switched to alternative mode"
    txt_infeas = "all samples collide" if args.paper else "all samples in collision"

    # columns: normal plan (B unseen) | B appears over that plan | replan | executed path
    panels = ["plan", "blocked", "replan", "passing", "path"]
    ncol = len(panels)
    fig, axes = plt.subplots(3, ncol, figsize=figsize, sharex=True, sharey=True)
    for row, c in enumerate(CTRLS):
        z = np.load(d / f"traj_{c}_seed{args.seed}.npz")
        e = json.loads(str(z["events"]))[args.event]
        A = np.asarray(z["pillar_a"])
        B = np.asarray(e["target"])
        tip, goal, hit = z["tip"], z["goal"], z["hit"]
        mt, em = z["mode_trajs"], z["exec_mode"]
        trig, cross = e["trig_step"], e["cross_step"]
        start = trig - 12
        end = min(len(tip), cross + 40)  # same 1 s evaluation window as the statistics
        plan = mt[trig, int(em[trig])]  # executed plan at the step B appears (computed without B)

        for col, kind in enumerate(panels):
            ax = axes[row, col]
            ax.plot(goal[start:end, 1], goal[start:end, 0], "k--", lw=0.8 * S, zorder=1)
            if kind != "path":
                s = {"plan": trig, "blocked": trig, "replan": trig + args.replan,
                     "passing": trig + args.passing}[kind]
                draw_pillars(ax, A, B, b_present=(kind != "plan"))
                # tip state at the start of step s is the end of step s-1
                ax.plot(tip[start:s, 1], tip[start:s, 0], color=C_TIP, lw=1.2 * S, zorder=2)
                ax.plot(tip[s - 1, 1], tip[s - 1, 0], "o", color="k", ms=4 * S, zorder=6)
                if kind == "replan":
                    ax.plot(plan[:, 1], plan[:, 0], color=C_EXEC, lw=1.6 * S, ls=(0, (3, 2)),
                            alpha=0.45, zorder=4)
                draw_modes(ax, mt[s], int(em[s]))
                if kind == "blocked":
                    inside = np.where(np.linalg.norm(plan[:, :2] - B, axis=1) < R_CYL + MARGIN)[0]
                    if len(inside):
                        k = inside[0]
                        ax.plot(plan[k, 1], plan[k, 0], "X", color="k", ms=5 * S, mew=0.5 * S,
                                mec="white", zorder=7)
                if kind == "replan" and side(mt[s, int(em[s])], A, e["d"]) != side(plan, A, e["d"]):
                    ax.text(0.02, 0.06, txt_switch, transform=ax.transAxes,
                            fontsize=7 * F, color="#2ca02c", ha="left", va="bottom")
                if kind in ("replan", "passing") and z["infeasible"][s]:
                    ax.text(0.02, 0.06, txt_infeas, transform=ax.transAxes,
                            fontsize=7 * F, color="#d62728", ha="left", va="bottom")
            else:
                draw_pillars(ax, A, B, True)
                seg = slice(start, end)
                ax.plot(tip[seg, 1], tip[seg, 0], color=C_TIP, lw=1.6 * S, zorder=2)
                hit_idx = np.where(hit[seg])[0] + start
                if len(hit_idx):
                    ax.plot(tip[hit_idx, 1], tip[hit_idx, 0], "x", color="k", ms=4 * S, zorder=6)
                ax.plot(tip[trig, 1], tip[trig, 0], "o", color=C_B, ms=4 * S, zorder=6)
                rel = tip[trig:end, :2] - A
                passed = np.any(rel @ np.asarray(e["d"]) > 0.10)
                outcome = "contact" if hit[trig:end].any() else ("passed" if passed else "stalled")
                ax.text(0.98, 0.92, outcome, transform=ax.transAxes, ha="right", va="top",
                        fontsize=9 * F, fontweight="bold",
                        color={"passed": "#2ca02c", "stalled": "#d62728", "contact": "#d62728"}[outcome])
            ax.set_aspect("equal")
            ax.set_xlim(0.05, 1.6)
            ax.set_ylim(-0.27, 0.27)
            ax.grid(alpha=0.25, lw=0.5 * S)
            ax.tick_params(labelsize=7.8 * F)
            ax.set_yticks([-0.2, 0.0, 0.2])
        axes[row, 0].set_ylabel(ROW_LABELS[c], fontsize=9 * F, rotation=0, ha="right", va="center", labelpad=6)

    titles = ["(a) planning (B not present)",
              "(b) B appears: executed plan blocked",
              f"(c) first replan with B ({args.replan * CTRL_DT * 1e3:+.0f} ms)",
              f"(d) bypassing ({args.passing * CTRL_DT * 1e3:+.0f} ms)",
              "(e) executed rope-tip path"]
    if args.paper:
        titles = ["(a) Planning", "(b) Pillar B appears",
                  f"(c) Replan ({args.replan * CTRL_DT * 1e3:+.0f} ms)",
                  f"(d) Bypass ({args.passing * CTRL_DT * 1e3:+.0f} ms)", "(e) Executed path"]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=9 * F)
    for col in range(ncol):
        axes[-1, col].set_xlabel("y (m)", fontsize=8 * F)
    # reference (dashed) and the "plan enters B" cross are explained in the caption
    h_exec = Line2D([], [], color=C_EXEC, lw=2.0 * S, label="executed mode")
    h_other = Line2D([], [], color=C_OTHER, lw=0.75 * S, label="other modes")
    handles = [
        h_exec,
        h_other,
        Line2D([], [], color=C_EXEC, lw=1.6 * S, ls=(0, (3, 2)), alpha=0.45, label="previous plan"),
        Line2D([], [], color=C_TIP, lw=1.4 * S, label="rope tip"),
        Line2D([], [], marker="o", color=C_A, ls="", ms=7 * S, label="pillar A"),
        Line2D([], [], marker="o", color=C_B, ls="", ms=7 * S, label="pillar B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8 * F, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.0, columnspacing=1.6,
               handler_map={h_exec: HandlerArrowLine(4.5 * S), h_other: HandlerArrowLine(2.4 * S)})
    if args.paper:
        fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.2, w_pad=0.3, h_pad=0.2)
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = args.run / f"scenario_b_modes_tau{args.tau:.2f}_seed{args.seed}_ev{args.event}"
    if args.paper:
        out = Path(str(out) + "_paper")
    fig.savefig(str(out) + ".png", dpi=250, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    print("saved", str(out) + ".png")


if __name__ == "__main__":
    main()
