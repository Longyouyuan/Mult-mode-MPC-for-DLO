"""Watch one Scenario B episode live in the MuJoCo viewer.

Edit the settings below and run this file directly (e.g. the VS Code Run button).
The scenario matches the main sweep (results/scenario_b/main_20261012), so the
same CONTROLLER / SEED / TAU reproduces the episode behind the reported numbers.

Viewer overlay: white dots = reference over the horizon, red = executed mode,
yellow = the other modes (n=3 only).  Pillar B pops up on the executed mode's
side TAU seconds before the reference reaches pillar A.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm2pc_scenario_b import ScenarioConfig, run_episode  # noqa: E402

# ===================== settings =====================
CONTROLLER = "beta0"   # "full" (BM2PC n=3) | "n1" (n=1) | "beta0" (beta=0, Top-3)
SEED = 0
TAU = 0.70            # obstacle lead time [s]; main sweep used 0.60 ... 0.85
BLOCK = True          # False: static pillar only (no pillar B)
SLOW = 2.0            # playback pacing: 1.0 = real time, 2.0 = half speed
# =====================================================

# Same scenario as run_scenario_b_sweep.sh
cfg = replace(
    ScenarioConfig(), block=BLOCK, tau=TAU,
    pillar_radius=0.05, line_vmax=1.2, min_event_step=100, n_steps=480,
    move_time=0.001, start_offset=0.10,
)
cfg = replace(cfg, pillar_xy=(0.0, cfg.line_amp))

ep, events = run_episode(CONTROLLER, SEED, cfg, viewer=True, slow=SLOW)
print(f"\n{CONTROLLER} seed={SEED} tau={TAU:.2f}: rmse={ep['rmse'] * 100:.1f} cm, "
      f"collision steps={ep['collision_steps']}")
for e in events:
    print(f"  event {e['event']}: blocked side {e['blocked_side']}, "
          f"contact={e['collision']}, max err={e['event_max_err'] * 100:.1f} cm")
