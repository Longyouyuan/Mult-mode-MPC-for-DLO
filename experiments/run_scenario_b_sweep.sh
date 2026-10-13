#!/usr/bin/env bash
# Scenario B main sweep: symmetric line pass, pillar A at the midpoint, pillar B
# appears on the controller's preferred side at lead time tau.
cd "$(dirname "$0")/.."
PY=${PY:-/d/Anaconda/envs/DRL/python.exe}
OUT=results/scenario_b/main_$(date +%Y%m%d)
COMMON="--seeds 0 1 2 3 4 5 6 7 8 9 --save-traj --set pillar_radius=0.05 line_vmax=1.2 min_event_step=100 n_steps=480 move_time=0.001 start_offset=0.10"
mkdir -p "$OUT"
for t in 0.60 0.65 0.70 0.75 0.80 0.85; do
  $PY experiments/bm2pc_scenario_b.py --out "$OUT/block_tau$t" --taus $t $COMMON > "$OUT/log_block_tau$t.txt" 2>&1 &
done
$PY experiments/bm2pc_scenario_b.py --out "$OUT/noblock" --no-block --taus 0.75 $COMMON > "$OUT/log_noblock.txt" 2>&1 &
wait
echo done
