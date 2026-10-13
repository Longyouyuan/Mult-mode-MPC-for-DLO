#!/usr/bin/env bash
# Relaunch individual Scenario B sweep jobs (staggered to avoid Warp cache races).
cd "$(dirname "$0")/.."
PY=${PY:-/d/Anaconda/envs/DRL/python.exe}
OUT=${OUT:-results/scenario_b/main_20261012}
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}
mkdir -p "$OUT"
COMMON="--seeds $SEEDS --save-traj --set pillar_radius=0.05 line_vmax=1.2 min_event_step=100 n_steps=480 move_time=0.001 start_offset=0.10"
for job in "$@"; do
  if [ "$job" = noblock ]; then
    $PY experiments/bm2pc_scenario_b.py --out "$OUT/noblock" --no-block --taus 0.75 $COMMON > "$OUT/log_noblock.txt" 2>&1 &
  else
    $PY experiments/bm2pc_scenario_b.py --out "$OUT/block_tau$job" --taus $job $COMMON > "$OUT/log_block_tau$job.txt" 2>&1 &
  fi
  sleep 8
done
wait
echo resume-done
