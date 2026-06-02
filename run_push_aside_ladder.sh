#!/usr/bin/env bash
# CRL-MAB ladder on arm_push_aside: reuse the frozen arm_push_hard CRL policy on a
# rightward push-aside task, sweeping the OOD goal offset x { ladder rungs 1-4 }.
#
#   rung 1 (CRL-only base)        -> read from each run's "base:" eval column
#   rung 2 (unconstrained resid)  -> barrier_type=none
#   rung 3 (Euclidean / isotropic)-> barrier_type=euclidean, isotropic
#   rung 4 (Mahalanobis ball)     -> barrier_type=ball
#
# Each (offset, barrier) is one train_residual_mab.py run. The frozen base policy
# is identical across all runs; only the env's goal offset and the barrier change.
#
# Usage:
#   CKPT=/path/to/..._final.pkl ./run_push_aside_ladder.sh
# Optional overrides: OFFSETS, BARRIERS, EPSILON, ZONE_RADIUS, EPISODE_LENGTH, ITERS, OUTDIR
set -euo pipefail

CKPT="${CKPT:-/Users/dylanzhou/Downloads/scaling-crl_runs_arm_push_hard_1000_20260309-010323_final.pkl}"
OFFSETS="${OFFSETS:-0.15 0.25 0.35 0.45}"   # 0.25 = red-bin edge (in-dist boundary); >0.25 OOD
BARRIERS="${BARRIERS:-none euclidean isotropic ball}"
EPSILON="${EPSILON:-0.3}"
ZONE_RADIUS="${ZONE_RADIUS:-0.12}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
ITERS="${ITERS:-20}"
OUTDIR="${OUTDIR:-results/push_aside_ladder_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "offset,barrier,epsilon,base_succ,corrected_succ,delta" > "$SUMMARY"
echo "[ladder] checkpoint: $CKPT"
echo "[ladder] offsets: $OFFSETS | barriers: $BARRIERS | epsilon: $EPSILON"
echo "[ladder] writing to: $OUTDIR"

for off in $OFFSETS; do
  for bar in $BARRIERS; do
    tag="off${off}_${bar}"
    log="$OUTDIR/${tag}.log"
    echo "[run] offset=$off barrier=$bar -> $log"
    uv run train_residual_mab.py \
      --checkpoint_path "$CKPT" \
      --env_id arm_push_aside \
      --aside_offset "$off" \
      --zone_radius "$ZONE_RADIUS" \
      --episode_length "$EPISODE_LENGTH" \
      --barrier_type "$bar" \
      --epsilon "$EPSILON" \
      --num_iterations "$ITERS" \
      --exp_name "$tag" \
      --out_dir "$OUTDIR" 2>&1 | tee "$log"

    # Parse the final "[done] base_succ=.. corrected_succ=.. delta=.." line.
    done_line="$(grep -E '^\[done\]' "$log" | tail -1 || true)"
    base="$(echo "$done_line"  | sed -nE 's/.*base_succ=([0-9.]+).*/\1/p')"
    corr="$(echo "$done_line"  | sed -nE 's/.*corrected_succ=([0-9.]+).*/\1/p')"
    delta="$(echo "$done_line" | sed -nE 's/.*delta=([+-][0-9.]+).*/\1/p')"
    echo "${off},${bar},${EPSILON},${base:-NA},${corr:-NA},${delta:-NA}" >> "$SUMMARY"
  done
done

echo
echo "[ladder] done. Summary:"
column -t -s, "$SUMMARY"
echo "[ladder] full summary at $SUMMARY"
