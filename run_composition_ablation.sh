#!/usr/bin/env bash
# Scripted-ball constrained-composition ablation on tidybot_hallway.
#
# Fits the two scripted skill balls (base + push) from controller rollouts, then
# runs the constrained hallway RL across the barrier ladder x seeds:
#   none       -> unconstrained RL baseline (no ball)
#   single     -> one fixed primitive's ball
#   euclidean  -> gated multi-primitive ball with Sigma = I
#   multi      -> gated multi-primitive Mahalanobis ball  (the method)
# The claim holds if `multi` reaches higher hallway success than `none` at a fixed
# step budget. Each run also logs a base->push->base gate selection histogram.
#
# Usage (defaults are tuned for a single NVIDIA GPU):
#   ./run_composition_ablation.sh
# Common overrides:
#   BARRIERS="none multi" SEEDS="0 1" ITERS=100 NUM_ENVS=512 ./run_composition_ablation.sh
#   COLLECT=0 ./run_composition_ablation.sh     # reuse existing skill_balls/*.pkl
set -euo pipefail

BARRIERS="${BARRIERS:-none single euclidean multi}"
SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-200}"
NUM_ENVS="${NUM_ENVS:-256}"
EPISODE_LENGTH="${EPISODE_LENGTH:-500}"
EPSILON="${EPSILON:-0.3}"

COLLECT="${COLLECT:-1}"                 # 1 = (re)fit the skill balls first
COLLECT_ENVS="${COLLECT_ENVS:-256}"
COLLECT_STEPS="${COLLECT_STEPS:-200}"
BASE_BALL="${BASE_BALL:-skill_balls/base.pkl}"
PUSH_BALL="${PUSH_BALL:-skill_balls/push.pkl}"

OUTDIR="${OUTDIR:-results/composition_ablation_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "barrier,seed,final_success" > "$SUMMARY"

echo "[ablation] barriers: $BARRIERS | seeds: $SEEDS | iters: $ITERS | envs: $NUM_ENVS"
echo "[ablation] balls: $BASE_BALL , $PUSH_BALL | writing to: $OUTDIR"

# ---- 1. fit the scripted skill balls ----
if [ "$COLLECT" = "1" ]; then
  echo "[ablation] fitting skill balls..."
  uv run collect_scripted_skill_data.py --skill base \
    --num_collect_envs "$COLLECT_ENVS" --collect_steps "$COLLECT_STEPS" \
    --out "$BASE_BALL" 2>&1 | tee "$OUTDIR/collect_base.log"
  uv run collect_scripted_skill_data.py --skill push \
    --num_collect_envs "$COLLECT_ENVS" --collect_steps "$COLLECT_STEPS" \
    --out "$PUSH_BALL" 2>&1 | tee "$OUTDIR/collect_push.log"
fi

# ---- 2. barrier ladder x seeds ----
for bar in $BARRIERS; do
  for seed in $SEEDS; do
    tag="${bar}_seed${seed}"
    log="$OUTDIR/${tag}.log"
    echo "[run] barrier=$bar seed=$seed -> $log"
    uv run train_constrained_composition.py \
      --env_id tidybot_hallway \
      --primitive_kind scripted scripted \
      --skill_ball_paths "$BASE_BALL" "$PUSH_BALL" \
      --barrier_type "$bar" \
      --epsilon "$EPSILON" \
      --num_iterations "$ITERS" \
      --num_envs "$NUM_ENVS" \
      --episode_length "$EPISODE_LENGTH" \
      --seed "$seed" \
      --exp_name "$tag" \
      --out_dir "$OUTDIR" 2>&1 | tee "$log"

    # Parse the final "[eval it=...] ... success=X" line.
    succ="$(grep -E '^\[eval it=' "$log" | tail -1 | sed -nE 's/.*success=([0-9.]+).*/\1/p')"
    echo "${bar},${seed},${succ:-NA}" >> "$SUMMARY"
  done
done

echo
echo "[ablation] done. Summary:"
column -t -s, "$SUMMARY"
echo "[ablation] full results in $OUTDIR"
