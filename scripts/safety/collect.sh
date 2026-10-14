#!/usr/bin/env bash
# Collect safety training data (Sec. 5.1, App. C.3):
#   roll the untrained MAS out G=8 times (8 seeds) on the safety training pool
#   (380 harmful AdvBench / HarmBench-val prompts + 300 benign MMLU-Pro questions),
#   temperature 0.8, top-p 0.95, 4096 tokens per agent, thinking mode;
#   turn every agent message into an example; label rollouts with the HarmBench reward.
#
#   bash scripts/safety/collect.sh {smas|hmas} {benign|compromised}
set -euo pipefail
source "$(dirname "$0")/../common.sh"
ARCH=$1; SETTING=$2
N_AGENTS=${N_AGENTS:-4}
N_QUERIES=${N_QUERIES:-680}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49"}
NAME=${NAME:-$ARCH-$SETTING-n$N_AGENTS}

for s in $SEEDS; do
  out=results/collect-$NAME-s$s.json
  [ -f "$out" ] || $PY run.py --method text_mas --task safety_train \
      --prompt $(prompt_of $ARCH) --n_agents $N_AGENTS $(agent_models_flag $ARCH $SETTING) \
      --think --temperature 0.8 --top_p 0.95 --max_new_tokens 4096 \
      --no_shuffle --reserve_tail 0 --max_samples $N_QUERIES --seed $s \
      --model_name $BACKBONE --run_id collect-$NAME-s$s --quiet
done
mkdir -p ckpt
$PY tools/build_fitset.py --src results/collect-$NAME-s*.json \
    --dst ckpt/fitset-$NAME.pt --n_agents $N_AGENTS --adapt_roles all
# Rollout reward + per-agent message condition (Sec. 4 / App. C.3).
$PY tools/judge_fitset.py ckpt/fitset-$NAME.pt
