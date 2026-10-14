#!/usr/bin/env bash
# Train one arm of Table 1 / Table 2 from a labelled fit set (scripts/safety/collect.sh).
#
#   bash scripts/safety/train.sh {smas|hmas} {benign|compromised} {admm|centralized|local|nodual}
#
# Settings are those of Table A2 (the train.py defaults): DPO + KL (beta 0.1 each),
# rank-16 B-only LoRA on all projections of all 36 layers, L=16 Adam steps
# (lr 1e-3, clip 0.01, batch 8) per x-step, rho=0.03, ring all-reduce,
# T = 24 rounds (H-MAS) / 20 rounds (S-MAS). All agents are trainable.
# FIT_MODE=sft or rsft switches the role-local objective to SFT.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
ARCH=$1; SETTING=$2; METHOD=$3
N_AGENTS=${N_AGENTS:-4}
NAME=${NAME:-$ARCH-$SETTING-n$N_AGENTS}
case $ARCH in hmas) T=${T:-24} ;; smas) T=${T:-20} ;; esac
case $METHOD in
  admm)        FLAGS="--consensus ${CONSENSUS:-ring} --topology ${TOPOLOGY:-ring} --gossip_steps ${GOSSIP_STEPS:-1}" ;;
  centralized) FLAGS="--pooled --consensus coordinator" ;;
  local)       FLAGS="--local_only" ;;
  nodual)      FLAGS="--no_dual" ;;
  *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac
mkdir -p ckpt
$PY train.py --task safety_train --prompt $(prompt_of $ARCH) --n_agents $N_AGENTS --think \
    --adapt_roles all --reward judge --fit_mode ${FIT_MODE:-dpo} --fitset_cache ckpt/fitset-$NAME.pt \
    --k_outer $T $FLAGS ${EXTRA:-} --model_name $BACKBONE --out ckpt/$NAME-$METHOD${SUFFIX:-}.pt

