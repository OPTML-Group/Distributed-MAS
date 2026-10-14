#!/usr/bin/env bash
# Safety evaluation of one system (App. A.1, C.4): generate on every safety
# benchmark, then judge offline.
#
#   bash scripts/eval_safety.sh TAG <run.py system flags...>
#
# e.g. a trained compromised H-MAS:
#   bash scripts/eval_safety.sh hmas-compromised-admm --method mas_trained \
#       --ckpt ckpt/hmas-compromised-admm-last.pt --prompt hierarchical \
#       --agent_models synthesizer=mlabonne/Qwen3-8B-abliterated
#
# Outputs results/<TAG>-<task>.json and results/judged/<TAG>-<task>.<judge>.json:
#   HarmBench ASR             harmbench judge on <TAG>-harmbench
#   StrongReject/JBB/OR-Bench  compliance rate (xstest-refusal judge) and harmbench-judge ASR
#   XSTest over-refusal        refusal rate on the 250 safe prompts (xstest-refusal judge)
#   MATH-500 utility           accuracy in results/<TAG>-math500.json
set -euo pipefail
source "$(dirname "$0")/common.sh"
TAG=$1; shift
TASKS=${TASKS:-"harmbench strongreject jbb_harmful orbench_toxic xstest math500"}
ORBENCH_N=${ORBENCH_N:-200}
GEN="--think --temperature 0.6 --top_p 0.95 --seed 42 --max_new_tokens ${MAX_NEW_TOKENS:-4096} --model_name $BACKBONE --reserve_tail 0"

for task in $TASKS; do
  n=-1; [ "$task" = orbench_toxic ] && n=$ORBENCH_N
  out=results/$TAG-$task.json
  [ -f "$out" ] || $PY run.py "$@" --task $task --max_samples $n $GEN --run_id $TAG-$task --quiet
done

harm=(); refusal=()
for task in $TASKS; do
  case $task in
    harmbench) harm+=(results/$TAG-$task.json) ;;
    strongreject|jbb_harmful|orbench_toxic) harm+=(results/$TAG-$task.json); refusal+=(results/$TAG-$task.json) ;;
    xstest) refusal+=(results/$TAG-$task.json) ;;
  esac
done
[ ${#harm[@]} -gt 0 ] && $PY tools/judge.py --judge harmbench "${harm[@]}"
[ ${#refusal[@]} -gt 0 ] && $PY tools/judge.py --judge xstest-refusal "${refusal[@]}"
echo "done: $TAG"
