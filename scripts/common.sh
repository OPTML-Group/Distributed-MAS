# Shared settings for the experiment scripts (sourced, not executed).
# Override any variable from the environment, e.g. `GPU=1 bash scripts/...`.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
BACKBONE=${BACKBONE:-Qwen/Qwen3-8B}
UNALIGNED=${UNALIGNED:-mlabonne/Qwen3-8B-abliterated}   # compromised agent's backbone

# arch -> --prompt, and the compromised agent of each architecture (Fig. 2a)
prompt_of()      { case $1 in smas) echo sequential_v2 ;; hmas) echo hierarchical ;; esac; }
compromised_of() { case $1 in smas) echo solver ;; hmas) echo synthesizer ;; esac; }
# compromised setting -> --agent_models flag (empty for the benign setting)
agent_models_flag() {  # arch setting
  [ "$2" = compromised ] && echo "--agent_models $(compromised_of $1)=$UNALIGNED" || true
}
