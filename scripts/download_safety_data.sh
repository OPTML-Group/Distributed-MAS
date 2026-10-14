#!/usr/bin/env bash
# Fetch the safety benchmark CSVs into $SAFETY_DATA_DIR (default $HF_HOME/safety_raw
# or ~/.cache/huggingface/safety_raw) and verify them against the copies used in the paper.
set -euo pipefail
DIR=${SAFETY_DATA_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/safety_raw}
mkdir -p "$DIR"
cd "$DIR"
get() {  # url  local-name  sha256
  [ -f "$2" ] || curl -sSL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet - || { echo "checksum mismatch: $2" >&2; exit 1; }
}
HB=https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets
get $HB/harmbench_behaviors_text_all.csv  harmbench_behaviors_text_all.csv  8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc
get $HB/harmbench_behaviors_text_val.csv  harmbench_behaviors_text_val.csv  7bfaaa5e2b9b95002c89e2d33c019f58e9e8496e2fc65d368f1eeb8179e69534
get $HB/harmbench_behaviors_text_test.csv harmbench_behaviors_text_test.csv 75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29
get https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv \
    xstest_prompts.csv 11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a
get https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv \
    harmful_behaviors.csv 6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1
get https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv \
    strongreject_dataset.csv 4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381
get https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv \
    jbb_harmful_behaviors.csv 4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1
get https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-toxic.csv \
    or-bench-toxic.csv 3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057
echo "safety data ready in $DIR"

