<div align='center'>

# Train Locally, Learn Together: Distributed Multi-Agent Training via ADMM

[![preprint](https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B)](https://arxiv.org/abs/XXXX.XXXXX)
[![Issues](https://img.shields.io/badge/Issues-Welcome!-fbbf24)](../../issues)
[![python](https://img.shields.io/badge/python-3.10-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](./LICENSE)

</div>

<table align="center">
  <tr>
    <td align="center">
      <img src="./images/teaser.png" alt="teaser" style="width: 1000px;"/>
      <br>
      <em style="font-size: 11px;"><strong style="font-size: 11px;">Figure 1:</strong> Schematic overview of centralized and distributed MAS training.</em>
    </td>
  </tr>
</table>

This is the official code repository for the paper [**Train Locally, Learn Together: Distributed Multi-Agent Training via ADMM**](https://arxiv.org/abs/XXXX.XXXXX).

Each agent of the MAS trains its own LoRA copy on role-local data, and agents only exchange parameters through ADMM consensus (ring all-reduce or gossip). No central trainer is needed.

## News

- 📢 [Oct 2026] We released the code and posted our paper on [arXiv](https://arxiv.org/abs/XXXX.XXXXX)! 🚀

## Installation

```bash
git clone https://github.com/OPTML-Group/Distributed-MAS
cd Distributed-MAS

conda create -n dmas python=3.10 -y
conda activate dmas
pip install -r requirements.txt

bash scripts/download_safety_data.sh
```

Models are pulled from Hugging Face: `Qwen/Qwen3-8B` (agents), `mlabonne/Qwen3-8B-abliterated` (compromised agent), `cais/HarmBench-Llama-2-13b-cls` and `Qwen/Qwen2.5-7B-Instruct` (judges).

## Quick Start

Safety training of the hierarchical MAS (H-MAS) with a compromised Synthesizer:

```bash
# collect rollouts and build the per-agent training data
bash scripts/safety/collect.sh hmas compromised

# train with ADMM (or: centralized / local / nodual)
bash scripts/safety/train.sh hmas compromised admm

# evaluate and judge
bash scripts/eval_safety.sh hmas-admm --method mas_trained \
    --ckpt ckpt/hmas-compromised-n4-admm-last.pt --prompt hierarchical \
    --agent_models synthesizer=mlabonne/Qwen3-8B-abliterated
```

Use `smas` / `sequential_v2` / `solver=...` for the sequential MAS, and `benign` for the setting without a compromised agent.

## Reproducing the Paper

The training arms differ only in how agents are coupled in `train.py`. Default hyper-parameters follow Table A2.

| Arm | `train.sh` | `train.py` flags |
|---|---|---|
| ADMM | `admm` | `--consensus ring` (or `--consensus gossip --gossip_steps τ`) |
| Centralized | `centralized` | `--pooled --consensus coordinator` |
| ADMM-Local | `local` | `--local_only` |
| ADMM-NoDual | `nodual` | `--no_dual` |

### Safety (Table 1, Table 2)

Run once per architecture (`smas`, `hmas`) and setting (`benign`, `compromised`). The example is the compromised H-MAS; for S-MAS use `--prompt sequential_v2` and `solver=...`, and for the benign setting drop `--agent_models`.

```bash
ARCH=hmas; SETTING=compromised
SYS="--prompt hierarchical --agent_models synthesizer=mlabonne/Qwen3-8B-abliterated"

# Base MAS (no training)
bash scripts/eval_safety.sh $ARCH-$SETTING-base --method text_mas $SYS

# training data: 680 queries x 8 rollouts, labeled with the HarmBench reward
bash scripts/safety/collect.sh $ARCH $SETTING

# Centralized and ADMM (Table 1), ADMM-Local and ADMM-NoDual (Table 2)
for M in centralized admm local nodual; do
  bash scripts/safety/train.sh $ARCH $SETTING $M
done

# evaluation (ADMM-Local has no consensus variable, so each agent uses its own adapter)
for M in centralized admm nodual; do
  bash scripts/eval_safety.sh $ARCH-$SETTING-$M --method mas_trained \
      --ckpt ckpt/$ARCH-$SETTING-n4-$M-last.pt $SYS
done
bash scripts/eval_safety.sh $ARCH-$SETTING-local --method mas_trained \
    --ckpt ckpt/$ARCH-$SETTING-n4-local.pt --deploy local $SYS
```

The Single-LLM rows use `--method baseline` (add `BACKBONE=mlabonne/Qwen3-8B-abliterated` for the compromised one):

```bash
bash scripts/eval_safety.sh single --method baseline
BACKBONE=mlabonne/Qwen3-8B-abliterated bash scripts/eval_safety.sh single-unaligned --method baseline
```

Numbers are read from `results/judged/<tag>-<benchmark>.harmbench.json` (ASR), `results/judged/<tag>-xstest.xstest-refusal.json` (XSTest refusal rate on safe prompts) and `results/<tag>-math500.json` (accuracy).

### Untrained systems (Fig. 2, Tables A1, A3, A5)

These only need `run.py`, through `eval_safety.sh` (`TASKS=harmbench` for the appendix tables):

```bash
# Table A1: another compromised position (any role, or several separated by commas)
TASKS=harmbench bash scripts/eval_safety.sh smas-verifier --method text_mas --prompt sequential_v2 \
    --agent_models verifier=mlabonne/Qwen3-8B-abliterated

# Table A3: a different number of agents
TASKS=harmbench bash scripts/eval_safety.sh hmas-n6 --method text_mas --prompt hierarchical --n_agents 6 \
    --agent_models synthesizer=mlabonne/Qwen3-8B-abliterated

# Table A5: Qwen3-4B backbone
BACKBONE=Qwen/Qwen3-4B TASKS=harmbench bash scripts/eval_safety.sh smas4b --method text_mas \
    --prompt sequential_v2 --agent_models solver=mlabonne/Qwen3-4B-abliterated
```

### Gossip (Fig. 5)

An eight-agent S-MAS with a compromised Solver, where nodes run τ rounds of gossip on a ring:

```bash
export N_AGENTS=8
bash scripts/safety/collect.sh smas compromised
for TAU in 1 2 4 8 16; do
  CONSENSUS=gossip GOSSIP_STEPS=$TAU SUFFIX=-tau$TAU bash scripts/safety/train.sh smas compromised admm
  TASKS=harmbench bash scripts/eval_safety.sh smas8-tau$TAU --method mas_trained \
      --ckpt ckpt/smas-compromised-n8-admm-tau$TAU.pt --deploy local --n_agents 8 \
      --prompt sequential_v2 --agent_models solver=mlabonne/Qwen3-8B-abliterated
done
```

The final primal residual is in `results/smas-compromised-n8-admm-tau<τ>-train.json`.

### Reasoning (Fig. 4)

Each benchmark is evaluated on the first 256 questions of a fixed shuffle with 8192 tokens per agent; training questions come from the train split (GSM8K) or from the rows after the evaluated ones.

```bash
TASK=mmlu_pro    # gsm8k: use --anchor_split train instead of --anchor_offset 256
EVAL="--task $TASK --max_samples 256 --reserve_tail 0 --think --max_new_tokens 8192"

python run.py --method baseline $EVAL --run_id $TASK-single
python run.py --method text_mas --prompt sequential_v2 $EVAL --run_id $TASK-base

python train.py --task $TASK --anchor_offset 256 --prompt sequential_v2 --think \
    --reward outcome --anchor_pool 64 --anchors 32 --anchor_pass_band 0,1 \
    --recollect_every 1 --k_outer 8 --rho 0.1 --max_new_tokens 8192 \
    --consensus ring --out ckpt/$TASK-admm.pt          # Centralized: --pooled --consensus coordinator
python run.py --method mas_trained --ckpt ckpt/$TASK-admm-last.pt --prompt sequential_v2 $EVAL --run_id $TASK-admm
```

GPQA (198 questions) and MATH-500 (500) are small, so lower `--max_samples`, `--anchor_offset`, `--anchors` and `--anchor_pool` accordingly.

## Cite This Work

If you found our code or paper helpful, please cite our work~

```
@article{chen2026train,
  title={Train Locally, Learn Together: Distributed Multi-Agent Training via ADMM},
  author={Chen, Yiwei and Wang, Changsheng and Lang, Yicheng and Qu, Qing and Hero, Alfred O. and Liu, Sijia},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
