# Polaris launch prompt — Headline binary-vs-diamond GRPO

Paste everything inside the fenced block below into a Claude Code session running
on your Polaris login node (`polaris.alcf.anl.gov`). Fill the four `<<<...>>>`
slots first.

```
You are on an ALCF Polaris login node. Kick off the FRS headline
binary-vs-diamond experiment as a PBS job.

## Fill-me-in
- Project/allocation:       <<<YOUR_ALCF_PROJECT>>>
- Queue:                    <<<QUEUE (preemptable | prod | debug-scaling)>>>
- Scratch / work root:      <<<e.g. /eagle/YOUR_ALCF_PROJECT/$USER>>>
- FRS repo path on Polaris: <<<e.g. /eagle/YOUR_ALCF_PROJECT/$USER/FormalRewardSignal>>>
  (if not present: see RSYNC below)

## Step 1 — sync FRS from my workstation (only if FRS isn't on Polaris yet)
On MY WORKSTATION (not Polaris), I will run:
    rsync -av --exclude=outputs/ --exclude=.git/ --exclude='*.gguf' \
      /home/espencer2/FormalRewardSignal/ \
      ericspencer@polaris.alcf.anl.gov:<<<FRS_PATH>>>/
Alternatively, the 500-train / 50-dev dataset alone is on GitHub at
https://github.com/LUC-AI4FM/ChatTLA/tree/claude/elated-hugle-8806ef/data/frs_tla_ralph_gen
and can be cloned instead if FRS hasn't been published to GitHub yet.

## Step 2 — build the PBS submit script at <FRS_PATH>/scripts/polaris/headline.pbs

Make it a single-node, 4x A100 GPU job, 6 hours. Two GRPO runs back-to-back
in the same job: first reward=binary, then reward=diamond. Same seed, same
data, same everything else. Checkpoint every 20 steps. Save logs per-run.

Concretely:

#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -q <<<QUEUE>>>
#PBS -A <<<YOUR_ALCF_PROJECT>>>
#PBS -N frs_headline_bin_vs_dia
#PBS -j oe
#PBS -o logs/headline.$PBS_JOBID.out

cd <<<FRS_PATH>>>
module load conda
conda activate frs || { conda env create -f environment.yml -n frs && conda activate frs; }

# Point training + eval at the 550-row dataset we just published.
# If FRS was rsynced, benchmarks/tla/train.jsonl already has it. Otherwise:
#   curl -sL https://raw.githubusercontent.com/LUC-AI4FM/ChatTLA/claude/elated-hugle-8806ef/data/frs_tla_ralph_gen/train.jsonl -o benchmarks/tla/train.jsonl
#   curl -sL https://raw.githubusercontent.com/LUC-AI4FM/ChatTLA/claude/elated-hugle-8806ef/data/frs_tla_ralph_gen/dev.jsonl   -o benchmarks/tla/dev.jsonl

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

mkdir -p outputs/headline_binary outputs/headline_diamond logs

# --- Run 1: binary reward ---
python -m frs.cli train configs/experiments/headline_binary_vs_diamond.yaml \
  --set grpo.reward=binary --set grpo.max_steps=200 --set grpo.mode=fullspec \
  --set data.train_split=benchmarks/tla/train.jsonl \
  --set data.eval_split=benchmarks/tla/dev.jsonl \
  --out outputs/headline_binary \
  2>&1 | tee logs/headline_binary.$PBS_JOBID.log

# --- Run 2: diamond reward (same seed, same data) ---
python -m frs.cli train configs/experiments/headline_binary_vs_diamond.yaml \
  --set grpo.reward=diamond --set grpo.max_steps=200 --set grpo.mode=fullspec \
  --set data.train_split=benchmarks/tla/train.jsonl \
  --set data.eval_split=benchmarks/tla/dev.jsonl \
  --out outputs/headline_diamond \
  2>&1 | tee logs/headline_diamond.$PBS_JOBID.log

# --- Post: vacuity-gap sweep over ckpts saved every 20 steps ---
python scripts/async/reference_vs_vacuous.py \
  --splits dev \
  --out outputs/headline_eval/ref_vs_vac.jsonl \
  --table outputs/headline_eval/ref_vs_vac_table.md

## Step 3 — submit and watch

    mkdir -p <<<FRS_PATH>>>/logs
    cd <<<FRS_PATH>>>
    qsub scripts/polaris/headline.pbs
    # returns JOBID; then:
    qstat -u $USER
    tail -F logs/headline.<JOBID>.out

## Step 4 — when both runs finish, generate the figure

    python scripts/ablations/weight_sensitivity.py \
      --runs outputs/headline_binary outputs/headline_diamond \
      --out  outputs/headline_eval/vacuity_gap.png

Report back the JOBID and the first 50 lines of the PBS .out file so I can
verify the run started cleanly.
```
