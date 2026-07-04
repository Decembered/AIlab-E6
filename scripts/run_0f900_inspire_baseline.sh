#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

CONDA_ROOT="${CONDA_ROOT:-/home/autovla/miniconda3}"
ENV_NAME="${ENV_NAME:-maniptrans}"
REPO_DIR="${REPO_DIR:-/data/autovla/ho_tracker_challenge/HO-Tracker-Baseline-Challenge}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_DIR}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p logs

run_retarget() {
  stdbuf -oL -eL python main/dataset/mano2dexhand.py \
    --data_idx 0f900@0 \
    --side left \
    --dexhand inspire \
    --headless \
    --iter 7000 \
    2>&1 | tee "logs/mano2dexhand_0f900_left_inspire_$(date +%Y%m%d_%H%M%S).log"
}

run_train_smoke() {
  stdbuf -oL -eL python main/rl/train.py \
    task=ResDexHand \
    dexhand=inspire \
    side=LH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    "dataIndices=[0f900@0]" \
    actionsMovingAverage=0.4 \
    experiment=baseline \
    max_iterations=2 \
    rl_train.params.config.save_frequency=1 \
    2>&1 | tee "logs/train_baseline_0f900_inspire_LH_smoke_$(date +%Y%m%d_%H%M%S).log"
}

run_train_full() {
  stdbuf -oL -eL python main/rl/train.py \
    task=ResDexHand \
    dexhand=inspire \
    side=LH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    "dataIndices=[0f900@0]" \
    actionsMovingAverage=0.4 \
    experiment=baseline \
    "$@" \
    2>&1 | tee "logs/train_baseline_0f900_inspire_LH_full_$(date +%Y%m%d_%H%M%S).log"
}

run_rollout_smoke() {
  python main/rl/eval_rollout.py \
    --tag baseline \
    --dexhand inspire \
    --gpu_id 4 \
    --extra num_rollouts_to_run=1024 num_rollouts_to_save=4 save_successful_rollouts_only=false \
    2>&1 | tee "logs/eval_rollout_baseline_0f900_inspire_smoke_$(date +%Y%m%d_%H%M%S).log"
}

run_score() {
  set +e
  python main/rl/eval_score.py --tag dump_baseline_ --dexhand inspire \
    2>&1 | tee "logs/eval_score_baseline_0f900_inspire_$(date +%Y%m%d_%H%M%S).log"
  local code="${PIPESTATUS[0]}"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "eval_score exited with code ${code}. This is expected for short smoke runs without successful rollouts."
  fi
}

case "${MODE}" in
  retarget)
    run_retarget
    ;;
  smoke)
    run_retarget
    run_train_smoke
    run_rollout_smoke
    run_score
    ;;
  train)
    run_train_full "${@:2}"
    ;;
  rollout)
    run_rollout_smoke
    ;;
  score)
    run_score
    ;;
  *)
    echo "Usage: $0 {retarget|smoke|train|rollout|score}" >&2
    exit 2
    ;;
esac

