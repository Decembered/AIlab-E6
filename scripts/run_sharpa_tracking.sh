#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"

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

check_sharpa() {
  python scripts/check_dexhand_support.py --dexhand sharpa --side left
  python scripts/check_dexhand_support.py --dexhand sharpa --side left --load-asset
}

retarget_sharpa() {
  stdbuf -oL -eL python main/dataset/mano2dexhand.py \
    --data_idx 0f900@0 \
    --side left \
    --dexhand sharpa \
    --headless \
    --iter "${RETARGET_ITERS:-7000}" \
    2>&1 | tee "logs/mano2dexhand_0f900_left_sharpa_$(date +%Y%m%d_%H%M%S).log"
}

train_sharpa_smoke() {
  stdbuf -oL -eL python main/rl/train.py \
    task=ResDexHand \
    dexhand=sharpa \
    side=LH \
    headless=true \
    num_envs="${NUM_ENVS:-256}" \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    "dataIndices=[0f900@0]" \
    actionsMovingAverage=0.4 \
    experiment=sharpa_smoke \
    max_iterations="${MAX_ITERATIONS:-1}" \
    rl_train.params.config.save_frequency=1 \
    wandb_activate="${WANDB_ACTIVATE:-true}" \
    wandb_project="${WANDB_PROJECT:-ho-tracker-baseline}" \
    2>&1 | tee "logs/train_sharpa_smoke_0f900_$(date +%Y%m%d_%H%M%S).log"
}

train_sharpa_full() {
  stdbuf -oL -eL python main/rl/train.py \
    task=ResDexHand \
    dexhand=sharpa \
    side=LH \
    headless=true \
    num_envs="${NUM_ENVS:-4096}" \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    "dataIndices=[0f900@0]" \
    actionsMovingAverage=0.4 \
    experiment=sharpa_baseline \
    wandb_activate="${WANDB_ACTIVATE:-true}" \
    wandb_project="${WANDB_PROJECT:-ho-tracker-baseline}" \
    "${@:2}" \
    2>&1 | tee "logs/train_sharpa_full_0f900_$(date +%Y%m%d_%H%M%S).log"
}

rollout_sharpa() {
  python main/rl/eval_rollout.py \
    --tag "${TAG:-sharpa_baseline}" \
    --dexhand sharpa \
    --gpu_id "${ROLLOUT_GPU_ID:-4}" \
    --extra num_rollouts_to_run="${NUM_ROLLOUTS_TO_RUN:-1024}" \
            num_rollouts_to_save="${NUM_ROLLOUTS_TO_SAVE:-4}" \
            save_successful_rollouts_only=false \
    2>&1 | tee "logs/eval_rollout_sharpa_0f900_$(date +%Y%m%d_%H%M%S).log"
}

score() {
  set +e
  python main/rl/eval_score.py --tag "${SCORE_TAG:-dump_sharpa_baseline_}" --dexhand sharpa \
    2>&1 | tee "logs/eval_score_sharpa_$(date +%Y%m%d_%H%M%S).log"
  local code="${PIPESTATUS[0]}"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "eval_score exited with code ${code}. Short smoke runs may have no successful rollouts."
  fi
}

case "${MODE}" in
  check)
    check_sharpa
    ;;
  retarget)
    check_sharpa
    retarget_sharpa
    ;;
  smoke)
    check_sharpa
    retarget_sharpa
    train_sharpa_smoke
    rollout_sharpa
    score
    ;;
  train)
    train_sharpa_full "$@"
    ;;
  rollout)
    rollout_sharpa
    ;;
  score)
    score
    ;;
  *)
    echo "Usage: $0 {check|retarget|smoke|train|rollout|score}" >&2
    exit 2
    ;;
esac

