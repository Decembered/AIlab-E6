#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/autovla/miniconda3}"
ENV_NAME="${ENV_NAME:-maniptrans}"
BASE_DIR="${BASE_DIR:-/data/autovla/ho_tracker_challenge}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/HO-Tracker-Baseline-Challenge}"
ISAACGYM_DIR="${ISAACGYM_DIR:-${BASE_DIR}/IsaacGym_Preview_4_Package/isaacgym/python}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8
fi

conda activate "${ENV_NAME}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip

python -m pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

if [[ ! -d "${ISAACGYM_DIR}" ]]; then
  echo "Missing IsaacGym Preview 4 python directory: ${ISAACGYM_DIR}" >&2
  echo "Download IsaacGym Preview 4 and extract it under ${BASE_DIR}/IsaacGym_Preview_4_Package first." >&2
  exit 1
fi

cd "${ISAACGYM_DIR}"
python -m pip install -e . --no-deps

cd "${REPO_DIR}"

grep -v -E "^(isaacgym|torch|torchvision|torchaudio|bps_torch|manotorch|chamfer_distance)" requirements.txt > /tmp/ho_tracker_requirements_min.txt
python -m pip install -r /tmp/ho_tracker_requirements_min.txt

python -m pip install "fvcore~=0.1.5"
python -m pip install --no-index --no-cache-dir pytorch3d==0.7.3 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

python -m pip install -e . --no-deps
python -m pip install -e maniptrans_envs --no-deps

cat <<'MSG'

Base environment is ready.

You still need bps_torch, chamfer_distance, and manotorch. If direct GitHub access works:

  pip install git+https://github.com/KailinLi/bps_torch.git
  pip install git+https://github.com/otaheri/chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846
  pip install git+https://github.com/lixiny/manotorch.git@933be97d6fa05c729656669640084251ce644b0a

On xmu75, GitHub cloning was unstable, so these packages were installed from downloaded source archives.
MSG

