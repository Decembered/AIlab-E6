#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${HOME}/.local/bin/uv"
ISAACLAB_ENV="${ROOT}/.venvs/isaaclab-py310"
ISAACGYM_ENV="${ROOT}/.venvs/isaacgym-py38"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_isaac_stack.sh prepare
      Create isolated Python environments:
        .venvs/isaaclab-py310  - Isaac Sim / Isaac Lab 4.x path
        .venvs/isaacgym-py38   - legacy Isaac Gym Preview 4 path

  scripts/setup_isaac_stack.sh install-isaacgym /path/to/IsaacGym_Preview_4_Package.tar.gz
      Install classic Isaac Gym Preview 4 from a locally downloaded NVIDIA package.
      This does not download the package because the NVIDIA license must be accepted manually.

  scripts/setup_isaac_stack.sh install-video2motion-deps
      Install the Python 3.8 dependencies needed by the Video2Motion2Action
      IsaacGym / HO-Tracker baseline path, without installing Isaac Gym itself.

  scripts/setup_isaac_stack.sh install-isaacsim45-rl --i-approve-large-download
      Install Isaac Sim 4.5 Python packages with the RL bundle into .venvs/isaaclab-py310.
      This can download more than 5 GB, so it requires the explicit flag above.

  scripts/setup_isaac_stack.sh check
      Run scripts/check_isaac_env.py.
EOF
}

ensure_uv() {
  if [[ -x "${UV}" ]]; then
    return
  fi
  python3 -m pip install --user uv
}

prepare_envs() {
  ensure_uv
  if [[ ! -x "${ISAACLAB_ENV}/bin/python" ]]; then
    "${UV}" venv --seed "${ISAACLAB_ENV}" --python 3.10
  else
    echo "Reusing ${ISAACLAB_ENV}"
  fi
  if [[ ! -x "${ISAACGYM_ENV}/bin/python" ]]; then
    "${UV}" venv --seed "${ISAACGYM_ENV}" --python 3.8
  else
    echo "Reusing ${ISAACGYM_ENV}"
  fi
}

install_isaacgym() {
  local package_path="${1:-}"
  if [[ -z "${package_path}" ]]; then
    echo "Missing path to Isaac Gym Preview 4 package or extracted directory." >&2
    exit 2
  fi

  prepare_envs
  mkdir -p "${ROOT}/.tools/isaacgym"

  local src
  if [[ -d "${package_path}" ]]; then
    src="${package_path}"
  elif [[ -f "${package_path}" ]]; then
    tar -xf "${package_path}" -C "${ROOT}/.tools/isaacgym"
    src="$(find "${ROOT}/.tools/isaacgym" -maxdepth 3 -type d -name python | head -1 | xargs -r dirname)"
  else
    echo "Path not found: ${package_path}" >&2
    exit 2
  fi

  if [[ ! -d "${src}/python" ]]; then
    echo "Could not find Isaac Gym python/ directory under: ${src}" >&2
    exit 3
  fi

  "${ISAACGYM_ENV}/bin/python" -m pip install --upgrade pip
  "${ISAACGYM_ENV}/bin/python" -m pip install "numpy<1.24" "gym==0.23.1" "Pillow" "matplotlib"
  "${ISAACGYM_ENV}/bin/python" -m pip install -e "${src}/python"

  echo
  echo "Installed legacy Isaac Gym package into ${ISAACGYM_ENV}."
  echo "Smoke test after activation:"
  echo "  source ${ISAACGYM_ENV}/bin/activate"
  echo "  python ${src}/python/examples/joint_monkey.py --sim_device cpu --graphics_device_id 0"
  echo
  echo "Note: GPU PhysX is likely incompatible with RTX 4060/Ada; prefer Isaac Lab for GPU RL."
}

install_video2motion_deps() {
  prepare_envs
  "${ISAACGYM_ENV}/bin/python" -m pip install \
    "numpy<1.24" \
    "gym==0.23.1" \
    "Pillow" \
    "matplotlib" \
    "pyyaml" \
    "tqdm" \
    "opencv-python-headless<4.9" \
    "trimesh[easy]" \
    "scipy<1.11" \
    "tensorboard"

  "${ISAACGYM_ENV}/bin/python" -m pip --default-timeout=1000 --retries 10 install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==1.13.1+cpu" \
    "torchvision==0.14.1+cpu"

  echo
  echo "Installed Video2Motion2Action baseline dependencies into ${ISAACGYM_ENV}."
  echo "Isaac Gym Preview 4 itself still requires the local NVIDIA package or contest image copy."
}

install_isaacsim45_rl() {
  local approval="${1:-}"
  if [[ "${approval}" != "--i-approve-large-download" ]]; then
    echo "Isaac Sim 4.5 packages and extension caches may exceed 5 GB." >&2
    echo "Re-run with: scripts/setup_isaac_stack.sh install-isaacsim45-rl --i-approve-large-download" >&2
    exit 2
  fi

  prepare_envs
  "${ISAACLAB_ENV}/bin/python" -m pip install --upgrade pip
  "${ISAACLAB_ENV}/bin/python" -m pip install "isaacsim[rl]==4.5.0" --extra-index-url https://pypi.nvidia.com

  echo
  echo "Installed Isaac Sim 4.5 RL bundle into ${ISAACLAB_ENV}."
  echo "Before first run, accept NVIDIA Omniverse EULA if appropriate:"
  echo "  export OMNI_KIT_ACCEPT_EULA=YES"
  echo "Then run:"
  echo "  source ${ISAACLAB_ENV}/bin/activate"
  echo "  python -c 'import isaacsim; print(\"isaacsim import ok\")'"
}

cmd="${1:-}"
case "${cmd}" in
  prepare)
    prepare_envs
    ;;
  install-isaacgym)
    shift
    install_isaacgym "$@"
    ;;
  install-video2motion-deps)
    install_video2motion_deps
    ;;
  install-isaacsim45-rl)
    shift
    install_isaacsim45_rl "$@"
    ;;
  check)
    python3 "${ROOT}/scripts/check_isaac_env.py"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
