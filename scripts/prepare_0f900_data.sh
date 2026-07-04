#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/autovla/ho_tracker_challenge/HO-Tracker-Baseline-Challenge}"
HF_BASE="${HF_BASE:-https://hf-mirror.com/datasets/kelvin34501/HO-Tracker-Challenge/resolve/main}"
SEQ_DIR="${REPO_DIR}/data/HO-Tracker/data/test_sample/h1o1/0f900@0"

mkdir -p "${SEQ_DIR}/left_urdf"
mkdir -p "${REPO_DIR}/data/HO-Tracker/data/train_sample"

download() {
  local rel="$1"
  local dst="${SEQ_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -s "${dst}" ]]; then
    echo "exists: ${dst}"
    return
  fi
  echo "download: ${rel}"
  curl -L --connect-timeout 20 --max-time 180 --retry 3 --retry-delay 2 \
    -o "${dst}" \
    "${HF_BASE}/HO-Tracker/data/test_sample/h1o1/0f900@0/${rel}"
}

download left_hand.pkl
download left_obj.pkl
download left_urdf/scan.ply

cat > "${SEQ_DIR}/left_urdf/scan.urdf" <<'EOF'
<?xml version="1.0"?>
<robot name="hotracker_scan">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="scan.ply" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="scan.ply" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
EOF

find "${SEQ_DIR}" -maxdepth 3 -type f -ls

