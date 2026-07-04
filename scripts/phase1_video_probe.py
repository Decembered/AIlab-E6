"""Phase 1: Video frame extraction + camera calibration verification."""
import sys, os, pickle
import numpy as np

SEQ = 'data/human_demo/weigh_bread__2026_0701_0044_30'

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system(f"{sys.executable} -m pip install opencv-python -q")
    import cv2

# 1. Video properties
print("=" * 60)
print("VIDEO PROPERTIES — weigh_bread")
print("=" * 60)
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    cap = cv2.VideoCapture(f'{SEQ}/video/{cam}.mkv')
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps > 0 else 0
    cap.release()
    print(f"  {cam}: {w}x{h}, {fps:.1f} fps, {n_frames} frames, {duration:.1f}s")

# 2. Camera calibration
print()
print("=" * 60)
print("CAMERA CALIBRATION")
print("=" * 60)
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    intr = pickle.load(open(f'{SEQ}/camera_calib/{cam}/cam_intr.pkl', 'rb'))
    extr = pickle.load(open(f'{SEQ}/camera_calib/{cam}/cam_extr.pkl', 'rb'))
    print(f"\n  {cam}:")
    print(f"    Intrinsic type: {type(intr).__name__}")
    if isinstance(intr, np.ndarray):
        print(f"    Intrinsic:\n{intr}")
    else:
        print(f"    Intrinsic: {intr}")
    print(f"    Extrinsic type: {type(extr).__name__}")
    if isinstance(extr, np.ndarray):
        print(f"    Extrinsic:\n{extr}")
    else:
        print(f"    Extrinsic: {extr}")

# 3. Extract sample frames
print()
print("=" * 60)
print("EXTRACTING FRAMES (every 5th frame)")
print("=" * 60)
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    cam_dir = os.path.join(OUT_DIR, cam)
    os.makedirs(cam_dir, exist_ok=True)
    cap = cv2.VideoCapture(f'{SEQ}/video/{cam}.mkv')
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            cv2.imwrite(os.path.join(cam_dir, f'frame_{count:06d}.jpg'), frame)
            saved += 1
        count += 1
    cap.release()
    print(f"  {cam}: {saved} frames saved (every 5th of {count} total)")

print()
print("Done!")
