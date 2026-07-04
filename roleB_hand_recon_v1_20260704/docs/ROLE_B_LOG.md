# Role B Hand Reconstruction Log


## 2026-07-04T09:21:14+00:00 - Start / Isolation

- Created independent role B workspace: .
- Stopped only  training PIDs identified by cwd/cmd.
- Will avoid modifying  shared files, official scoring logic, shared conda env, and A-role /.
- Preferred light GPU for role B experiments: ; CPU-only for data inspection.

## 2026-07-04T09:21:34+00:00 - Dataset Requirement Check

- README says to download HO-Tracker-Challenge from Hugging Face and extract to data/HO-Tracker and data/human_demo.
- Initial data directory size is tiny; checking whether full data exists elsewhere.

## 2026-07-04T09:22:05+00:00 - Correction To Initial Log

- Workspace: /data/autovla/ho_tracker_challenge/roleB_hand_recon.
- Stopped only /data/autovla/beyondmimic/whole_body_tracking training PIDs identified by cwd/cmd.
- Will avoid modifying shared HO-Tracker-Baseline-Challenge files, official scoring logic, shared conda env, and A-role runs/logs.
- Preferred light GPU for role B experiments: CUDA_VISIBLE_DEVICES=5; CPU-only for data inspection.
- Data present in baseline repo: only sample 0f900@0 pkl/ply/urdf/retargeting files; no videos and no data/human_demo directory found.

## 2026-07-04T09:24:43+00:00 - HF Dataset Metadata

- Cloned HO-Tracker-Challenge metadata under roleB/hf_metadata with GIT_LFS_SKIP_SMUDGE=1.
- Dataset file list includes HO-Tracker test samples plus human_demo sequences with camera_side_1, camera_side_2, camera_top videos, camera intr/extr pkl, and pose_3d.hdf5.
- Next check determines whether files are actual payloads or LFS pointers before downloading any large video.

## 2026-07-04T09:25:28+00:00 - Initial Role B Target Sequence

- Selected human_demo/weigh_bread__2026_0701_0044_30 for first role B pass because bread is one of the required object categories and it has three camera views plus pose_3d.hdf5.
- Plan: download only this session first, inspect videos/calibration, then build hand reconstruction/overlay artifacts in roleB outputs.

## 2026-07-04T09:26:12+00:00 - Download Single Human Demo Session

- Running low-priority git lfs pull for human_demo/weigh_bread__2026_0701_0044_30 only (~272 MB videos + tiny calibration files).

## 2026-07-04T09:28:08+00:00 - Download Attempt Failed Via HF Mirror

- git lfs pull for weigh_bread failed because us.aws.cdn.hf-mirror.org could not be resolved.
- No GPU was used; files remain as LFS pointers. Next fallback is official huggingface.co remote for the same include path.

## 2026-07-04T09:29:03+00:00 - Network Check For Dataset Download

- Official huggingface.co timed out from xmu75.
- hf-mirror.com page works, but LFS payload redirects to us.aws.cdn.hf-mirror.org, which failed DNS resolution.
- Checking whether proxy_on/proxy_off exists before retrying.

## 2026-07-04T09:30:50+00:00 - Direct Resolve Download

- Direct hf-mirror resolve URLs are reachable, so downloading selected session into roleB/data without touching the main repo.
- Direct resolve download completed for selected weigh_bread session.

## 2026-07-04T09:31:46+00:00 - Download Completed

- Downloaded selected session into /data/autovla/ho_tracker_challenge/roleB_hand_recon/data/human_demo/weigh_bread__2026_0701_0044_30.
- Files: 3 MKV videos (~86 MB, 83 MB, 90 MB), 6 camera calibration pkl files, pose_3d.hdf5.
- Next: verify media/calibration and run a lightweight hand trajectory baseline in roleB outputs.

## 2026-07-04T09:33:30+00:00 - MediaPipe Baseline Start

- Running CPU MediaPipe Hands on all three views, max 235 frames, resize width 640.
- Outputs are under roleB/outputs/weigh_bread__2026_0701_0044_30; no writes to main HO-Tracker runs/logs.
- MediaPipe baseline completed; summary generated in terminal and will be added to next log section.

## 2026-07-04T09:34:30+00:00 - MediaPipe Baseline Result

- camera_side_1: 235/235 valid hand detections.
- camera_side_2: 233/235 valid hand detections.
- camera_top: 235/235 valid hand detections.
- Raw 2D/3D MediaPipe trajectories, smoothed 3D skeleton trajectories, and overlay videos are under roleB/outputs/weigh_bread__2026_0701_0044_30.
- Next: generate thumbnail QA and a simple 3D skeleton playback; then inspect whether overlay quality is acceptable enough as baseline evidence.

## 2026-07-04T09:36:06+00:00 - Proxy 2D Mask Baseline

- Generated proxy visible-hand masks for camera_side_1 and camera_top using dilated convex hulls of MediaPipe 21 landmarks.
- This is a baseline mask definition, not a final segmentation model; outputs are hand_mask_proxy_overlay.mp4 and sampled mask PNGs under mask_proxy/.
- camera_side_1 and camera_top both have 235/235 proxy masks.

## 2026-07-04T09:37:02+00:00 - QA Artifacts Generated

- Generated overlay thumbnails for frames 0, 80, 160, 234 across all three cameras.
- Generated side_1 3D skeleton playback: camera_side_1_pybullet_hand_playback.mp4.
- Generated side_1/top proxy mask overlays and sampled binary masks.
- Next: visual QA and write a compact role B status summary.

## 2026-07-04T09:41:07+00:00 - Visual QA Summary

- Inspected side_1/top frame 80 thumbnails locally. Skeletons align with the right hand.
- Proxy mask aligns spatially but overlaps the bread; recorded as known baseline limitation.
- Wrote compact status summary: /data/autovla/ho_tracker_challenge/roleB_hand_recon/ROLE_B_STATUS.md.

## 2026-07-04T10:07:04+00:00 - Role B Pipeline Run

- sessions=weigh_bread__2026_0701_0044_30, weigh_bread__left__2026_0701_0046_02, weigh_drink_yykx__2026_0701_0051_12, grasp_pipette_press__2026_0701_0028_11
- download=True
- max_frames=235, resize_width=640

## 2026-07-04T10:07:04+00:00 - Download Session weigh_bread__2026_0701_0044_30

- Downloading only this human_demo session via hf-mirror direct resolve URLs.
- Using resume and retry; files are stored under roleB/data, not the main HO-Tracker repo.

## 2026-07-04T10:07:34+00:00 - Processed Session weigh_bread__2026_0701_0044_30

- task=None side=None obj=None
- best_cam=camera_top; valid frames: camera_side_1 235/235, camera_side_2 233/235, camera_top 235/235
- triangulation=world_to_camera median_reprojection_px=2.31 valid_points=4935
- outputs=/data/autovla/ho_tracker_challenge/roleB_hand_recon/outputs/weigh_bread__2026_0701_0044_30

## 2026-07-04T10:07:34+00:00 - Download Session weigh_bread__left__2026_0701_0046_02

- Downloading only this human_demo session via hf-mirror direct resolve URLs.
- Using resume and retry; files are stored under roleB/data, not the main HO-Tracker repo.

## 2026-07-04T10:09:02+00:00 - Processed Session weigh_bread__left__2026_0701_0046_02

- task=None side=None obj=None
- best_cam=camera_side_2; valid frames: camera_side_1 201/217, camera_side_2 217/217, camera_top 217/217
- triangulation=world_to_camera median_reprojection_px=2.12 valid_points=4557
- outputs=/data/autovla/ho_tracker_challenge/roleB_hand_recon/outputs/weigh_bread__left__2026_0701_0046_02

## 2026-07-04T10:09:02+00:00 - Download Session weigh_drink_yykx__2026_0701_0051_12

- Downloading only this human_demo session via hf-mirror direct resolve URLs.
- Using resume and retry; files are stored under roleB/data, not the main HO-Tracker repo.

## 2026-07-04T10:10:34+00:00 - Processed Session weigh_drink_yykx__2026_0701_0051_12

- task=None side=None obj=None
- best_cam=camera_side_1; valid frames: camera_side_1 235/235, camera_side_2 55/235, camera_top 235/235
- triangulation=world_to_camera median_reprojection_px=1.72 valid_points=4935
- outputs=/data/autovla/ho_tracker_challenge/roleB_hand_recon/outputs/weigh_drink_yykx__2026_0701_0051_12

## 2026-07-04T10:10:34+00:00 - Download Session grasp_pipette_press__2026_0701_0028_11

- Downloading only this human_demo session via hf-mirror direct resolve URLs.
- Using resume and retry; files are stored under roleB/data, not the main HO-Tracker repo.

## 2026-07-04T10:12:12+00:00 - Processed Session grasp_pipette_press__2026_0701_0028_11

- task=None side=None obj=None
- best_cam=camera_top; valid frames: camera_side_1 235/235, camera_side_2 205/235, camera_top 235/235
- triangulation=world_to_camera median_reprojection_px=3.67 valid_points=4935
- outputs=/data/autovla/ho_tracker_challenge/roleB_hand_recon/outputs/grasp_pipette_press__2026_0701_0028_11

## HDF5 Metadata Enrichment

- Used maniptrans Python with h5py to fill task_name, side, obj_id, and object-pose frame counts in SESSION_METRICS.csv.

## 2026-07-04T10:15:00+00:00 - Multi-Session Optimization Summary

- Processed four sessions across bread, drink, and pipette tasks.
- Added scripted pipeline, automatic view scoring, proxy masks, thumbnails, metrics table, and multi-view triangulation.
- Best views and reprojection errors are summarized in ROLE_B_STATUS.md and SESSION_METRICS.csv.
