# Task2 Curated Visualization Videos

This directory contains selected GitHub-safe MP4 artifacts for judge review and team integration. Full generated output directories are intentionally not committed because they contain raw frames, per-frame figures, masks, and videos that are too large for normal Git history.

## Summary

- `summary/weigh_drink_yykx_left_side2_success_review.mp4`: compact success-demo review for the standard side-2 sequence.

## Standard Side-2 Sequence

Source run: `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2`.

- `standard_side2/mediapipe_overlay.mp4`: 2D hand keypoint overlay.
- `standard_side2/hand_mask_overlay.mp4`: generated visible hand mask overlay.
- `standard_side2/hand_3d_skeleton.mp4`: MediaPipe world-landmark 3D skeleton replay.
- `standard_side2/task2_scoring_review.mp4`: combined scoring review video for the standard result.

## Failure And Backup Views

- `failure_side1/mediapipe_overlay.mp4`: side-1 failure-case 2D overlay for comparison.
- `failure_side1/hand_mask_overlay.mp4`: side-1 mask overlay for failure analysis.
- `top_view/mediapipe_overlay.mp4`: top-view backup 2D overlay.
- `top_view/hand_3d_skeleton.mp4`: top-view backup 3D skeleton replay.

## Not Included

The following are intentionally excluded from Git:

- Full `task2/outputs/videos/task2_scoring_review.mp4` because it exceeds GitHub's 100 MB single-file limit.
- Large by-view scoring review videos over or near 100 MB.
- Raw videos, extracted frames, per-frame figures, generated masks, checkpoints, and caches.

For full local outputs, use the timestamped handoff package under `/mnt/workspace/TEMP-FILE-STATION` in the shared workspace.
