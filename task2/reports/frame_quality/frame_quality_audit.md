# 帧级质量审计报告

- 输入轨迹：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/trajectories/hand_traj.npz`
- 输入 JSON：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/trajectories/mediapipe_landmarks.json`
- mask 目录：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/masks`
- 总帧数：534
- 有效帧：534 / 534 (100.00%)
- 插值帧：0
- CSV：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/frame_metrics.csv`

## 序列级指标

- quality_score：{'min': 0.9827730059623718, 'p05': 0.9890462160110474, 'median': 0.9944167137145996, 'p95': 0.997831404209137, 'max': 0.9992327094078064, 'mean': 0.9940402507781982}
- temporal_jump_score：{'min': 0.0, 'p05': 0.02544254995882511, 'median': 0.11856149137020111, 'p95': 0.9998170733451843, 'max': 1.4220778942108154, 'mean': 0.3084977865219116}
- bone_length_error：{'min': 0.0012762366095557809, 'p05': 0.001582800643518567, 'median': 0.004030367359519005, 'p95': 0.006350989453494549, 'max': 0.007815880700945854, 'mean': 0.003721945220604539}
- wrist_step：{'min': 0.0, 'p05': 0.0002611939562484622, 'median': 0.0012171557173132896, 'p95': 0.010264151729643345, 'max': 0.014599093236029148, 'mean': 0.0031670473981648684}
- fingertip_step_max：{'min': 0.0, 'p05': 0.00041183584835380316, 'median': 0.0013414080021902919, 'p95': 0.02050802670419216, 'max': 0.023414477705955505, 'mean': 0.005684144329279661}
- mask_area：{'min': 8496.0, 'p05': 10055.650390625, 'median': 11916.0, 'p95': 13445.1005859375, 'max': 14091.0, 'mean': 11815.8779296875}
- bbox_area：{'min': 3567.5224609375, 'p05': 3729.54931640625, 'median': 4813.2177734375, 'p95': 6760.19921875, 'max': 7846.76171875, 'mean': 5105.3828125}

## Top 风险帧

| frame | time | risk | reasons | quality | temporal | wrist_step | fingertip_max | bone | mask_area |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 0.267 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9934 | 0.0623 | 0.000640 | 0.001003 | 0.003776 | 11798.0 |
| 5 | 0.333 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9935 | 0.0509 | 0.000522 | 0.000810 | 0.003823 | 11777.0 |
| 6 | 0.400 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9944 | 0.0167 | 0.000171 | 0.000524 | 0.003872 | 11701.0 |
| 7 | 0.467 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9927 | 0.1927 | 0.001978 | 0.002007 | 0.004101 | 11778.0 |
| 8 | 0.533 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9935 | 0.1550 | 0.001592 | 0.001420 | 0.004294 | 11781.0 |
| 9 | 0.600 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9957 | 0.0874 | 0.000898 | 0.001055 | 0.004410 | 11768.0 |
| 10 | 0.667 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9938 | 0.1163 | 0.001194 | 0.001204 | 0.004489 | 11815.0 |
| 11 | 0.733 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9948 | 0.0573 | 0.000589 | 0.000637 | 0.004459 | 11766.0 |
| 12 | 0.800 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9946 | 0.0710 | 0.000729 | 0.000864 | 0.004459 | 11875.0 |
| 13 | 0.867 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9936 | 0.1792 | 0.001839 | 0.001166 | 0.004506 | 11851.0 |
| 14 | 0.933 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9948 | 0.1501 | 0.001541 | 0.001171 | 0.004526 | 11868.0 |
| 15 | 1.000 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9940 | 0.0573 | 0.000589 | 0.000830 | 0.004552 | 11955.0 |
| 16 | 1.067 | critical | bone_length_outlier;wrist_step_outlier;fingertip_step_outlier;mask_bbox_mismatch | 0.9946 | 0.4881 | 0.005011 | 0.012758 | 0.004274 | 11874.0 |
| 17 | 1.133 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9941 | 0.1390 | 0.001427 | 0.001505 | 0.004034 | 12019.0 |
| 18 | 1.200 | critical | bone_length_outlier;mask_bbox_mismatch;handedness_switch | 0.9924 | 0.1162 | 0.001193 | 0.000958 | 0.004132 | 11999.0 |
| 19 | 1.267 | critical | bone_length_outlier;mask_bbox_mismatch;handedness_switch | 0.9914 | 0.0952 | 0.000977 | 0.001437 | 0.004073 | 12116.0 |
| 20 | 1.333 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9928 | 0.1748 | 0.001795 | 0.001177 | 0.003988 | 12013.0 |
| 21 | 1.400 | critical | bone_length_outlier;wrist_step_outlier;fingertip_step_outlier;mask_bbox_mismatch | 0.9920 | 0.7571 | 0.007772 | 0.011438 | 0.004117 | 11995.0 |
| 22 | 1.467 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9914 | 0.0848 | 0.000871 | 0.001201 | 0.004167 | 12059.0 |
| 23 | 1.533 | critical | bone_length_outlier;mask_bbox_mismatch | 0.9923 | 0.1426 | 0.001464 | 0.001468 | 0.004197 | 12143.0 |

## 图表

- quality scores：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/quality_scores.png`
- temporal motion：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/temporal_motion.png`
- bone length error：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/bone_length_error.png`
- mask area：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/mask_area.png`
- mask alignment：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/mask_alignment.png`
- keyframe panel：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/figures/quality_keyframe_panel.jpg`

## 说明

这些 step / velocity 指标来自 MediaPipe non-metric world landmarks，只用于同序列内部相对质量审计，不能解释为真实米制位移。
