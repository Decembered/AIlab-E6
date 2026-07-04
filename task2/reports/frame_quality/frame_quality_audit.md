# 帧级质量审计报告

- 输入轨迹：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/trajectories/hand_traj.npz`
- 输入 JSON：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/trajectories/mediapipe_landmarks.json`
- mask 目录：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/masks`
- 总帧数：182
- 有效帧：182 / 182 (100.00%)
- 插值帧：0
- CSV：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/frame_metrics.csv`

## 序列级指标

- quality_score：{'min': 0.9461156725883484, 'p05': 0.9558365941047668, 'median': 0.9782912731170654, 'p95': 0.9918906688690186, 'max': 0.9965959191322327, 'mean': 0.9764806628227234}
- temporal_jump_score：{'min': 0.0, 'p05': 0.12364140152931213, 'median': 0.3713555932044983, 'p95': 0.9982684254646301, 'max': 1.5692588090896606, 'mean': 0.42870765924453735}
- bone_length_error：{'min': 0.001152382930740714, 'p05': 0.0015764770796522498, 'median': 0.002791483886539936, 'p95': 0.003727610455825925, 'max': 0.004079287871718407, 'mean': 0.0027616210281848907}
- wrist_step：{'min': 0.0, 'p05': 0.0004464532248675823, 'median': 0.0013409173116087914, 'p95': 0.0036046188324689865, 'max': 0.005666391924023628, 'mean': 0.001548008294776082}
- fingertip_step_max：{'min': 0.0, 'p05': 0.0006710032466799021, 'median': 0.002252408303320408, 'p95': 0.005763830151408911, 'max': 0.007763710338622332, 'mean': 0.0026521775871515274}
- mask_area：{'min': 8774.0, 'p05': 9452.75, 'median': 13047.5, 'p95': 15567.2998046875, 'max': 16408.0, 'mean': 12789.4775390625}
- bbox_area：{'min': 7750.70849609375, 'p05': 8532.17578125, 'median': 13803.390625, 'p95': 18087.25, 'max': 19254.21484375, 'mean': 13450.296875}

## Top 风险帧

| frame | time | risk | reasons | quality | temporal | wrist_step | fingertip_max | bone | mask_area |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32 | 2.133 | critical | fingertip_step_outlier | 0.9909 | 0.3381 | 0.001221 | 0.006808 | 0.002898 | 15575.0 |
| 33 | 2.200 | critical | fingertip_step_outlier | 0.9860 | 0.4869 | 0.001758 | 0.006686 | 0.002850 | 16106.0 |
| 34 | 2.267 | critical | fingertip_step_outlier | 0.9892 | 0.2946 | 0.001064 | 0.007180 | 0.002611 | 16291.0 |
| 35 | 2.333 | critical | fingertip_step_outlier | 0.9898 | 0.1948 | 0.000703 | 0.007764 | 0.002307 | 16378.0 |
| 43 | 2.867 | critical | temporal_jump | 0.9709 | 0.8609 | 0.003109 | 0.004039 | 0.001262 | 14972.0 |
| 66 | 4.400 | critical | low_confidence;bone_length_outlier | 0.9535 | 0.4663 | 0.001684 | 0.002959 | 0.003733 | 14009.0 |
| 67 | 4.467 | critical | low_confidence;bone_length_outlier | 0.9511 | 0.5529 | 0.001996 | 0.004078 | 0.003965 | 13829.0 |
| 68 | 4.533 | critical | bone_length_outlier | 0.9697 | 0.2777 | 0.001003 | 0.004269 | 0.004079 | 13651.0 |
| 69 | 4.600 | critical | bone_length_outlier | 0.9824 | 0.4114 | 0.001486 | 0.003377 | 0.003998 | 13272.0 |
| 70 | 4.667 | critical | bone_length_outlier | 0.9846 | 0.7157 | 0.002584 | 0.003953 | 0.003945 | 13090.0 |
| 71 | 4.733 | critical | bone_length_outlier | 0.9762 | 0.1665 | 0.000601 | 0.005684 | 0.004054 | 12730.0 |
| 72 | 4.800 | critical | bone_length_outlier | 0.9718 | 0.5482 | 0.001979 | 0.003849 | 0.003976 | 12501.0 |
| 73 | 4.867 | critical | low_confidence;bone_length_outlier | 0.9461 | 0.3730 | 0.001347 | 0.003456 | 0.003955 | 11978.0 |
| 74 | 4.933 | critical | low_confidence;bone_length_outlier | 0.9510 | 0.5450 | 0.001968 | 0.003261 | 0.004018 | 11892.0 |
| 75 | 5.000 | critical | low_confidence;bone_length_outlier | 0.9498 | 0.2178 | 0.000786 | 0.001995 | 0.003768 | 11693.0 |
| 78 | 5.200 | critical | low_confidence | 0.9514 | 0.6002 | 0.002167 | 0.001651 | 0.002793 | 11450.0 |
| 91 | 6.067 | critical | low_confidence | 0.9577 | 0.2519 | 0.000909 | 0.001538 | 0.002645 | 11302.0 |
| 94 | 6.267 | critical | low_confidence | 0.9558 | 0.1546 | 0.000558 | 0.001728 | 0.002343 | 10859.0 |
| 95 | 6.333 | critical | low_confidence | 0.9583 | 0.6420 | 0.002318 | 0.002034 | 0.002214 | 10624.0 |
| 97 | 6.467 | critical | low_confidence | 0.9581 | 0.1606 | 0.000580 | 0.001337 | 0.002251 | 10153.0 |

## 图表

- quality scores：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/quality_scores.png`
- temporal motion：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/temporal_motion.png`
- bone length error：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/bone_length_error.png`
- mask area：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/mask_area.png`
- mask alignment：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/mask_alignment.png`
- keyframe panel：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/reports/frame_quality/figures/quality_keyframe_panel.jpg`

## 说明

这些 step / velocity 指标来自 MediaPipe non-metric world landmarks，只用于同序列内部相对质量审计，不能解释为真实米制位移。
