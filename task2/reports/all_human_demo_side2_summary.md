# 全量 human_demo Task2 Baseline 汇总

本报告基于 MediaPipe pseudo-label baseline。所有 3D 仍为 non-metric MediaPipe world landmarks，不是 GT。

- 总视角数：12
- 成功视角数：12
- CSV：`task2/reports/all_human_demo_side2_metrics.csv`

## 每视角指标

| sequence | view | status | frames | valid_ratio | quality_mean | risk_frames | mask_nonempty |
|---|---|---|---:|---:|---:|---:|---:|
| `grasp_drink_yykx_2026_0701_0054_45` | `camera_side_2` | ok | 265 | 78.49% | 0.7108 | 222 | 208 |
| `grasp_pipette_press_2026_0701_0028_11` | `camera_side_2` | ok | 315 | 90.48% | 0.8294 | 314 | 285 |
| `grasp_pipette_rotate_2026_0701_0025_42` | `camera_side_2` | ok | 355 | 93.80% | 0.8925 | 346 | 333 |
| `grasp_pipette_stand_2026_0701_0019_19` | `camera_side_2` | ok | 260 | 100.00% | 0.9397 | 242 | 260 |
| `pipette_rh_beaker_2026_0701_0035_47` | `camera_side_2` | ok | 715 | 94.97% | 0.9211 | 390 | 679 |
| `pipette_rh_beaker_testtube_2026_0701_0039_28` | `camera_side_2` | ok | 534 | 100.00% | 0.9909 | 531 | 534 |
| `weigh_bread_2026_0701_0044_30` | `camera_side_2` | ok | 235 | 100.00% | 0.9664 | 120 | 235 |
| `weigh_bread_left_2026_0701_0046_02` | `camera_side_2` | ok | 217 | 100.00% | 0.9865 | 94 | 217 |
| `weigh_drink_ad_2026_0701_0047_56` | `camera_side_2` | ok | 243 | 37.04% | 0.3105 | 243 | 90 |
| `weigh_drink_ad_left_2026_0701_0049_04` | `camera_side_2` | ok | 209 | 100.00% | 0.9789 | 81 | 209 |
| `weigh_drink_yykx_2026_0701_0051_12` | `camera_side_2` | ok | 242 | 25.62% | 0.2155 | 242 | 62 |
| `weigh_drink_yykx_left_2026_0701_0052_53` | `camera_side_2` | ok | 182 | 100.00% | 0.9765 | 46 | 182 |

## 每序列推荐视角

| sequence | best_run | valid_ratio | quality_mean |
|---|---|---:|---:|
| `grasp_drink_yykx_2026_0701_0054_45` | `grasp_drink_yykx_2026_0701_0054_45_camera_side_2` | 78.49% | 0.7108 |
| `grasp_pipette_press_2026_0701_0028_11` | `grasp_pipette_press_2026_0701_0028_11_camera_side_2` | 90.48% | 0.8294 |
| `grasp_pipette_rotate_2026_0701_0025_42` | `grasp_pipette_rotate_2026_0701_0025_42_camera_side_2` | 93.80% | 0.8925 |
| `grasp_pipette_stand_2026_0701_0019_19` | `grasp_pipette_stand_2026_0701_0019_19_camera_side_2` | 100.00% | 0.9397 |
| `pipette_rh_beaker_2026_0701_0035_47` | `pipette_rh_beaker_2026_0701_0035_47_camera_side_2` | 94.97% | 0.9211 |
| `pipette_rh_beaker_testtube_2026_0701_0039_28` | `pipette_rh_beaker_testtube_2026_0701_0039_28_camera_side_2` | 100.00% | 0.9909 |
| `weigh_bread_2026_0701_0044_30` | `weigh_bread_2026_0701_0044_30_camera_side_2` | 100.00% | 0.9664 |
| `weigh_bread_left_2026_0701_0046_02` | `weigh_bread_left_2026_0701_0046_02_camera_side_2` | 100.00% | 0.9865 |
| `weigh_drink_ad_2026_0701_0047_56` | `weigh_drink_ad_2026_0701_0047_56_camera_side_2` | 37.04% | 0.3105 |
| `weigh_drink_ad_left_2026_0701_0049_04` | `weigh_drink_ad_left_2026_0701_0049_04_camera_side_2` | 100.00% | 0.9789 |
| `weigh_drink_yykx_2026_0701_0051_12` | `weigh_drink_yykx_2026_0701_0051_12_camera_side_2` | 25.62% | 0.2155 |
| `weigh_drink_yykx_left_2026_0701_0052_53` | `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2` | 100.00% | 0.9765 |
