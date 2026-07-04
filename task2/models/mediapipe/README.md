# MediaPipe 模型目录

当前环境中的 `mediapipe 0.10.35` 是 Tasks-only 版本，没有旧版 `mp.solutions.hands` 接口。运行 `task2/scripts/02_run_mediapipe_hands.py` 时需要 HandLandmarker `.task` 模型文件。

推荐文件名：

- `hand_landmarker.task`

推荐放置路径：

- `task2/models/mediapipe/hand_landmarker.task`

该模型是 MediaPipe 官方 Hand Landmarker task 模型，体积较小，不属于 HaMeR/SAM2 这类大模型权重。若不能联网下载，请手动放置后重新运行脚本。
