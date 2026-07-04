# heikesong 附件清点记录

记录时间：2026-07-04  
位置：

- 解压目录：`/home/ruan/research/Hackthon/heikesong/heikesong`
- 原始压缩包：`/home/ruan/Downloads/heikesong.zip`

## 总体结论

`heikesong` 不是 Track E 的 IsaacGym/灵巧手附件，而是一个 **Wan2.1 视频生成模型推理缓存优化** 附件包。它包含：

- Wan2.1 视频生成源码副本；
- SenCache 加速/缓存脚本；
- 预计算 sensitivity 文件；
- 3 条 prompt；
- baseline 与 SenCache 的 6 个对比输出视频；
- Python/conda 环境描述。

该附件对应赛事页面里的“视频生成模型推理缓存优化挑战（A卷·个人）”方向，而不是 `Video2Motion2Action：灵巧操作技能迁移（A卷·团队）`。

## 规模

- 解压目录大小：约 28 MB
- 原始 zip：约 27 MB
- 文件数：64
- 目录数：16
- 主要文件类型：
  - Python：39 个
  - MP4：6 个
  - Markdown：4 个
  - JSON 文本 prompt：2 个
  - NPZ sensitivity：1 个
  - JPG 输入示例：1 个
  - YAML/TOML/requirements/shell 若干

## 目录结构

```text
heikesong/heikesong/
  environment.yml
  Wan2.1/
    README.md
    INSTALL.md
    LICENSE.txt
    requirements.txt
    pyproject.toml
    generate.py
    sencache.py
    sensitivity_wan21.npz
    test_prompt.json
    examples/i2v_input.JPG
    gradio/
    tests/
    wan/
      configs/
      distributed/
      modules/
      utils/
  SenCache/
    Wan2.1/
      README.md
      sencache.py
      sensitivity_calculation.py
      test_prompt.json
  test_results_video/
    wan2.1/
      case_01_baseline.mp4
      case_02_baseline.mp4
      case_03_baseline.mp4
    sencache/
      case_01_sencache.mp4
      case_02_sencache.mp4
      case_03_encache.mp4
```

`__MACOSX` 与 `.DS_Store` 是 macOS 打包产生的元数据，可忽略。

## 核心内容

### Wan2.1

`Wan2.1/` 是 Wan2.1 视频生成代码，支持：

- Text-to-Video：`t2v-1.3B`、`t2v-14B`
- Image-to-Video：`i2v-14B`
- First/Last Frame to Video：`flf2v-14B`
- VACE：`vace-1.3B`、`vace-14B`
- Text-to-Image：`t2i-14B`
- 单卡、多卡、FSDP、xDiT/Ulysses/Ring 推理入口
- Gradio demo 入口

主要入口：

- `Wan2.1/generate.py`：原始 Wan2.1 推理入口
- `Wan2.1/sencache.py`：已合入 SenCache 的推理入口
- `Wan2.1/tests/test.sh`：多任务测试脚本，需要本地模型目录和 GPU 数

### SenCache

`SenCache/Wan2.1/README.md` 说明用法：将 `sencache.py` 和 sensitivity 权重放入 Wan2.1 根目录后运行。

示例命令：

```bash
python sencache.py \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size 832*480 \
  --output_dir ./output \
  --offload_model True \
  --prompt_file ./test_prompt.json \
  --frame_num 81 \
  --sample_steps 50 \
  --sencache_K 3 \
  --sencache_thresh_main 2 \
  --sencache_thresh_start 0.045
```

关键参数：

- `--prompt_file`：prompt 文本文件路径。文件实际是“一行一个 prompt”，不是标准 JSON 数组。
- `--sencache_K`：最大连续跳过/缓存复用步数。
- `--sencache_thresh_start`：早期 step 的缓存误差阈值。
- `--sencache_thresh_main`：后期 step 的缓存误差阈值。
- `--no_sencache`：关闭 SenCache，走标准推理。

`SenCache/Wan2.1/sensitivity_calculation.py` 用于重新计算 sensitivity，当前脚本限定 `t2v` 任务。

### Sensitivity 文件

文件：`Wan2.1/sensitivity_wan21.npz`

内容：

- `timesteps`：shape `(50,)`，int64，范围 92 到 999
- `J_x_norm`：shape `(50,)`，float32，范围约 1.0287 到 31.9340
- `J_t_norm`：shape `(50,)`，float32，范围约 1.9452 到 30.9178

`sencache.py` 会从当前目录读取 `./sensitivity_wan21.npz`。

### Prompt 文件

文件：

- `Wan2.1/test_prompt.json`
- `SenCache/Wan2.1/test_prompt.json`

二者内容相同，包含 3 条英文 prompt：

1. 滑板选手在夕阳下踢翻下楼梯。
2. 金毛犬沿浅海岸线奔跑并溅起水花。
3. 人手在工作台上组装机械怀表。

注意：虽然扩展名是 `.json`，但内容不是 JSON，而是纯文本每行一个 prompt。

### 样例视频

共有 6 个 MP4：

| 文件 | 分辨率 | 帧率 | 帧数 | 时长 |
| --- | --- | --- | --- | --- |
| `test_results_video/wan2.1/case_01_baseline.mp4` | 832x480 | 16 fps | 81 | 5.0625s |
| `test_results_video/wan2.1/case_02_baseline.mp4` | 832x480 | 16 fps | 81 | 5.0625s |
| `test_results_video/wan2.1/case_03_baseline.mp4` | 832x480 | 16 fps | 81 | 5.0625s |
| `test_results_video/sencache/case_01_sencache.mp4` | 832x480 | 16 fps | 81 | 5.0625s |
| `test_results_video/sencache/case_02_sencache.mp4` | 832x480 | 16 fps | 81 | 5.0625s |
| `test_results_video/sencache/case_03_encache.mp4` | 832x480 | 16 fps | 81 | 5.0625s |

命名上 `case_03_encache.mp4` 可能是 `case_03_sencache.mp4` 的拼写遗漏。

## 环境要求

`environment.yml` 声明：

- Python 3.10
- PyTorch 2.6.*
- torchvision 0.21.*
- pytorch-cuda 12.4
- diffusers >= 0.31.0
- transformers >= 4.49.0
- accelerate >= 1.1.1
- opencv-python >= 4.9
- gradio >= 5.0
- flash-attn == 2.8.3
- modelscope、dashscope、decord 等

`Wan2.1/requirements.txt` 声明 torch >= 2.4.0、torchvision >= 0.19.0、flash_attn 等。

## 当前可运行性判断

附件本身没有包含 Wan2.1 模型权重，例如：

- `Wan2.1-T2V-1.3B`
- `Wan2.1-T2V-14B`
- `Wan2.1-I2V-14B-480P`
- `Wan2.1-I2V-14B-720P`

因此当前目录可以做代码审查、参数分析、已有视频对比和环境规划，但不能直接跑完整推理。下载模型权重会很大，需要单独确认。

## 和当前黑客松工作的关系

如果做 Track E 的 `Video2Motion2Action：灵巧操作技能迁移`，这个附件不是直接需要的 IsaacGym/Sharpa/HO-Tracker 资料。

如果切换到“视频生成模型推理缓存优化挑战”，这个附件就是核心起点，下一步应围绕：

1. 搭建 `sencache-wan21` 环境；
2. 下载最小可跑的 `Wan2.1-T2V-1.3B` 权重；
3. 复现 3 条 prompt 的 baseline 与 SenCache 输出；
4. 记录速度、显存、跳步次数、视频质量差异；
5. 调 `sencache_K`、`sencache_thresh_start`、`sencache_thresh_main` 做 ablation。
