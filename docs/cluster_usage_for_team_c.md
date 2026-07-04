# 队员 C 集群使用指南

记录时间：2026-07-04  
来源：

- `/home/ruan/Downloads/P集群Hackthon.pdf`
- `/home/ruan/Downloads/阿里云（4090）集群__黑客松集群入门.pdf`

## 你的目标

你负责团队题的子任务 3：物体形状重建与 IsaacGym Asset，以及和 A/B 队员的接口协同。

现在连接集群的目的不是立刻训练，而是做最小验证：

1. 确认官方镜像/集群里是否有 IsaacGym Preview 4。
2. 跑通 `joint_monkey.py` 或等价 IsaacGym smoke test。
3. 找到 HO-Tracker / Sharpa / dexhand / baseline / asset / 数据路径。
4. 确认物体 mesh/URDF 能在 IsaacGym 加载。
5. 约定和 A/B 的交付接口：物体 mesh、collision mesh、URDF、scale、mass/inertia、object pose trajectory。

不要一上来跑长训练、下载大模型、重装系统环境或把数据放在临时盘里。

## 两套集群怎么选

### 优先：阿里云 4090 DSW

适合：

- 交互式调试；
- VNC 可视化；
- IsaacGym 可视化验证；
- 小规模 asset 加载和 rollout；
- 查看官方镜像里已有的 HO-Tracker / IsaacGym 内容。

文档建议：

- 登录前连接上海人工智能实验室 VPN。
- 登录入口：`https://qke4t1cn.aliyunidaas.com/portal/user/page/index.html`
- 进入“我的应用：阿里云4090”。
- 搜索 `PAI`，进入“人工智能平台 PAI”。
- 区域选择：`cn-beijing`。
- 使用“交互式建模 DSW”做调试。
- 新建 DSW 时系统盘建议选择“云盘”，容量提高到 100 GiB。
- 启用 SSH，把本机 SSH 公钥加入 DSW。
- 对 CPU/内存性能敏感时启用 CPU 亲和性。
- 重要文件放到 CPFS 数据集，不要只放临时 DSW/DLC 本地盘。

### 辅助：P 集群

适合：

- Slurm 作业；
- 共享存储；
- 后续需要队列化运行的任务。

文档要点：

```bash
ssh [username]@jump.pjlab.org.cn
cinfo -g
squeue
squeue -u $USER
scancel <jobid>
```

Hackathon 开放的是 `wam_data` 分区资源。提交任务建议使用 `--quotatype=auto`。

存储：

- Petrel-OSS / Ceph：大数据和 checkpoint。
- Petrel-FS：`/mnt/petrelfs/$USER`，适合代码和环境，小容量需要申请扩容。
- 公用环境通常在 `/mnt/petrelfs/share`。

P 集群管理节点近期不能直接访问公网，GitHub/环境安装通常需要代理；代理和 Ceph 可能冲突，建议分开终端使用。

## 今天建议做的事情

### 1. 创建 DSW 实例

在阿里云 PAI 里新建 DSW：

- GPU：4090，先 1 张即可；
- CPU/内存：按默认或 1 GPU 对应 16 核、32-64G 的规格；
- 系统盘：云盘，100 GiB；
- 启用 SSH；
- 添加本机 SSH 公钥；
- 优先级：文档提到调试 DSW 可设为 9，若界面可配置则使用 9；
- 如果有官方镜像选择，优先找 `ho-tracker-v3` 或与题目相关的镜像。

赛事页面曾提到的镜像名：

```text
pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/zhanxinyu:ho-tracker-v3
```

如果界面没有这个镜像，先用默认镜像启动，再查是否已有题目环境。

### 2. 登录 DSW

在 DSW 实例页面复制 SSH 公网访问命令。形式大概是：

```bash
ssh root@<DSW_PUBLIC_IP> -p <DSW_PUBLIC_PORT>
```

连接后先做只读检查：

```bash
hostname
whoami
pwd
nvidia-smi
df -h
free -h
python --version
python3 --version
python3.8 --version || true
conda env list || true
```

### 3. 检查题目环境和 IsaacGym

```bash
find ~ /opt /root -maxdepth 5 \( \
  -iname '*isaacgym*' -o \
  -iname '*isaac_gym*' -o \
  -iname '*HO-Tracker*' -o \
  -iname '*ManipTrans*' -o \
  -iname '*Sharpa*' -o \
  -iname '*dexhand*' \
\) 2>/dev/null | sort | head -200
```

重点找：

```text
~/opt/isaac_gym/isaacgym
HO-Tracker
ManipTrans
Sharpa
dexhand
```

如果存在 IsaacGym：

```bash
cd ~/opt/isaac_gym/isaacgym/python/examples
python3.8 joint_monkey.py
```

如果没有可视化环境，先跑导入检查：

```bash
python3.8 - <<'PY'
import sys
print(sys.version)
import isaacgym
print("isaacgym import ok", isaacgym)
PY
```

### 4. 给 C 任务做 asset 预检

找物体资产和 URDF：

```bash
find ~ /root /workspace /mnt -maxdepth 6 \( \
  -iname '*.obj' -o \
  -iname '*.ply' -o \
  -iname '*.stl' -o \
  -iname '*.urdf' -o \
  -iname '*.xml' \
\) 2>/dev/null | sort | head -200
```

找评测脚本和训练入口：

```bash
find ~ /root /workspace /mnt -maxdepth 6 -type f | \
  grep -Ei 'eval|score|rollout|train|tracking|retarget|asset|urdf|sharpa|inspire' | \
  head -200
```

记录你需要给 A/B 的接口：

- object name；
- mesh 路径；
- collision mesh 路径；
- URDF 路径；
- scale；
- mass / inertia；
- initial pose；
- per-frame object pose trajectory；
- IsaacGym load test 截图/日志；
- 如果失败，失败原因和修复计划。

### 5. VNC 可视化

如果官方镜像已经有 TurboVNC：

```bash
/opt/TurboVNC/bin/vncserver :0 -geometry 1280x720
```

本机开端口转发：

```bash
ssh -L 5900:127.0.0.1:5900 root@<DSW_PUBLIC_IP> -p <DSW_PUBLIC_PORT>
```

然后用 RealVNC Viewer 连接：

```text
127.0.0.1:5900
```

如果镜像没有 TurboVNC，文档给的是 apt 安装方案，但这会改系统环境。优先确认官方镜像是否已经配置好；确实需要装时再单独记录。

## 必须保存的日志

在集群上建一个轻量记录目录：

```bash
EXP=cluster_probe_$(date +%F_%H%M)
mkdir -p "$EXP"
{
  date
  hostname
  whoami
  pwd
  nvidia-smi
  df -h
  free -h
  python --version || true
  python3 --version || true
  python3.8 --version || true
  conda env list || true
} 2>&1 | tee "$EXP/env_check.txt"
```

IsaacGym 检查：

```bash
{
  find ~ /opt /root -maxdepth 5 \( -iname '*isaacgym*' -o -iname '*isaac_gym*' \) 2>/dev/null
  python3.8 - <<'PY'
try:
    import isaacgym
    print("isaacgym import ok", isaacgym)
except Exception as e:
    print("isaacgym import failed:", type(e).__name__, e)
PY
} 2>&1 | tee "$EXP/isaacgym_check.txt"
```

最后把 `EXP` 目录同步/下载回来，或者放进持久化 CPFS/Petrel 存储。

## 不要做的事

- 不要在临时盘只保存唯一副本。
- 不要下载大模型或完整大数据，除非团队确认需要。
- 不要默认跑长训练。
- 不要随意 `sudo apt upgrade` 或重装 conda。
- 不要把 AD 密码、AK/SK、token 写进仓库或日志。
- 不要把 VLA/策略输出描述成可直接上真机；真实执行需要安全过滤、规划器、控制器、状态估计和急停链路。

## 你今天完成的标准

今天只需要做到：

1. DSW 能 SSH 登录；
2. `nvidia-smi` 正常；
3. 找到或确认缺失 IsaacGym Preview 4；
4. 找到或确认缺失 HO-Tracker / Sharpa / dexhand / baseline；
5. 跑通 `joint_monkey.py` 或至少 `import isaacgym`；
6. 保存 `env_check.txt` 和 `isaacgym_check.txt`；
7. 把路径和结果发给 A/B，约定 C 的 asset 输出格式。
