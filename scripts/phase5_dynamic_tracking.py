#!/usr/bin/env python3
"""Phase 5: Dynamic-prompt SAM mask tracking + image-plane pose trajectory.

用 tracking-by-detection 替代固定坐标 SAM prompt：
  Frame 0: 已知优质 prompt（来自 phase2_multiview_all.py）
  Frame t: 上一帧有效 mask 质心 → 动态 SAM prompt → 空间选 mask → 质量门控 → pose

输出（每个物体每个相机）：
  - 逐帧 mask NPY 序列
  - object_trajectory_mask_pose.json  （2D image-plane poses + 质量指标）
  - object_trajectory_mask_pose.npz   （poses 数组 + valid 标记）
  - trajectory_quality_report.json    （汇总统计）
  - trajectory_plot.png               （三面板诊断图）
  - mask_sequence_debug/              （关键帧调试叠加图）

用法: python phase5_dynamic_tracking.py --object bread --camera camera_top
"""

import sys, json, argparse, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')

# ── 配置 ────────────────────────────────────────────────────────────────
OBJECTS = {
    'bread': {
        'seq': 'weigh_bread__2026_0701_0044_30',
        'area_range': [3000, 35000],
        'max_centroid_jump_px': 50.0,
        'max_ar_change': 0.6,
    },
    'pipette': {
        'seq': 'grasp_pipette_stand__2026_0701_0019_19',
        'area_range': [5000, 40000],
        'max_centroid_jump_px': 60.0,
        'max_ar_change': 0.7,
    },
    'drink_ad': {
        'seq': 'weigh_drink_ad__2026_0701_0047_56',
        'area_range': [4000, 30000],
        'max_centroid_jump_px': 50.0,
        'max_ar_change': 0.6,
    },
    'drink_yykx': {
        'seq': 'weigh_drink_yykx__2026_0701_0051_12',
        'area_range': [4000, 30000],
        'max_centroid_jump_px': 50.0,
        'max_ar_change': 0.6,
    },
}

FRAME_STRIDE = 3            # 每隔 N 帧处理一帧
MAX_CONSECUTIVE_INVALID = 5  # 连续失败次数超过此值触发重捕获
BOX_EXPAND_MARGIN = 0.25     # 动态 prompt 的 bbox 相对上一帧扩大比例
MAX_DEBUG_OVERLAYS = 12
SPATIAL_WEIGHT = 0.6         # 空间距离权重（vs SAM score）用于选 mask


# ── SAM 工具函数 ────────────────────────────────────────────────────────
def load_sam():
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device='cpu')
    return SamPredictor(sam)


def get_skin_mask(frame_rgb):
    """检测肤色像素，用于负 prompt。"""
    r = frame_rgb[:, :, 0].astype(float)
    g = frame_rgb[:, :, 1].astype(float)
    b = frame_rgb[:, :, 2].astype(float)
    skin = ((r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15) &
            (r > g + 10) & (r > b + 10))
    return skin


def run_sam_multimask(predictor, pos, neg, box):
    """运行 SAM，返回全部 3 个 mask 及其 scores。"""
    all_pts = np.vstack([pos, neg])
    all_labels = np.array([1]*len(pos) + [0]*len(neg))

    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_labels,
        box=box[None, :],
        multimask_output=True,
    )
    return masks, scores


def select_best_mask(masks, scores, expected_centroid, frame_w, frame_h):
    """从 SAM 的 3 个候选 mask 中挑选最佳者。

    策略：综合考虑 SAM score 和空间距离（mask 质心离预期位置的距离）。
    避免选中桌面/背景等大面积区域。
    """
    best_mask = None
    best_score = -1.0
    best_idx = -1
    best_stats = None

    max_dist = np.sqrt(frame_w**2 + frame_h**2)  # 对角线作为归一化因子

    for i in range(len(masks)):
        mask = masks[i]
        # 保留最大连通分量
        cleaned, _ = clean_mask(mask)
        stats = mask_stats(cleaned)
        if stats is None:
            continue

        # 空间距离分数（越近越好）
        dist = np.linalg.norm(stats['centroid'] - expected_centroid)
        spatial_score = 1.0 - min(dist / max_dist, 1.0)

        # 综合分数：SAM score 和空间 score 加权
        sam_score = float(scores[i])
        combined = SPATIAL_WEIGHT * spatial_score + (1 - SPATIAL_WEIGHT) * sam_score

        if combined > best_score:
            best_score = combined
            best_mask = cleaned
            best_idx = i
            best_stats = stats

    return best_mask, best_idx, best_score, best_stats


def clean_mask(mask):
    """只保留最大连通分量。"""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask, 0
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = labeled == (np.argmax(sizes) + 1)
    return largest, n


def mask_stats(mask):
    """返回 mask 的质心、bbox、面积、宽高比。"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ys, xs = np.where(mask)
    area = len(ys)
    w, h = xmax - xmin, ymax - ymin
    ar = w / max(1, h)
    return {
        'centroid': np.array([xs.mean(), ys.mean()]),
        'bbox': np.array([xmin, ymin, xmax, ymax]),
        'area': area,
        'aspect_ratio': ar,
        'width': w,
        'height': h,
    }


def mask_pca_theta(mask, centroid):
    """PCA 主方向角（弧度）。"""
    ys, xs = np.where(mask)
    if len(ys) < 3:
        return 0.0
    centered = np.column_stack([xs - centroid[0], ys - centroid[1]])
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, -1]
    return float(np.arctan2(major_axis[1], major_axis[0]))


# ── Frame-0 prompts（来自 phase2_multiview_all.py）─────────────────────
def get_frame0_prompts(obj, w, h, skin_mask):
    """返回 frame 0 的 (pos, neg, box)。"""
    neg_common = np.array([
        [10, 10], [w-10, 10], [10, h-10], [w-10, h-10],
    ])

    skin_ys, skin_xs = np.where(skin_mask)
    neg_skin = []
    if len(skin_ys) > 0:
        indices = np.linspace(0, len(skin_ys)-1, min(6, len(skin_ys)), dtype=int)
        for idx in indices:
            neg_skin.append([skin_xs[idx], skin_ys[idx]])

    if obj == 'bread':
        pos = np.array([
            [int(w*0.50), int(h*0.48)],
            [int(w*0.46), int(h*0.50)],
            [int(w*0.54), int(h*0.50)],
        ])
        neg = np.array([
            [int(w*0.50), int(h*0.25)],
            [int(w*0.50), int(h*0.72)],
            [int(w*0.30), int(h*0.50)],
            [int(w*0.70), int(h*0.50)],
            [int(w*0.15), int(h*0.50)],
            [int(w*0.85), int(h*0.50)],
        ])
        box = np.array([int(w*0.25), int(h*0.28), int(w*0.75), int(h*0.68)])

    elif obj == 'pipette':
        cx, cy = w//2, h//2
        pos = np.array([
            [cx, cy],
            [int(cx - w*0.12), int(cy - 5)],
            [int(cx + w*0.12), int(cy + 5)],
            [int(cx - w*0.20), int(cy - 3)],
            [int(cx + w*0.20), int(cy + 3)],
        ])
        neg = np.vstack([
            neg_common,
            np.array([[cx, int(cy - h*0.10)],
                      [cx, int(cy + h*0.15)],
                      [cx, int(cy + h*0.25)],
                      [int(cx - w*0.25), cy],
                      [int(cx + w*0.25), cy]]),
        ])
        if neg_skin:
            neg = np.vstack([neg, np.array(neg_skin)])
        box = np.array([int(w*0.22), int(h*0.38), int(w*0.78), int(h*0.58)])

    elif obj in ('drink_ad', 'drink_yykx'):
        pos = np.array([
            [int(w*0.50), int(h*0.45)],
            [int(w*0.48), int(h*0.30)],
            [int(w*0.52), int(h*0.60)],
            [int(w*0.46), int(h*0.50)],
            [int(w*0.54), int(h*0.50)],
        ])
        neg = np.array([
            [int(w*0.50), int(h*0.15)],
            [int(w*0.50), int(h*0.80)],
            [int(w*0.20), int(h*0.45)],
            [int(w*0.80), int(h*0.45)],
        ])
        box = np.array([int(w*0.30), int(h*0.18), int(w*0.70), int(h*0.72)])

    else:
        raise ValueError(f"Unknown object: {obj}")

    return pos, neg, box


# ── 动态 prompts（frame t > 0）──────────────────────────────────────────
def get_dynamic_prompts(prev_stats, prev_centroid_history, w, h, skin_mask):
    """根据上一帧 mask 构建动态 prompt。

    使用 prev_centroid_history（最近几个有效帧的质心列表）来预测运动方向。
    """
    cx, cy = prev_stats['centroid']
    xmin, ymin, xmax, ymax = prev_stats['bbox']

    # 用最近两帧质心估计运动速度
    if len(prev_centroid_history) >= 2:
        v = prev_centroid_history[-1] - prev_centroid_history[-2]
        # 预测下一帧质心
        pred_cx = cx + v[0]
        pred_cy = cy + v[1]
    else:
        pred_cx, pred_cy = cx, cy

    # 扩大 bbox
    bw, bh = xmax - xmin, ymax - ymin
    margin_x = bw * BOX_EXPAND_MARGIN
    margin_y = bh * BOX_EXPAND_MARGIN
    # 加入运动预测的不确定性
    pred_offset = 0 if len(prev_centroid_history) < 2 else np.linalg.norm(v) * 1.5
    margin_x += pred_offset
    margin_y += pred_offset

    # 确保最小 margin（至少 30px）
    margin_x = max(margin_x, 30)
    margin_y = max(margin_y, 30)

    box = np.array([
        max(0, int(xmin - margin_x)),
        max(0, int(ymin - margin_y)),
        min(w, int(xmax + margin_x)),
        min(h, int(ymax + margin_y)),
    ])

    # 正点：预测质心 + 左右偏移
    pos = np.array([
        [int(pred_cx), int(pred_cy)],
        [int(pred_cx - bw*0.2), int(pred_cy)],
        [int(pred_cx + bw*0.2), int(pred_cy)],
        [int(pred_cx), int(pred_cy - bh*0.2)],
        [int(pred_cx), int(pred_cy + bh*0.2)],
    ])

    # 负点：四角 + 框外上下左右 + 肤色
    neg_parts = [
        np.array([[10, 10], [w-10, 10], [10, h-10], [w-10, h-10]]),
        np.array([
            [int(pred_cx), max(0, ymin - int(bh*0.4))],
            [int(pred_cx), min(h, ymax + int(bh*0.4))],
            [max(0, xmin - int(bw*0.4)), int(pred_cy)],
            [min(w, xmax + int(bw*0.4)), int(pred_cy)],
        ]),
    ]

    skin_ys, skin_xs = np.where(skin_mask)
    if len(skin_ys) > 0:
        indices = np.linspace(0, len(skin_ys)-1, min(6, len(skin_ys)), dtype=int)
        neg_skin = np.array([[skin_xs[i], skin_ys[i]] for i in indices])
        neg_parts.append(neg_skin)

    neg = np.vstack(neg_parts)
    return pos, neg, box


# ── 质量门控 ────────────────────────────────────────────────────────────
def quality_check(stats, ref_stats, prev_centroid, cfg):
    """返回 (ok, reason)。"""
    area = stats['area']
    ar = stats['aspect_ratio']
    centroid = stats['centroid']

    # 面积门控
    lo, hi = cfg['area_range']
    if area < lo or area > hi:
        return False, f"area_out_of_range ({area} ∉ [{lo},{hi}])"

    # 宽高比门控（相对于 frame-0 参考值）
    if ref_stats is not None:
        ref_ar = ref_stats['aspect_ratio']
        ar_change = abs(ar - ref_ar) / max(ref_ar, 0.1)
        if ar_change > cfg['max_ar_change']:
            return False, f"ar_change ({ar_change:.2f} > {cfg['max_ar_change']:.2f})"

    # 质心跳跃门控
    if prev_centroid is not None:
        jump = np.linalg.norm(centroid - prev_centroid)
        if jump > cfg['max_centroid_jump_px']:
            return False, f"centroid_jump ({jump:.1f} > {cfg['max_centroid_jump_px']:.0f}px)"

    return True, None


# ── 调试叠加图 ──────────────────────────────────────────────────────────
def save_debug_overlay(frame_rgb, mask, stats, valid, frame_idx, out_path,
                       sam_area=0, expected_centroid=None):
    """保存调试图：原图 + mask 叠加 + 质心标记。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(frame_rgb)
    if mask is not None and mask.any():
        overlay = np.zeros((*mask.shape, 4))
        color = [0.2, 0.8, 0.2, 0.35] if valid else [0.8, 0.2, 0.2, 0.35]
        overlay[mask] = color
        ax1.imshow(overlay)

    if stats is not None:
        cx, cy = stats['centroid']
        ax1.plot(cx, cy, 'y+', markersize=20, markeredgewidth=2)
        xmin, ymin, xmax, ymax = stats['bbox']
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             fill=False, color='yellow', linewidth=2)
        ax1.add_patch(rect)

    if expected_centroid is not None:
        ax1.plot(expected_centroid[0], expected_centroid[1], 'c.', markersize=12)

    status = 'VALID' if valid else 'INVALID'
    ax1.set_title(f'Frame {frame_idx} | {status} | Area={stats["area"] if stats else sam_area}px')
    ax1.axis('off')

    if mask is not None:
        ax2.imshow(mask, cmap='gray')
    ax2.set_title(f'Mask | {status}')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


# ── 轨迹图 ──────────────────────────────────────────────────────────────
def save_trajectory_plot(records, out_path, obj_name, camera):
    """三面板诊断图：轨迹、mask 面积、theta。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    frames = [r['frame_id'] for r in records]
    valid = np.array([r['valid'] for r in records])
    centroids = np.array([r['centroid'] for r in records])
    areas = [r.get('mask_area', 0) or 0 for r in records]
    thetas = [abs(r.get('theta_deg', 0) or 0) for r in records]
    colors = ['green' if v else 'red' for v in valid]

    # X/Y 轨迹
    axes[0].plot(centroids[:, 0], centroids[:, 1], color='0.65', linewidth=1)
    axes[0].scatter(centroids[valid, 0], centroids[valid, 1],
                    c='green', s=18, label='valid')
    if (~valid).any():
        axes[0].scatter(centroids[~valid, 0], centroids[~valid, 1],
                        c='red', marker='x', s=35, label='invalid')
    axes[0].set_title('Image-Plane Trajectory (px)')
    axes[0].set_xlabel('x (px)')
    axes[0].set_ylabel('y (px)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Mask 面积
    axes[1].bar(frames, areas, color=colors, alpha=0.7, width=FRAME_STRIDE)
    axes[1].axhline(y=records[0]['mask_area'], color='blue', linestyle='--', alpha=0.5, label='ref')
    axes[1].set_title('Mask Area')
    axes[1].set_xlabel('frame')
    axes[1].set_ylabel('pixels')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    # Theta
    axes[2].plot(frames, thetas, color='0.3', linewidth=1)
    axes[2].scatter(frames, thetas, c=colors, s=18)
    axes[2].set_title('|Theta| (PCA orientation)')
    axes[2].set_xlabel('frame')
    axes[2].set_ylabel('deg')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Dynamic SAM Tracking: {obj_name} / {camera}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 主追踪循环 ──────────────────────────────────────────────────────────
def track_object(obj, camera, stride=FRAME_STRIDE):
    """对单个物体的单个相机执行动态 prompt SAM 追踪。"""
    cfg = OBJECTS[obj]
    seq = cfg['seq']
    exp_dir = Path(f'/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_{obj}')
    video_path = Path(f'/home/ruan/research/Hackthon/data/human_demo/{seq}/video/{camera}.mkv')
    out_dir = exp_dir / 'pose_tracking'
    mask_seq_dir = out_dir / 'mask_sequence'
    debug_dir = out_dir / 'mask_sequence_debug'
    for d in [out_dir, mask_seq_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 5 动态追踪: {obj} / {camera}")
    print(f"视频: {video_path}")
    print(f"输出: {out_dir}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_process = total_frames // stride
    print(f"总帧数: {total_frames}, FPS: {fps}, 步长: {stride}, 将处理 {n_process} 帧")

    print("加载 SAM vit_b (CPU)...")
    t0 = time.time()
    predictor = load_sam()
    print(f"  加载耗时 {time.time()-t0:.1f}s")

    # ── Frame 0: 已知优质 prompt ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_bgr = cap.read()
    if not ret:
        print("ERROR: 无法读取 frame 0")
        cap.release()
        return
    frame0_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame0_rgb.shape[:2]

    skin0 = get_skin_mask(frame0_rgb)
    predictor.set_image(frame0_rgb)
    pos0, neg0, box0 = get_frame0_prompts(obj, w, h, skin0)

    masks0, scores0 = run_sam_multimask(predictor, pos0, neg0, box0)
    # Frame 0: 用 np.argmax(scores) 选 mask（prompt 已经很紧，最高分就是目标物体）
    best_idx0 = int(np.argmax(scores0))
    mask0, _ = clean_mask(masks0[best_idx0])
    stats0 = mask_stats(mask0)
    best_score0 = float(scores0[best_idx0])

    if stats0 is None or stats0['area'] < 100:
        print("ERROR: Frame 0 mask 为空或太小")
        cap.release()
        return

    print(f"\nFrame   0: ✓ area={stats0['area']}px score={scores0[best_idx0]:.3f} "
          f"AR={stats0['aspect_ratio']:.1f} c=[{stats0['centroid'][0]:.0f},{stats0['centroid'][1]:.0f}]")

    # 自适应面积范围
    ref_area = stats0['area']
    area_range = [max(2000, int(ref_area * 0.35)), int(ref_area * 3.0)]
    cfg['area_range'] = area_range
    print(f"  自适应面积范围: {area_range}")

    # 保存 frame 0
    np.save(mask_seq_dir / 'frame_000000.npy', mask0)
    save_debug_overlay(frame0_rgb, mask0, stats0, True, 0,
                       debug_dir / 'frame_000000.jpg')

    # ── 追踪状态 ──
    prev_stats = stats0
    prev_centroid = stats0['centroid'].copy()
    prev_valid_mask = mask0.copy()
    prev_theta = mask_pca_theta(mask0, prev_centroid)
    centroid_history = [prev_centroid.copy()]  # 最近几个有效质心
    consecutive_invalid = 0

    records = [{
        'frame_id': 0,
        'valid': True,
        'invalid_reason': None,
        'mask_area': int(stats0['area']),
        'centroid': [float(prev_centroid[0]), float(prev_centroid[1]), 0.0],
        'theta_rad': float(prev_theta),
        'theta_deg': float(np.degrees(prev_theta)),
        'centroid_jump_px': 0.0,
        'theta_jump_deg': 0.0,
        'sam_score': float(scores0[best_idx0]),
        'bbox': stats0['bbox'].tolist(),
        'aspect_ratio': float(stats0['aspect_ratio']),
        'sam_raw_area': int(stats0['area']),
    }]

    n_valid = 1
    n_processed = 1
    debug_stride = max(1, n_process // MAX_DEBUG_OVERLAYS)
    t_start = time.time()

    # ── 逐帧循环 ──
    for i_f, frame_idx in enumerate(range(stride, total_frames, stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        skin = get_skin_mask(frame_rgb)
        predictor.set_image(frame_rgb)
        expected_centroid = prev_centroid.copy()

        # 构建 prompt
        if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
            # 重捕获：用 frame-0 风格 prompt 但中心在最后已知位置
            pos, neg, box = get_frame0_prompts(obj, w, h, skin)
            # 把 box 平移到最后已知位置
            cx_old = (box[0] + box[2]) / 2
            cy_old = (box[1] + box[3]) / 2
            dx = expected_centroid[0] - cx_old
            dy = expected_centroid[1] - cy_old
            box = np.array([
                max(0, int(box[0] + dx)),
                max(0, int(box[1] + dy)),
                min(w, int(box[2] + dx)),
                min(h, int(box[3] + dy)),
            ])
            reacq = True
        else:
            pos, neg, box = get_dynamic_prompts(
                prev_stats, centroid_history, w, h, skin)
            reacq = False

        # 运行 SAM（获取 3 个候选 mask）
        masks, scores = run_sam_multimask(predictor, pos, neg, box)

        # 策略 1: 空间选 mask（离预期位置最近）
        mask_spatial, idx_spatial, score_spatial, stats_spatial = select_best_mask(
            masks, scores, expected_centroid, w, h)

        # 策略 2: argmax(scores) 作为备选
        idx_argmax = int(np.argmax(scores))
        mask_argmax, _ = clean_mask(masks[idx_argmax])
        stats_argmax = mask_stats(mask_argmax)
        score_argmax = float(scores[idx_argmax])

        # 记录 SAM 原始输出（调试用）
        sam_raw_area = stats_spatial['area'] if stats_spatial else 0

        valid = False
        invalid_reason = "empty_mask"
        mask_candidate = None
        stats_candidate = None
        selected_strategy = "none"

        ref_ar = records[0]['aspect_ratio']

        # 先试空间选 mask
        if stats_spatial is not None and stats_spatial['area'] > 100:
            valid, invalid_reason = quality_check(
                stats_spatial, {'aspect_ratio': ref_ar}, prev_centroid, cfg)
            if valid:
                mask_candidate = mask_spatial
                stats_candidate = stats_spatial
                selected_strategy = "spatial"

        # 空间 mask 失败 → 尝试 argmax mask（放松质心跳跃门限）
        if not valid and stats_argmax is not None and stats_argmax['area'] > 100:
            # 用宽松门限重试：质心跳跃放宽到 1.5 倍
            relaxed_cfg = dict(cfg)
            relaxed_cfg['max_centroid_jump_px'] = cfg['max_centroid_jump_px'] * 1.5
            relaxed_cfg['max_ar_change'] = cfg['max_ar_change'] * 1.3
            valid2, reason2 = quality_check(
                stats_argmax, {'aspect_ratio': ref_ar}, prev_centroid, relaxed_cfg)
            if valid2:
                mask_candidate = mask_argmax
                stats_candidate = stats_argmax
                selected_strategy = "argmax_relaxed"
                valid = True
                invalid_reason = None
            else:
                # 记录两个策略都失败的原因
                invalid_reason = f"spatial: {invalid_reason}; argmax: {reason2}"

        if mask_candidate is None:
            mask_candidate = mask_spatial  # 用于后续显示
            stats_candidate = stats_spatial

        # 更新状态
        centroid_jump = np.linalg.norm(
            stats_candidate['centroid'] - prev_centroid) if stats_candidate else 0.0
        theta_jump = 0.0

        if valid:
            prev_stats = stats_candidate
            prev_centroid = stats_candidate['centroid'].copy()
            prev_valid_mask = mask_candidate.copy()
            prev_theta = mask_pca_theta(mask_candidate, prev_centroid)
            centroid_history.append(prev_centroid.copy())
            if len(centroid_history) > 5:
                centroid_history.pop(0)
            consecutive_invalid = 0
            n_valid += 1
            mask_final = mask_candidate
            stats_final = stats_candidate
        else:
            consecutive_invalid += 1
            # 保持上一有效帧的 mask（hold）
            mask_final = prev_valid_mask
            stats_final = prev_stats
            # 但质心跳跃用候选 mask 的
            if n_processed > 0 and records[-1]['valid']:
                theta_jump = abs(float(np.degrees(prev_theta)) -
                                 float(np.degrees(records[-1]['theta_rad'])))

        n_processed += 1

        # 保存 mask（始终保存当帧 SAM 实际输出）
        np.save(mask_seq_dir / f'frame_{frame_idx:06d}.npy', mask_final)

        # 调试叠加图
        if n_processed <= 3 or (i_f + 1) % debug_stride == 0:
            save_debug_overlay(frame_rgb, mask_final, stats_final, valid, frame_idx,
                               debug_dir / f'frame_{frame_idx:06d}.jpg',
                               sam_area=sam_raw_area,
                               expected_centroid=expected_centroid)

        # 日志
        tag = " [REACQ]" if reacq else ""
        status = '✓' if valid else '✗'
        eta = (time.time() - t_start) / (i_f + 1) * (n_process - i_f - 1) if i_f > 0 else 0
        sel_tag = f" [{selected_strategy}]" if selected_strategy and selected_strategy != "none" else ""
        print(f"  Frame {frame_idx:3d}: {status} SAM_area={sam_raw_area:5d}px "
              f"final_area={stats_final['area']:5d}px "
              f"c_jump={centroid_jump:.0f}px "
              f"score={max(scores):.3f} "
              f"inv={consecutive_invalid}{tag}{sel_tag} "
              f"[ETA {eta:.0f}s]")

        records.append({
            'frame_id': int(frame_idx),
            'valid': bool(valid),
            'invalid_reason': invalid_reason if not valid else None,
            'mask_area': int(stats_final['area']),
            'centroid': [float(stats_final['centroid'][0]),
                         float(stats_final['centroid'][1]), 0.0],
            'theta_rad': float(prev_theta),
            'theta_deg': float(np.degrees(prev_theta)),
            'centroid_jump_px': float(centroid_jump),
            'theta_jump_deg': float(theta_jump),
            'sam_score': float(max(scores)),
            'bbox': stats_final['bbox'].tolist(),
            'aspect_ratio': float(stats_final['aspect_ratio']),
            'sam_raw_area': int(sam_raw_area),
            'reacquisition': reacq,
            'reason': invalid_reason if not valid else None,
        })

    cap.release()
    elapsed = time.time() - t_start
    print(f"\n完成！{n_valid}/{len(records)} 帧有效 "
          f"({n_valid/max(len(records),1)*100:.0f}%) "
          f"耗时 {elapsed:.0f}s ({elapsed/len(records):.0f}s/帧)")

    # ── 汇总拒绝原因 ──
    invalid_reasons = {}
    for r in records:
        reason = r.get('reason') or r.get('invalid_reason')
        if reason:
            key = str(reason).split(' (', 1)[0]
            invalid_reasons[key] = invalid_reasons.get(key, 0) + 1

    # ── 构建 poses 数组 ──
    poses = np.stack([np.eye(4) for _ in records])
    for i, r in enumerate(records):
        poses[i, 0, 3] = r['centroid'][0]
        poses[i, 1, 3] = r['centroid'][1]
        th = r['theta_rad']
        poses[i, 0, 0] = np.cos(th)
        poses[i, 0, 1] = -np.sin(th)
        poses[i, 1, 0] = np.sin(th)
        poses[i, 1, 1] = np.cos(th)

    valid_arr = np.array([r['valid'] for r in records], dtype=bool)
    timestamps = np.array([r['frame_id'] for r in records], dtype='float64')

    # ── 保存 NPZ ──
    npz_path = out_dir / 'object_trajectory_mask_pose.npz'
    np.savez(npz_path, poses=poses, raw_poses=poses,
             timestamps=timestamps, valid=valid_arr)
    print(f"  NPZ: {npz_path}")

    # ── 保存 JSON ──
    json_path = out_dir / 'object_trajectory_mask_pose.json'
    # 清理 records 中不可序列化的字段
    clean_records = []
    for r in records:
        cr = {k: v for k, v in r.items() if k != 'reason'}
        clean_records.append(cr)

    with open(json_path, 'w') as f:
        json.dump({
            'object_name': obj,
            'sequence_name': seq,
            'camera': camera,
            'method': 'image_plane_pose',
            'transform_name': 'T_image_object',
            'num_frames': len(records),
            'num_valid': n_valid,
            'yaw_ambiguous': obj in ('drink_ad', 'drink_yykx'),
            'limitation': (
                '无 depth 数据；pose 为 2D 图像平面坐标，Z 固定为 0。'
                '平移单位是像素（非米制 3D）。'
            ),
            'canonical_mesh': None,
            'tracking_config': {
                'frame_stride': stride,
                'area_range': list(area_range),
                'max_centroid_jump_px': cfg['max_centroid_jump_px'],
                'max_ar_change': cfg['max_ar_change'],
                'sam_model': 'vit_b',
                'prompt_mode': 'dynamic（上一帧质心 + 空间选 mask）',
                'spatial_weight': SPATIAL_WEIGHT,
            },
            'poses': poses.tolist(),
            'timestamps': timestamps.tolist(),
            'frames': clean_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # ── 质量报告 ──
    report_path = out_dir / 'trajectory_quality_report.json'
    metrics = {}
    for key in ('mask_area', 'centroid_jump_px', 'theta_jump_deg', 'sam_score', 'aspect_ratio'):
        vals = [r.get(key) for r in records if r.get(key) is not None]
        if vals:
            metrics[key] = {
                'min': float(min(vals)),
                'max': float(max(vals)),
                'mean': float(sum(vals) / len(vals)),
            }
        else:
            metrics[key] = None

    with open(report_path, 'w') as f:
        json.dump({
            'object_name': obj,
            'sequence_name': seq,
            'camera': camera,
            'method': 'image_plane_pose',
            'num_frames': len(records),
            'num_valid': n_valid,
            'valid_ratio': n_valid / max(len(records), 1),
            'invalid_reasons': invalid_reasons,
            'yaw_ambiguity': (
                '物体近似轴对称；yaw 会计算但不可靠。'
                if obj in ('drink_ad', 'drink_yykx') else None
            ),
            'limitation': '无 depth 数据；仅为 2D 图像平面追踪。',
            'metrics': metrics,
            'elapsed_seconds': elapsed,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")

    # ── 轨迹图 ──
    plot_path = out_dir / 'trajectory_plot.png'
    save_trajectory_plot(records, plot_path, obj, camera)
    print(f"  Plot: {plot_path}")

    print(f"\n{'='*60}")
    print(f"Phase 5 完成: {obj} / {camera}")
    print(f"有效: {n_valid}/{len(records)} ({n_valid/max(len(records),1)*100:.0f}%)")
    if invalid_reasons:
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            print(f"  拒绝 ({reason}): {count}")
    print(f"{'='*60}")

    return records


def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: 动态 prompt SAM mask 追踪 + 图像平面 pose')
    parser.add_argument('--object', required=True,
                        choices=['bread', 'pipette', 'drink_ad', 'drink_yykx'])
    parser.add_argument('--camera', default='camera_top',
                        choices=['camera_top', 'camera_side_1', 'camera_side_2'])
    parser.add_argument('--stride', type=int, default=FRAME_STRIDE,
                        help=f'每隔 N 帧处理一帧（默认: {FRAME_STRIDE}）')
    args = parser.parse_args()

    track_object(args.object, args.camera, args.stride)


if __name__ == '__main__':
    main()
