#!/usr/bin/env python3.8
"""
Mask Extraction V2: SAM-based object mask extraction from multi-view videos.

Approach:
1. For each object sequence, process each camera video
2. Sample frames every STRIDE frames (~1 fps)
3. For the first frame, use manual region-of-interest prompts
4. For subsequent frames, propagate prompts using motion flow
5. Run SAM to refine masks
6. Save masks (.png) and overlay visualizations

Mask definition: visible-region mask of the target object.
"""
import os, sys, json, argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

CKPT_PATH = os.path.expanduser('~/.cache/sam/sam_vit_b_01ec64.pth')
MODEL_TYPE = 'vit_b'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FRAME_STRIDE = 5  # process ~3 frames per second at 15fps

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')

OBJECT_CONFIGS = {
    'bread': {
        'sequences': ['weigh_bread__2026_0701_0044_30', 'weigh_bread__left__2026_0701_0046_02'],
        'cameras': ['camera_top', 'camera_side_1', 'camera_side_2'],
        'prompts': {  # (cx, cy, w, h) for top view at frame 0, in image coords
            'camera_top': {'bbox': (500, 280, 700, 420)},   # bread region
            'camera_side_1': {'bbox': (400, 250, 750, 480)},
            'camera_side_2': {'bbox': (400, 250, 750, 480)},
        }
    },
    'pipette': {
        'sequences': [
            'grasp_pipette_stand__2026_0701_0019_19',
            'grasp_pipette_rotate__2026_0701_0025_42',
            'grasp_pipette_press__2026_0701_0028_11',
            'pipette_rh_beaker__2026_0701_0035_47',
            'pipette_rh_beaker_testtube__2026_0701_0039_28',
        ],
        'cameras': ['camera_top', 'camera_side_1', 'camera_side_2'],
        'prompts': {
            'camera_top': {'bbox': (480, 280, 720, 390)},
            'camera_side_1': {'bbox': (450, 250, 750, 500)},
            'camera_side_2': {'bbox': (450, 250, 750, 500)},
        }
    },
    'drink_ad': {
        'sequences': ['weigh_drink_ad__2026_0701_0047_56', 'weigh_drink_ad__left__2026_0701_0049_04'],
        'cameras': ['camera_top', 'camera_side_1', 'camera_side_2'],
        'prompts': {
            'camera_top': {'bbox': (500, 280, 680, 420)},
            'camera_side_1': {'bbox': (450, 200, 750, 500)},
            'camera_side_2': {'bbox': (450, 200, 750, 500)},
        }
    },
    'drink_yykx': {
        'sequences': [
            'weigh_drink_yykx__2026_0701_0051_12',
            'weigh_drink_yykx__left__2026_0701_0052_53',
            'grasp_drink_yykx__2026_0701_0054_45',
        ],
        'cameras': ['camera_top', 'camera_side_1', 'camera_side_2'],
        'prompts': {
            'camera_top': {'bbox': (500, 280, 680, 420)},
            'camera_side_1': {'bbox': (450, 200, 750, 500)},
            'camera_side_2': {'bbox': (450, 200, 750, 500)},
        }
    },
}


def load_sam():
    print(f"Loading SAM {MODEL_TYPE} on {DEVICE}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


def get_sam_mask(predictor, bbox=None):
    """Run SAM with bbox prompt, return best mask."""
    box = np.array(bbox) if bbox is not None else None
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :] if box is not None else None,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    return masks[best_idx], scores[best_idx]


def compute_background_mask(frame, bg_subtractor):
    """Use MOG2 to detect foreground moving region."""
    fg_mask = bg_subtractor.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def get_object_bbox_from_masks(mask_prev, bg_roi):
    """Combine previous mask and background motion to estimate new bbox."""
    combined = mask_prev.astype(np.uint8) * 255
    if bg_roi is not None:
        combined = cv2.bitwise_or(combined, bg_roi)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # Expand slightly
    margin = 30
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(1280 - x, w + 2 * margin)
    h = min(720 - y, h + 2 * margin)
    return (x, y, x + w, y + h)


def extract_masks_for_sequence(obj_name, seq_name, cameras, prompts, predictor, stride=FRAME_STRIDE):
    out_dir = os.path.join(OUT_ROOT, obj_name, seq_name)
    overlay_dir = os.path.join(out_dir, 'mask_overlays')
    mask_dir = os.path.join(out_dir, 'masks')
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    results = {}

    for cam in cameras:
        video_path = os.path.join(DATA_ROOT, seq_name, 'video', f'{cam}.mkv')
        if not os.path.exists(video_path):
            print(f"  WARNING: {video_path} not found, skipping")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=36, detectShadows=False
        )

        frame_indices = list(range(0, total_frames, stride))
        masks_out = {}
        prev_mask = None

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            bbox = None
            if fidx == 0:
                # First frame: use manual prompt
                prompt = prompts.get(cam, {})
                if 'bbox' in prompt:
                    x1, y1, x2, y2 = prompt['bbox']
                    bbox = (x1, y1, x2, y2)
            else:
                # Subsequent frames: use motion + previous mask
                bg_roi = compute_background_mask(frame, bg_subtractor)
                if prev_mask is not None and bg_roi is not None:
                    bbox = get_object_bbox_from_masks(prev_mask, bg_roi)

            mask, score = get_sam_mask(predictor, bbox)

            # Post-process: keep largest connected component
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                if len(areas) > 0:
                    largest_label = np.argmax(areas) + 1
                    mask = (labels_im == largest_label)

            prev_mask = mask.astype(np.uint8) * 255

            # Save mask
            mask_key = f'{cam}_frame_{fidx:06d}'
            mask_path = os.path.join(mask_dir, f'{mask_key}.png')
            cv2.imwrite(mask_path, prev_mask)
            masks_out[mask_key] = {'frame': fidx, 'mask_pixels': int(mask.sum()), 'sam_score': float(score)}

            # Save overlay visualization (composite mask on frame)
            overlay = frame.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2) if bbox else None
            cv2.putText(overlay, f'frame {fidx} | cam {cam} | score {score:.3f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            overlay_path = os.path.join(overlay_dir, f'{mask_key}_overlay.jpg')
            cv2.imwrite(overlay_path, overlay)

        cap.release()
        results[cam] = {'num_masks': len(masks_out), 'masks': masks_out}
        print(f"    {cam}: {len(masks_out)} masks extracted")

    # Save metadata
    meta = {
        'object': obj_name,
        'sequence': seq_name,
        'frame_stride': stride,
        'cameras': results,
    }
    with open(os.path.join(out_dir, 'mask_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=['bread', 'pipette', 'drink_ad', 'drink_yykx'],
                        help='Objects to process')
    parser.add_argument('--stride', type=int, default=FRAME_STRIDE, help='Frame sampling stride')
    args = parser.parse_args()

    stride = args.stride

    predictor = load_sam()

    for obj_name in args.objects:
        config = OBJECT_CONFIGS.get(obj_name)
        if not config:
            print(f"Unknown object: {obj_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {obj_name}")
        print(f"{'='*60}")

        for seq_name in config['sequences']:
            print(f"  Sequence: {seq_name}")
            extract_masks_for_sequence(
                obj_name, seq_name,
                config['cameras'], config['prompts'],
                predictor, stride
            )

    print(f"\nDone. Outputs in: {OUT_ROOT}")


if __name__ == '__main__':
    main()
