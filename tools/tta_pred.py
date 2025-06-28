import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from copy import deepcopy

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from torchvision.ops import nms

def normalize_bbox(cxcywh, image_size):
    w_img, h_img = image_size
    cxcywh[:, 0] /= w_img
    cxcywh[:, 2] /= w_img
    cxcywh[:, 1] /= h_img
    cxcywh[:, 3] /= h_img
    return cxcywh

def tta_inference(model, img_path):
    all_instances = []
    base_pipeline = deepcopy(model.cfg.test_pipeline)
    orig_resize = base_pipeline[1]
    
    scales = [(640, 640), (800, 800), (1024, 1024)]
    flips = [False, True]

    for scale in scales:
        for flip in flips:
            # Modify pipeline inplace for current TTA setting
            base_pipeline[1]['scale'] = scale
            base_pipeline[1]['keep_ratio'] = True
            if len(base_pipeline) > 2 and 'flip' in base_pipeline[2]:
                base_pipeline[2]['flip'] = flip

            model.cfg.test_pipeline = deepcopy(base_pipeline)
            result = inference_detector(model, img_path)

            if result.pred_instances.bboxes.shape[0] > 0:
                all_instances.append(result.pred_instances)

    if not all_instances:
        return None

    # Merge predictions with NMS
    bboxes = torch.cat([r.bboxes for r in all_instances], dim=0)
    scores = torch.cat([r.scores for r in all_instances], dim=0)
    labels = torch.cat([r.labels for r in all_instances], dim=0)

    final_bboxes, final_scores, final_labels = [], [], []

    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = cls_scores > 0.01
        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]
        if cls_boxes.numel() == 0:
            continue
        keep_idx = nms(cls_boxes, cls_scores, iou_threshold=0.5)
        final_bboxes.append(cls_boxes[keep_idx])
        final_scores.append(cls_scores[keep_idx])
        final_labels.append(torch.full((len(keep_idx),), cls, dtype=torch.int64))

    if not final_bboxes:
        return None

    merged = InstanceData()
    merged.bboxes = torch.cat(final_bboxes)
    merged.scores = torch.cat(final_scores)
    merged.labels = torch.cat(final_labels)
    return merged

def main(config_file, checkpoint_file, test_dir, output_txt, device='cuda:0'):
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = init_detector(cfg, checkpoint_file, device=device)

    images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    with open(output_txt, 'w') as out_f:
        for fname in tqdm(images):
            img_path = os.path.join(test_dir, fname)
            result = tta_inference(model, img_path)
            if result is None or result.bboxes.shape[0] == 0:
                continue

            img = Image.open(img_path)
            w_img, h_img = img.size

            bboxes = bbox_xyxy_to_cxcywh(result.bboxes.clone())
            bboxes = normalize_bbox(bboxes.cpu().numpy(), (w_img, h_img))
            labels = result.labels.cpu().numpy()
            scores = result.scores.cpu().numpy()
            img_id = os.path.splitext(fname)[0]

            for label, bbox, score in zip(labels, bboxes, scores):
                label = int(label) + 1  # convert to 1-based label
                x_c, y_c, w, h = bbox.tolist()
                out_f.write(f"{img_id} {label} {x_c:.3f} {y_c:.3f} {w:.3f} {h:.3f} {score:.3f}\n")

    print(f"\n✅ Done. TTA predictions saved to: {output_txt}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_txt', type=str, default='tta_predictions.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.test_dir, args.output_txt, args.device)
