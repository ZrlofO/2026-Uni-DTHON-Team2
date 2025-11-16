"""
Metrics for evaluating bounding box predictions
Main metric: mIoU (mean Intersection over Union)
"""

import numpy as np
import torch


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes

    Args:
        box1: [x, y, w, h] format (top-left corner)
        box2: [x, y, w, h] format (top-left corner)

    Returns:
        IoU score (float)
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Compute intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def compute_miou(pred_boxes, gt_boxes):
    """
    Compute mean IoU across multiple samples

    Args:
        pred_boxes: List of predicted boxes [[x, y, w, h], ...]
        gt_boxes: List of ground truth boxes [[x, y, w, h], ...]

    Returns:
        mIoU score (float)
    """
    assert len(pred_boxes) == len(gt_boxes), "Number of predictions and ground truths must match"

    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        iou = compute_iou(pred, gt)
        ious.append(iou)

    miou = np.mean(ious)
    return miou


def compute_miou_tensor(pred_boxes, gt_boxes):
    """
    Compute mIoU using PyTorch tensors (for batch processing)

    Args:
        pred_boxes: Tensor of shape [N, 4] in [x, y, w, h] format
        gt_boxes: Tensor of shape [N, 4] in [x, y, w, h] format

    Returns:
        mIoU score (float)
    """
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()

    return compute_miou(pred_boxes, gt_boxes)


def evaluate_predictions(predictions_df, ground_truth_df):
    """
    Evaluate predictions against ground truth

    Args:
        predictions_df: DataFrame with columns [query_id, pred_x, pred_y, pred_w, pred_h]
        ground_truth_df: DataFrame with columns [instance_id, bounding_box]

    Returns:
        Dictionary with evaluation metrics
    """
    # Parse bounding boxes
    pred_boxes = predictions_df[['pred_x', 'pred_y', 'pred_w', 'pred_h']].values

    # Parse ground truth boxes from string format
    gt_boxes = []
    for bbox_str in ground_truth_df['bounding_box'].values:
        # Parse "[x, y, w, h]" string
        bbox = eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
        gt_boxes.append(bbox)
    gt_boxes = np.array(gt_boxes)

    # Compute mIoU
    miou = compute_miou(pred_boxes, gt_boxes)

    # Compute IoU distribution
    ious = [compute_iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]

    # Compute accuracy at different IoU thresholds
    iou_thresholds = [0.25, 0.5, 0.75]
    accuracies = {}
    for threshold in iou_thresholds:
        acc = np.mean([iou >= threshold for iou in ious])
        accuracies[f'acc@{threshold}'] = acc

    results = {
        'mIoU': miou,
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'min_iou': np.min(ious),
        'max_iou': np.max(ious),
        **accuracies
    }

    return results


if __name__ == '__main__':
    # Test
    box1 = [100, 100, 200, 150]  # [x, y, w, h]
    box2 = [150, 120, 200, 150]

    iou = compute_iou(box1, box2)
    print(f"IoU: {iou:.4f}")

    # Test batch
    pred_boxes = [[100, 100, 200, 150], [50, 50, 100, 100]]
    gt_boxes = [[150, 120, 200, 150], [50, 50, 100, 100]]

    miou = compute_miou(pred_boxes, gt_boxes)
    print(f"mIoU: {miou:.4f}")
