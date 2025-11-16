"""
Hybrid Loss Functions for Bounding Box Regression

Combines IoU-based loss with L1 loss for better alignment with mIoU metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou_loss(pred_bbox, target_bbox):
    """
    Compute Generalized IoU (GIoU) Loss
    
    Args:
        pred_bbox: (B, 4) in format [cx, cy, w, h] normalized to [0, 1]
        target_bbox: (B, 4) in format [cx, cy, w, h] normalized to [0, 1]
    
    Returns:
        Loss tensor (scalar)
    """
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    def convert_format(bbox):
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    pred_box = convert_format(pred_bbox)  # (B, 4)
    target_box = convert_format(target_bbox)  # (B, 4)
    
    # Extract coordinates
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box.unbind(1)
    target_x1, target_y1, target_x2, target_y2 = target_box.unbind(1)
    
    # Compute areas
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-8)
    
    # Enclosing box for GIoU
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)
    
    # Loss (1 - GIoU, so higher IoU = lower loss)
    loss = 1.0 - giou
    
    return loss.mean()


def compute_ciou_loss(pred_bbox, target_bbox):
    """
    Compute Complete IoU (CIoU) Loss
    Adds penalty for aspect ratio and center distance
    
    Reference: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    
    Args:
        pred_bbox: (B, 4) in format [cx, cy, w, h] normalized to [0, 1]
        target_bbox: (B, 4) in format [cx, cy, w, h] normalized to [0, 1]
    
    Returns:
        Loss tensor (scalar)
    """
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    def convert_format(bbox):
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    pred_box = convert_format(pred_bbox)  # (B, 4)
    target_box = convert_format(target_bbox)  # (B, 4)
    
    # Extract coordinates
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box.unbind(1)
    target_x1, target_y1, target_x2, target_y2 = target_box.unbind(1)
    
    # Compute areas and dimensions
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    target_w = target_x2 - target_x1
    target_h = target_y2 - target_y1
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-8)
    
    # Center distance
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2
    
    center_distance = ((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2).sqrt()
    
    # Diagonal of enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    diagonal = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).sqrt()
    
    # DIoU term
    diou = iou - (center_distance ** 2) / (diagonal ** 2 + 1e-8)
    
    # Aspect ratio consistency
    atan_pred = torch.atan(pred_w / (pred_h + 1e-8))
    atan_target = torch.atan(target_w / (target_h + 1e-8))
    v = (4 / (3.14159 ** 2)) * ((atan_pred - atan_target) ** 2)
    
    # CIoU
    alpha = v / ((1 - iou).detach() + v + 1e-8)
    ciou = diou - alpha * v
    
    # Loss (1 - CIoU, so higher IoU = lower loss)
    loss = 1.0 - ciou
    
    return loss.mean()


class HybridBBoxLoss(nn.Module):
    """
    Hybrid Loss combining Smooth L1 and IoU-based loss
    
    Total Loss = α * Smooth_L1 + β * IoU_Loss
    
    This aligns better with mIoU metric during evaluation
    """
    
    def __init__(self, l1_weight=0.5, iou_weight=0.5, use_ciou=True):
        """
        Args:
            l1_weight: Weight for Smooth L1 loss
            iou_weight: Weight for IoU loss
            use_ciou: Use CIoU (True) or GIoU (False)
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        self.use_ciou = use_ciou
        
        print(f"HybridBBoxLoss initialized:")
        print(f"  L1 weight: {l1_weight}")
        print(f"  IoU weight: {iou_weight}")
        print(f"  IoU type: {'CIoU' if use_ciou else 'GIoU'}")
    
    def forward(self, pred_bbox, target_bbox):
        """
        Args:
            pred_bbox: (B, 4) predicted bbox in [cx, cy, w, h]
            target_bbox: (B, 4) target bbox in [cx, cy, w, h]
        
        Returns:
            Scalar loss
        """
        # L1 component
        l1_loss = F.smooth_l1_loss(pred_bbox, target_bbox, beta=1.0)
        
        # IoU component
        if self.use_ciou:
            iou_loss = compute_ciou_loss(pred_bbox, target_bbox)
        else:
            iou_loss = compute_iou_loss(pred_bbox, target_bbox)
        
        # Hybrid loss
        total_loss = self.l1_weight * l1_loss + self.iou_weight * iou_loss
        
        return total_loss


class DynamicHybridBBoxLoss(nn.Module):
    """
    Dynamic Hybrid Loss that adjusts weights during training
    
    Early epochs: More weight on L1 for stable training
    Later epochs: More weight on IoU for metric alignment
    """
    
    def __init__(self, total_epochs=15, use_ciou=True):
        """
        Args:
            total_epochs: Total training epochs
            use_ciou: Use CIoU (True) or GIoU (False)
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.use_ciou = use_ciou
        self.current_epoch = 0
        
        print(f"DynamicHybridBBoxLoss initialized:")
        print(f"  Total epochs: {total_epochs}")
        print(f"  IoU type: {'CIoU' if use_ciou else 'GIoU'}")
        print(f"  Schedule: L1 weight decreases from 1.0 to 0.3")
        print(f"  Schedule: IoU weight increases from 0.0 to 0.7")
    
    def set_epoch(self, epoch):
        """Update current epoch for weight adjustment"""
        self.current_epoch = epoch
    
    def get_weights(self):
        """Get current loss weights based on epoch"""
        # Linear interpolation: epoch 0 -> (1.0, 0.0), epoch N -> (0.3, 0.7)
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        
        l1_weight = 1.0 - 0.7 * progress
        iou_weight = 0.7 * progress
        
        return l1_weight, iou_weight
    
    def forward(self, pred_bbox, target_bbox):
        """
        Args:
            pred_bbox: (B, 4) predicted bbox in [cx, cy, w, h]
            target_bbox: (B, 4) target bbox in [cx, cy, w, h]
        
        Returns:
            Scalar loss
        """
        l1_weight, iou_weight = self.get_weights()
        
        # L1 component
        l1_loss = F.smooth_l1_loss(pred_bbox, target_bbox, beta=1.0)
        
        # IoU component
        if self.use_ciou:
            iou_loss = compute_ciou_loss(pred_bbox, target_bbox)
        else:
            iou_loss = compute_iou_loss(pred_bbox, target_bbox)
        
        # Dynamic hybrid loss
        total_loss = l1_weight * l1_loss + iou_weight * iou_loss
        
        return total_loss


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    
    # Random predictions and targets (normalized to [0, 1])
    pred = torch.rand(batch_size, 4) * 0.5 + 0.25  # Between 0.25-0.75
    target = torch.rand(batch_size, 4) * 0.5 + 0.25
    
    print("Testing Loss Functions")
    print("=" * 50)
    
    # Test GIoU
    print("\n1. GIoU Loss:")
    giou_loss = compute_iou_loss(pred, target)
    print(f"   GIoU Loss: {giou_loss.item():.4f}")
    
    # Test CIoU
    print("\n2. CIoU Loss:")
    ciou_loss = compute_ciou_loss(pred, target)
    print(f"   CIoU Loss: {ciou_loss.item():.4f}")
    
    # Test Hybrid Loss
    print("\n3. Hybrid Loss (50% L1 + 50% CIoU):")
    hybrid_loss = HybridBBoxLoss(l1_weight=0.5, iou_weight=0.5, use_ciou=True)
    loss = hybrid_loss(pred, target)
    print(f"   Hybrid Loss: {loss.item():.4f}")
    
    # Test Dynamic Hybrid Loss
    print("\n4. Dynamic Hybrid Loss:")
    dynamic_loss = DynamicHybridBBoxLoss(total_epochs=15, use_ciou=True)
    
    for epoch in range(0, 16, 3):
        dynamic_loss.set_epoch(epoch)
        l1_w, iou_w = dynamic_loss.get_weights()
        loss = dynamic_loss(pred, target)
        print(f"   Epoch {epoch:2d}: L1_weight={l1_w:.3f}, IoU_weight={iou_w:.3f}, Loss={loss.item():.4f}")
    
    print("\n✓ All loss functions tested successfully!")
