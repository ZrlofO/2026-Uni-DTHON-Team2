"""
Training script for Fusion Visual Grounding Model
Korean-optimized with BiGRU + Cross-Attention
Much better than CLIP for Korean text!
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

from utils.dataset import VisualGroundingDataset, get_train_transforms, get_val_transforms
from models.fusion_model import FusionVisualGroundingModel, Vocabulary
from utils.metrics import compute_miou
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Train Fusion Visual Grounding Model')

    # Data paths
    parser.add_argument('--train_csv', type=str, default='/home/elicer/train_valid/train.csv')
    parser.add_argument('--val_csv', type=str, default='/home/elicer/train_valid/val.csv')
    parser.add_argument('--train_img_dir', type=str, default='/home/elicer/train_valid/train/jpg')
    parser.add_argument('--val_img_dir', type=str, default='/home/elicer/train_valid/valid/jpg')

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--min_freq', type=int, default=2, help='Min word frequency for vocab')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--max_query_len', type=int, default=64)

    # Optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--no_pretrain', action='store_true', help='Disable ImageNet pretrained weights')

    # Class balancing
    parser.add_argument('--use_class_weights', action='store_true', default=True, help='Use class-balanced loss')
    parser.add_argument('--use_weighted_sampling', action='store_true', default=True, help='Use weighted random sampling')
    parser.add_argument('--class_weights_json', type=str, default='class_weights.json', help='Path to class weights JSON')
    parser.add_argument('--beta', type=float, default=0.9999, help='Beta for effective number of samples')
    parser.add_argument('--loss_type', type=str, default='giou', choices=['iou', 'giou', 'smooth_l1', 'l1', 'mse'],
                        help='Loss function type (default: giou)')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_fusion')
    parser.add_argument('--log_dir', type=str, default='./logs_fusion')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


class FusionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Fusion model
    Tokenizes query text using custom Vocabulary
    """
    def __init__(self, csv_path, image_dir, vocab, transform, max_len=64):
        import pandas as pd
        import ast
        from PIL import Image

        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def _create_query(self, row):
        """Create query from class_name + instruction + answer"""
        parts = []
        if pd.notna(row['class_name']):
            parts.append(f"유형: {row['class_name']}")
        if pd.notna(row['visual_instruction']):
            parts.append(f"질문: {row['visual_instruction']}")
        if pd.notna(row['visual_answer']):
            parts.append(f"설명: {row['visual_answer']}")
        return " ".join(parts)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import ast
        from PIL import Image
        import numpy as np

        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, row['source_data_name_jpg'])
        image = Image.open(img_path).convert('RGB')
        W, H = image.size

        # Create query
        query_text = self._create_query(row)

        # Tokenize
        token_ids = self.vocab.encode(query_text, max_len=self.max_len)
        length = len(token_ids)

        # Parse bbox (normalized to [0, 1] as cx, cy, w, h)
        bbox_str = row['bounding_box']
        if isinstance(bbox_str, str):
            bbox = ast.literal_eval(bbox_str)
        else:
            bbox = bbox_str

        x, y, w, h = bbox
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        target = [cx, cy, nw, nh]

        # Transform image
        image_np = np.array(image)
        if self.transform:
            # Albumentations doesn't support our bbox format well, so manual
            import cv2
            image_np = cv2.resize(image_np, (512, 512))
            image_np = image_np.astype(np.float32) / 255.0
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np - mean) / std
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float()
        else:
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return {
            'image': image_t,
            'tokens': torch.tensor(token_ids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32),
            'query_text': query_text,
            'original_size': (W, H),
            'class_name': row['class_name']
        }


def collate_fn(batch):
    """Collate function with padding"""
    max_len = max(b['length'].item() for b in batch)

    images = torch.stack([b['image'] for b in batch])
    targets = torch.stack([b['target'] for b in batch])
    lengths = torch.tensor([b['length'].item() for b in batch], dtype=torch.long)
    class_names = [b['class_name'] for b in batch]

    # Pad tokens
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b['length'].item()
        tokens[i, :l] = b['tokens'][:l]

    return {
        'images': images,
        'tokens': tokens,
        'lengths': lengths,
        'targets': targets,
        'class_names': class_names
    }


def build_vocab(csv_path):
    """Build vocabulary from training data"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    vocab = Vocabulary(min_freq=2)

    texts = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row['class_name']):
            parts.append(str(row['class_name']))
        if pd.notna(row['visual_instruction']):
            parts.append(str(row['visual_instruction']))
        if pd.notna(row['visual_answer']):
            parts.append(str(row['visual_answer']))
        texts.append(" ".join(parts))

    vocab.build(texts)
    print(f"Built vocabulary with {len(vocab)} tokens")

    return vocab


def load_class_weights(json_path, method='effective_num'):
    """Load class weights from JSON file"""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Class weights will not be used.")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if method == 'effective_num':
        weights = data['effective_num_weights']
    else:
        weights = data['inverse_freq_weights']

    print(f"\nLoaded class weights (method={method}):")
    for class_name, weight in weights.items():
        print(f"  {class_name}: {weight:.4f}")

    return weights


def calculate_class_balanced_weights(csv_path, beta=0.9999):
    """Calculate class-balanced weights on-the-fly"""
    df = pd.read_csv(csv_path)
    class_counts = df['class_name'].value_counts().to_dict()

    weights = {}
    for class_name, count in class_counts.items():
        effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
        weights[class_name] = 1.0 / effective_num

    # Normalize
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}

    return weights


def create_weighted_sampler(dataset, class_weights):
    """Create WeightedRandomSampler for balanced sampling"""
    sample_weights = []

    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        class_name = row['class_name']
        weight = class_weights.get(class_name, 1.0)
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def compute_iou_loss(pred, target):
    """
    Compute IoU loss (1 - IoU) for bounding boxes
    pred: [B, 4] - predicted boxes in format [cx, cy, w, h] (normalized)
    target: [B, 4] - target boxes in format [cx, cy, w, h] (normalized)
    Returns: IoU loss (lower is better)
    """
    # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # Calculate intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # Calculate union area
    pred_area = pred[:, 2] * pred[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union_area = pred_area + target_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    # IoU loss = 1 - IoU
    iou_loss = 1.0 - iou

    return iou_loss


def compute_giou_loss(pred, target):
    """
    Compute GIoU loss (Generalized IoU)
    More stable than IoU loss, especially for non-overlapping boxes
    pred: [B, 4] - predicted boxes in format [cx, cy, w, h] (normalized)
    target: [B, 4] - target boxes in format [cx, cy, w, h] (normalized)
    """
    # Convert from center format to corner format
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # Calculate intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # Calculate union
    pred_area = pred[:, 2] * pred[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union_area = pred_area + target_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    # Calculate enclosing box (smallest box containing both pred and target)
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    enclose_area = enclose_w * enclose_h

    # GIoU = IoU - (enclose_area - union_area) / enclose_area
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)

    # GIoU loss = 1 - GIoU
    giou_loss = 1.0 - giou

    return giou_loss


def compute_class_balanced_loss(pred, target, class_names, class_weights, loss_type='giou'):
    """
    Compute loss with class-based weighting
    pred: [B, 4]
    target: [B, 4]
    class_names: list of B class names
    class_weights: dict mapping class_name -> weight
    loss_type: 'iou', 'giou', 'smooth_l1', 'l1', or 'mse'
    """
    if loss_type == 'iou':
        # IoU loss
        losses = compute_iou_loss(pred, target)
    elif loss_type == 'giou':
        # GIoU loss (recommended - more stable)
        losses = compute_giou_loss(pred, target)
    elif loss_type == 'smooth_l1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        losses = loss_fn(pred, target).mean(dim=1)
    elif loss_type == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='none')
        losses = loss_fn(pred, target).mean(dim=1)
    else:
        loss_fn = torch.nn.MSELoss(reduction='none')
        losses = loss_fn(pred, target).mean(dim=1)

    # Apply class weights
    if class_weights is not None:
        weights = torch.tensor(
            [class_weights.get(name, 1.0) for name in class_names],
            device=losses.device,
            dtype=losses.dtype
        )
        losses = losses * weights

    return losses.mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, args, class_weights=None):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    scaler = GradScaler() if args.mixed_precision else None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths'].to(device)
        targets = batch['targets'].to(device)
        class_names = batch['class_names']

        optimizer.zero_grad()

        # Forward
        if args.mixed_precision:
            with autocast():
                pred = model(images, tokens, lengths)
                # Use class-balanced loss if enabled
                if args.use_class_weights and class_weights is not None:
                    loss = compute_class_balanced_loss(pred, targets, class_names, class_weights, loss_type=args.loss_type)
                else:
                    # Use specified loss type
                    if args.loss_type == 'iou':
                        loss = compute_iou_loss(pred, targets).mean()
                    elif args.loss_type == 'giou':
                        loss = compute_giou_loss(pred, targets).mean()
                    else:
                        loss = model.compute_loss(pred, targets, loss_type=args.loss_type)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images, tokens, lengths)
            # Use class-balanced loss if enabled
            if args.use_class_weights and class_weights is not None:
                loss = compute_class_balanced_loss(pred, targets, class_names, class_weights, loss_type=args.loss_type)
            else:
                # Use specified loss type
                if args.loss_type == 'iou':
                    loss = compute_iou_loss(pred, targets).mean()
                elif args.loss_type == 'giou':
                    loss = compute_giou_loss(pred, targets).mean()
                else:
                    loss = model.compute_loss(pred, targets, loss_type=args.loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / num_batches
        })

        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer):
    """Validate model with class-wise mIoU"""
    model.eval()

    all_pred_bboxes = []
    all_gt_bboxes = []
    all_class_names = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    for batch in pbar:
        images = batch['images'].to(device)
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths'].to(device)
        targets = batch['targets'].cpu().numpy()
        class_names = batch['class_names']

        # Forward
        pred = model(images, tokens, lengths)
        pred = pred.cpu().numpy()

        # Convert normalized bbox to pixel coordinates for mIoU
        # For validation, we keep normalized format
        all_pred_bboxes.extend(pred)
        all_gt_bboxes.extend(targets)
        all_class_names.extend(class_names)

    # Compute overall mIoU (on normalized bbox)
    # Convert cx,cy,w,h to x,y,w,h for IoU calculation
    pred_boxes_xywh = []
    gt_boxes_xywh = []

    for pred, gt in zip(all_pred_bboxes, all_gt_bboxes):
        # pred/gt are [cx, cy, w, h] normalized
        pred_x = (pred[0] - pred[2] / 2.0)
        pred_y = (pred[1] - pred[3] / 2.0)
        pred_boxes_xywh.append([pred_x, pred_y, pred[2], pred[3]])

        gt_x = (gt[0] - gt[2] / 2.0)
        gt_y = (gt[1] - gt[3] / 2.0)
        gt_boxes_xywh.append([gt_x, gt_y, gt[2], gt[3]])

    miou = compute_miou(pred_boxes_xywh, gt_boxes_xywh)

    # Compute class-wise mIoU
    class_wise_ious = defaultdict(list)
    for pred_box, gt_box, class_name in zip(pred_boxes_xywh, gt_boxes_xywh, all_class_names):
        iou = compute_miou([pred_box], [gt_box])
        class_wise_ious[class_name].append(iou)

    print(f"\nValidation mIoU: {miou:.4f}")
    print("\nClass-wise mIoU:")
    print("-" * 50)

    class_mious = {}
    for class_name in sorted(class_wise_ious.keys()):
        ious = class_wise_ious[class_name]
        class_miou = np.mean(ious)
        class_mious[class_name] = class_miou
        print(f"  {class_name:<25} {class_miou:.4f} (n={len(ious)})")

    if writer:
        writer.add_scalar('val/mIoU', miou, epoch)
        for class_name, class_miou in class_mious.items():
            safe_name = class_name.replace('/', '_').replace('(', '_').replace(')', '_')
            writer.add_scalar(f'val/class_mIoU/{safe_name}', class_miou, epoch)

    return miou, class_mious


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(args.train_csv)

    # Load or calculate class weights
    class_weights = None
    if args.use_class_weights or args.use_weighted_sampling:
        print("\n" + "="*50)
        print("Loading class weights for balanced training...")
        print("="*50)

        # Try to load from JSON first
        if os.path.exists(args.class_weights_json):
            class_weights = load_class_weights(args.class_weights_json, method='effective_num')
        else:
            # Calculate on-the-fly
            print(f"Class weights JSON not found. Calculating from {args.train_csv}...")
            class_weights = calculate_class_balanced_weights(args.train_csv, beta=args.beta)
            print("\nCalculated class weights:")
            for class_name, weight in class_weights.items():
                print(f"  {class_name}: {weight:.4f}")

    # Create datasets
    print("\n" + "="*50)
    print("Loading datasets...")
    print("="*50)
    train_dataset = FusionDataset(
        args.train_csv,
        args.train_img_dir,
        vocab,
        transform=True,
        max_len=args.max_query_len
    )

    val_dataset = FusionDataset(
        args.val_csv,
        args.val_img_dir,
        vocab,
        transform=True,
        max_len=args.max_query_len
    )

    # Create weighted sampler if enabled
    train_sampler = None
    shuffle = True
    if args.use_weighted_sampling and class_weights is not None:
        print("\nCreating WeightedRandomSampler for balanced training...")
        train_sampler = create_weighted_sampler(train_dataset, class_weights)
        shuffle = False  # Don't shuffle when using sampler
        print("WeightedRandomSampler created!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create model
    print("Creating model...")
    model = FusionVisualGroundingModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        pretrained_backbone=not args.no_pretrain
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    best_miou = 0.0

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    print(f"Loss type: {args.loss_type.upper()}")
    print(f"Class-balanced loss: {args.use_class_weights}")
    print(f"Weighted sampling: {args.use_weighted_sampling}")
    print("="*50)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, args, class_weights)
        print(f"Train Loss: {train_loss:.4f}")

        val_miou, class_mious = validate(model, val_loader, device, epoch, writer)

        if val_miou > best_miou:
            best_miou = val_miou
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_itos': vocab.itos,
                'vocab_stoi': vocab.stoi,
                'miou': val_miou,
                'class_mious': class_mious,
                'args': vars(args)
            }, save_path)
            print(f"\nSaved best model with mIoU: {val_miou:.4f}")

        scheduler.step()

    writer.close()
    print(f"\n{'='*50}")
    print(f"Training completed! Best mIoU: {best_miou:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
