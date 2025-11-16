"""
Two-Stage Visual Grounding Model
Stage 1: Faster-RCNN for table/chart detection
Stage 2: CLIP-based query matching and selection
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import clip
from typing import List, Tuple


class VisualGroundingModel(nn.Module):
    def __init__(
        self,
        num_classes=10,  # Background + 9 visual element types
        clip_model_name='ViT-B/32',
        detection_score_threshold=0.3,
        top_k_proposals=10,
        device='cuda'
    ):
        super().__init__()

        self.device = device
        self.detection_score_threshold = detection_score_threshold
        self.top_k_proposals = top_k_proposals

        # Stage 1: Faster R-CNN for detection
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace the classifier head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Stage 2: CLIP for text-image matching
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)

        # Freeze CLIP (only fine-tune detector)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print(f"Model initialized with {num_classes} classes")
        print(f"CLIP model: {clip_model_name}")

    def forward_detector(self, images):
        """
        Stage 1: Run Faster-RCNN detection

        Args:
            images: Tensor [B, 3, H, W]

        Returns:
            List of detections per image
        """
        self.detector.eval()
        with torch.no_grad():
            detections = self.detector(images)

        return detections

    def forward_clip_matching(self, images, query_texts, proposals):
        """
        Stage 2: CLIP-based query matching

        Args:
            images: Tensor [B, 3, H, W] - original images
            query_texts: List[str] - query texts
            proposals: List of proposals per image

        Returns:
            List of best matching bboxes per image
        """
        batch_size = len(images)
        results = []

        # Encode text queries
        text_tokens = clip.tokenize(query_texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i in range(batch_size):
            image = images[i]
            text_feature = text_features[i:i+1]
            image_proposals = proposals[i]

            # Get top-k proposals
            boxes = image_proposals['boxes']
            scores = image_proposals['scores']
            labels = image_proposals['labels']

            # Filter by score and take top-k
            valid_idx = scores > self.detection_score_threshold
            boxes = boxes[valid_idx]
            scores = scores[valid_idx]
            labels = labels[valid_idx]

            if len(boxes) == 0:
                # No proposals found, return zero box
                results.append(torch.tensor([0, 0, 1, 1], device=self.device, dtype=torch.float32))
                continue

            # Take top-k
            if len(boxes) > self.top_k_proposals:
                top_idx = torch.topk(scores, k=self.top_k_proposals).indices
                boxes = boxes[top_idx]
                scores = scores[top_idx]
                labels = labels[top_idx]

            # Crop and encode each proposal
            proposal_features = []
            for box in boxes:
                # Crop region
                x1, y1, x2, y2 = box.int().tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[2], x2), min(image.shape[1], y2)

                cropped = image[:, y1:y2, x1:x2]

                # Resize and normalize for CLIP
                # CLIP expects 224x224 images
                cropped = torch.nn.functional.interpolate(
                    cropped.unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                # Encode with CLIP
                with torch.no_grad():
                    img_feature = self.clip_model.encode_image(cropped.unsqueeze(0))
                    img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)

                proposal_features.append(img_feature)

            # Compute similarities
            proposal_features = torch.cat(proposal_features, dim=0)
            similarities = (proposal_features @ text_feature.T).squeeze(-1)

            # Select best matching proposal
            best_idx = torch.argmax(similarities)
            best_box = boxes[best_idx]

            # Convert to [x, y, w, h] format
            x1, y1, x2, y2 = best_box
            bbox = torch.tensor([x1, y1, x2 - x1, y2 - y1], device=self.device, dtype=torch.float32)

            results.append(bbox)

        return torch.stack(results)

    def forward(self, images, query_texts=None):
        """
        Full forward pass

        Args:
            images: Tensor [B, 3, H, W]
            query_texts: List[str] - query texts (required for inference)

        Returns:
            If query_texts provided: predicted bboxes [B, 4] in [x, y, w, h] format
            If query_texts not provided: detector outputs (for training detector)
        """
        # Stage 1: Detection
        if self.training:
            # During training, return detector outputs for loss computation
            return self.detector(images)
        else:
            # Inference
            detections = self.forward_detector(images)

            if query_texts is None:
                return detections

            # Stage 2: CLIP matching
            predicted_bboxes = self.forward_clip_matching(images, query_texts, detections)

            return predicted_bboxes

    def train_detector(self, images, targets):
        """
        Train only the Faster-RCNN detector

        Args:
            images: List of Tensors [3, H, W]
            targets: List of dicts with 'boxes' and 'labels'

        Returns:
            loss_dict
        """
        self.detector.train()
        loss_dict = self.detector(images, targets)
        return loss_dict


def build_detector_targets(bboxes, image_sizes, num_classes=10):
    """
    Build targets for Faster-RCNN training

    Args:
        bboxes: Tensor [B, 4] in [x, y, w, h] format
        image_sizes: List of (width, height) tuples
        num_classes: Number of classes

    Returns:
        List of target dicts
    """
    batch_size = len(bboxes)
    targets = []

    for i in range(batch_size):
        bbox = bboxes[i]

        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # Class 1 (visual element)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        targets.append(target)

    return targets


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VisualGroundingModel(device=device).to(device)

    # Test forward pass
    dummy_images = torch.randn(2, 3, 800, 800).to(device)
    dummy_queries = ["유형: 차트(꺾은선형) 질문: 매출 추이 차트는?", "유형: 표 질문: 예산 표는?"]

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images, dummy_queries)

    print(f"Output shape: {outputs.shape}")
    print(f"Predicted bboxes:\n{outputs}")
