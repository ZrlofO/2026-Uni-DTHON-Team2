"""
Dataset class for Visual Element Grounding
Query: class_name + visual_instruction + visual_answer (visual_context excluded)
"""

import os
import ast
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VisualGroundingDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        transform=None,
        is_test=False
    ):
        """
        Args:
            csv_path: Path to CSV file (train.csv, val.csv, or test.csv)
            image_dir: Directory containing images
            transform: Albumentations transform
            is_test: Whether this is test dataset (no bounding box)
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def _parse_bbox(self, bbox_str):
        """Parse bounding box from string format"""
        if pd.isna(bbox_str):
            return None

        # Handle string format: "[x, y, w, h]"
        if isinstance(bbox_str, str):
            bbox = ast.literal_eval(bbox_str)
        else:
            bbox = bbox_str

        return np.array(bbox, dtype=np.float32)

    def _create_query(self, row):
        """
        Create query text from class_name + visual_instruction + visual_answer
        Note: visual_context is EXCLUDED per requirements
        """
        class_name = str(row['class_name']) if pd.notna(row['class_name']) else ""
        instruction = str(row['visual_instruction']) if pd.notna(row['visual_instruction']) else ""
        answer = str(row['visual_answer']) if pd.notna(row['visual_answer']) else ""

        # Combine into single query
        query_parts = []
        if class_name:
            query_parts.append(f"유형: {class_name}")
        if instruction:
            query_parts.append(f"질문: {instruction}")
        if answer:
            query_parts.append(f"설명: {answer}")

        query = " ".join(query_parts)
        return query

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_name = row['source_data_name_jpg']
        image_path = os.path.join(self.image_dir, image_name)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (800, 600), color='white')

        image = np.array(image)
        original_height, original_width = image.shape[:2]

        # Create query text (class_name + visual_instruction + visual_answer)
        query_text = self._create_query(row)

        # Parse bounding box if not test set
        if not self.is_test:
            bbox = self._parse_bbox(row['bounding_box'])

            # Apply transforms
            if self.transform:
                transformed = self.transform(
                    image=image,
                    bboxes=[bbox],
                    labels=[0]  # Dummy label for albumentations
                )
                image = transformed['image']

                # Get transformed bbox (normalized to [0, 1])
                if len(transformed['bboxes']) > 0:
                    bbox = np.array(transformed['bboxes'][0], dtype=np.float32)
                else:
                    # If bbox was cropped out, use original
                    bbox = self._parse_bbox(row['bounding_box'])

            return {
                'image': image,
                'query_text': query_text,
                'bbox': torch.tensor(bbox, dtype=torch.float32),
                'image_name': image_name,
                'original_size': (original_width, original_height)
            }
        else:
            # Test set
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            instance_id = row.get('instance_id', f"{image_name}_{idx}")

            return {
                'image': image,
                'query_text': query_text,
                'image_name': image_name,
                'instance_id': instance_id,
                'original_size': (original_width, original_height)
            }


def get_train_transforms(image_size=800):
    """Training augmentations"""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            value=(255, 255, 255)
        ),
        A.HorizontalFlip(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',  # [x, y, w, h]
        label_fields=['labels']
    ))


def get_val_transforms(image_size=800):
    """Validation/Test transforms (no augmentation)"""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            value=(255, 255, 255)
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels']
    ))


def get_test_transforms(image_size=800):
    """Test transforms (no bbox params)"""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            value=(255, 255, 255)
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    query_texts = [item['query_text'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]

    result = {
        'images': images,
        'query_texts': query_texts,
        'image_names': image_names,
        'original_sizes': original_sizes
    }

    # Add bboxes if not test set
    if 'bbox' in batch[0]:
        bboxes = torch.stack([item['bbox'] for item in batch])
        result['bboxes'] = bboxes
    else:
        instance_ids = [item['instance_id'] for item in batch]
        result['instance_ids'] = instance_ids

    return result


if __name__ == '__main__':
    # Test dataset
    csv_path = "val.csv"
    image_dir = "."

    dataset = VisualGroundingDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=get_val_transforms(),
        is_test=False
    )

    print(f"Dataset size: {len(dataset)}")

    # Test first sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Query: {sample['query_text'][:100]}...")
    print(f"BBox: {sample['bbox']}")
    print(f"Original size: {sample['original_size']}")
