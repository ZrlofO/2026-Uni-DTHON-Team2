"""
Simplified Fusion Model for Korean Document Visual Grounding
Using simple cross-attention (no multi-head complexity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.cross_attention import (
    MultiHeadCrossAttention,
    SpatialCrossAttention,
    AttentionWithResidual
)



class Vocabulary:
    """Simple vocabulary for Korean text tokenization"""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq = {}
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, texts: List[str]):
        """Build vocabulary from texts"""
        for text in texts:
            for tok in self._tokenize(text):
                self.freq[tok] = self.freq.get(tok, 0) + 1

        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def _tokenize(self, text: str) -> List[str]:
        """Simple Korean-aware tokenization"""
        text = text.replace("##", " ").replace(",", " ")
        text = text.replace("(", " ").replace(")", " ")
        text = text.replace(":", " ").replace("?", " ")
        text = text.replace("·", " ")
        return [t for t in text.strip().split() if t]

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._tokenize(text)[:max_len]
        if not tokens:
            return [1]  # <unk>
        return [self.stoi.get(t, 1) for t in tokens]

    def __len__(self):
        return len(self.itos)


class TextEncoder(nn.Module):
    """BiGRU-based text encoder for Korean queries"""
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.1
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        out = self.proj(h)
        out = self.norm(out)
        return out


class ImageEncoder(nn.Module):
    """ResNet-based image encoder"""
    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        if pretrained:
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet34(pretrained=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x


# SimpleCrossAttention 클래스는 유지하되, 옵션으로 변경 가능하게
class SimpleCrossAttention(nn.Module):
    """
    Simple cross-attention (no multi-head)
    Much simpler and guaranteed to work!
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, dim) text features
            feature_map: (B, dim, H, W) image features

        Returns:
            (B, dim) attended features
        """
        B, D, H, W = feature_map.shape

        # Query projection
        Q = self.q_proj(query)  # (B, dim)

        # Key/Value projection
        K = self.k_proj(feature_map)  # (B, dim, H, W)
        V = self.v_proj(feature_map)  # (B, dim, H, W)

        # Flatten spatial dimensions
        K = K.flatten(2)  # (B, dim, H*W)
        V = V.flatten(2)  # (B, dim, H*W)

        # Compute attention: Q @ K^T
        # (B, dim) @ (B, dim, HW) -> (B, HW)
        attn = torch.bmm(Q.unsqueeze(1), K).squeeze(1) * self.scale  # (B, HW)
        attn = F.softmax(attn, dim=-1)  # (B, HW)

        # Apply attention to values
        # (B, dim, HW) @ (B, HW, 1) -> (B, dim)
        out = torch.bmm(V, attn.unsqueeze(-1)).squeeze(-1)  # (B, dim)

        # Output projection with residual
        out = self.out_proj(out)
        out = self.norm(out + query)

        return out

class FusionVisualGroundingModel(nn.Module):
    """
    Simplified Fusion model with multi-head cross-attention option
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        pretrained_backbone: bool = True,
        use_enhanced_attention: bool = True,  # 추가: 향상된 attention 사용 여부
        attention_type: str = 'multihead'  # 'simple', 'multihead', 'spatial'
    ):
        super().__init__()
        
        self.text_encoder = TextEncoder(vocab_size, embed_dim, embed_dim)
        self.image_encoder = ImageEncoder(embed_dim, pretrained_backbone)
        
        # Cross-attention: 타입에 따라 선택
        if use_enhanced_attention:
            if attention_type == 'multihead':
                self.cross_attention = MultiHeadCrossAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    add_spatial_embedding=True
                )
            elif attention_type == 'spatial':
                self.cross_attention = SpatialCrossAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1
                )
            elif attention_type == 'residual':
                self.cross_attention = AttentionWithResidual(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=embed_dim * 2,
                    dropout=0.1
                )
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
            
            print(f"Using enhanced {attention_type} cross-attention")
        
        else:
            # 기존 SimpleCrossAttention 사용
            from fusion_model import SimpleCrossAttention
            self.cross_attention = SimpleCrossAttention(embed_dim)
            print(f"Using simple cross-attention")
        
        self.bbox_head = BBoxHead(embed_dim)
        
        print(f"Fusion Model initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Pretrained backbone: {pretrained_backbone}")
        print(f"  Enhanced attention: {use_enhanced_attention}")
    
    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        # Text encoding
        text_features = self.text_encoder(tokens, lengths)  # (B, embed_dim)
        
        # Image encoding
        image_features = self.image_encoder(images)  # (B, embed_dim, H', W')
        
        # Cross-attention fusion (Multi-head 버전도 동일한 interface)
        fused_features = self.cross_attention(text_features, image_features)  # (B, embed_dim)
        
        # Bbox prediction
        bbox_pred = self.bbox_head(fused_features)  # (B, 4)
        
        return bbox_pred

class BBoxHead(nn.Module):
    """Bounding box regression head"""
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bbox = self.mlp(x)
        bbox = torch.sigmoid(bbox)
        return bbox


class FusionVisualGroundingModel(nn.Module):
    """
    Simplified Fusion model with single-head cross-attention
    More stable and easier to debug
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,  # Ignored, kept for compatibility
        pretrained_backbone: bool = True
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, embed_dim, embed_dim)
        self.image_encoder = ImageEncoder(embed_dim, pretrained_backbone)
        self.cross_attention = SimpleCrossAttention(embed_dim)  # Simple version!
        self.bbox_head = BBoxHead(embed_dim)

        print(f"Fusion Model (Simplified) initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Attention: Single-head (simplified)")
        print(f"  Pretrained backbone: {pretrained_backbone}")

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        # Encode text
        text_features = self.text_encoder(tokens, lengths)  # (B, embed_dim)

        # Encode image
        image_features = self.image_encoder(images)  # (B, embed_dim, H', W')

        # Cross-attention fusion
        fused_features = self.cross_attention(text_features, image_features)  # (B, embed_dim)

        # Predict bbox
        bbox_pred = self.bbox_head(fused_features)  # (B, 4)

        return bbox_pred

    def compute_loss(
        self,
        pred_bbox: torch.Tensor,
        target_bbox: torch.Tensor,
        loss_type: str = 'smooth_l1'
    ) -> torch.Tensor:
        if loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_bbox, target_bbox)
        elif loss_type == 'l1':
            loss = F.l1_loss(pred_bbox, target_bbox)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred_bbox, target_bbox)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss


if __name__ == '__main__':
    # Test
    vocab_size = 10000
    batch_size = 4
    embed_dim = 256

    print(f"Testing simplified model with embed_dim={embed_dim}")

    model = FusionVisualGroundingModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        use_enhanced_attention=True,      # 원할 때만 True
        attention_type='multihead'        # 원할 때만 변경
    )

    # Dummy inputs
    images = torch.randn(batch_size, 3, 512, 512)
    tokens = torch.randint(0, vocab_size, (batch_size, 32))
    lengths = torch.tensor([30, 25, 32, 20])
    targets = torch.rand(batch_size, 4)

    # Forward
    print("\nRunning forward pass...")
    pred = model(images, tokens, lengths)
    print(f"✓ Prediction shape: {pred.shape}")

    # Loss
    loss = model.compute_loss(pred, targets)
    print(f"✓ Loss: {loss.item():.4f}")

    # Backward
    print("\nTesting backward pass...")
    loss.backward()
    print("✓ Backward pass successful!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("\n✓ All tests passed!")
