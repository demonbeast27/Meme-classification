"""
multimodal_model.py
===================
Decoupled Multimodal Meme Classifier architecture for Phase 2.

Architecture Overview:
- Backbones: Frozen CLIP Vision Transformer (512-d) and Text Transformer (512-d).
- Trainable Gated Fusion Layer: Dynamically balances visual and textual evidence.
- Multi-Label Stylistic Head: Independent Sigmoid outputs for stylistic attributes.
- Decoupled Hate Head: Separate binary head for hate/harm detection.
- Masked BCE Loss: Strictly excludes unknown / unannotated categories (-1) from loss computation.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits supporting masked / unknown targets (-1).
    Ignores entries where target == -1, preventing backpropagation into unsupported classes.
    """
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Tensor of shape (B, C)
            targets: Tensor of shape (B, C) with values in {0, 1, -1}
        """
        mask = (targets != -1).float()
        valid_targets = torch.clamp(targets, min=0.0, max=1.0)

        # Standard BCE with logits per element
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            valid_targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )
        masked_loss = bce * mask
        num_valid = mask.sum()

        if num_valid > 0:
            return masked_loss.sum() / num_valid
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


class GatedMultimodalFusion(nn.Module):
    """
    Gated cross-modal fusion layer combining visual and textual features.
    Learns a dimension-wise gating vector to dynamically weight visual vs text cues.
    """
    def __init__(self, visual_dim: int = 512, text_dim: int = 512, fusion_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.proj_v = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Context-dependent gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

        # Post-fusion residual block
        self.fusion_dense = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        v_h = self.proj_v(visual_feats)
        t_h = self.proj_t(text_feats)

        # Compute dimension-wise gating factor
        gate = self.gate_net(torch.cat([v_h, t_h], dim=-1))
        fused = gate * v_h + (1.0 - gate) * t_h

        # Residual connection
        out = self.layer_norm(fused + self.fusion_dense(fused))
        return out


class MultimodalMemeClassifier(nn.Module):
    """
    Multimodal meme classifier with decoupled stylistic and hate heads.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        stylistic_categories: list[str] = None,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if stylistic_categories is None:
            stylistic_categories = ["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"]
        self.stylistic_categories = stylistic_categories
        self.num_stylistic = len(stylistic_categories)

        # Load CLIP backbone
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()
        self.freeze_backbone = freeze_backbone

        visual_dim = self.clip.config.projection_dim  # 512 for clip-vit-base-patch32
        text_dim = self.clip.config.projection_dim    # 512

        # Trainable Fusion Layer
        self.fusion = GatedMultimodalFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        # Head 1: Multi-Label Stylistic Classification (independent sigmoids)
        self.stylistic_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, self.num_stylistic),
        )

        # Head 2: Decoupled Hate / Harm Detection (binary classification)
        self.hate_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def extract_backbone_features(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                vision_out = self.clip.get_image_features(pixel_values=pixel_values)
                text_out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            vision_out = self.clip.get_image_features(pixel_values=pixel_values)
            text_out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # Unpack tensor from BaseModelOutputWithPooling if returned by transformers
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            vision_out = vision_out.pooler_output
        elif hasattr(vision_out, "image_embeds") and vision_out.image_embeds is not None:
            vision_out = vision_out.image_embeds
        elif isinstance(vision_out, (tuple, list)):
            vision_out = vision_out[0]

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_out = text_out.pooler_output
        elif hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
            text_out = text_out.text_embeds
        elif isinstance(text_out, (tuple, list)):
            text_out = text_out[0]

        # L2-normalize backbone features
        vision_out = vision_out / vision_out.norm(p=2, dim=-1, keepdim=True)
        text_out = text_out / text_out.norm(p=2, dim=-1, keepdim=True)
        return vision_out, text_out

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v_feat, t_feat = self.extract_backbone_features(pixel_values, input_ids, attention_mask)
        fused = self.fusion(v_feat, t_feat)

        stylistic_logits = self.stylistic_head(fused)
        hate_logits = self.hate_head(fused).squeeze(-1)

        return {
            "stylistic_logits": stylistic_logits,
            "stylistic_probs": torch.sigmoid(stylistic_logits),
            "hate_logits": hate_logits,
            "hate_probs": torch.sigmoid(hate_logits),
        }
