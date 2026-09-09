import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        vit_dim: int = 1024,
        llm_dim: int = 3200,
        n_layers: int = 24,
        n_heads: int = 16,
        num_image_tokens: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_image_tokens = num_image_tokens
        self.vit_dim = vit_dim

        self.patch_embed = nn.Conv2d(
            3, vit_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, vit_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=n_heads,
            dim_feedforward=vit_dim * 4,
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.projector = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim),
        )

        self.perceiver = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, num_image_tokens * llm_dim),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_image_tokens, vit_dim) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = self.patch_embed(images)
        patches = patches.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)

        queries = self.query_tokens.expand(batch_size, -1, -1)
        attn_weights = torch.matmul(queries, x.transpose(1, 2))
        attn_weights = F.softmax(attn_weights / (self.vit_dim**0.5), dim=-1)
        x_pooled = torch.matmul(attn_weights, x)

        projected = self.projector(x_pooled)
        return projected

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return self.forward(image_tensor)


class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, num_queries: int = 32):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, vision_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(vision_dim, num_heads=8, batch_first=True)
        self.linear = nn.Linear(vision_dim, llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(vision_features.shape[0], -1, -1)
        attended, _ = self.cross_attn(queries, vision_features, vision_features)
        return self.linear(attended)
