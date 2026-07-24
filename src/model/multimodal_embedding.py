import torch
import torch.nn as nn
from typing import List, Optional, Union

from .vision_encoder import VisionEncoder


class MultimodalEmbedding(nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        llm_dim: int,
        vision_encoder: Optional[VisionEncoder] = None,
        image_token_id: int = 128010,
        num_image_tokens: int = 256,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens

        self.text_embed = nn.Embedding(text_vocab_size, llm_dim)
        self.vision_encoder = vision_encoder

        if vision_encoder is not None:
            self.image_placeholder = nn.Parameter(torch.randn(1, num_image_tokens, llm_dim) * 0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_embeds = self.text_embed(input_ids)

        if images is not None and self.vision_encoder is not None:
            image_embeds = self.vision_encoder(images)

            final_embeds = []
            for i in range(input_ids.shape[0]):
                seq = []
                img_idx = 0
                text_pos = 0
                while text_pos < input_ids.shape[1]:
                    if input_ids[i, text_pos].item() == self.image_token_id:
                        seq.append(image_embeds[img_idx:img_idx+1].squeeze(0))
                        img_idx += 1
                        text_pos += 1
                    else:
                        seq.append(text_embeds[i, text_pos:text_pos+1])
                        text_pos += 1
                final_embeds.append(torch.cat(seq, dim=0))

            max_len = max(e.shape[0] for e in final_embeds)
            padded = []
            for e in final_embeds:
                if e.shape[0] < max_len:
                    e = torch.cat([e, torch.zeros(max_len - e.shape[0], e.shape[1], device=e.device)])
                padded.append(e)
            return torch.stack(padded)

        return text_embeds

    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder not initialized")
        return self.vision_encoder(image_tensor)
