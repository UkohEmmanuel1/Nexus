from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms


class ImageProcessor:
    def __init__(self, image_size: int = 224, mean=None, std=None):
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)

    def load_images(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        tensors = [self.load_image(p) for p in image_paths]
        return torch.cat(tensors, dim=0)

    def process_numpy(self, image_array) -> torch.Tensor:
        img = Image.fromarray(image_array).convert("RGB")
        return self.transform(img).unsqueeze(0)


class MultimodalEngine:
    def __init__(self, model, tokenizer, image_processor: ImageProcessor, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        image_tensor = self.image_processor.load_image(image_path).to(self.device)
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        img_embeds = self.model.module.token_embedding.get_image_embedding(image_tensor) \
            if hasattr(self.model, 'module') else self.model.token_embedding.get_image_embedding(image_tensor)

        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        return self.tokenizer.decode(output_ids[0].tolist())
