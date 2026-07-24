# Multimodal (Vision) Guide

Add vision understanding capabilities to the model. The system uses a Vision Transformer (ViT) encoder with a Q-Former projector to convert images into token embeddings that the LLM can process.

## Architecture

```
Image (224×224)
    │
    ▼
ViT Encoder (patch_size=14)
    │
    ▼
Image Features (num_patches + 1 × vit_dim)
    │
    ▼
Q-Former Projector (cross-attention)
    │
    ▼
Image Tokens (256 × llm_dim)
    │
    ▼
┌─────────────────────────────────────┐
│ Multimodal Embedding               │
│  [Text Tokens ... ] + [Image Tokens] │
└──────────────┬──────────────────────┘
               │
               ▼
            Transformer
```

## Components

### Vision Encoder (`VisionEncoder`)
- **Backbone**: ViT (Vision Transformer)
- **Input**: 224×224 RGB image
- **Patch size**: 14×14 (256 patches)
- **Output**: 257 tokens (256 patch + 1 CLS) of dimension `vit_dim`
- **Layers**: 24 transformer layers
- **Heads**: 16 attention heads

### Q-Former Projector
- Learned queries cross-attend to ViT output
- Compresses 257 vision tokens to 256 compact tokens
- 2-layer MLP projects `vit_dim → llm_dim`

### Multimodal Embedding (`MultimodalEmbedding`)
- Replaces image token IDs with projected vision features
- Supports interleaved text and image inputs
- Special image token (`<image>`) marks image positions

## Usage

### Inference with Image
```python
from src.inference.multimodal import ImageProcessor, MultimodalEngine
from src.model.vision_encoder import VisionEncoder

# Initialize
image_processor = ImageProcessor(image_size=224)
vision_encoder = VisionEncoder(
    image_size=224,
    patch_size=14,
    vit_dim=1024,
    llm_dim=model.config.dim,
    num_image_tokens=256,
)

# Load and process image
image_tensor = image_processor.load_image("photo.jpg")

# Generate with context
response = engine.generate_with_image(
    prompt="Describe this image in detail",
    image_path="photo.jpg",
    max_new_tokens=256,
)
```

### REST API
```json
POST /v1/chat/completions
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQ..."
        }}
      ]
    }
  ],
  "max_tokens": 256
}
```

### Image Processing Pipeline
```python
from PIL import Image
from src.inference.multimodal import ImageProcessor

processor = ImageProcessor(
    image_size=224,
    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
    std=[0.229, 0.224, 0.225],
)

# Load from file
tensor = processor.load_image("photo.jpg")

# Load multiple images
tensors = processor.load_images(["photo1.jpg", "photo2.jpg"])

# Process numpy array (e.g., from cv2)
import cv2
frame = cv2.imread("frame.jpg")
tensor = processor.process_numpy(frame)
```

## Training

### Stage 1: Vision Encoder Pretraining
Train the ViT and Q-Former on image-text pairs:
- Dataset: CC3M, CC12M, LAION-400M
- Objective: ITC (Image-Text Contrastive) + ITM (Image-Text Matching) + LM (Language Modeling)
- Duration: 50K–100K steps

### Stage 2: Multimodal Fine-Tuning
Jointly fine-tune the vision encoder and LLM:
- Dataset: multimodal instruction data
- Objective: next-token prediction on text with image context
- Duration: 5K–10K steps

## Configuration

```yaml
model:
  vision:
    enabled: true
    image_size: 224
    patch_size: 14
    vit_dim: 1024
    vit_layers: 24
    vit_heads: 16
    num_image_tokens: 256
    image_token_id: 128010
```

## Performance

| Image Size | Patches | ViT Dim | Tokens | Perf |
|-----------|---------|---------|--------|------|
| 224×224 | 256 | 1024 | 256 | Fast |
| 336×336 | 576 | 1024 | 256 | Better quality |
| 448×448 | 1024 | 1280 | 512 | Best quality |

## Known Limitations

- **Single image modality**: Text + image only (no audio/video yet)
- **Training data**: Requires image-text paired datasets
- **Latency**: Vision encoding adds ~50ms per image on GPU
- **Resolution**: Currently fixed at 224×224; high-res support planned
