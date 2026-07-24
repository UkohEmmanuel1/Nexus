# Data Pipeline

Process raw text data into tokenized training examples. The pipeline handles dataset discovery, downloading, cleaning, tokenization, and sequence packing.

## Pipeline Overview

```
Raw Sources (web, books, code, etc.)
    │
    ▼
Data Discovery ──→ Find and register datasets
    │
    ▼
Data Download ──→ Download from HuggingFace/sources
    │
    ▼
Data Cleaning ──→ Dedup, filter, normalize
    │
    ▼
Data Split ──→ Train / Validation / Test (0.98 / 0.01 / 0.01)
    │
    ▼
Tokenize ──→ SentencePiece (128k vocab)
    │
    ▼
Sequence Pack ──→ Concatenate + truncate to max_seq_len
    │
    ▼
Save (.pt / Arrow)
```

## Components

### Pipeline (`src/data/pipeline.py`)

```python
from src.data.pipeline import DataPipeline

pipeline = DataPipeline(
    name="pretrain",
    tokenizer_path="tokenizer/tokenizer.model",
    max_seq_len=4096,
    split_ratio=(0.98, 0.01, 0.01),
)

# Load and prepare
pipeline.load_datasets(
    sources=["huggingface:wikitext", "local:data/raw/my_corpus"]
)

# Split and save
pipeline.split_and_save(output_dir="data/tokenized")

# Get dataloaders
train_loader, val_loader, test_loader = pipeline.get_dataloaders(
    batch_size=8,
    num_workers=4,
)
```

### Tokenization (`src/data/tokenize_dataset.py`)

```bash
# Tokenize a raw text corpus
python -m src.data.tokenize_dataset \
    --input data/raw/wikitext \
    --output data/tokenized/wikitext \
    --tokenizer tokenizer/tokenizer.model \
    --max_seq_len 4096 \
    --format text
```

Output structure:
```
data/tokenized/
├── train/
│   ├── shard_0001.pt
│   ├── shard_0002.pt
│   └── ...
├── val/
│   └── shard_0001.pt
└── test/
    └── shard_0001.pt
```

Each `.pt` file:
```python
{
    "input_ids": torch.tensor([...]),   # (seq_len,)
    "labels": torch.tensor([...]),       # (seq_len,) - same as input_ids for LM
}
```

## Dataset Sources

| Source | Format | Command |
|--------|--------|---------|
| HuggingFace | Auto | `source: "huggingface:dataset_name"` |
| Local text | `.txt` | `source: "local:path/to/dir"` |
| JSONL | `.jsonl` | `source: "jsonl:path/to/file.jsonl"` |
| Parquet | `.parquet` | `source: "parquet:path/to/file.parquet"` |

## Supported Transforms

```yaml
pipeline:
  transforms:
    - dedup_exact: true           # Remove exact duplicates
    - dedup_minhash: false        # MinHash dedup (for large datasets)
    - filter_too_short: 200       # Remove <200 tokens
    - filter_too_long: 1048576    # Remove >1M tokens
    - filter_high_perplexity: 15  # Filter low-quality docs
    - normalize_whitespace: true  # Normalize spaces
    - normalize_unicode: true     # NFKC normalization
```

## Sequence Packing

Multiple short sequences are concatenated to fill the max_seq_len:

```
Before:
[doc1_tokens] [SEP] [doc2_tokens] [SEP] [doc3_partial_tokens] [PAD...]

After packing:
[doc1_tokens] [SEP] [doc2_tokens] [SEP] [doc3_tokens] [SEP] [doc4_partial...]
```

Packing improves training efficiency by minimizing padding.

## Data Configuration

```yaml
pipeline:
  name: pretrain
  tokenizer_path: tokenizer/tokenizer.model
  max_seq_len: 4096
  split_ratio: [0.98, 0.01, 0.01]
  transforms:
    dedup_exact: true
    normalize_whitespace: true
  dataloader:
    batch_size: 8
    num_workers: 4
    prefetch_factor: 2
  output_dir: data/tokenized
```

## Best Practices

1. **Deduplicate**: Removes 10–30% of Common Crawl data, significantly improving quality.
2. **Filter low-quality**: Use perplexity-based filtering with a small reference model.
3. **Balance domains**: Mix web, books, code, and academic sources for diverse training.
4. **Shuffle globally**: Distribute shards to avoid temporal bias from ordered data.
5. **Cache tokenized data**: Tokenization is expensive; cache the results on fast storage.
