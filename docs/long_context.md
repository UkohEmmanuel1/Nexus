# Long Context Processing

Process and generate over documents up to 1M+ tokens using chunking, hierarchical compression, and iterative generation.

## Why Long Context Matters

Standard transformers have O(n²) memory usage and are limited to 4K–128K tokens. Long context processing enables:
- Book/document summarization (500K+ tokens)
- Multi-document analysis across entire codebases
- Long conversation histories
- Scientific paper analysis

## Strategy

The processor uses a three-stage approach:

```
Long Text (1M+ tokens)
    │
    ▼
Chunking ──→ Split into 4K token chunks
    │
    ├── Chunk 1 ──→ Summarize ──→ Summary 1 (256 tokens)
    ├── Chunk 2 ──→ Summarize ──→ Summary 2 (256 tokens)
    ├── ...
    └── Chunk N ──→ Summarize ──→ Summary N (256 tokens)
    │
    ▼
Hierarchical Summarization (if needed)
    │
    ├── Chunk Summaries ──→ Summarize ──→ Meta Summary (256 tokens)
    │
    ▼
Query + Full context of relevant chunks + summaries
    │
    ▼
Generate Answer
```

## Components

### Chunking
Splits documents into overlapping 4K token chunks:

```python
from src.inference.long_context import LongContextProcessor

processor = LongContextProcessor(model, tokenizer, config)

# Process a long document
answer = processor.process_long_document(
    long_text=book_text,          # 500K+ tokens
    query="Summarize the plot",
    max_new_tokens=512,
)
```

### Summarization Mode
When the document exceeds the generation window:

```python
# Chunk-then-summarize approach
answer = processor.process_long_document(
    long_text=very_long_document,  # 1M+ tokens
    query="What are the key findings?",
    mode="summarize",
)
```

### Chunked Generation Mode
When detailed answers from specific sections are needed:

```python
# Find relevant chunks, then answer
answer = processor.process_long_document(
    long_text=document,
    query="What did the experiment show about temperature?",
    mode="chunked",
    top_k_chunks=5,
)
```

## Configuration

```yaml
long_context:
  chunk_size: 4096
  chunk_overlap: 128
  summary_max_tokens: 256
  top_k_chunks: 5
  max_input_tokens: 8192
  hierarchical_summarization: true
```

## Usage Examples

### Single Long Document
```python
processor = LongContextProcessor(model, tokenizer, config)

with open("war_and_peace.txt") as f:
    text = f.read()

# Summarization
summary = processor.process_long_document(
    text, query="Summarize this novel in 3 paragraphs."
)

# Q&A
answer = processor.process_long_document(
    text, query="Describe the relationship between Natasha and Pierre."
)
```

### Multi-Document
```python
document_texts = [
    "doc1.txt: ...",
    "doc2.txt: ...",
]

# Concatenate with document markers
combined = "\n[DOC 1]\n" + document_texts[0] + "\n[DOC 2]\n" + document_texts[1]

answer = processor.process_long_document(
    combined,
    query="What are the key differences between these documents?"
)
```

### Streaming
```python
for chunk in processor.process_long_document_stream(
    long_text=text,
    query="Explain the main argument",
):
    print(chunk, end="", flush=True)
```

## Performance

| Document Size | Strategy | Time | Memory (GPU) |
|--------------|----------|------|--------------|
| 10K tokens | Direct | ~1s | 2 GB |
| 100K tokens | Summarize | ~30s | 4 GB |
| 500K tokens | Summarize | ~3min | 8 GB |
| 1M+ tokens | Hierarchical | ~10min | 12 GB |

## Best Practices

1. **Use direct mode for <8K tokens**: Most efficient for small documents.
2. **Use summarize for >100K tokens**: Hierarchical summarization preserves key information.
3. **Adjust chunk_size based on GPU memory**: Larger chunks give better context but use more memory.
4. **Overlap chunks (128–256 tokens)**: Prevents information loss at chunk boundaries.
5. **Use focused queries**: Specific questions get better answers than vague ones.
