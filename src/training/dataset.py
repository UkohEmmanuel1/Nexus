from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TokenizedDataset(Dataset):
    """Dataset over tokenized `.pt` shards.

    Each shard must contain an ``input_ids`` key (a sequence or a batch of
    sequences) and optionally ``labels``. Sequences are flattened into a flat
    list so dataloader batching behaves like classic pretraining data loading.
    """

    def __init__(self, shard_paths: list[str]):
        self.sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
        for path in shard_paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            ids = state["input_ids"]
            labels = state.get("labels", ids)
            if torch.is_tensor(ids):
                ids = ids.tolist()
                if torch.is_tensor(labels):
                    labels = labels.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                for seq, lab in zip(ids, labels):
                    self.sequences.append(
                        (torch.tensor(seq, dtype=torch.long), torch.tensor(lab, dtype=torch.long))
                    )
            else:
                self.sequences.append(
                    (torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        input_ids, labels = self.sequences[idx]
        return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch: list[dict], pad_id: int = 0, ignore_index: int = 0) -> dict:
    """Pad a batch of sequences to the longest one and return a dict batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels = [], []
    for item in batch:
        x, y = item["input_ids"], item["labels"]
        input_ids.append(torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id))
        labels.append(torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=ignore_index))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}


def load_shard_paths(shard_dir: str) -> list[str]:
    files = sorted(Path(shard_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt shards found in {shard_dir}")
    return [str(f) for f in files]


def build_train_loader(
    shard_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokenizedDataset(load_shard_paths(shard_dir))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
