from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch.utils.data import Dataset


def read_text(path: Path, encoding: str = "utf-8") -> str:
    text = path.read_text(encoding=encoding).strip()
    if not text:
        raise ValueError(f"Training text is empty: {path}")
    return text


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int]]:
    vocab = sorted(set(text))
    stoi = {char: index for index, char in enumerate(vocab)}
    return vocab, stoi


def encode_text(text: str, stoi: Dict[str, int], fallback_char: str = " ") -> List[int]:
    fallback_index = stoi.get(fallback_char, 0)
    return [stoi.get(char, fallback_index) for char in text]


def decode_tokens(token_ids: Sequence[int], vocab: Sequence[str]) -> str:
    return "".join(vocab[index] for index in token_ids)


class TextWindowDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], seq_len: int, start_positions: Sequence[int]) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if not start_positions:
            raise ValueError("No training windows were generated. Add more data or reduce seq_len.")
        self.token_ids = list(token_ids)
        self.seq_len = seq_len
        self.start_positions = list(start_positions)

    def __len__(self) -> int:
        return len(self.start_positions)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.start_positions[index]
        stop = start + self.seq_len + 1
        window = self.token_ids[start:stop]
        inputs = torch.tensor(window[:-1], dtype=torch.long)
        targets = torch.tensor(window[1:], dtype=torch.long)
        return inputs, targets


def build_datasets(
    token_ids: Sequence[int],
    seq_len: int,
    stride: int,
    val_ratio: float,
    seed: int,
) -> Tuple[TextWindowDataset, TextWindowDataset | None]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    max_start = len(token_ids) - seq_len - 1
    if max_start < 0:
        raise ValueError("Dataset is too small for the configured seq_len.")

    all_positions = list(range(0, max_start + 1, stride))
    if not all_positions:
        raise ValueError("No training windows were generated from the dataset.")

    if len(all_positions) == 1 or val_ratio <= 0:
        return TextWindowDataset(token_ids, seq_len, all_positions), None

    rng = random.Random(seed)
    rng.shuffle(all_positions)

    raw_val_count = int(len(all_positions) * val_ratio)
    val_count = max(1, raw_val_count)
    if val_count >= len(all_positions):
        val_count = len(all_positions) - 1

    val_positions = sorted(all_positions[:val_count])
    train_positions = sorted(all_positions[val_count:])

    train_dataset = TextWindowDataset(token_ids, seq_len, train_positions)
    val_dataset = TextWindowDataset(token_ids, seq_len, val_positions) if val_positions else None
    return train_dataset, val_dataset
