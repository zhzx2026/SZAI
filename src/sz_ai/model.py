from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from sz_ai.data import decode_tokens, encode_text


class CharLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.rnn(embedded)
        return self.head(hidden_states)


def build_model(model_config: Dict[str, Any], vocab_size: int) -> CharLanguageModel:
    return CharLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=model_config["embedding_dim"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    config: Dict[str, Any],
    vocab: Sequence[str],
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab": list(vocab),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(payload["config"]["model"], len(payload["vocab"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / temperature
    if 0 < top_k < scaled.shape[-1]:
        top_values, top_indices = torch.topk(scaled, top_k, dim=-1)
        probabilities = F.softmax(top_values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return int(top_indices.gather(-1, sampled_index).item())

    probabilities = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate_text(
    model: nn.Module,
    prompt: str,
    vocab: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    stoi = {char: index for index, char in enumerate(vocab)}
    prompt_ids = encode_text(prompt, stoi)
    fallback_id = stoi.get(" ", 0)
    generated_ids = list(prompt_ids) if prompt_ids else [fallback_id]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            generated_ids.append(next_token)

    output = decode_tokens(generated_ids, vocab)
    if not prompt_ids and generated_ids and generated_ids[0] == fallback_id:
        return output[1:]
    return output
