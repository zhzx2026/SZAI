from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sz_ai.data import build_datasets, build_vocab, encode_text, read_text
from sz_ai.model import build_model, generate_text, load_checkpoint, resolve_device, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SZ-AI-R1/V1 character-level model.")
    parser.add_argument("--config", default="configs/sz-ai-r1-v1.json", help="Path to the JSON config file.")
    parser.add_argument("--dataset", help="Optional dataset path override.")
    parser.add_argument("--output-dir", default="artifacts/SZ-AI-R1-V1", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--device", help="Device override, for example cpu, cuda, or mps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop each train/eval loop after N batches.")
    parser.add_argument("--sample-prompt", help="Prompt used for the post-training sample.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_data_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    max_steps: int,
) -> tuple[float, int]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    steps = 0
    vocab_size = model.head.out_features

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        if is_train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        steps += 1

        if max_steps and steps >= max_steps:
            break

    average_loss = total_loss / max(steps, 1)
    return average_loss, steps


def main() -> None:
    args = parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)

    if args.dataset:
        config["dataset"]["path"] = args.dataset
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.sample_prompt:
        config["generation"]["sample_prompt"] = args.sample_prompt

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (PROJECT_ROOT / config["dataset"]["path"]).resolve()
    text = read_text(dataset_path, encoding=config["dataset"].get("encoding", "utf-8"))
    vocab, stoi = build_vocab(text)
    token_ids = encode_text(text, stoi)

    set_seed(config.get("seed", 42))
    device = resolve_device(args.device)

    train_dataset, val_dataset = build_datasets(
        token_ids=token_ids,
        seq_len=config["dataset"]["seq_len"],
        stride=config["dataset"]["stride"],
        val_ratio=config["dataset"].get("val_ratio", 0.1),
        seed=config.get("seed", 42),
    )

    train_loader = create_data_loader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = (
        create_data_loader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )
        if val_dataset
        else None
    )

    model = build_model(config["model"], vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_score = math.inf
    history = []
    effective_config_path = output_dir / "effective-config.json"
    effective_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_steps = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=config["training"]["grad_clip"],
            max_steps=args.max_steps,
        )

        val_loss = None
        val_steps = 0
        if val_loader:
            with torch.no_grad():
                val_loss, val_steps = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    grad_clip=0.0,
                    max_steps=args.max_steps,
                )

        score = val_loss if val_loss is not None else train_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_steps": train_steps,
                "val_loss": val_loss,
                "val_steps": val_steps,
            }
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
        )

        if score < best_score:
            best_score = score
            save_checkpoint(
                checkpoint_path=output_dir / "model.pt",
                model=model,
                config=config,
                vocab=vocab,
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "device": str(device),
                },
            )

    metrics = {
        "history": history,
        "best_score": best_score,
        "best_score_type": "val_loss" if val_loader else "train_loss",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model_name": config["name"],
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "dataset_characters": len(text),
        "vocab_size": len(vocab),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset) if val_dataset else 0,
        "epochs": config["training"]["epochs"],
        "best_score": best_score,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_model, payload = load_checkpoint(output_dir / "model.pt", device=device)
    sample_prompt = config["generation"]["sample_prompt"]
    sample = generate_text(
        model=best_model,
        prompt=sample_prompt,
        vocab=payload["vocab"],
        device=device,
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")

    print(f"saved_checkpoint={output_dir / 'model.pt'}")
    print(f"saved_sample={output_dir / 'sample.txt'}")


if __name__ == "__main__":
    main()
