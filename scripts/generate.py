from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sz_ai.model import generate_text, load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from the trained SZ-AI-R1/V1 checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/SZ-AI-R1-V1/model.pt", help="Checkpoint path.")
    parser.add_argument("--prompt", default="SZ-AI: ", help="Prompt prefix.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Number of new tokens to sample.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k sampling cutoff.")
    parser.add_argument("--device", help="Device override, for example cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    device = resolve_device(args.device)

    model, payload = load_checkpoint(checkpoint_path, device=device)
    text = generate_text(
        model=model,
        prompt=args.prompt,
        vocab=payload["vocab"],
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
