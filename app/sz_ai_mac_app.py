from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sz_ai.model import generate_text, load_checkpoint, resolve_device


APP_TITLE = "SZ-AI Mac Tester"
DEFAULT_PROMPT = "def solve("
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 24


class SZAIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x720")
        self.root.minsize(860, 620)

        self.checkpoint_var = tk.StringVar(value=self._find_default_checkpoint())
        self.max_new_tokens_var = tk.StringVar(value=str(DEFAULT_MAX_NEW_TOKENS))
        self.temperature_var = tk.StringVar(value=str(DEFAULT_TEMPERATURE))
        self.top_k_var = tk.StringVar(value=str(DEFAULT_TOP_K))
        self.device_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Ready")

        self.prompt_text: scrolledtext.ScrolledText | None = None
        self.output_text: scrolledtext.ScrolledText | None = None
        self.generate_button: ttk.Button | None = None

        self._model_cache: dict[str, object] = {}
        self._is_generating = False

        self._build_ui()

    def _find_default_checkpoint(self) -> str:
        candidates = [
            PROJECT_ROOT / "artifacts" / "SZ-AI-Code-R1-V1" / "model.pt",
            PROJECT_ROOT / "artifacts" / "code-smoke" / "model.pt",
            PROJECT_ROOT / "artifacts" / "SZ-AI-R1-V1" / "model.pt",
            PROJECT_ROOT / "artifacts" / "smoke-test" / "model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)
        container.rowconfigure(5, weight=1)

        title = ttk.Label(container, text=APP_TITLE, font=("Helvetica", 20, "bold"))
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            container,
            text="Pick a model checkpoint, enter a prompt, and generate text locally on your Mac.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 14))

        settings = ttk.LabelFrame(container, text="Model Settings", padding=12)
        settings.grid(row=2, column=0, sticky="nsew")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Checkpoint").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        checkpoint_entry = ttk.Entry(settings, textvariable=self.checkpoint_var)
        checkpoint_entry.grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(settings, text="Browse", command=self._browse_checkpoint).grid(row=0, column=2, padx=(8, 0), pady=6)

        ttk.Label(settings, text="Device").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        device_combo = ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=["auto", "cpu", "mps"],
            state="readonly",
        )
        device_combo.grid(row=1, column=1, sticky="w", pady=6)

        numeric_row = ttk.Frame(settings)
        numeric_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for index in range(6):
            numeric_row.columnconfigure(index, weight=1 if index % 2 else 0)

        ttk.Label(numeric_row, text="Max New Tokens").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(numeric_row, textvariable=self.max_new_tokens_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(numeric_row, text="Temperature").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Entry(numeric_row, textvariable=self.temperature_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(numeric_row, text="Top-K").grid(row=0, column=4, sticky="w", padx=(16, 8))
        ttk.Entry(numeric_row, textvariable=self.top_k_var, width=10).grid(row=0, column=5, sticky="w")

        prompt_frame = ttk.LabelFrame(container, text="Prompt", padding=12)
        prompt_frame.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=10, font=("Menlo", 12))
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)

        actions = ttk.Frame(container)
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 12))
        actions.columnconfigure(2, weight=1)

        self.generate_button = ttk.Button(actions, text="Generate", command=self._start_generation)
        self.generate_button.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Clear Output", command=self._clear_output).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

        output_frame = ttk.LabelFrame(container, text="Output", padding=12)
        output_frame.grid(row=5, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Menlo", 12))
        self.output_text.grid(row=0, column=0, sticky="nsew")

    def _browse_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select model.pt checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.checkpoint_var.set(selected)

    def _clear_output(self) -> None:
        if self.output_text is not None:
            self.output_text.delete("1.0", tk.END)

    def _start_generation(self) -> None:
        if self._is_generating:
            return

        try:
            raw_checkpoint = self.checkpoint_var.get().strip()
            if not raw_checkpoint:
                raise ValueError("Please choose a model.pt checkpoint first.")
            checkpoint_path = Path(raw_checkpoint).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint must be a file, not a directory: {checkpoint_path}")
            if checkpoint_path.suffix != ".pt":
                raise ValueError(f"Checkpoint should usually be a .pt file: {checkpoint_path.name}")
            max_new_tokens = int(self.max_new_tokens_var.get().strip())
            temperature = float(self.temperature_var.get().strip())
            top_k = int(self.top_k_var.get().strip())
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            if not prompt:
                raise ValueError("Prompt cannot be empty.")
        except Exception as error:
            messagebox.showerror(APP_TITLE, str(error))
            return

        device_name = self.device_var.get().strip() or "auto"
        self._is_generating = True
        self.status_var.set("Generating...")
        if self.generate_button is not None:
            self.generate_button.state(["disabled"])

        worker = threading.Thread(
            target=self._generate_worker,
            args=(checkpoint_path, prompt, max_new_tokens, temperature, top_k, device_name),
            daemon=True,
        )
        worker.start()

    def _get_cached_model(self, checkpoint_path: Path, device_name: str):
        cache_key = f"{checkpoint_path}:{device_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        device = resolve_device(None if device_name == "auto" else device_name)
        model, payload = load_checkpoint(checkpoint_path, device=device)
        cached = {"model": model, "payload": payload, "device": device}
        self._model_cache = {cache_key: cached}
        return cached

    def _generate_worker(
        self,
        checkpoint_path: Path,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        device_name: str,
    ) -> None:
        try:
            cached = self._get_cached_model(checkpoint_path, device_name)
            text = generate_text(
                model=cached["model"],
                prompt=prompt,
                vocab=cached["payload"]["vocab"],
                device=cached["device"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            metadata = cached["payload"].get("metadata", {})
            model_name = cached["payload"].get("config", {}).get("name", "Unknown model")
            device_label = str(cached["device"])
            epoch = metadata.get("epoch", "n/a")
            self.root.after(
                0,
                lambda: self._on_generation_success(
                    text=text,
                    model_name=model_name,
                    device_label=device_label,
                    epoch=epoch,
                ),
            )
        except Exception:
            error_message = traceback.format_exc()
            self.root.after(0, lambda: self._on_generation_error(error_message))

    def _on_generation_success(self, text: str, model_name: str, device_label: str, epoch: object) -> None:
        self._is_generating = False
        if self.generate_button is not None:
            self.generate_button.state(["!disabled"])
        self.status_var.set(f"Done on {device_label}")

        if self.output_text is not None:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(
                "1.0",
                f"# {model_name}\n# device: {device_label}\n# checkpoint epoch: {epoch}\n\n{text}",
            )

    def _on_generation_error(self, error_message: str) -> None:
        self._is_generating = False
        if self.generate_button is not None:
            self.generate_button.state(["!disabled"])
        self.status_var.set("Failed")
        if self.output_text is not None:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", error_message)
        messagebox.showerror(APP_TITLE, "Generation failed. The traceback is shown in the output panel.")


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    app = SZAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
