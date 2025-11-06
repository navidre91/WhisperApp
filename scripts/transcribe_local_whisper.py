#!/usr/bin/env python3
"""
Transcribe audio locally using the open-source Whisper model (offline).

Usage examples:
  # Transcribe pairs like 1.wav and 1-denoised.wav under samples/
  python scripts/transcribe_local_whisper.py --dir samples --write

  # Transcribe specific files
  python scripts/transcribe_local_whisper.py samples/1.wav samples/1-denoised.wav --write

Notes:
  - Requires ffmpeg installed on your system.
  - Requires PyTorch and the open-source 'openai-whisper' package.
  - The first run will download the specified model to your cache.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple


def _import_whisper():
    try:
        import whisper  # type: ignore
        return whisper
    except Exception as e:
        print(
            "ERROR: Could not import 'whisper'.\n"
            "Install dependencies first:\n"
            "  pip install torch torchvision torchaudio\n"
            "  pip install -U openai-whisper\n"
            "Also ensure ffmpeg is installed (e.g., 'brew install ffmpeg' on macOS).",
            file=sys.stderr,
        )
        raise


def _pick_device(user_choice: str) -> str:
    # Try to pick best device if 'auto'
    if user_choice != "auto":
        return user_choice
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")


def is_audio(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXTS)


def find_pairs(root: str) -> List[Tuple[str, str]]:
    """Find (original, denoised) pairs like name.wav and name-denoised.wav under root."""
    all_audio: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if is_audio(full):
                all_audio.append(os.path.abspath(full))

    by_base: Dict[str, Dict[str, str]] = {}
    for path in all_audio:
        stem, _ = os.path.splitext(os.path.basename(path))
        if stem.endswith("-denoised"):
            base = stem[: -len("-denoised")]
            kind = "denoised"
        else:
            base = stem
            kind = "original"
        by_base.setdefault(base, {})[kind] = path

    pairs: List[Tuple[str, str]] = []
    for _, items in by_base.items():
        if "original" in items and "denoised" in items:
            pairs.append((items["original"], items["denoised"]))

    pairs.sort(key=lambda t: t[0])
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Local Whisper transcription (offline)")
    ap.add_argument("--dir", default="samples", help="Directory to scan for pairs")
    ap.add_argument("--all", action="store_true", help="Transcribe all audio in --dir")
    ap.add_argument("paths", nargs="*", help="Specific files or directories to transcribe")
    ap.add_argument(
        "--model",
        default="small",
        help="Whisper model: tiny | base | small | medium | large | large-v2 | large-v3",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto | cpu | cuda | mps",
    )
    ap.add_argument("--language", default=None, help="Force language code (e.g., en, es). Auto if unset")
    ap.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to run (translate to English)",
    )
    ap.add_argument("--write", action="store_true", help="Write .transcript.txt next to audio")
    ap.add_argument("--verbose", action="store_true", help="Show verbose decoding output")
    args = ap.parse_args()

    # Enable MPS CPU fallback early when user requests/auto-chooses MPS.
    if args.device in ("mps", "auto"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = _pick_device(args.device)
    whisper = _import_whisper()

    # Collect targets
    targets: List[str] = []
    if args.paths:
        for p in args.paths:
            if os.path.isdir(p):
                for dirpath, _, filenames in os.walk(p):
                    for fn in filenames:
                        full = os.path.join(dirpath, fn)
                        if is_audio(full):
                            targets.append(os.path.abspath(full))
            elif is_audio(p):
                targets.append(os.path.abspath(p))
    else:
        if args.all:
            for dirpath, _, filenames in os.walk(args.dir):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    if is_audio(full):
                        targets.append(os.path.abspath(full))
        else:
            pairs = find_pairs(args.dir)
            if not pairs:
                print(
                    f"No (original, denoised) pairs found under {args.dir}. Try --all or pass files.",
                    file=sys.stderr,
                )
                sys.exit(1)
            for orig, den in pairs:
                targets.append(orig)
                targets.append(den)

    # De-dupe preserve order
    seen = set()
    ordered: List[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    print(f"Loading Whisper model '{args.model}' on device '{device}'...")
    try:
        model = whisper.load_model(args.model, device=device)
    except NotImplementedError as e:
        if device == "mps":
            print(
                "MPS op not implemented; falling back to CPU.\n"
                "Tip: keep MPS fallback enabled (PYTORCH_ENABLE_MPS_FALLBACK=1) or run with --device cpu.")
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
        else:
            raise
    except RuntimeError as e:
        msg = str(e)
        if device == "mps" and ("SparseMPS" in msg or "not implemented" in msg or "_sparse_coo_tensor" in msg):
            print(
                "MPS backend hit an unsupported op; falling back to CPU.\n"
                "Tip: run with --device cpu to avoid this on Apple Silicon.")
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
        else:
            raise

    # fp16 only on CUDA by default; on CPU/MPS use fp32 for stability
    fp16 = device == "cuda"

    for path in ordered:
        print(f"Transcribing: {path}")
        try:
            result = model.transcribe(
                path,
                language=args.language,
                task=args.task,
                fp16=fp16,
                verbose=args.verbose,
            )
            text = result.get("text", "").strip()
        except Exception as e:
            print(f"Failed transcribing {path}: {e}", file=sys.stderr)
            continue

        print("--- Transcript start ---")
        print(text)
        print("--- Transcript end ---\n")

        if args.write:
            out_path = f"{os.path.splitext(path)[0]}.transcript.txt"
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                print(f"Wrote: {out_path}")
            except Exception as e:
                print(f"Could not write {out_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


