#!/usr/bin/env python3
"""
Transcribe original and denoised audio files using OpenAI Whisper.

Usage examples:
  # Transcribe pairs like 1.wav and 1-denoised.wav under samples/
  python scripts/transcribe_openai.py --dir samples --write

  # Transcribe specific files
  python scripts/transcribe_openai.py samples/1.wav samples/1-denoised.wav --write

Requires environment variable OPENAI_API_KEY.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

from openai import OpenAI


AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")


def is_audio(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXTS)


def find_pairs(root: str) -> List[Tuple[str, str]]:
    """Find (original, denoised) pairs like name.wav and name-denoised.wav under root.
    Returns a sorted list of tuples.
    """
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
    for base, items in by_base.items():
        if "original" in items and "denoised" in items:
            pairs.append((items["original"], items["denoised"]))

    pairs.sort(key=lambda t: t[0])
    return pairs


def transcribe(client: OpenAI, path: str, model: str) -> str:
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
        )
    return getattr(resp, "text", str(resp))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcribe original and denoised audio with OpenAI Whisper"
    )
    ap.add_argument(
        "--dir",
        default="samples",
        help="Directory to scan for pairs like name.wav and name-denoised.wav",
    )
    ap.add_argument(
        "--model",
        default="whisper-1",
        help="OpenAI model to use (e.g., whisper-1)",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write .transcript.txt files next to audio",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Transcribe all audio files in --dir (not just pairs)",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Specific audio files or directories to transcribe (overrides --dir)",
    )
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(api_key=api_key)

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

    # De-duplicate while preserving order
    seen = set()
    ordered_targets: List[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            ordered_targets.append(t)

    for path in ordered_targets:
        print(f"Transcribing: {path}")
        try:
            text = transcribe(client, path, args.model).strip()
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

