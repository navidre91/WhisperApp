#!/usr/bin/env python3
"""Stream microphone audio into Whisper and print partial transcripts."""

from __future__ import annotations

import argparse
import queue
import sys
import time


def _import_whisper():
    try:
        import whisper  # type: ignore
        return whisper
    except Exception as exc:  # pragma: no cover - guidance only
        print(
            "ERROR: Could not import 'whisper'.\n"
            "Install dependencies first:\n"
            "  pip install torch torchvision torchaudio\n"
            "  pip install -U openai-whisper\n"
            "Also ensure ffmpeg is installed (e.g., 'brew install ffmpeg' on macOS).",
            file=sys.stderr,
        )
        raise exc


def _pick_device(user_choice: str) -> str:
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


def _device_arg(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stream Whisper transcription from the microphone")
    ap.add_argument("--model", default="small", help="Whisper model size to load")
    ap.add_argument("--device", default="auto", help="Device to use: auto | cpu | cuda | mps")
    ap.add_argument("--language", default="English", help="Force language code; auto-detect if unset")
    ap.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to run (translate emits English)",
    )
    ap.add_argument("--sample-rate", type=int, default=16_000, help="Microphone sample rate (Hz)")
    ap.add_argument("--block-ms", type=int, default=50, help="Audio callback size in milliseconds")
    ap.add_argument("--chunk-ms", type=int, default=2_000, help="Audio window per inference call")
    ap.add_argument("--stride-ms", type=int, default=500, help="Context overlap kept between chunks")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to Whisper (0 for deterministic)",
    )
    ap.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Segments above this probability are skipped as silence",
    )
    ap.add_argument(
        "--energy-threshold",
        type=float,
        default=0.0001,
        help="Skip chunks whose RMS energy falls below this value",
    )
    ap.add_argument(
        "--keep-context",
        action="store_true",
        help="Feed previous text back in via initial_prompt",
    )
    ap.add_argument(
        "--context-limit",
        type=int,
        default=200,
        help="Character budget retained when --keep-context is enabled",
    )
    ap.add_argument(
        "--condition-on-prev-text",
        action="store_true",
        help="Let Whisper condition on past tokens (slightly higher latency)",
    )
    ap.add_argument("--queue-size", type=int, default=8, help="Audio queue capacity")
    ap.add_argument(
        "--input-device",
        type=_device_arg,
        default=None,
        help="sounddevice input device index or name",
    )
    ap.add_argument("--max-chunks", type=int, default=None, help="Stop after N processed chunks")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--verbose", action="store_true", help="Print diagnostic information")
    # Rolling window logging options
    ap.add_argument("--rolling-log", action="store_true", help="Enable rolling 30s window logging (3x10s)")
    ap.add_argument("--window-chunks", type=int, default=1, help="Chunks per window (rolling mode)")
    ap.add_argument("--log-file", default="stream.log", help="Append rolling transcripts to this file")
    return ap.parse_args()


def _prepare_audio_backend(args: argparse.Namespace):
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover - guidance only
        print("ERROR: sounddevice is required (pip install sounddevice)", file=sys.stderr)
        raise exc

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    blocksize = max(1, int(args.sample_rate * args.block_ms / 1000))
    return sd, blocksize


def _stream_chunks(args: argparse.Namespace) -> None:
    import numpy as np

    sd, blocksize = _prepare_audio_backend(args)

    # Configure chunk and stride for modes
    if args.rolling_log:
        # Enforce 10s chunks for requested behavior
        args.chunk_ms = 10_000
    chunk_samples = max(1, int(args.sample_rate * args.chunk_ms / 1000))
    stride_samples = int(args.sample_rate * args.stride_ms / 1000)
    if args.rolling_log:
        stride_samples = 0
    if not args.rolling_log and stride_samples >= chunk_samples:
        raise SystemExit("--stride-ms must be smaller than --chunk-ms")

    if args.device in ("mps", "auto"):
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = _pick_device(args.device)
    whisper = _import_whisper()
    print(f"Loading Whisper model '{args.model}' on device '{device}'...")
    try:
        model = whisper.load_model(args.model, device=device)
    except NotImplementedError:
        if device == "mps":
            print(
                "MPS backend missing an op; falling back to CPU.\n"
                "Tip: run with --device cpu to skip this warning next time.",
                file=sys.stderr,
            )
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
        else:
            raise
    except RuntimeError as exc:
        msg = str(exc)
        if device == "mps" and ("SparseMPS" in msg or "_sparse_coo_tensor" in msg or "not implemented" in msg):
            print(
                "MPS backend hit an unsupported op; falling back to CPU.\n"
                "Tip: run with --device cpu to avoid this on Apple Silicon.",
                file=sys.stderr,
            )
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
        else:
            raise
    fp16 = device == "cuda"

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.queue_size)

    def _callback(indata, frames, _time, status):
        if status and args.verbose:
            print(f"sounddevice status: {status}", file=sys.stderr)
        mono = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(mono)
        except queue.Full:
            if args.verbose:
                print("Audio queue full; dropping chunk", file=sys.stderr)

    stream = sd.InputStream(
        samplerate=args.sample_rate,
        channels=1,
        callback=_callback,
        blocksize=blocksize,
        device=args.input_device,
        dtype="float32",
        latency="low",
    )

    print("Mic stream started. Press Ctrl+C to stop.\n")
    start_time = time.time()
    chunk_index = 0

    if args.rolling_log:
        # Rolling window logging: 10s chunk, N-chunk window (default 3)
        window_chunks = args.window_chunks if args.window_chunks and args.window_chunks > 0 else 3
        if window_chunks == 1:
            window_chunks = 3
        window_samples = window_chunks * chunk_samples
        window_buf = np.zeros(0, dtype="float32")
        since_last = 0
        processed_windows = 0
        context_text: str = ""
        try:
            with stream, open(args.log_file, "a", encoding="utf-8") as logfh:
                print(
                    f"Rolling log: chunk={args.chunk_ms/1000:.0f}s, window={window_chunks*args.chunk_ms/1000:.0f}s -> {args.log_file}"
                )
                while True:
                    try:
                        block = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    window_buf = np.concatenate((window_buf, block))
                    since_last += len(block)

                    # Keep only the last window worth of audio
                    if len(window_buf) > window_samples:
                        window_buf = window_buf[-window_samples:]

                    # Trigger once per full chunk worth of new audio
                    while since_last >= chunk_samples:
                        since_last -= chunk_samples
                        chunk_index += 1

                        if len(window_buf) < window_samples:
                            if args.verbose:
                                secs = len(window_buf) / args.sample_rate
                                print(f"[{chunk_index:04d}] waiting for full window ({secs:.1f}s)", file=sys.stderr)
                            continue

                        # Optional energy gate on newest chunk to avoid logging long silence
                        tail = window_buf[-chunk_samples:]
                        rms = float(np.sqrt(np.mean(np.square(tail))))
                        if rms < args.energy_threshold:
                            if args.verbose:
                                print(
                                    f"[{chunk_index:04d}] skipped (tail energy={rms:.4f} < {args.energy_threshold})",
                                    file=sys.stderr,
                                )
                            continue

                        transcribe_kwargs = dict(
                            language=args.language,
                            task=args.task,
                            fp16=fp16,
                            condition_on_previous_text=args.condition_on_prev_text,
                            temperature=args.temperature,
                            no_speech_threshold=args.no_speech_threshold,
                            verbose=False,
                        )

                        if args.keep_context and context_text:
                            transcribe_kwargs["initial_prompt"] = context_text[-args.context_limit :]

                        result = model.transcribe(window_buf, **transcribe_kwargs)
                        text = (result.get("text") or "").strip()
                        if not text:
                            segs = result.get("segments") or []
                            parts = [s.get("text", "").strip() for s in segs]
                            text = " ".join(p for p in parts if p).strip()

                        if not text:
                            if args.verbose:
                                print(f"[{chunk_index:04d}] silence", file=sys.stderr)
                            continue

                        processed_windows += 1
                        elapsed = time.time() - start_time
                        line = text.replace("\n", " ").strip()
                        logfh.write(line + "\n")
                        logfh.flush()
                        print(
                            f"[win {processed_windows:04d} | +{elapsed:7.2f}s] wrote window {window_chunks}x{args.chunk_ms/1000:.0f}s"
                        )

                        if args.keep_context:
                            context_text = (context_text + " " + line).strip()
                            if len(context_text) > args.context_limit:
                                context_text = context_text[-args.context_limit :]

                        if args.max_chunks is not None and chunk_index >= args.max_chunks:
                            raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("\nStopping stream...")
    else:
        # Original incremental stdout mode
        audio_buffer = np.zeros(0, dtype="float32")
        context_text: str = ""
        try:
            with stream:
                while args.max_chunks is None or chunk_index < args.max_chunks:
                    try:
                        block = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    audio_buffer = np.concatenate((audio_buffer, block))

                    while len(audio_buffer) >= chunk_samples:
                        chunk = audio_buffer[:chunk_samples]
                        step = chunk_samples - stride_samples if stride_samples > 0 else chunk_samples
                        audio_buffer = audio_buffer[step:]

                        chunk_index += 1
                        guard = args.stride_ms / 1000.0 if chunk_index > 1 else 0.0
                        rms = float(np.sqrt(np.mean(np.square(chunk))))
                        if rms < args.energy_threshold:
                            if args.verbose:
                                print(
                                    f"[{chunk_index:04d}] skipped (energy={rms:.4f} < {args.energy_threshold})",
                                    file=sys.stderr,
                                )
                            continue

                        transcribe_kwargs = dict(
                            language=args.language,
                            task=args.task,
                            fp16=fp16,
                            condition_on_previous_text=args.condition_on_prev_text,
                            temperature=args.temperature,
                            no_speech_threshold=args.no_speech_threshold,
                            verbose=False,
                        )

                        if args.keep_context and context_text:
                            transcribe_kwargs["initial_prompt"] = context_text[-args.context_limit :]

                        result = model.transcribe(chunk, **transcribe_kwargs)
                        segments = result.get("segments") or []

                        chunk_parts = []
                        for seg in segments:
                            if seg.get("end", 0.0) <= guard:
                                continue
                            if seg.get("no_speech_prob", 0.0) > args.no_speech_threshold:
                                continue
                            text_part = seg.get("text", "").strip()
                            if text_part:
                                chunk_parts.append(text_part)

                        if not chunk_parts:
                            text = result.get("text", "").strip()
                            chunk_parts = [text] if text else []

                        chunk_text = " ".join(chunk_parts).strip()
                        if not chunk_text:
                            if args.verbose:
                                print(f"[{chunk_index:04d}] silence", file=sys.stderr)
                            continue

                        if args.keep_context:
                            context_text = (context_text + " " + chunk_text).strip()
                            if len(context_text) > args.context_limit:
                                context_text = context_text[-args.context_limit :]

                        elapsed = time.time() - start_time
                        print(f"[{chunk_index:04d} | {elapsed:7.2f}s] {chunk_text}")
                        sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nStopping stream...")


def main() -> None:
    args = parse_args()
    _stream_chunks(args)


if __name__ == "__main__":
    main()
