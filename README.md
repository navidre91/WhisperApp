# AirPods Voice Recorder (macOS)

Python GUI application that captures audio from a selected CoreAudio input (e.g. AirPods) and saves it to a WAV file. Optional real-time voice isolation is available when the `rnnoise` Python package is installed.

## Features
- Lists all available microphone inputs and preselects AirPods when present.
- Simple record/stop toggle that writes 16-bit WAV files.
- Non-blocking recording using a worker thread so the UI stays responsive.
- Optional voice isolation: RNNoise real-time (48 kHz recommended) or built-in spectral gating fallback.

## Getting Started

1. Create a virtual environment (Apple Silicon friendly helper script):
   ```bash
   ./scripts/bootstrap_venv.sh
   ```

   If you prefer manual steps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install RNNoise for real-time isolation (where wheels are available):
   ```bash
   pip install rnnoise
   ```
   Without this dependency the app still offers a built-in spectral-gating pass that runs after each recording.

4. Run the app:
   ```bash
   python app.py
   ```

The first launch will trigger macOS microphone permissions for Python (or the bundled app if you package it). Ensure you grant access.

## Packaging (Optional)

Use PyInstaller to bundle a `.app`:
```bash
pyinstaller --windowed --name AirPodsRecorder app.py
```

Before distributing, open the generated `AirPodsRecorder.app/Contents/Info.plist` and add an `NSMicrophoneUsageDescription` key so macOS shows a friendly permission prompt message.

## Notes
- RNNoise expects 48 kHz mono audio frames. If your selected device uses a different sample rate, recording still works but real-time suppression is skipped.
- Apple’s proprietary “Voice Isolation” mode from Control Center is not exposed via a public API. RNNoise provides a cross-platform alternative. If you need the native VoiceProcessingIO pipeline, consider bridging to Swift/Objective-C using `PyObjC`.
- The app runs RNNoise in real time when the Python bindings are available; otherwise it post-processes recordings with a lightweight spectral gate and writes a `*-denoised.wav` copy.
- On Apple Silicon (M1/M2/M3), ensure you install Homebrew packages for building native wheels if RNNoise installation fails: `brew install autoconf automake libtool pkg-config`.

## Transcribe with OpenAI Whisper

You can transcribe the original and denoised WAVs using OpenAI’s Whisper API.

1. Install the OpenAI SDK:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your API key (get one from platform.openai.com):
   ```bash
   export OPENAI_API_KEY=sk-...  # macOS/Linux
   # on Windows (PowerShell):
   # setx OPENAI_API_KEY "sk-..."
   ```

3. Transcribe pairs found under `samples/` (creates `.transcript.txt` files alongside audio):
   ```bash
   python scripts/transcribe_openai.py --dir samples --write
   ```

   Or transcribe specific files:
   ```bash
   python scripts/transcribe_openai.py samples/1.wav samples/1-denoised.wav --write
   ```

By default the script looks for pairs like `name.wav` and `name-denoised.wav`. Use `--all` to transcribe every audio file in a directory, or pass specific paths.

## Local (Offline) Whisper

If you prefer to run Whisper locally without the API, use the provided script that wraps the open-source model.

Prerequisites:
- Install ffmpeg (macOS): `brew install ffmpeg`
- Install PyTorch (choose the right wheel for your platform): https://pytorch.org/get-started/locally/
- Install Whisper:
  ```bash
  pip install -U openai-whisper
  ```

Usage:
```bash
# Transcribe pairs found under samples/ and write transcripts
python scripts/transcribe_local_whisper.py --dir samples --write

# Transcribe specific files with a chosen model and auto device
python scripts/transcribe_local_whisper.py samples/1.wav samples/1-denoised.wav \
  --model small --device auto --write

# Force language or switch to translation task
python scripts/transcribe_local_whisper.py --dir samples --language en --task translate --write
```

Notes:
- The first run will download the selected model (e.g., `small`, `medium`, `large-v3`) into your cache.
- `--device auto` will prefer CUDA if available, then Apple `mps`, else CPU.
- On CPU or Apple `mps`, the script uses fp32 for stability; on CUDA it uses fp16.
- For multilingual examples and advanced settings, see the official notebook: https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb
