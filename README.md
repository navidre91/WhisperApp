# AirPods Voice Recorder (macOS)

Python GUI application that captures audio from a selected CoreAudio input (e.g. AirPods) and saves it to a WAV file. Optional real-time voice isolation is available when the `rnnoise` Python package is installed.

## Features
- Lists all available microphone inputs and preselects AirPods when present.
- Simple record/stop toggle that writes 16-bit WAV files.
- Non-blocking recording using a worker thread so the UI stays responsive.
- Optional RNNoise-based voice isolation (48 kHz input recommended).

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

3. (Optional) Enable voice isolation by installing RNNoise:
   ```bash
   pip install rnnoise-cli
   ```

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
- RNNoise expects 48 kHz mono audio frames. If your selected device uses a different sample rate, recording still works but noise suppression is skipped.
- Apple’s proprietary “Voice Isolation” mode from Control Center is not exposed via a public API. RNNoise provides a cross-platform alternative. If you need the native VoiceProcessingIO pipeline, consider bridging to Swift/Objective-C using `PyObjC`.
- On Apple Silicon (M1/M2/M3), ensure you install Homebrew packages for building native wheels if RNNoise installation fails: `brew install autoconf automake libtool pkg-config`.
