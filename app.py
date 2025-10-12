"""
Cross-platform microphone recorder GUI for macOS.

Features:
    * Enumerate CoreAudio inputs and try to preselect AirPods.
    * Record audio directly to WAV while keeping the UI responsive.
    * Optional real-time voice isolation powered by RNNoise if available.

Dependencies (install inside a venv):
    pip install PySide6 sounddevice soundfile numpy
    # Optional
    pip install rnnoise
"""

from __future__ import annotations

import datetime
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QWidget,
)

try:
    import rnnoise  # type: ignore
except ImportError:
    rnnoise = None


@dataclass
class InputDevice:
    index: int
    name: str
    hostapi: str
    default_samplerate: int


class AudioRecorder:
    """Handles audio capture, optional RNNoise processing, and file writing."""

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

        self._device_index: Optional[int] = None
        self._samplerate: Optional[int] = None
        self._channels = 1  # AirPods expose mono input; stick to mono to simplify NS.

        self._target_path: Optional[str] = None
        self._soundfile: Optional[sf.SoundFile] = None

        self._running = False
        self._start_ts = 0.0

        self._use_ns = False
        self._ns_engine = rnnoise.RNNoise() if rnnoise else None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def target_path(self) -> Optional[str]:
        return self._target_path

    def available_devices(self) -> List[InputDevice]:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        inputs: List[InputDevice] = []
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                inputs.append(
                    InputDevice(
                        index=idx,
                        name=info["name"],
                        hostapi=hostapis[info["hostapi"]]["name"],
                        default_samplerate=int(info["default_samplerate"]),
                    )
                )
        return inputs

    def choose_device(self, device_index: int) -> None:
        self._device_index = device_index

    def enable_noise_suppression(self, enabled: bool) -> None:
        self._use_ns = enabled and self._ns_engine is not None

    def start(self, target_path: str, samplerate: Optional[int] = None) -> None:
        if self._device_index is None:
            raise RuntimeError("No input device selected")

        device_info = sd.query_devices(self._device_index)
        sr = int(samplerate or device_info["default_samplerate"])

        self._samplerate = sr
        self._target_path = target_path
        self._soundfile = sf.SoundFile(
            target_path,
            mode="w",
            samplerate=sr,
            channels=self._channels,
            subtype="PCM_16",
        )

        self._running = True
        self._start_ts = time.time()

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        blocksize = 480 if (self._use_ns and sr == 48_000) else 1024

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
            if status:
                print(str(status), flush=True)
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=sr,
            channels=self._channels,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None

        if self._soundfile:
            self._soundfile.flush()
            self._soundfile.close()
            self._soundfile = None

    def elapsed_seconds(self) -> int:
        if not self._running:
            return 0
        return int(time.time() - self._start_ts)

    def _writer_loop(self) -> None:
        assert self._soundfile is not None
        while self._running or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            processed = self._process(chunk)
            self._soundfile.write(processed)

        # Flush any trailing data just in case
        while not self._queue.empty():
            self._soundfile.write(self._process(self._queue.get()))

    def _process(self, data: np.ndarray) -> np.ndarray:
        if not self._use_ns or self._ns_engine is None:
            return data

        if self._samplerate != 48_000:
            # RNNoise expects 48 kHz. We skip processing but keep recording.
            return data

        flat = data.reshape(-1)
        frame_size = 480  # 10 ms @ 48 kHz
        n_frames = len(flat) // frame_size
        if n_frames == 0:
            return data

        processed = np.empty(n_frames * frame_size, dtype=np.float32)
        for idx in range(n_frames):
            frame = flat[idx * frame_size : (idx + 1) * frame_size]
            processed[idx * frame_size : (idx + 1) * frame_size] = self._ns_engine.process_frame(frame)

        # Append unprocessed remainder if chunk length is not divisible by frame_size
        remainder = flat[n_frames * frame_size :]
        if remainder.size:
            processed = np.concatenate([processed, remainder.astype(np.float32)], axis=0)

        return processed.reshape(data.shape)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirPods Recorder")
        self._recorder = AudioRecorder()

        self._device_combo = QComboBox()
        self._refresh_btn = QPushButton("Refresh")
        self._record_btn = QPushButton("Record")
        self._status_label = QLabel("Idle")
        self._ns_checkbox = QCheckBox("Enable voice isolation (RNNoise)")

        self._setup_layout()
        self._bind_signals()

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._update_elapsed)

        self._populate_devices()

    def _setup_layout(self) -> None:
        central = QWidget(self)
        layout = QGridLayout(central)

        layout.addWidget(QLabel("Input device"), 0, 0)
        layout.addWidget(self._device_combo, 0, 1)
        layout.addWidget(self._refresh_btn, 0, 2)

        layout.addWidget(self._ns_checkbox, 1, 0, 1, 2)

        layout.addWidget(self._record_btn, 2, 0, 1, 3)
        layout.addWidget(self._status_label, 3, 0, 1, 3)

        self.setCentralWidget(central)
        self.setMinimumWidth(460)

        if rnnoise is None:
            self._ns_checkbox.setChecked(False)
            self._ns_checkbox.setEnabled(False)
            self._ns_checkbox.setToolTip("Install the 'rnnoise' package to enable voice isolation.")

    def _bind_signals(self) -> None:
        self._refresh_btn.clicked.connect(self._populate_devices)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._record_btn.clicked.connect(self._toggle_recording)
        self._ns_checkbox.toggled.connect(self._recorder.enable_noise_suppression)

    def _populate_devices(self) -> None:
        devices = self._recorder.available_devices()
        self._device_combo.blockSignals(True)
        self._device_combo.clear()

        preferred_index = -1
        for idx, dev in enumerate(devices):
            label = f"{dev.name} — {dev.hostapi} @ {dev.default_samplerate} Hz"
            self._device_combo.addItem(label, dev.index)
            if preferred_index == -1 and "airpods" in dev.name.lower():
                preferred_index = idx

        self._device_combo.blockSignals(False)

        if devices:
            self._device_combo.setCurrentIndex(preferred_index if preferred_index != -1 else 0)
            self._on_device_changed(self._device_combo.currentIndex())
            self._status_label.setText("Ready")
        else:
            self._status_label.setText("No input devices found")

    def _on_device_changed(self, combo_index: int) -> None:
        device_index = self._device_combo.itemData(combo_index)
        if device_index is not None:
            self._recorder.choose_device(int(device_index))

    def _toggle_recording(self) -> None:
        if not self._recorder.running:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        default_name = f"recording-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save recording",
            dir=".",
            filter="WAV files (*.wav)",
            selectedFilter="WAV files (*.wav)",
            options=QFileDialog.Options(QFileDialog.DontUseNativeDialog),
        )

        if not target_path:
            return

        if not target_path.lower().endswith(".wav"):
            target_path += ".wav"

        try:
            self._recorder.start(target_path)
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")
            return

        self._record_btn.setText("Stop")
        self._status_label.setText(f"Recording → {target_path}")
        self._timer.start()

    def _stop_recording(self) -> None:
        self._recorder.stop()
        self._record_btn.setText("Record")
        self._status_label.setText("Idle")
        self._timer.stop()

    def _update_elapsed(self) -> None:
        elapsed = self._recorder.elapsed_seconds()
        destination = self._recorder.target_path or "<unknown>"
        self._status_label.setText(f"Recording ({elapsed // 60:02d}:{elapsed % 60:02d}) → {destination}")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
