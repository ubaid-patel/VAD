# ===================== pyqt_vad.py =====================
import sys
import time
import queue
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# ================= CONFIG =================
SR = 16000
FRAME_LEN = 0.025
HOP_LEN = 0.01
N_MELS = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================
class VADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=64 * 120,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.gru(x)
        return self.fc(x).squeeze(-1)  # logits

# ================= FEATURES =================
def extract_features(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=int(FRAME_LEN * SR),
        hop_length=int(HOP_LEN * SR),
        win_length=int(FRAME_LEN * SR),
        n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    return np.vstack([logmel, delta, delta2]).T

def smooth(x, win):
    if len(x) < win:
        return x
    return np.convolve(x, np.ones(win) / win, mode="same")

# ================= AUDIO =================
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].copy())

# ================= UI =================
class VADApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Voice Activity Detection")
        self.resize(520, 420)

        self.model = VADModel().to(DEVICE)
        self.model.load_state_dict(torch.load("vad_model.pt", map_location=DEVICE))
        self.model.eval()

        self.audio_buffer = np.zeros(int(SR * 1.0))
        self.prob_hist = []
        self.speech_state = False

        self.on_th = 0.6
        self.off_th = 0.4
        self.smooth_win = 7

        self._build_ui()
        self._init_audio()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_vad)
        self.timer.start(20)  # 50 FPS UI

    def _build_ui(self):
        layout = QVBoxLayout()

        self.status = QLabel("🔇 SILENCE")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-size: 26px; color: red;")
        layout.addWidget(self.status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(0, 1)
        self.plot_curve = self.plot.plot(pen="g")
        layout.addWidget(self.plot)

        sliders = QHBoxLayout()

        self.on_slider = QSlider(Qt.Orientation.Horizontal)
        self.on_slider.setRange(40, 90)
        self.on_slider.setValue(60)
        self.on_slider.valueChanged.connect(lambda v: setattr(self, "on_th", v / 100))
        sliders.addWidget(QLabel("Speech ON"))
        sliders.addWidget(self.on_slider)

        self.off_slider = QSlider(Qt.Orientation.Horizontal)
        self.off_slider.setRange(10, 60)
        self.off_slider.setValue(40)
        self.off_slider.valueChanged.connect(lambda v: setattr(self, "off_th", v / 100))
        sliders.addWidget(QLabel("Speech OFF"))
        sliders.addWidget(self.off_slider)

        layout.addLayout(sliders)
        self.setLayout(layout)

    def _init_audio(self):
        self.stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            blocksize=int(SR * HOP_LEN),
            callback=audio_callback
        )
        self.stream.start()

    def update_vad(self):
        while not audio_q.empty():
            chunk = audio_q.get()
            self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
            self.audio_buffer[-len(chunk):] = chunk

        feats = extract_features(self.audio_buffer)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        self.prob_hist.extend(probs.tolist())
        self.prob_hist = self.prob_hist[-100:]

        smoothed = smooth(np.array(self.prob_hist), self.smooth_win)
        conf = float(np.mean(smoothed))

        if not self.speech_state and conf > self.on_th:
            self.speech_state = True
        elif self.speech_state and conf < self.off_th:
            self.speech_state = False

        self.bar.setValue(int(conf * 100))
        self.plot_curve.setData(self.prob_hist)

        if self.speech_state:
            self.status.setText("🗣️ SPEECH")
            self.status.setStyleSheet("font-size: 26px; color: green;")
        else:
            self.status.setText("🔇 SILENCE")
            self.status.setStyleSheet("font-size: 26px; color: red;")

# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    vad = VADApp()
    vad.show()
    sys.exit(app.exec())
