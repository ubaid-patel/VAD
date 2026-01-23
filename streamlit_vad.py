# ===================== streamlit_vad.py =====================
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import queue
import time

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
        x = x.permute(0,2,1,3).reshape(b, t, c*f)
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
    return np.convolve(x, np.ones(win)/win, mode="same")

# ================= AUDIO =================
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].copy())

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Live VAD", layout="centered")
st.title("🎤 Live Voice Activity Detection")

on_th = st.slider("Speech ON threshold", 0.4, 0.9, 0.6, 0.01)
off_th = st.slider("Speech OFF threshold", 0.1, 0.6, 0.4, 0.01)
smooth_win = st.slider("Smoothing window (frames)", 3, 15, 7)

start = st.button("▶ Start VAD")
stop = st.button("⏹ Stop VAD")

status_box = st.empty()
conf_bar = st.progress(0)
chart = st.line_chart([])

@st.cache_resource
def load_model():
    m = VADModel().to(DEVICE)
    m.load_state_dict(torch.load("vad_model19.pt", map_location=DEVICE))
    m.eval()
    return m

model = load_model()

if start:
    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=int(SR * HOP_LEN),
        callback=audio_callback
    )
    stream.start()

    audio_buffer = np.zeros(int(SR * 1.0))
    probs_hist = []
    speech_state = False

    while not stop:
        while not audio_q.empty():
            chunk = audio_q.get()
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

        feats = extract_features(audio_buffer)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        probs_hist.extend(probs.tolist())
        probs_hist = probs_hist[-50:]

        smoothed = smooth(np.array(probs_hist), smooth_win)
        confidence = float(np.mean(smoothed))

        if not speech_state and confidence > on_th:
            speech_state = True
        elif speech_state and confidence < off_th:
            speech_state = False

        label = "🗣️ SPEECH" if speech_state else "🔇 SILENCE"
        color = "green" if speech_state else "red"

        status_box.markdown(f"### <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        conf_bar.progress(min(int(confidence * 100), 100))
        chart.add_rows([confidence])

        time.sleep(0.02)

    stream.stop()
    stream.close()
