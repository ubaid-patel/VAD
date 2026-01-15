import librosa
import numpy as np
import torch
from collections import deque
from model import CNNLSTMVAD

# ================= CONFIG =================
SR = 16000
FRAME = int(0.025 * SR)
HOP = int(0.010 * SR)
MFCC = 13
SEQ = 25

LABELS = {
    0: "Silence",
    1: "Speech",
    2: "Noise"
}

# 🔴 CHANGE THIS FILE TO TEST DIFFERENT CASES
# Speech example:
# AUDIO_FILE = "data/Audio/Female/TMIT/SA1.wav"

# Noise example (Noizeus):
AUDIO_FILE = "data/Audio/Male/TMIT/SI532.wav"

# ================= LOAD MODEL =================
print("Loading model...")
model = CNNLSTMVAD(
    n_features=MFCC + 1,
    n_noise_types=7
)
model.load_state_dict(
    torch.load("best_vad.pt", map_location="cpu")
)
model.eval()
print("✅ Model loaded")

# ================= LOAD AUDIO =================
print(f"\nLoading audio: {AUDIO_FILE}")
audio, _ = librosa.load(AUDIO_FILE, sr=SR)

# ================= FEATURE EXTRACTION =================
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SR,
    n_mfcc=MFCC,
    n_fft=FRAME,
    hop_length=HOP
)

rms = librosa.feature.rms(
    y=audio,
    hop_length=HOP
)

# Stack MFCC + RMS (same as training)
features = np.vstack([mfcc, rms]).T

# CMVN normalization (CRITICAL)
features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

# ================= FRAME-BY-FRAME INFERENCE =================
buffer = deque(maxlen=SEQ)
predictions = []

for frame in features:
    buffer.append(frame)

    if len(buffer) == SEQ:
        x = torch.from_numpy(
            np.array(buffer, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            vad_logits, _ = model(x)
            pred = torch.argmax(vad_logits, dim=1).item()

        predictions.append(pred)

# ================= RESULTS =================
unique_preds = set(predictions)

print("\n===== RESULTS =====")
print("Unique predictions:", {LABELS[p] for p in unique_preds})

counts = {LABELS[i]: predictions.count(i) for i in range(3)}
print("Prediction counts:", counts)

total = len(predictions)
print("\nPercentages:")
for k, v in counts.items():
    print(f"{k:8s}: {100*v/total:.2f}%")

print("\nDone.")
