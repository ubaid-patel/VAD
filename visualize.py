import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from vad_model import LSTMVAD

SR = 16000
FRAME_LEN = int(0.025 * SR)
HOP_LEN = int(0.010 * SR)
N_MFCC = 13
SEQ_LEN = 20

model = LSTMVAD(N_MFCC)
model.load_state_dict(torch.load("lstm_vad_model.pth"))
model.eval()

audio, _ = librosa.load("example.wav", sr=SR)
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SR,
    n_mfcc=N_MFCC,
    hop_length=HOP_LEN,
    n_fft=FRAME_LEN
).T

X = np.array([mfcc[i:i+SEQ_LEN] for i in range(len(mfcc)-SEQ_LEN)])

with torch.no_grad():
    preds = torch.argmax(
        model(torch.tensor(X, dtype=torch.float32)), dim=1
    ).numpy()

t = np.arange(len(audio)) / SR
plt.figure(figsize=(14,4))
plt.plot(t, audio, alpha=0.6)

for i, p in enumerate(preds):
    if p == 1:
        start = i * HOP_LEN / SR
        end = start + HOP_LEN / SR
        plt.axvspan(start, end, color="red", alpha=0.3)

plt.title("Speech Activity Detection (Red = Speech)")
plt.show()
