import sounddevice as sd
import librosa
import numpy as np
import torch
from collections import deque
from model import LSTMVAD

SR = 16000
FRAME_LEN = int(0.025 * SR)
HOP_LEN = int(0.010 * SR)
N_MFCC = 13
SEQ_LEN = 20

model = LSTMVAD(N_MFCC)
model.load_state_dict(torch.load("lstm_vad_model.pth", map_location=torch.device('cpu')))
model.eval()

buffer = deque(maxlen=SEQ_LEN)

def callback(indata, frames, time, status):
    audio = indata[:,0]
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        hop_length=HOP_LEN,
        n_fft=FRAME_LEN
    ).T

    for f in mfcc:
        buffer.append(f)
        if len(buffer) == SEQ_LEN:
            x = torch.tensor(
                np.array(buffer)[None,:,:], dtype=torch.float32
            )
            with torch.no_grad():
                pred = torch.argmax(model(x), dim=1).item()
            print("🎤 SPEECH" if pred else "— silence")

with sd.InputStream(
    samplerate=SR,
    channels=1,
    blocksize=FRAME_LEN,
    callback=callback
):
    print("🎧 Real-time VAD running... Press ENTER to stop")
    input()
