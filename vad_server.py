import torch
import torch.nn as nn
import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio

# ================= CONFIG (Matches live_vad.py) =================
SR = 16000
FRAME_LEN = 0.025
HOP_LEN = 0.01
N_MELS = 80

SMOOTH_WIN = 7          # 70 ms
SPEECH_ON = 0.6
SPEECH_OFF = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ================= MODEL (Matches live_vad.py) =================
class VADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),   # 240 -> 120
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
        x = x.unsqueeze(1)              # (B,1,T,240)
        x = self.cnn(x)                 # (B,64,T,120)
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).reshape(b, t, c*f)
        x, _ = self.gru(x)
        return self.fc(x).squeeze(-1)   # LOGITS

# ================= FEATURES (Matches live_vad.py) =================
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
    return np.vstack([logmel, delta, delta2]).T  # (T,240)

# ================= SMOOTHING (Matches live_vad.py) =================
def smooth(x, win=7):
    if len(x) < win:
        return x
    return np.convolve(x, np.ones(win)/win, mode="same")

# ================= SERVER =================
app = FastAPI()

# 1. Load Model (Updated to vad_model19.pt)
model = VADModel().to(DEVICE)
try:
    # Changed from 'vad_model.pt' to 'vad_model19.pt'
    model.load_state_dict(torch.load("vad_model19.pt", map_location=DEVICE))
    print("✅ Model 'vad_model19.pt' loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load 'vad_model19.pt'. {e}")
model.eval()

@app.websocket("/vad")
async def vad_ws(ws: WebSocket):
    await ws.accept()

    # Buffer for 1 second of audio
    audio_buffer = np.zeros(int(SR * 1.0), dtype=np.float32)
    prob_hist = []
    speech_state = False
    
    # Event loop for threading
    loop = asyncio.get_running_loop()

    try:
        while True:
            # 1. Receive data
            data = await ws.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)

            # 2. Update Buffer (Exactly like live_vad.py)
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # 3. Threaded Feature Extraction (Prevents Timeout/Latency)
            feats = await loop.run_in_executor(None, extract_features, audio_buffer)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 4. Inference
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # 5. Smoothing Logic (Exact Copy from live_vad.py)
            prob_hist.extend(probs.tolist())
            prob_hist = prob_hist[-50:] # Keep last 50 frames

            # EXACT LOGIC COPY:
            smoothed = smooth(np.array(prob_hist), SMOOTH_WIN)
            confidence = float(np.mean(smoothed))

            # 6. Hysteresis (Exact Copy from live_vad.py)
            if not speech_state and confidence > SPEECH_ON:
                speech_state = True
            elif speech_state and confidence < SPEECH_OFF:
                speech_state = False

            # 7. Send Result
            await ws.send_json({
                "speech": speech_state,
                "confidence": round(confidence, 3)
            })
            
            # Yield control to prevent event loop blocking
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_timeout=60)