import numpy as np
import librosa
import torch
import sounddevice as sd
from collections import deque
import time
from model import CNNLSTMVAD
import torch.nn.functional as F

# ================= CONFIG =================
SR = 16000
FRAME = int(0.025 * SR)   # 25 ms
HOP = int(0.010 * SR)
MFCC = 13
SEQ = 25

LOG_INTERVAL = 2.0        # 🔥 seconds

LABELS = {0: "Silence", 1: "Speech", 2: "Noise"}

# ================= LOAD MODEL =================
print("Loading model...")
model = CNNLSTMVAD(
    n_features=MFCC + 1,
    n_noise_types=7
)
model.load_state_dict(torch.load("best_vad.pt", map_location="cpu"))
model.eval()
print("✅ Model loaded")

# ================= BUFFERS =================
feature_buffer = deque(maxlen=SEQ)

logits_buffer = []
probs_buffer = []
last_log_time = 0.0

# ================= FEATURE EXTRACTION =================
def extract_features(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=MFCC,
        n_fft=FRAME,
        hop_length=HOP
    )
    rms = librosa.feature.rms(y=audio, hop_length=HOP)
    feat = np.vstack([mfcc, rms]).T

    # ⚠️ SAME NORMALIZATION AS TRAINING (for diagnosis)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-6)

    return feat

# ================= AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    global last_log_time, logits_buffer, probs_buffer

    audio = indata[:, 0]
    features = extract_features(audio)

    for f in features:
        feature_buffer.append(f)

        if len(feature_buffer) < SEQ:
            continue

        x = torch.from_numpy(
            np.array(feature_buffer, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            vad_logits, _ = model(x)
            probs = F.softmax(vad_logits, dim=1)

        logits_buffer.append(vad_logits.squeeze(0).cpu().numpy())
        probs_buffer.append(probs.squeeze(0).cpu().numpy())

    now = time.time()

    # -------- LOG EVERY 2 SECONDS --------
    if now - last_log_time >= LOG_INTERVAL and len(logits_buffer) > 0:
        mean_logits = np.mean(np.stack(logits_buffer), axis=0)
        mean_probs = np.mean(np.stack(probs_buffer), axis=0)
        pred = int(np.argmax(mean_probs))

        print(
            f"\n⏱ 2s WINDOW\n"
            f"MEAN LOGITS: {mean_logits}\n"
            f"MEAN PROBS : {mean_probs}\n"
            f"PRED      : {LABELS[pred]}\n"
        )

        logits_buffer.clear()
        probs_buffer.clear()
        last_log_time = now

# ================= MAIN =================
if __name__ == "__main__":
    print("\n🎤 Live Mic – RAW MODEL OUTPUT (2s aggregation)")
    print("Press ENTER to start microphone...")
    input()

    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME,
        dtype="float32",
        callback=audio_callback
    )

    try:
        stream.start()
        print("🎙 Microphone started")
        print("Logging raw model outputs every 2 seconds...\n")

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped")

    finally:
        stream.stop()
        stream.close()
        print("✅ Stream closed")
