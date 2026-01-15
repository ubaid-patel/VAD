import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm  # <--- NEW IMPORT
from model import CNNLSTMVAD

# ================= CONFIG =================
SR = 16000
FRAME = int(0.025 * SR)
HOP = int(0.010 * SR)
MFCC = 13
SEQ = 25
BATCH = 64
EPOCHS = 50
PATIENCE = 5

AUDIO = "data/Audio"
ANNOT = "data/Annotation"

NOISES = ["Babble", "Car", "Restaurant", "Station", "Street", "Train", "NoNoise"]
NMAP = {n: i for i, n in enumerate(NOISES)}

# ================= TEXTGRID PARSER =================
def parse_tg(path):
    intervals = []
    xmin = xmax = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("xmin ="):
                xmin = float(line.split("=")[1])
            elif line.startswith("xmax ="):
                xmax = float(line.split("=")[1])
            elif line.startswith("text ="):
                label = int(line.split("=")[1].replace('"', ''))
                intervals.append((xmin, xmax, label))

    return intervals

# ================= LOAD DATA =================
print("Loading and processing data...") # Added status print
X, Yv, Yn = [], [], []

for root, _, files in os.walk(AUDIO):
    for file in files:
        if not file.endswith(".wav"):
            continue

        wav_path = os.path.join(root, file)
        tg_path = os.path.join(
            ANNOT,
            os.path.relpath(wav_path, AUDIO)
        ).replace(".wav", ".TextGrid")

        if not os.path.exists(tg_path):
            continue

        # ---- Load audio (librosa >= 0.10 safe)
        audio, _ = librosa.load(wav_path, sr=SR)

        # ---- MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SR,
            n_mfcc=MFCC,
            n_fft=FRAME,
            hop_length=HOP
        )

        # ---- RMS energy (SNR-aware feature)
        rms = librosa.feature.rms(
            y=audio,
            hop_length=HOP
        )

        # ---- Feature stack
        feat = np.vstack([mfcc, rms]).T  # (frames, features)

        # ---- CMVN normalization (CRITICAL)
        feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-6)

        # ---- Labels
        intervals = parse_tg(tg_path)
        vad = np.zeros(len(feat), dtype=np.int64)
        noise = np.full(len(feat), -1, dtype=np.int64)

        is_noisy = "Noizeus" in wav_path
        noise_type = -1
        for n in NOISES:
            if n in wav_path:
                noise_type = NMAP[n]

        for xmin, xmax, lab in intervals:
            s = int(xmin * SR / HOP)
            e = int(xmax * SR / HOP)
            
            # Bound checks
            if s >= len(vad): continue
            e = min(e, len(vad))

            if lab == 1:
                vad[s:e] = 1                 # Speech
            else:
                vad[s:e] = 2 if is_noisy else 0
                if is_noisy:
                    noise[s:e] = noise_type

        # ---- Build sequences
        for i in range(len(feat) - SEQ):
            X.append(feat[i:i+SEQ])
            Yv.append(vad[i+SEQ-1])
            Yn.append(noise[i+SEQ-1])

print(f"Data Loaded: {len(X)} sequences") # Added status print

# ================= FAST TENSOR CONVERSION =================
X = torch.from_numpy(np.array(X, dtype=np.float32))
Yv = torch.from_numpy(np.array(Yv, dtype=np.int64))
Yn = torch.from_numpy(np.array(Yn, dtype=np.int64))

# ================= SPLIT =================
Xt, Xv, Yvt, Yvv, Ynt, Ynv = train_test_split(
    X, Yv, Yn,
    test_size=0.2,
    stratify=Yv,
    random_state=42
)

train_loader = DataLoader(
    TensorDataset(Xt, Yvt, Ynt),
    batch_size=BATCH,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(Xv, Yvv, Ynv),
    batch_size=BATCH
)

# ================= MODEL =================
model = CNNLSTMVAD(
    n_features=X.shape[2],
    n_noise_types=len(NOISES)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_vad = nn.CrossEntropyLoss()
loss_noise = nn.CrossEntropyLoss()

best_val = np.inf
patience_counter = 0
train_losses, val_losses = [], []

# ================= TRAIN =================
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    # ---- Wrapped with tqdm for progress bar
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for xb, yv, yn in train_loop:
        optimizer.zero_grad()
        out_vad, out_noise = model(xb)

        loss = loss_vad(out_vad, yv)

        mask = yn != -1
        if mask.any():
            loss += loss_noise(out_noise[mask], yn[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item()
        
        # Update progress bar with current batch loss
        train_loop.set_postfix(loss=loss.item())

    # ---- Validation
    model.eval()
    val_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        # Iterate over validation without progress bar (optional, usually fast)
        for xb, yv, yn in val_loader:
            out_vad, out_noise = model(xb)
            loss = loss_vad(out_vad, yv)

            mask = yn != -1
            if mask.any():
                loss += loss_noise(out_noise[mask], yn[mask])

            val_loss += loss.item()
            preds.extend(out_vad.argmax(1).cpu().numpy())
            trues.extend(yv.cpu().numpy())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    acc = accuracy_score(trues, preds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print summary after the progress bar closes
    print(
        f"   -> Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {acc:.4f}"
    )

    # ---- Early stopping
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_vad.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered")
            break

# ================= VISUALIZATION =================
print("Training complete. Showing graph...")
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.tight_layout()
plt.show()