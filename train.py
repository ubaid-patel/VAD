import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm  # NEW: For progress bars
from model import LSTMVAD

# PARAMETERS
SR = 16000
FRAME_LEN = int(0.025 * SR)
HOP_LEN = int(0.010 * SR)
N_MFCC = 13
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 2

AUDIO_ROOT = "data/Audio"
ANNOT_ROOT = "data/Annotation"

# ---------------- TEXTGRID PARSER ----------------
def parse_textgrid(path):
    intervals = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    xmin = xmax = label = None
    for line in lines:
        line = line.strip()
        if line.startswith("xmin ="):
            try:
                xmin = float(line.split("=")[1])
            except IndexError: pass
        elif line.startswith("xmax ="):
            try:
                xmax = float(line.split("=")[1])
            except IndexError: pass
        elif line.startswith("text ="):
            try:
                # Assumes label is "1" or "0" in TextGrid
                val = line.split("=")[1].replace('"', '').strip()
                if val: # Ensure not empty
                    label = int(val) 
                    intervals.append((xmin, xmax, label))
            except ValueError:
                pass # Skip non-integer labels
    return intervals

def intervals_to_frames(intervals, n_frames):
    labels = np.zeros(n_frames)
    for xmin, xmax, lab in intervals:
        s = int(xmin * SR / HOP_LEN)
        e = int(xmax * SR / HOP_LEN)
        if s < n_frames:
            labels[s:e] = lab
    return labels

def create_sequences(X, y):
    Xs, ys = [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN-1])
    return np.array(Xs), np.array(ys)

# ---------------- LOAD DATASET ----------------
print(f"🔍 Scanning {AUDIO_ROOT} for files...")
X_all, y_all = [], []
file_count = 0

for root, _, files in os.walk(AUDIO_ROOT):
    for f in files:
        if not f.endswith(".wav"):
            continue

        wav_path = os.path.join(root, f)
        rel = os.path.relpath(wav_path, AUDIO_ROOT)
        tg_path = os.path.join(ANNOT_ROOT, rel).replace(".wav", ".TextGrid")

        if not os.path.exists(tg_path):
            continue

        # Load Audio
        audio, _ = librosa.load(wav_path, sr=SR)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SR,
            n_mfcc=N_MFCC,
            hop_length=HOP_LEN,
            n_fft=FRAME_LEN
        ).T

        # Load Labels
        intervals = parse_textgrid(tg_path)
        if not intervals: 
            continue
            
        labels = intervals_to_frames(intervals, len(mfcc))
        
        X_all.append(mfcc)
        y_all.append(labels)
        file_count += 1
        
        print(f"\r   Processed {file_count} files...", end="")

print(f"\n✅ Data processing complete. Stacking arrays...")

X = np.vstack(X_all)
y = np.hstack(y_all)

print(f"📊 Creating sequences (Seq Len: {SEQ_LEN})...")
X, y = create_sequences(X, y)

print(f"✂️ Splitting dataset (Total samples: {len(X)})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- TRAIN ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Starting training on {device}...")

model = LSTMVAD(N_MFCC).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    # Progress bar for training loop
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress bar with current loss
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    
    # ---------------- VALIDATION STEP ----------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ---------------- FINAL EVALUATION ----------------
print("\n📝 Running final evaluation...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = torch.argmax(model(xb), dim=1)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.cpu().numpy())

print("-" * 30)
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print("-" * 30)

save_path = "lstm_vad_model.pth"
torch.save(model.state_dict(), save_path)
print(f"💾 Model saved to: {save_path}")