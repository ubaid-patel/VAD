# ================= COLAB SETUP =================
# Check if running in Colab to install dependencies
import sys
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # For progress bars

# ================= PARAMETERS =================
SR = 16000
FRAME_LEN = int(0.025 * SR)
HOP_LEN = int(0.010 * SR)
N_MFCC = 13
SEQ_LEN = 25

# Hyperparameters for Auto-Epochs
MAX_EPOCHS = 100        # Maximum limit if early stopping doesn't trigger
PATIENCE = 7            # Stop if validation loss doesn't improve for 7 epochs
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

AUDIO_ROOT = "data/Audio"
ANNOT_ROOT = "data/Annotation"

NOISE_TYPES = ["Babble", "Car", "Restaurant", "Station", "Street", "Train", "NoNoise"]
NOISE_MAP = {n: i for i, n in enumerate(NOISE_TYPES)}

# ================= MODEL DEFINITION (Inline) =================
# Defining the model here so you don't need a separate file
class CNNLSTMVAD(nn.Module):
    def __init__(self, n_mfcc, n_noise_types, hidden_dim=64, n_layers=1):
        super(CNNLSTMVAD, self).__init__()
        
        # 1. CNN Feature Extractor (1D Convolution over features)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_mfcc, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        )
        
        # 2. LSTM for Temporal Context
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=False
        )
        
        # 3. Heads
        # VAD Head: 3 classes (0: Silence, 1: Speech, 2: Noise)
        self.fc_vad = nn.Linear(hidden_dim, 3)
        # Noise Head: n_noise_types classes
        self.fc_noise = nn.Linear(hidden_dim, n_noise_types)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features) -> needs (Batch, Features, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.cnn(x)
        
        # Permute back for LSTM: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        # LSTM output
        _, (hn, _) = self.lstm(x)
        
        # Use the final hidden state
        feat = hn[-1] 
        
        vad_logits = self.fc_vad(feat)
        noise_logits = self.fc_noise(feat)
        
        return vad_logits, noise_logits

# ================= HELPER FUNCTIONS =================
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'best_model.pth')

def parse_textgrid(path):
    intervals = []
    try:
        with open(path) as f:
            lines = f.readlines()
        xmin = xmax = label = None
        for l in lines:
            l = l.strip()
            if l.startswith("xmin ="):
                xmin = float(l.split("=")[1])
            elif l.startswith("xmax ="):
                xmax = float(l.split("=")[1])
            elif l.startswith("text ="):
                val = l.split("=")[1].replace('"', '').strip()
                if val: # Check if empty
                    try:
                        label = int(val)
                        intervals.append((xmin, xmax, label))
                    except ValueError:
                        pass # Handle non-integer labels if any
        return intervals
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return []

def intervals_to_labels(intervals, n_frames, is_noisy, noise_type):
    vad = np.zeros(n_frames, dtype=np.int64)
    noise = np.full(n_frames, -1, dtype=np.int64)

    for xmin, xmax, lab in intervals:
        s = int(xmin * SR / HOP_LEN)
        e = int(xmax * SR / HOP_LEN)
        
        # Safety clamp
        s = max(0, s)
        e = min(n_frames, e)

        if lab == 1:
            vad[s:e] = 1
        else:
            if is_noisy:
                vad[s:e] = 2
                noise[s:e] = noise_type
            else:
                vad[s:e] = 0
    return vad, noise

# ================= DATA PROCESSING =================
print("Starting Data Processing...")
X, y_vad, y_noise = [], [], []

# Verify directories exist
if not os.path.exists(AUDIO_ROOT):
    print(f"WARNING: Directory {AUDIO_ROOT} not found.")
    print("Please upload your 'data' folder to Colab or adjust AUDIO_ROOT.")
    # creating dummy data just to let code run if no data found
    print("Creating dummy data for demonstration...")
    X = np.random.randn(100, SEQ_LEN, N_MFCC+1)
    y_vad = np.random.randint(0, 3, 100)
    y_noise = np.random.randint(0, len(NOISE_TYPES), 100)
else:
    file_list = []
    for root, _, files in os.walk(AUDIO_ROOT):
        for f in files:
            if f.endswith(".wav"):
                file_list.append((root, f))

    print(f"Found {len(file_list)} audio files.")
    
    for root, f in tqdm(file_list, desc="Extracting Features"):
        wav = os.path.join(root, f)
        rel = os.path.relpath(wav, AUDIO_ROOT)
        tg = os.path.join(ANNOT_ROOT, rel).replace(".wav", ".TextGrid")

        if not os.path.exists(tg):
            continue

        try:
            audio, _ = librosa.load(wav, sr=SR)
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LEN, n_fft=FRAME_LEN)
            energy = librosa.feature.rms(y=audio, hop_length=HOP_LEN)
            mfcc = np.vstack([mfcc, energy])
            mfcc = mfcc.T

            is_noisy = "Noizeus" in wav
            noise_type_id = -1
            for n in NOISE_TYPES:
                if n in wav:
                    noise_type_id = NOISE_MAP[n]

            vad_labels, noise_labels = intervals_to_labels(
                parse_textgrid(tg), len(mfcc), is_noisy, noise_type_id
            )

            # Sequence generation
            for i in range(0, len(mfcc) - SEQ_LEN, SEQ_LEN): # Stride=SEQ_LEN to reduce overlap for speed
                X.append(mfcc[i:i+SEQ_LEN])
                y_vad.append(vad_labels[i+SEQ_LEN-1])
                y_noise.append(noise_labels[i+SEQ_LEN-1])
        except Exception as e:
            print(f"Skipping {f}: {e}")

X = torch.tensor(np.array(X), dtype=torch.float32)
y_vad = torch.tensor(np.array(y_vad), dtype=torch.long)
y_noise = torch.tensor(np.array(y_noise), dtype=torch.long)

print(f"Dataset shape: {X.shape}")

# ================= SPLIT =================
if len(X) > 0:
    # 80% Train, 20% Temp (Test + Val)
    X_tr, X_temp, yv_tr, yv_temp, yn_tr, yn_temp = train_test_split(
        X, y_vad, y_noise, test_size=0.2, random_state=42, stratify=y_vad
    )
    # Split Temp into 50% Val, 50% Test (so 10% total each)
    X_val, X_te, yv_val, yv_te, yn_val, yn_te = train_test_split(
        X_temp, yv_temp, yn_temp, test_size=0.5, random_state=42, stratify=yv_temp
    )

    train_ds = TensorDataset(X_tr, yv_tr, yn_tr)
    val_ds = TensorDataset(X_val, yv_val, yn_val)
    test_ds = TensorDataset(X_te, yv_te, yn_te)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
else:
    print("No data loaded. Exiting.")
    sys.exit()

# ================= TRAINING SETUP =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = CNNLSTMVAD(n_mfcc=N_MFCC+1, n_noise_types=len(NOISE_TYPES)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_vad_fn = nn.CrossEntropyLoss()
loss_noise_fn = nn.CrossEntropyLoss(ignore_index=-1)

early_stopping = EarlyStopping(patience=PATIENCE, delta=0.001)

# History for plotting
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

# ================= AUTO EPOCH LOOP =================
print("\nStarting Training with Auto-Epochs (Early Stopping)...")

for epoch in range(MAX_EPOCHS):
    # --- TRAIN STEP ---
    model.train()
    train_loss = 0
    
    # Tqdm for progress bar within the epoch
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]", leave=False)
    
    for xb, yv, yn in pbar:
        xb, yv, yn = xb.to(device), yv.to(device), yn.to(device)
        
        opt.zero_grad()
        vad_out, noise_out = model(xb)
        
        l = loss_vad_fn(vad_out, yv) + loss_noise_fn(noise_out, yn)
        l.backward()
        opt.step()
        
        train_loss += l.item()
        pbar.set_postfix({'loss': l.item()})
    
    avg_train_loss = train_loss / len(train_dl)

    # --- VALIDATION STEP ---
    model.eval()
    val_loss = 0
    correct_vad = 0
    total_vad = 0
    
    with torch.no_grad():
        for xb, yv, yn in val_dl:
            xb, yv, yn = xb.to(device), yv.to(device), yn.to(device)
            
            v_out, n_out = model(xb)
            l = loss_vad_fn(v_out, yv) + loss_noise_fn(n_out, yn)
            val_loss += l.item()
            
            # Accuracy metric
            preds = torch.argmax(v_out, dim=1)
            correct_vad += (preds == yv).sum().item()
            total_vad += yv.size(0)

    avg_val_loss = val_loss / len(val_dl)
    val_accuracy = correct_vad / total_vad
    
    # Store history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_accuracy)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    # --- CHECK EARLY STOPPING ---
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load best model
print("Loading best model weights...")
model.load_state_dict(torch.load('best_model.pth'))

# ================= PLOTTING THE PROCESS =================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Val Accuracy', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ================= EVALUATION ON TEST SET =================
print("\nRunning Final Evaluation on Test Set...")
model.eval()
vad_true, vad_pred = [], []

with torch.no_grad():
    for xb, yv, yn in test_dl:
        xb = xb.to(device)
        v, _ = model(xb)
        p = torch.argmax(v, dim=1)
        vad_true.extend(yv.cpu().numpy())
        vad_pred.extend(p.cpu().numpy())

print(classification_report(
    vad_true,
    vad_pred,
    target_names=["Silence", "Speech", "Noise"]
))

# ================= ONNX EXPORT =================
dummy = torch.randn(1, SEQ_LEN, N_MFCC+1).to(device)
torch.onnx.export(
    model,
    dummy,
    "vad_cnn_lstm.onnx",
    input_names=["audio_features"],
    output_names=["vad_logits", "noise_logits"],
    opset_version=12
)

print("✅ Process Complete. Best model saved as 'best_model.pth' and 'vad_cnn_lstm.onnx'")