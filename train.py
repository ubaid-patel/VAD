import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from model import CNNLSTMVAD  # Ensure your new model is in model.py

# ================= CONFIGURATION =================
CONFIG = {
    "SR": 16000,
    "FRAME": int(0.025 * 16000),
    "HOP": int(0.010 * 16000),
    "MFCC": 13,
    "SEQ_LEN": 25,           # Number of frames per sample
    "BATCH_SIZE": 64,        # Increase this if you have a good GPU
    "EPOCHS": 50,
    "LR": 1e-3,
    "PATIENCE": 7,           # Early stopping patience
    "NUM_WORKERS": 4,        # Set to 0 if on Windows and getting errors
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NOISE_TYPES": ["Babble", "Car", "Restaurant", "Station", "Street", "Train", "NoNoise"]
}

# ================= AUGMENTATION (SpecAugment) =================
class SpecAugment:
    """Randomly masks time and frequency bands to force the model to be robust."""
    def __init__(self, time_mask_param=5, freq_mask_param=2):
        self.time_mask = time_mask_param
        self.freq_mask = freq_mask_param

    def __call__(self, spec):
        # spec shape: [Features, Time]
        cloned = spec.copy()
        n_feats, n_time = cloned.shape

        # Frequency masking
        f = np.random.randint(0, self.freq_mask)
        f0 = np.random.randint(0, n_feats - f)
        cloned[f0:f0+f, :] = 0

        # Time masking
        t = np.random.randint(0, self.time_mask)
        t0 = np.random.randint(0, n_time - t)
        cloned[:, t0:t0+t] = 0

        return cloned

# ================= DATASET =================
class VADDataset(Dataset):
    def __init__(self, file_list, labels, noise_labels=None, augment=False):
        """
        file_list: List of paths to .wav files
        labels: List of integers [0=Silence, 1=Speech, 2=Noise]
        noise_labels: List of integers for specific noise type (optional)
        """
        self.files = file_list
        self.labels = labels
        self.noise_labels = noise_labels if noise_labels is not None else [-1]*len(labels)
        self.augment = SpecAugment() if augment else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Load Audio
        path = self.files[idx]
        y, _ = librosa.load(path, sr=CONFIG["SR"])

        # 2. Extract Features (Same logic as visualize.py)
        # Pad or truncate to ensure we get exactly SEQ_LEN frames
        target_len = (CONFIG["SEQ_LEN"] - 1) * CONFIG["HOP"] + CONFIG["FRAME"]
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=CONFIG["SR"], n_mfcc=CONFIG["MFCC"], 
                                    n_fft=CONFIG["FRAME"], hop_length=CONFIG["HOP"])
        rms = librosa.feature.rms(y=y, hop_length=CONFIG["HOP"])
        feat = np.vstack([mfcc, rms]) # Shape: [14, Seq]
        
        # 3. Transpose to [Seq, Features] for standardization
        feat = feat.T 
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)

        # 4. Augmentation (Apply on feature map)
        if self.augment:
            # Transpose back for spec augment [Freq, Time]
            feat_T = feat.T
            feat_T = self.augment(feat_T)
            feat = feat_T.T

        return {
            "x": torch.tensor(feat, dtype=torch.float32), # [Seq, Feat]
            "y_vad": torch.tensor(self.labels[idx], dtype=torch.long),
            "y_noise": torch.tensor(self.noise_labels[idx], dtype=torch.long)
        }

# ================= EARLY STOPPING =================
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_vad.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ================= TRAINING LOOP =================
def run_training():
    print(f"🚀 Training on {CONFIG['DEVICE'].upper()}")
    
    # --- MOCK DATA GENERATION (REPLACE THIS WITH YOUR REAL DATA LOADING) ---
    print("Generating dummy data for demonstration...")
    num_samples = 1000
    dummy_files = ["dummy.wav"] * num_samples # Create a real dummy.wav file if running
    # Create a 1 second silence wav just so code runs if you test it immediately
    import soundfile as sf
    sf.write('dummy.wav', np.zeros(16000), 16000)
    
    dummy_vad_labels = np.random.randint(0, 3, num_samples)
    dummy_noise_labels = np.random.randint(0, len(CONFIG["NOISE_TYPES"]), num_samples)
    
    # Split Train/Val
    split = int(0.8 * num_samples)
    train_dataset = VADDataset(dummy_files[:split], dummy_vad_labels[:split], dummy_noise_labels[:split], augment=True)
    val_dataset = VADDataset(dummy_files[split:], dummy_vad_labels[split:], dummy_noise_labels[split:], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    # -----------------------------------------------------------------------

    # Init Model
    model = CNNLSTMVAD(n_features=CONFIG["MFCC"]+1, n_noise_types=len(CONFIG["NOISE_TYPES"]))
    model = model.to(CONFIG["DEVICE"])

    # Loss & Optimizer
    # Weight handling for imbalanced classes (Optional)
    # weights = torch.tensor([1.0, 2.0, 1.5]).to(CONFIG["DEVICE"]) 
    criterion_vad = nn.CrossEntropyLoss() 
    criterion_noise = nn.CrossEntropyLoss(ignore_index=-1) # Ignore noise loss for silence/speech samples if needed

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=CONFIG["PATIENCE"], path='best_vad.pt')
    scaler = GradScaler() # For Mixed Precision

    # Loop
    for epoch in range(CONFIG["EPOCHS"]):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]")
        
        for batch in loop:
            x = batch["x"].to(CONFIG["DEVICE"])
            y_vad = batch["y_vad"].to(CONFIG["DEVICE"])
            y_noise = batch["y_noise"].to(CONFIG["DEVICE"])

            # Zero Gradients
            optimizer.zero_grad()

            # Forward (Mixed Precision)
            with autocast(enabled=(CONFIG["DEVICE"] == 'cuda')):
                out_vad, out_noise = model(x)
                
                loss1 = criterion_vad(out_vad, y_vad)
                # Only calculate noise loss if the ground truth is actually noise (Label 2)
                # Or simply train it on all, relying on ignore_index if labels are missing
                loss2 = criterion_noise(out_noise, y_noise)
                
                loss = loss1 + (0.5 * loss2) # Weight the tasks

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item()
            probs = torch.softmax(out_vad, dim=1)
            preds = torch.argmax(probs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y_vad.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(CONFIG["DEVICE"])
                y_vad = batch["y_vad"].to(CONFIG["DEVICE"])
                y_noise = batch["y_noise"].to(CONFIG["DEVICE"])

                out_vad, out_noise = model(x)
                loss1 = criterion_vad(out_vad, y_vad)
                loss2 = criterion_noise(out_noise, y_noise)
                loss = loss1 + (0.5 * loss2)

                val_loss += loss.item()
                preds = torch.argmax(out_vad, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_vad.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Logging
        print(f"\nResults: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Early Stopping Check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered")
            break

    print("Training Complete. Best model saved as 'best_vad.pt'")

if __name__ == "__main__":
    # Create dummy file to ensure it runs out of the box
    import soundfile as sf
    if not os.path.exists("dummy.wav"):
        sf.write('dummy.wav', np.zeros(16000), 16000)
        
    run_training()