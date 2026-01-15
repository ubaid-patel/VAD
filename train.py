# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VADDataset
import glob
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL ================= #
class VADModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),     # 240 → 120
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # CNN output: (B, 64, T, 120) → 64×120
        self.gru = nn.GRU(
            input_size=64 * 120,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, T, 240)
        x = x.unsqueeze(1)          # (B,1,T,240)
        x = self.cnn(x)             # (B,64,T,120)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)

        x, _ = self.gru(x)          # (B,T,256)
        x = torch.sigmoid(self.fc(x)).squeeze(-1)

        return x

# ================= TRAIN ================= #
def main():
    wavs = sorted(glob.glob("data/Audio/**/*.wav", recursive=True))
    grids = sorted(glob.glob("data/Annotation/**/*.TextGrid", recursive=True))

    dataset = VADDataset(wavs, grids)

    loader = DataLoader(
        dataset,
        batch_size=1,        # variable-length safe
        shuffle=True,
        num_workers=0
    )

    model = VADModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 🔥 CRITICAL FIX: class-weighted loss
    # Speech frames are rare → boost their importance
    criterion = nn.BCELoss(
        pos_weight=torch.tensor([3.0], device=DEVICE)
    )

    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            unit="file"
        )

        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), "vad_model.pt")
    print("🎉 Training finished — model saved as vad_model.pt")

if __name__ == "__main__":
    main()
