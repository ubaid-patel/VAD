# train_vad.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VADDataset
import glob
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=64 * 40,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).reshape(b, t, -1)
        x, _ = self.gru(x)
        x = torch.sigmoid(self.fc(x)).squeeze(-1)
        return x

def main():
    wavs = sorted(glob.glob("data/Audio/**/*.wav", recursive=True))
    grids = sorted(glob.glob("data/Annotation/**/*.TextGrid", recursive=True))

    dataset = VADDataset(wavs, grids)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = VADModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(30):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "vad_model.pt")
    print("✅ Model saved as vad_model.pt")

if __name__ == "__main__":
    main()
