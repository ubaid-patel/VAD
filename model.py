import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMVAD(nn.Module):
    def __init__(self, n_features, n_noise_types):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            64, 128, num_layers=2,
            bidirectional=True, batch_first=True
        )

        self.vad_head = nn.Linear(256, 3)
        self.noise_head = nn.Linear(256, n_noise_types)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        out, _ = self.lstm(x)
        feat = out[:, -1]

        return self.vad_head(feat), self.noise_head(feat)
