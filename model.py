import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMVAD(nn.Module):
    def __init__(self, n_mfcc=13, n_noise_types=7):
        super().__init__()

        # CNN (local spectral patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM (temporal modeling)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Heads
        self.vad_head = nn.Linear(256, 3)          # Silence / Speech / Noise
        self.noise_head = nn.Linear(256, n_noise_types)  # Noise types

    def forward(self, x):
        # x: (B, T, MFCC)
        x = x.transpose(1, 2)       # (B, MFCC, T)
        x = self.cnn(x)
        x = x.transpose(1, 2)       # (B, T, 64)

        lstm_out, _ = self.lstm(x)
        feat = lstm_out[:, -1, :]   # last frame

        vad_logits = self.vad_head(feat)
        noise_logits = self.noise_head(feat)

        return vad_logits, noise_logits

    def confidence_scores(self, vad_logits):
        probs = F.softmax(vad_logits, dim=1)
        return {
            "silence": probs[:, 0],
            "speech":  probs[:, 1],
            "noise":   probs[:, 2]
        }
