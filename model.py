import torch
import torch.nn as nn

class LSTMVAD(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
