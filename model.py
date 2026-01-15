import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        # weights shape: [batch, seq_len, 1]
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        
        # Context vector: Weighted sum of all frames
        # shape: [batch, hidden_dim]
        context = torch.sum(x * weights, dim=1)
        return context, weights

class CNNLSTMVAD(nn.Module):
    def __init__(self, n_features, n_noise_types):
        super().__init__()
        
        # 1. Deep Feature Extraction (ResNet Style)
        # Input: [Batch, Features, Seq_Len]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            nn.Dropout(0.2)
        )

        # 2. Temporal Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=128, 
            num_layers=2,
            bidirectional=True, 
            batch_first=True,
            dropout=0.2
        )

        # 3. Attention Mechanism
        self.attention = Attention(hidden_dim=256) # 128 * 2 (Bi-directional)

        # 4. Classification Heads
        self.fc_common = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.vad_head = nn.Linear(128, 3)          # Silence, Speech, Noise
        self.noise_head = nn.Linear(128, n_noise_types) # Specific noise type

    def forward(self, x):
        # Input x: [Batch, Seq_Len, Features]
        x = x.transpose(1, 2)  # Conv1d expects [Batch, Channels, Seq_Len]
        
        # CNN
        x = self.cnn(x)        # Out: [Batch, 128, Seq_Len]
        x = x.transpose(1, 2)  # LSTM expects [Batch, Seq_Len, Features]
        
        # LSTM
        self.lstm.flatten_parameters() # Optimization for CUDA
        out, _ = self.lstm(x)  # Out: [Batch, Seq_Len, 256]
        
        # Attention Pooling (Instead of taking just the last step)
        context, attn_weights = self.attention(out) # Context: [Batch, 256]
        
        # Heads
        feat = self.fc_common(context)
        
        return self.vad_head(feat), self.noise_head(feat)