# dataset.py
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from textgrid_utils import load_intervals

SR = 16000
FRAME_LEN = 0.02   # 20 ms
HOP_LEN = 0.01     # 10 ms

def extract_features(wav, sr):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=int(0.025 * sr),
        hop_length=int(HOP_LEN * sr),
        win_length=int(FRAME_LEN * sr),
        n_mels=80
    )
    logmel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    return np.vstack([logmel, delta, delta2]).T  # (T, 240)

def frame_labels(intervals, n_frames):
    labels = np.zeros(n_frames)
    for i in range(n_frames):
        t = i * HOP_LEN
        for xmin, xmax, lab in intervals:
            if xmin <= t < xmax:
                labels[i] = lab
                break
    return labels

class VADDataset(Dataset):
    def __init__(self, wav_paths, textgrid_paths):
        self.samples = []

        for wav_path, tg_path in zip(wav_paths, textgrid_paths):
            wav, sr = librosa.load(wav_path, sr=SR)
            feats = extract_features(wav, sr)
            intervals = load_intervals(tg_path)
            labels = frame_labels(intervals, len(feats))

            self.samples.append((
                torch.tensor(feats).float(),
                torch.tensor(labels).float()
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
