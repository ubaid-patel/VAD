import torch
import numpy as np
from model import CNNLSTMVAD

# ================= CONFIG =================
SEQ = 25
N_FEATURES = 14   # 13 MFCC + 1 RMS
N_NOISE_TYPES = 7

# ================= LOAD MODEL =================
print("Loading model...")
model = CNNLSTMVAD(
    n_features=N_FEATURES,
    n_noise_types=N_NOISE_TYPES
)

state = torch.load("best_vad.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
print("✅ Model loaded")

# ================= TEST 1: RANDOM INPUT =================
print("\nTest 1: Random input")

x = torch.randn(1, SEQ, N_FEATURES)

with torch.no_grad():
    vad_logits, noise_logits = model(x)

print("VAD logits:", vad_logits.numpy())
print("Noise logits:", noise_logits.numpy())

# ================= TEST 2: ZERO INPUT =================
print("\nTest 2: Zero input")

x_zero = torch.zeros(1, SEQ, N_FEATURES)

with torch.no_grad():
    vad_logits_zero, _ = model(x_zero)

print("VAD logits (zero):", vad_logits_zero.numpy())

# ================= TEST 3: DIFFERENT INPUT =================
print("\nTest 3: Different input")

x2 = torch.randn(1, SEQ, N_FEATURES) * 5.0

with torch.no_grad():
    vad_logits_2, _ = model(x2)

print("VAD logits (different):", vad_logits_2.numpy())

# ================= DECISION =================
if torch.allclose(vad_logits, vad_logits_2):
    print("\n❌ Model output is constant → model is NOT learning")
else:
    print("\n✅ Model outputs vary → model forward pass works")
