import dearpygui.dearpygui as dpg
import threading
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import queue
import time

# ================= CONFIG (Matches live_vad.py) =================
SR = 16000
FRAME_LEN = 0.025
HOP_LEN = 0.01
N_MELS = 80

SMOOTH_WIN = 7
SPEECH_ON = 0.6
SPEECH_OFF = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= SHARED STATE =================
# This allows the VAD thread to speak to the GUI thread
state = {
    "confidence": 0.0,
    "speech": False,
    "running": True
}

# ================= MODEL (Matches live_vad.py) =================
class VADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=64 * 120,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).reshape(b, t, c*f)
        x, _ = self.gru(x)
        return self.fc(x).squeeze(-1)

# ================= FEATURES (Matches live_vad.py) =================
def extract_features(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=int(FRAME_LEN * SR),
        hop_length=int(HOP_LEN * SR),
        win_length=int(FRAME_LEN * SR),
        n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    return np.vstack([logmel, delta, delta2]).T

# ================= SMOOTHING (Matches live_vad.py) =================
def smooth(x, win=7):
    if len(x) < win:
        return x
    return np.convolve(x, np.ones(win)/win, mode="same")

# ================= AUDIO CALLBACK =================
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if state["running"]:
        audio_q.put(indata[:, 0].copy())

# ================= VAD LOGIC THREAD =================
# This functions EXACTLY like 'main()' in your script
def vad_thread_target():
    print("🔊 Loading model...")
    model = VADModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load("vad_model19.pt", map_location=DEVICE))
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()

    print("🎤 Starting microphone...")
    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=int(SR * HOP_LEN),
        callback=audio_callback
    )
    stream.start()

    audio_buffer = np.zeros(int(SR * 1.0))
    prob_hist = []
    speech_state = False

    while state["running"]:
        # 1. Queue Draining (Crucial for Low Latency)
        # If the GUI slows down, this loop catches up instantly
        # exactly like your original script
        has_new_data = False
        while not audio_q.empty():
            chunk = audio_q.get()
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            has_new_data = True

        if has_new_data:
            # 2. Inference
            feats = extract_features(audio_buffer)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            prob_hist.extend(probs.tolist())
            prob_hist = prob_hist[-50:]

            smoothed = smooth(np.array(prob_hist), SMOOTH_WIN)
            confidence = float(np.mean(smoothed))

            # 3. Hysteresis
            if not speech_state and confidence > SPEECH_ON:
                speech_state = True
            elif speech_state and confidence < SPEECH_OFF:
                speech_state = False

            # 4. Update Shared State
            state["confidence"] = confidence
            state["speech"] = speech_state
        else:
            # Prevents CPU burning when no audio is present
            time.sleep(0.005)

    stream.stop()
    stream.close()

# ================= GUI MAIN =================
def main_gui():
    dpg.create_context()
    dpg.create_viewport(title="VAD Live Monitor", width=500, height=250)

    # --- Theme Setup ---
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
    dpg.bind_theme(global_theme)

    # --- Layout ---
    with dpg.window(label="Main", tag="Primary Window"):
        dpg.add_text("Real-Time Voice Activity Detection", color=[150, 150, 150])
        dpg.add_spacer(height=10)

        # 1. STATUS TEXT (Fixed crash by removing 'size')
        dpg.add_text("WAITING...", tag="status_text") 
        dpg.add_spacer(height=10)

        # 2. CONFIDENCE BAR
        dpg.add_text("Confidence Level:")
        # We use a larger height to make it readable
        dpg.add_progress_bar(tag="conf_bar", default_value=0.0, width=-1, height=40)
        
        # 3. THRESHOLD MARKERS
        with dpg.group(horizontal=True):
            dpg.add_text(f"OFF ({SPEECH_OFF})", color=[100,100,100])
            dpg.add_spacer(width=280)
            dpg.add_text(f"ON ({SPEECH_ON})", color=[100,100,100])

    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # --- Start VAD Thread ---
    t = threading.Thread(target=vad_thread_target, daemon=True)
    t.start()

    # --- GUI Render Loop ---
    try:
        while dpg.is_dearpygui_running():
            conf = state["confidence"]
            is_speech = state["speech"]

            # Instant UI Update
            dpg.set_value("conf_bar", conf)

            if is_speech:
                dpg.set_value("status_text", "🗣 SPEECH DETECTED")
                dpg.configure_item("status_text", color=[0, 255, 0]) # Bright Green
                dpg.configure_item("conf_bar", overlay="Speech")
            else:
                dpg.set_value("status_text", "🔇 SILENCE")
                dpg.configure_item("status_text", color=[255, 50, 50]) # Red
                dpg.configure_item("conf_bar", overlay="Silence")
            
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        pass
    
    state["running"] = False
    t.join(timeout=1.0)
    dpg.destroy_context()

if __name__ == "__main__":
    main_gui()