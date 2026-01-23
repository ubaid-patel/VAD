import dearpygui.dearpygui as dpg
import threading
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import queue
import time

# ================= CONFIG (Unchanged) =================
SR = 16000
FRAME_LEN = 0.025
HOP_LEN = 0.01
N_MELS = 80

SMOOTH_WIN = 7
SPEECH_ON = 0.6
SPEECH_OFF = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= SHARED STATE =================
state = {
    "confidence": 0.0,
    "speech": False,
    "running": True
}

# ================= MODEL =================
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
        self.gru = nn.GRU(64 * 120, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).reshape(b, t, c*f)
        x, _ = self.gru(x)
        return self.fc(x).squeeze(-1)

# ================= FEATURES =================
def extract_features(wav):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=int(FRAME_LEN * SR), hop_length=int(HOP_LEN * SR),
        win_length=int(FRAME_LEN * SR), n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    return np.vstack([logmel, delta, delta2]).T

# ================= SMOOTHING =================
def smooth(x, win=7):
    if len(x) < win: return x
    return np.convolve(x, np.ones(win)/win, mode="same")

# ================= AUDIO CALLBACK =================
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if state["running"]:
        audio_q.put(indata[:, 0].copy())

# ================= VAD THREAD (Logic) =================
def vad_thread_target():
    print("🔊 Loading model...")
    model = VADModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load("vad_model19.pt", map_location=DEVICE))
    except:
        print("❌ Error loading model.")
        return

    model.eval()
    stream = sd.InputStream(samplerate=SR, channels=1, blocksize=int(SR * HOP_LEN), callback=audio_callback)
    stream.start()

    audio_buffer = np.zeros(int(SR * 1.0))
    prob_hist = []
    speech_state = False

    while state["running"]:
        # --- Queue Draining (Latency Fix) ---
        has_new_data = False
        while not audio_q.empty():
            chunk = audio_q.get()
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            has_new_data = True

        if has_new_data:
            feats = extract_features(audio_buffer)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            prob_hist.extend(probs.tolist())
            prob_hist = prob_hist[-50:]
            smoothed = smooth(np.array(prob_hist), SMOOTH_WIN)
            confidence = float(np.mean(smoothed))

            if not speech_state and confidence > SPEECH_ON: speech_state = True
            elif speech_state and confidence < SPEECH_OFF: speech_state = False

            state["confidence"] = confidence
            state["speech"] = speech_state
        else:
            time.sleep(0.005)

    stream.stop()
    stream.close()

# ================= GUI MAIN =================
def main_gui():
    dpg.create_context()
    dpg.create_viewport(title="Professional VAD Monitor", width=600, height=450)
    
    # Increase global font size for better readability
    dpg.set_global_font_scale(1.25)

    # --- THEME (Dark & Round) ---
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_PlotRounding, 6)
            # Make the plot line green
            dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 255, 100), category=dpg.mvThemeCat_Plots)
    dpg.bind_theme(global_theme)

    # --- DATA BUFFERS FOR PLOT ---
    # We will show 200 frames of history
    plot_x = list(range(200))
    plot_y = [0.0] * 200

    # --- LAYOUT ---
    with dpg.window(label="Main", tag="Primary Window"):
        dpg.add_text("VOICE ACTIVITY DETECTION", color=[150, 150, 150])
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # 1. STATUS "LED" & TEXT
        with dpg.group(horizontal=True):
            # We draw a circle button that acts as an LED
            dpg.add_button(label="", width=30, height=30, tag="status_led")
            dpg.add_spacer(width=10)
            dpg.add_text("INITIALIZING...", tag="status_text", size=30) # Font scale handles size

        dpg.add_spacer(height=15)

        # 2. CONFIDENCE BAR
        dpg.add_text("Confidence:")
        dpg.add_progress_bar(tag="conf_bar", default_value=0.0, width=-1, height=30)
        
        # Thresholds
        with dpg.group(horizontal=True):
            dpg.add_text(f"OFF ({SPEECH_OFF})", color=[100,100,100], size=15)
            dpg.add_spacer(width=350)
            dpg.add_text(f"ON ({SPEECH_ON})", color=[100,100,100], size=15)

        dpg.add_spacer(height=15)

        # 3. REAL-TIME PLOT
        with dpg.plot(label="Confidence History", height=-1, width=-1, no_mouse_pos=True):
            dpg.add_plot_legend()
            # X Axis
            dpg.add_plot_axis(dpg.mvXAxis, label="Time", no_tick_labels=True)
            # Y Axis (Fixed 0 to 1)
            with dpg.add_plot_axis(dpg.mvYAxis, label="Probability"):
                dpg.set_axis_limits(dpg.last_item(), 0, 1.1)
                # The line series
                dpg.add_line_series(plot_x, plot_y, tag="plot_series", label="Confidence")
                # Add threshold lines
                dpg.add_shade_series(plot_x, [SPEECH_ON]*200, y2=[SPEECH_ON]*200, color=[0, 255, 0, 50], label="ON Threshold")

    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # --- START THREAD ---
    t = threading.Thread(target=vad_thread_target, daemon=True)
    t.start()

    # --- RENDER LOOP ---
    try:
        while dpg.is_dearpygui_running():
            conf = state["confidence"]
            is_speech = state["speech"]

            # Update Plot Data
            plot_y.pop(0)
            plot_y.append(conf)
            dpg.set_value("plot_series", [plot_x, plot_y])

            # Update Bar
            dpg.set_value("conf_bar", conf)

            # Update LED & Status Text
            if is_speech:
                dpg.set_value("status_text", "SPEECH DETECTED")
                # Green LED
                dpg.configure_item("status_led", background_color=[0, 255, 0]) 
                dpg.configure_item("status_text", color=[100, 255, 100])
                dpg.configure_item("conf_bar", overlay=f"{conf:.2f}")
            else:
                dpg.set_value("status_text", "SILENCE")
                # Red LED (Dimmed)
                dpg.configure_item("status_led", background_color=[100, 0, 0])
                dpg.configure_item("status_text", color=[255, 100, 100])
                dpg.configure_item("conf_bar", overlay=f"{conf:.2f}")

            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        pass
    
    state["running"] = False
    t.join(timeout=1.0)
    dpg.destroy_context()

if __name__ == "__main__":
    main_gui()