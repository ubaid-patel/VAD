import numpy as np
import sounddevice as sd
import librosa
import torch
import pyqtgraph as pg
# CHANGED: Added QtGui to imports
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from collections import deque
from model import CNNLSTMVAD
import torch.nn.functional as F
import time

# ================= CONFIG =================
SR = 16000
FRAME = int(0.025 * SR)
HOP = int(0.010 * SR)
MFCC = 13
SEQ = 25

WINDOW_SEC = 4
MAX_SAMPLES = SR * WINDOW_SEC
MAX_FRAMES = MAX_SAMPLES // HOP

LABELS = ["Silence", "Speech", "Noise"]
NOISE_TYPES = ["Babble", "Car", "Restaurant", "Station", "Street", "Train", "NoNoise"]

COLORS = {
    0: (130, 130, 130),
    1: (0, 220, 0),
    2: (220, 0, 0)
}

# ================= LOAD MODEL =================
model = CNNLSTMVAD(n_features=MFCC + 1, n_noise_types=len(NOISE_TYPES))
try:
    model.load_state_dict(torch.load("best_vad.pt", map_location="cpu"))
except FileNotFoundError:
    print("Warning: 'best_vad.pt' not found. Using random weights for demo.")
model.eval()

# ================= BUFFERS =================
audio_buffer = np.zeros(MAX_SAMPLES, dtype=np.float32)
label_buffer = np.zeros(MAX_FRAMES, dtype=np.int8)

audio_ptr = 0
frame_ptr = 0
feature_buffer = deque(maxlen=SEQ)

current_label = 0
current_conf = 0.0
current_noise_type = ""

start_time = time.time()

# ================= FEATURE EXTRACTION =================
def extract_features(audio):
    if len(audio) < FRAME:
        audio = np.pad(audio, (0, FRAME - len(audio)))
        
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=MFCC,
        n_fft=FRAME, hop_length=HOP
    )
    rms = librosa.feature.rms(y=audio, hop_length=HOP)
    feat = np.vstack([mfcc, rms]).T
    
    if feat.shape[0] == 0:
        return np.zeros((1, MFCC+1))
        
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
    return feat

# ================= AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    global audio_ptr, frame_ptr, current_label, current_conf, current_noise_type

    audio = indata[:, 0]
    n = len(audio)

    if audio_ptr + n > MAX_SAMPLES:
        first_chunk = MAX_SAMPLES - audio_ptr
        audio_buffer[audio_ptr:] = audio[:first_chunk]
        audio_buffer[:n-first_chunk] = audio[first_chunk:]
        audio_ptr = n - first_chunk
    else:
        audio_buffer[audio_ptr:audio_ptr+n] = audio
        audio_ptr += n

    try:
        feats = extract_features(audio)
        
        for f in feats:
            feature_buffer.append(f)
            if len(feature_buffer) == SEQ:
                x = torch.from_numpy(np.array(feature_buffer, dtype=np.float32)).unsqueeze(0)

                with torch.no_grad():
                    vad_logits, noise_logits = model(x)
                    vad_probs = F.softmax(vad_logits, dim=1)[0]
                    noise_probs = F.softmax(noise_logits, dim=1)[0]

                pred = int(torch.argmax(vad_probs))
                current_label = pred
                current_conf = float(vad_probs[pred])

                if pred == 2:
                    current_noise_type = NOISE_TYPES[int(torch.argmax(noise_probs))]
                else:
                    current_noise_type = ""

                label_buffer[frame_ptr % MAX_FRAMES] = pred
                frame_ptr += 1
    except Exception as e:
        print(f"Error in callback: {e}")

# ================= UI SETUP =================
app = QtWidgets.QApplication([])
pg.setConfigOptions(background="#0f1117", foreground="w", antialias=True)

# 1. Main Container
main_win = QtWidgets.QWidget()
main_win.setWindowTitle("Live VAD – Pro UI")
main_win.resize(1300, 700)
main_layout = QtWidgets.QGridLayout()
main_win.setLayout(main_layout)

# 2. Graphics Area
graphics_win = pg.GraphicsLayoutWidget()
main_layout.addWidget(graphics_win, 0, 0, 1, 4)

# ---- Waveform plot
plot = graphics_win.addPlot(row=0, col=0)
plot.setYRange(-1, 1)
plot.hideAxis("bottom")
plot.hideButtons()
plot.setMouseEnabled(x=False, y=False)

wave_curve = plot.plot(pen=pg.mkPen("w", width=1))

image = pg.ImageItem()
image.setOpacity(0.35)
image.setRect(QtCore.QRectF(0, -1, MAX_SAMPLES, 2)) 
plot.addItem(image)

# ---- Confidence bar
conf_plot = graphics_win.addPlot(row=1, col=0)
conf_plot.setMaximumHeight(50)
conf_plot.setYRange(0, 1)
conf_plot.setXRange(0, 1)
conf_plot.hideAxis("left")
conf_plot.hideAxis("bottom")
conf_plot.setMouseEnabled(x=False, y=False)
conf_bar = pg.BarGraphItem(x=[0.5], height=[0], width=1, brush="g")
conf_plot.addItem(conf_bar)

# ---- Labels (FIXED SECTION)
label_text = pg.TextItem("", anchor=(0, 0), color="w")

# FIXED: Use QtGui.QFont instead of QtCore.QFont
font = QtGui.QFont("Arial", 16)
font.setBold(True)
label_text.setFont(font)

plot.addItem(label_text)
label_text.setPos(0, 0.9)

time_text = pg.TextItem("", anchor=(1, 0), color="w")
plot.addItem(time_text)
time_text.setPos(MAX_SAMPLES, 0.9)

# 3. Control Area
def create_slider(name, minv, maxv, init):
    container = QtWidgets.QWidget()
    l = QtWidgets.QHBoxLayout()
    container.setLayout(l)
    
    lbl = QtWidgets.QLabel(name)
    lbl.setStyleSheet("color: white; font-weight: bold;")
    
    sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setMinimum(minv)
    sl.setMaximum(maxv)
    sl.setValue(init)
    
    val_lbl = QtWidgets.QLabel(str(init))
    val_lbl.setStyleSheet("color: #aaa; width: 30px;")
    
    sl.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    
    l.addWidget(lbl)
    l.addWidget(sl)
    l.addWidget(val_lbl)
    return container, sl

s_widget, s_slider = create_slider("Silence Thresh", 1, 50, 5)
ni_widget, ni_slider = create_slider("Noise In", 10, 100, 30)
no_widget, no_slider = create_slider("Noise Out", 5, 80, 15)

main_layout.addWidget(s_widget, 1, 0)
main_layout.addWidget(ni_widget, 1, 1)
main_layout.addWidget(no_widget, 1, 2)

# ================= UPDATE LOOP =================
def update():
    display_audio = np.roll(audio_buffer, -audio_ptr)
    wave_curve.setData(display_audio)
    
    display_labels = np.roll(label_buffer, -frame_ptr)
    color_data = np.zeros((MAX_FRAMES, 1, 3), dtype=np.uint8)
    for i, l in enumerate(display_labels):
        color_data[i, 0] = COLORS[l]
    
    image.setImage(color_data, autoLevels=False)
    image.setRect(QtCore.QRectF(0, -1, MAX_SAMPLES, 2)) 

    conf_bar.setOpts(
        height=[current_conf],
        brush=pg.mkColor(*COLORS[current_label])
    )

    label_text.setText(
        f"{LABELS[current_label]}  "
        f"{current_conf*100:.0f}%  "
        f"{current_noise_type}",
        color=pg.mkColor(*COLORS[current_label])
    )

    t = time.time() - start_time
    time_text.setText(f"{t:.1f} s")

# ================= START =================
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    blocksize=FRAME,
    callback=audio_callback
)

main_win.show()
stream.start()
app.exec_()

stream.stop()
stream.close()