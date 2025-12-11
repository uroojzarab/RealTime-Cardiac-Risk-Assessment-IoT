import threading
import time
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox

import serial
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tensorflow as tf
from scipy.signal import butter, filtfilt, resample

# =====================================================
# CONFIG
# =====================================================
PORT = "COM8"
BAUD = 115200
MODEL_PATH = "fusion_model.keras"

TARGET_FS = 125
WINDOW_SIZE = 360
PRED_INTERVAL = 1.0
SAVE_CSV = True
CSV_FILE = "realtime_recording.csv"
MAX_RAW_SECONDS = 10.0
PLOT_MAX = 600

# =====================================================
# FILTERS
# =====================================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return b, a

def bandpass_ecg(x, fs, low=0.5, high=35, order=3):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)

def lowpass_ppg(x, fs, cutoff=8, order=3):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, x)

def normalize_window(x):
    x = np.array(x, dtype=np.float32)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# =====================================================
# WORKER THREAD
# =====================================================
class RealtimeEngine(threading.Thread):
    def __init__(self, ui_event_queue, on_status, on_prediction):
        super().__init__(daemon=True)
        self.ui_event_queue = ui_event_queue
        self.on_status = on_status
        self.on_prediction = on_prediction
        self.stop_flag = threading.Event()

        self.t_raw, self.ecg_raw, self.ir_raw, self.red_raw = deque(), deque(), deque(), deque()
        self.plot_t = deque(maxlen=PLOT_MAX)
        self.plot_ecg = deque(maxlen=PLOT_MAX)
        self.plot_ir = deque(maxlen=PLOT_MAX)
        self.last_hr = None
        self.last_spo2 = None
        self.last_pred_time = time.time()
        self.ser = None
        self.csv_file = None
        self.model = None

    def run(self):
        try:
            self.on_status("Loading model...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.on_status("Model loaded successfully.")

            self.ser = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            self.on_status(f"Connected to {PORT}")

            if SAVE_CSV:
                self.csv_file = open(CSV_FILE, "w")
                self.csv_file.write("timestamp,ecg,hr,ir,red,spo2,prediction\n")

            while not self.stop_flag.is_set():
                try:
                    rawline = self.ser.readline().decode(errors='ignore').strip()
                    if not rawline or not rawline[0].isdigit():
                        continue

                    parts = rawline.split(",")
                    if len(parts) != 6:
                        continue

                    t_ms, ecg_str, hr_str, ir_str, red_str, spo2_str = parts
                    t_ms = float(t_ms)
                    ecg_val = float(ecg_str)
                    hr_val = int(hr_str)
                    ir_val = float(ir_str)
                    red_val = float(red_str)
                    spo2_val = int(spo2_str)
                    t_s = t_ms / 1000.0

                    self.t_raw.append(t_s)
                    self.ecg_raw.append(ecg_val)
                    self.ir_raw.append(ir_val)
                    self.red_raw.append(red_val)
                    self.last_hr = hr_val
                    self.last_spo2 = spo2_val

                    while self.t_raw and (self.t_raw[-1] - self.t_raw[0]) > MAX_RAW_SECONDS:
                        self.t_raw.popleft(); self.ecg_raw.popleft()
                        self.ir_raw.popleft(); self.red_raw.popleft()

                    self.plot_t.append(t_s)
                    self.plot_ecg.append(ecg_val)
                    self.plot_ir.append(ir_val)

                    if self.ui_event_queue.qsize() < 4:
                        self.ui_event_queue.put(("plot_update", list(self.plot_ecg), list(self.plot_ir)))

                    if len(self.t_raw) < 5:
                        continue

                    dt = np.diff(np.array(self.t_raw))
                    fs_real = 1.0 / np.median(dt)
                    ecg_arr, ir_arr = np.array(self.ecg_raw), np.array(self.ir_raw)

                    p1, p99 = np.percentile(ir_arr, 1), np.percentile(ir_arr, 99)
                    ir_arr = np.clip(ir_arr, p1, p99)

                    n_new = int(round(len(ecg_arr) * TARGET_FS / fs_real))
                    if n_new < WINDOW_SIZE: continue

                    ecg_rs, ir_rs = resample(ecg_arr, n_new), resample(ir_arr, n_new)
                    ecg_f, ppg_f = bandpass_ecg(ecg_rs, TARGET_FS), lowpass_ppg(ir_rs, TARGET_FS)

                    if len(ecg_f) < WINDOW_SIZE: continue

                    ecg_in = normalize_window(ecg_f[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE)
                    ppg_in = normalize_window(ppg_f[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE)

                    now = time.time()
                    if (now - self.last_pred_time) >= PRED_INTERVAL:
                        self.last_pred_time = now
                        probs = self.model.predict([ecg_in, ppg_in], verbose=0)[0]
                        pred_class = int(np.argmax(probs))
                        labels = ["NORMAL", "WARNING", "CRITICAL"]
                        prediction = labels[pred_class]

                        self.on_prediction(f"{hr_val} BPM", f"{spo2_val}%", prediction, probs)
                        if SAVE_CSV:
                            self.csv_file.write(f"{t_ms},{ecg_val},{hr_val},{ir_val},{red_val},{spo2_val},{prediction}\n")
                            self.csv_file.flush()

                except Exception as e:
                    self.on_status(f"Loop error: {e}")

        except Exception as e:
            self.on_status(f"Fatal error: {e}")

        finally:
            if self.csv_file: self.csv_file.close()
            if self.ser and self.ser.is_open: self.ser.close()
            self.on_status("Engine stopped.")

    def stop(self):
        self.stop_flag.set()


# =====================================================
# MODERN UI
# =====================================================
class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("❤️ ECG + SpO₂ Fusion Dashboard")
        self.geometry("1100x750")
        self.configure(bg="#0d1117")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", background="#238636", foreground="white", padding=6, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#0d1117", foreground="#f0f6fc", font=("Segoe UI", 11))
        style.configure("Card.TFrame", background="#161b22", relief="ridge", borderwidth=2)

        top = ttk.Frame(self, style="Card.TFrame")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text=f"Port: {PORT} | Baud: {BAUD} | Model: {MODEL_PATH}", font=("Consolas", 10)).pack(side=tk.LEFT)
        self.btn_start = ttk.Button(top, text="▶ Start", command=self.start_engine)
        self.btn_start.pack(side=tk.RIGHT, padx=5)
        self.btn_stop = ttk.Button(top, text="⏹ Stop", command=self.stop_engine, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=5)

        info_frame = ttk.Frame(self, style="Card.TFrame")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        self.hr_var = tk.StringVar(value="HR: -")
        self.sp_var = tk.StringVar(value="SpO₂: -")
        self.pred_var = tk.StringVar(value="Prediction: -")
        self.prob_var = tk.StringVar(value="Probabilities: -")

        for var in [self.hr_var, self.sp_var, self.pred_var]:
            ttk.Label(info_frame, textvariable=var, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=40, pady=10)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        self.fig, self.axes = fig, axes
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.grid(color="#30363d")
            ax.tick_params(colors="#f0f6fc")
            ax.title.set_color("#58a6ff")

        self.l_ecg, = axes[0].plot([], [], label="ECG", color="#58a6ff")
        self.l_ir,  = axes[1].plot([], [], label="IR PPG", color="#f85149")
        axes[0].set_title("ECG Signal (Filtered)")
        axes[1].set_title("IR PPG Signal (Filtered)")
        axes[0].legend(facecolor="#161b22", edgecolor="#161b22", labelcolor="#58a6ff")
        axes[1].legend(facecolor="#161b22", edgecolor="#161b22", labelcolor="#f85149")

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor=tk.W, font=("Consolas", 10)).pack(fill=tk.X, padx=15, pady=5)

        self.ui_event_queue = queue.Queue()
        self.engine = None
        self.after(100, self._update_ui)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_status(self, txt): self.status_var.set(txt)

    def on_prediction(self, hr_txt, spo2_txt, prediction, probs):
        self.hr_var.set(f"❤️ HR: {hr_txt}")
        self.sp_var.set(f"🩸 SpO₂: {spo2_txt}")
        self.pred_var.set(f"🧠 Status: {prediction}")
        self.prob_var.set(f"Probabilities: {np.round(probs, 3)}")

    def start_engine(self):
        if self.engine and self.engine.is_alive():
            messagebox.showinfo("Info", "Already running!")
            return
        self.engine = RealtimeEngine(self.ui_event_queue, self.on_status, self.on_prediction)
        self.engine.start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.on_status("Engine started.")

    def stop_engine(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.on_status("Engine stopped.")

    def _update_ui(self):
        try:
            while True:
                item = self.ui_event_queue.get_nowait()
                if not item: break
                tag, ecg, ir = item
                x = range(len(ecg))
                self.l_ecg.set_data(x, ecg)
                self.l_ir.set_data(x, ir)
                for ax in self.axes:
                    ax.relim(); ax.autoscale_view()
                self.canvas.draw_idle()
        except queue.Empty:
            pass
        self.after(60, self._update_ui)

    def on_close(self):
        if self.engine: self.engine.stop()
        self.destroy()

if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
