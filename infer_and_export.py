import numpy as np
import json
import time
import argparse
import os
import matplotlib.pyplot as plt

# Keras
from tensorflow.keras.models import load_model

# ONNX
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ------------------------
# Command-line args
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='sample_ecg.npy')
parser.add_argument('--out', type=str, default='ecg_output.json')
parser.add_argument('--onnx', action='store_true', help='Use ONNX model if available')
args = parser.parse_args()

# ------------------------
# Load ECG
# ------------------------
ecg = np.load(args.input).astype(np.float32)
if len(ecg) < 250:
    ecg = np.pad(ecg, (0, 250-len(ecg)))
elif len(ecg) > 250:
    ecg = ecg[:250]

# Normalize
ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg)+1e-6)
x = ecg_norm.reshape(1,250,1)

# ------------------------
# Keras inference
# ------------------------
model = load_model("models/cnn_ecg.h5")
t0 = time.time()
probs_keras = model.predict(x)[0]
t_keras = (time.time()-t0)*1000

class_map = ['normal', 'arrhythmia', 'tachycardia', 'bradycardia']
pred_label_keras = class_map[np.argmax(probs_keras)]
pred_prob_keras = float(np.max(probs_keras))

# ------------------------
# ONNX inference (optional)
# ------------------------
probs_onnx = None
pred_label_onnx = None
pred_prob_onnx = None
t_onnx = None

if args.onnx and ort is not None:
    onnx_file = "cnn_ecg_quant.onnx" if os.path.exists("cnn_ecg_quant.onnx") else "cnn_ecg.onnx"
    if os.path.exists(onnx_file):
        try:
            providers = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_file, providers=providers)
            t0 = time.time()
            out = sess.run(None, {sess.get_inputs()[0].name: x})
            t_onnx = (time.time()-t0)*1000
            probs_onnx = out[0][0]
            pred_label_onnx = class_map[np.argmax(probs_onnx)]
            pred_prob_onnx = float(np.max(probs_onnx))
        except Exception as e:
            print("ONNX inference failed, falling back to Keras:", e)

# ------------------------
# Heart rate estimation
# ------------------------
from scipy.signal import find_peaks
peaks, _ = find_peaks(ecg_norm, distance=20, height=0.5)
hr = float(len(peaks)*60.0)  # 1 sec of data → bpm

# ------------------------
# Map urgency
# ------------------------
urgency_map = {'normal':'low', 'arrhythmia':'high', 'tachycardia':'critical', 'bradycardia':'high'}
urgency = urgency_map[pred_label_keras]

# ------------------------
# Save JSON
# ------------------------
out_dict = {
    "heart_rate": hr,
    "ecg_label": pred_label_keras,
    "probability": pred_prob_keras,
    "urgency": urgency,
    "inference_time_keras_ms": round(t_keras,2)
}
if probs_onnx is not None:
    out_dict.update({
        "inference_time_onnx_ms": round(t_onnx,2),
        "pred_label_onnx": pred_label_onnx,
        "probability_onnx": pred_prob_onnx
    })

with open(args.out, 'w') as f:
    json.dump(out_dict, f, indent=2)

print(f" Saved output to {args.out}")
print(out_dict)

# ------------------------
# Plot ECG
# ------------------------
plt.figure(figsize=(8,4))
plt.plot(ecg)
plt.title("ECG Sample")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
