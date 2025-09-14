import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_ecg.h5")

SEQ_LEN = 250
NUM_CLASSES = 4  # normal, arrhythmia, tachycardia, bradycardia

# --- Step 1: Synthetic dataset (demo only) ---
def make_synthetic_ecg(n_samples=500, seq_len=SEQ_LEN):
    X = np.zeros((n_samples, seq_len))
    y = np.zeros(n_samples, dtype=int)
    rng = np.random.RandomState(1)
    for i in range(n_samples):
        cls = rng.randint(0, NUM_CLASSES)
        y[i] = cls
        t = np.linspace(0, 1, seq_len)
        base = 0.1 * rng.randn(seq_len)
        if cls == 0:   # normal
            sig = 0.8*np.sin(2*np.pi*5*t) + base
            for p in range(5, seq_len, 50):
                sig[p:p+3] += 1.0
        elif cls == 1: # arrhythmia
            sig = 0.5*np.sin(2*np.pi*5*t) + base
            for p in rng.randint(10, seq_len-10, size=6):
                sig[p:p+3] += 1.2
        elif cls == 2: # tachycardia
            sig = 0.7*np.sin(2*np.pi*9*t) + base
            for p in range(5, seq_len, 30):
                sig[p:p+3] += 1.1
        else:          # bradycardia
            sig = 0.6*np.sin(2*np.pi*3*t) + base
            for p in range(10, seq_len, 80):
                sig[p:p+3] += 1.0
        X[i] = sig
    return X.astype(np.float32), y

# --- Step 2: Build CNN ---
def build_model(input_shape=(SEQ_LEN,1), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv1D(16, 7, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(32, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Step 3: Training ---
print("Generating synthetic ECG dataset...")
X, y = make_synthetic_ecg(n_samples=1000)
X = X[..., np.newaxis]
y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=2)

model = build_model()
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=10, batch_size=32, verbose=2)

model.save(MODEL_PATH)
print("Model saved at:", MODEL_PATH)
