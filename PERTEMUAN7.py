#Pertemuan 7

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set seed global untuk reproduktifitas
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"Seed acak global telah disetel ke: {SEED}")
print("-" * 60)

try:
    df = pd.read_csv("processed_kelulusan.csv")
    print(f"Data dimuat. Jumlah baris: {len(df)}")
except FileNotFoundError:
    print("⚠️ ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    print("Menggunakan data dummy untuk melanjutkan kode.")
    data_dummy = np.random.rand(1000, 15)
    labels_dummy = np.random.randint(0, 2, 1000)
    df = pd.DataFrame(data_dummy, columns=[f'feature_{i}' for i in range(15)])
    df['Lulus'] = labels_dummy

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}")
print("-" * 60)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])

print("Arsitektur Model (Baseline):")
model.summary()
print("-" * 60)

es = keras.callbacks.EarlyStopping(
    monitor="val_loss",           
    patience=10,                  
    restore_best_weights=True     
)

print("Memulai Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,                   
    batch_size=32,
    callbacks=[es],
    verbose=1                     
)

print(f"\nTraining selesai. Dihentikan pada epoch ke-{len(history.history['loss'])}")
print("-" * 60)

loss, acc, auc_keras = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Acc: {acc:.4f}")
print(f"Test AUC (dari Keras): {auc_keras:.4f}")

y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

auc_roc_manual = roc_auc_score(y_test, y_proba)
print(f"Test ROC-AUC (Manual): {auc_roc_manual:.4f}")

print("\nConfusion Matrix (Threshold 0.5):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Termasuk F1-score):")
print(classification_report(y_test, y_pred, digits=3))
print("-" * 60)

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve: Train vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

print("Learning curve telah disimpan sebagai 'learning_curve.png'.")
print("-" * 60)