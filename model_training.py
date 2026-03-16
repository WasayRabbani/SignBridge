# ============================================================
# MUST BE FIRST — before any imports
# ============================================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_PATH = r"/content/drive/MyDrive/FYP NEW/Codes"

# ============================================================
# 1. LOAD
# ============================================================
X_train = np.load(os.path.join(OUTPUT_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(OUTPUT_PATH, 'y_train.npy'))
X_test  = np.load(os.path.join(OUTPUT_PATH, 'X_test.npy'))
y_test  = np.load(os.path.join(OUTPUT_PATH, 'y_test.npy'))

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}  ← unseen, no augmentation")

num_classes = y_train.shape[1]

# ============================================================
# 2. MODEL
# ============================================================
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# ============================================================
# 3. CALLBACKS
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# ============================================================
# 4. TRAINING
# ============================================================
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# ============================================================
# 5. EVALUATION
# ============================================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"TRUE Test Loss     : {loss:.4f}")
print(f"TRUE Test Accuracy : {accuracy * 100:.2f}%")
print(f"{'='*50}")

# ============================================================
# 6. SAVE — three formats
# ============================================================
# Format 1 — full model h5
model.save(os.path.join(OUTPUT_PATH, 'signbridge_model.h5'))
print("Saved: signbridge_model.h5")

# Format 2 — weights h5
model.save_weights(os.path.join(OUTPUT_PATH, 'signbridge_weights.weights.h5'))
print("Saved: signbridge_weights.weights.h5")

# Format 3 — numpy weights (most compatible across all Keras versions)
weights = model.get_weights()
np.save(os.path.join(OUTPUT_PATH, 'model_weights.npy'), np.array(weights, dtype=object))
print("Saved: model_weights.npy")

print("\n✅ Download model_weights.npy — works on any local Keras version")