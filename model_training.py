import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# --- 1. LOAD MASTER TENSORS ---
# Ensure these files are in your current directory
X = np.load('X_train.npy')
y = np.load('y_train.npy')

# --- 2. TRAIN-TEST SPLIT ---
# 80% to train, 20% to test. 'stratify' ensures labels are balanced in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Total Samples: {len(X)}")
print(f"Training on: {len(X_train)} samples")
print(f"Testing on: {len(X_test)} samples")

# --- 3. BiLSTM MODEL ARCHITECTURE ---
# Input shape: (30 frames, 258 landmarks)
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 258)),
    Dropout(0.2),
    
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    Dropout(0.2),
    
    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax') # y.shape[1] is the number of words (classes)
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. TRAINING ---
# EarlyStopping stops training if the model stops improving, preventing overfitting.
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\nStarting Training...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# --- 5. FINAL EVALUATION ---
res = model.predict(X_test)
print(f"\nModel Accuracy on Test Data: {history.history['categorical_accuracy'][-1] * 100:.2f}%")

# Save the brain
model.save('sign_language_model.h5')
print("Model saved as 'sign_language_model.h5'")