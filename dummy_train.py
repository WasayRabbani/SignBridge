import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

# 1. Load your current data
X = np.load('X_train.npy') # Your 77 videos
y = np.load('y_train.npy') # Your labels

# 2. Split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Simple Model to check data quality
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 258)),
    Bidirectional(LSTM(32, return_sequences=False, activation='relu')),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Train for just 20-30 epochs
print("Checking data quality...")
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

final_acc = history.history['val_categorical_accuracy'][-1]
if final_acc > 0.85:
    print(f"\nSUCCESS: Data quality is high ({final_acc*100:.2f}%). Keep recording!")
else:
    print(f"\nWARNING: Accuracy is low ({final_acc*100:.2f}%). Check your lighting or framing.")