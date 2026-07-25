import os
import sys
import subprocess
import numpy as np

WEIGHTS_PATH = "lstm_weights.npy"
PREPROCESSED_PATH = r"D:\Preprocessed"
SEQUENCE_LENGTH = 122
ACTIONS_FILE = "actions_list.npy"


print("="*50)
print(" SIGNBRIDGE MASTER RETRAIN PIPELINE ")
print("="*50)

# STEP 0: Auto-Rename new videos
print("\n[0/4] Auto-renaming videos in D:\\Signs...")
subprocess.run([sys.executable, "rename_files.py"], check=True)

# STEP 1: Extract coordinates from new MP4s
# Smart pre-check: only run extraction if there are actually new videos to process.
# This avoids the Windows multiprocessing deadlock caused by MediaPipe workers
# being spawned and then hanging on cleanup even when everything is skipped.
print("\n[1/4] Checking for new videos to extract...")
SIGNS_FOLDER     = r"D:\Signs"
EXTRACTED_FOLDER = r"D:\Extracted"
new_videos_found = False
for word_folder in os.listdir(SIGNS_FOLDER):
    signs_word_path     = os.path.join(SIGNS_FOLDER, word_folder)
    extracted_word_path = os.path.join(EXTRACTED_FOLDER, word_folder)
    if not os.path.isdir(signs_word_path):
        continue
    for f in os.listdir(signs_word_path):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            npy_path = os.path.join(extracted_word_path, os.path.splitext(f)[0] + '.npy')
            if not os.path.exists(npy_path):
                new_videos_found = True
                break
    if new_videos_found:
        break

if new_videos_found:
    print("  New videos found. Running extraction...")
    subprocess.run([sys.executable, "extract_raw_coordinates.py"], check=True)
else:
    print("  All videos already extracted. Skipping (no deadlock risk).")


# STEP 2: Preprocess all data (this detects new words and creates X_train/y_train)
print("\n[2/4] Preprocessing and augmenting data...")
subprocess.run([sys.executable, "preprocess_train.py"], check=True)

# STEP 3: Load Data and Actions
# Import Keras here — AFTER all subprocesses are done.
# This prevents TensorFlow from being loaded in memory when
# extract_raw_coordinates.py spawns its multiprocessing workers,
# which causes a Windows deadlock.
print("\n[3/4] Preparing data for Fine-Tuning...")
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam

def build_old_model(seq_len, n_classes):
    model = Sequential([
        Input(shape=(seq_len, 288)),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax', name='old_dense'),
    ])
    return model


X_train = np.load(os.path.join(PREPROCESSED_PATH, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(PREPROCESSED_PATH, "y_train.npy"))
new_actions = np.load(ACTIONS_FILE, allow_pickle=True)
new_num_classes = len(new_actions)
print(f"Dataset Shape: X={X_train.shape}, y={y_train.shape}")
print(f"Total vocabulary size: {new_num_classes}")

# STEP 4: Transfer Learning
print("\n[4/4] Starting Transfer Learning (Fine-Tuning)...")

# Detect old number of classes from weights
try:
    old_weights = np.load(WEIGHTS_PATH, allow_pickle=True)
    old_num_classes = old_weights[-1].shape[0]
    print(f"Old vocabulary size: {old_num_classes}")
except Exception as e:
    print(f"Could not load old weights. Error: {e}")
    print("Assuming training from scratch.")
    old_num_classes = 0

if new_num_classes == old_num_classes:
    print("\nNo new words detected. Training the entire model to improve accuracy on existing words...")
    model = build_old_model(SEQUENCE_LENGTH, new_num_classes)
    model.set_weights(old_weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=32)
    new_weights = model.get_weights()

elif old_num_classes > 0:
    print(f"\nAdding {new_num_classes - old_num_classes} new words. Rebuilding classification head...")
    old_model = build_old_model(SEQUENCE_LENGTH, old_num_classes)
    old_model.set_weights(old_weights)
    
    # Freeze the LSTM feature extractor
    for layer in old_model.layers[:-1]: # All layers except the last Dense(n)
        layer.trainable = False
        
    # Get the output of the Dense(64) layer
    base_output = old_model.layers[-2].output 
    
    # Add new output layer
    new_output = Dense(new_num_classes, activation='softmax', name='new_predictions')(base_output)
    
    # Create the new model
    new_model = Model(inputs=old_model.input, outputs=new_output)
    new_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print("\nTraining ONLY the new classification layer (Fast Fine-Tuning)...")
    new_model.fit(X_train, y_train, epochs=15, batch_size=32)
    new_weights = new_model.get_weights()

else:
    print("\nTraining entirely from scratch...")
    model = build_old_model(SEQUENCE_LENGTH, new_num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    new_weights = model.get_weights()

# Save the new weights
np.save("lstm_weights_v2.npy", np.array(new_weights, dtype=object))
# Backup old weights just in case
if os.path.exists(WEIGHTS_PATH):
    if os.path.exists("lstm_weights_backup.npy"):
        os.remove("lstm_weights_backup.npy")
    os.rename(WEIGHTS_PATH, "lstm_weights_backup.npy")
os.rename("lstm_weights_v2.npy", WEIGHTS_PATH)

print("\n" + "="*50)
print(" DONE! The model is updated.")
print(" 1. Your new vocabulary is saved in actions_list.npy")
print(" 2. Your new weights are saved in lstm_weights.npy")
print(" 3. Your old weights are backed up as lstm_weights_backup.npy")
print(" Just restart your Flask server and it will instantly know the new words!")
print("="*50)
