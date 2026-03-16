"""
PREPROCESSING — TEST SET
- Uses last 20% of videos per word (never seen during training)
- NO augmentation — raw preprocessing only
- Saves X_test.npy, y_test.npy

This gives you TRUE test accuracy with no data leakage.
"""

import numpy as np
import os
from keras.utils import to_categorical

# ============================================================
# CONFIGURATION — must match preprocess_train.py exactly
# ============================================================
DATA_PATH       = r"D:\Extracted"
OUTPUT_PATH     = r"D:\Preprocessed"

ACTIONS         = np.array(['I', 'Need', 'Food', 'Water', 'Nothing'])
SEQUENCE_LENGTH = 142
FEATURE_SIZE    = 144
TEST_SPLIT      = 0.2   # must match train script

POSE_END  = 18
LH_START  = 18
LH_END    = 81
RH_START  = 81
RH_END    = 144

# ============================================================
# PREPROCESSING FUNCTIONS — same as train, NO augmentation
# ============================================================
def interpolate_missing(sequence):
    sequence      = sequence.copy()
    num_frames    = len(sequence)
    missing       = np.array([np.all(sequence[i, LH_START:RH_END] == 0) for i in range(num_frames)])
    valid_indices = np.where(~missing)[0]
    if len(valid_indices) == 0:
        return sequence
    for i in np.where(missing)[0]:
        before = valid_indices[valid_indices < i]
        after  = valid_indices[valid_indices > i]
        if len(before) > 0 and len(after) > 0:
            b, a  = before[-1], after[0]
            alpha = (i - b) / (a - b)
            sequence[i] = (1 - alpha) * sequence[b] + alpha * sequence[a]
    return sequence

def normalize(sequence):
    sequence = sequence.copy()
    for i in range(len(sequence)):
        frame = sequence[i]
        lh = frame[LH_START:LH_END].reshape(21, 3)
        if not np.all(lh == 0):
            wrist = lh[0].copy()
            lh    = lh - wrist
            size  = np.linalg.norm(lh[12])
            if size > 0: lh = lh / size
            sequence[i, LH_START:LH_END] = lh.flatten()
        rh = frame[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            wrist = rh[0].copy()
            rh    = rh - wrist
            size  = np.linalg.norm(rh[12])
            if size > 0: rh = rh / size
            sequence[i, RH_START:RH_END] = rh.flatten()
        pose = frame[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            mid = (pose[0] + pose[1]) / 2
            pose = pose - mid
            sequence[i, 0:POSE_END] = pose.flatten()
    return sequence

def add_velocity(sequence):
    velocity     = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, velocity], axis=1)

def pad_or_truncate(sequence, length=SEQUENCE_LENGTH):
    current_len = len(sequence)
    feature_dim = sequence.shape[1]
    if current_len > length:
        start = (current_len - length) // 2
        return sequence[start: start + length]
    elif current_len < length:
        pad = np.zeros((length - current_len, feature_dim))
        return np.vstack([sequence, pad])
    return sequence

# ============================================================
# MAIN
# ============================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)
label_map = {label: num for num, label in enumerate(ACTIONS)}
sequences = []
labels    = []
total     = 0

print("TEST SET PREPROCESSING (No Augmentation)\n")

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"⚠️  Folder not found: {action_path}")
        continue

    npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    npy_files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))

    # ── TAKE ONLY LAST 20% FOR TESTING ──
    split_idx  = int(len(npy_files) * (1 - TEST_SPLIT))
    test_files = npy_files[split_idx:]

    print(f"--- [{action}] {len(test_files)} test sequences ---")

    for npy_file in test_files:
        raw = np.load(os.path.join(action_path, npy_file))

        # Same preprocessing as training — NO augmentation
        seq = interpolate_missing(raw)
        seq = normalize(seq)
        seq = add_velocity(seq)
        seq = pad_or_truncate(seq)

        sequences.append(seq)
        labels.append(label_map[action])
        total += 1
        print(f"  ✅ {npy_file}")

    print()

X_test = np.array(sequences, dtype=np.float32)
y_test = to_categorical(labels, num_classes=len(ACTIONS)).astype(np.float32)

np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)

print("=" * 50)
print(f"TEST SET DONE")
print(f"  Total test samples : {total}")
print(f"  X_test shape       : {X_test.shape}")
print(f"  y_test shape       : {y_test.shape}")
print(f"  Saved to           : {OUTPUT_PATH}")
print(f"\nThese are UNSEEN videos — accuracy on this is your REAL accuracy")