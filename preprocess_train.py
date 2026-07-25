"""
PREPROCESSING — TRAINING SET
- Uses first 80% of videos per word
- Applies full augmentation (5x multiplier)
- Saves X_train.npy, y_train.npy

NO DATA LEAKAGE — test videos never touched here
"""

import numpy as np
import os
import gc
from keras.utils import to_categorical
    
# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH       = r"D:\Extracted"
OUTPUT_PATH     = r"D:\Preprocessed"

# Dynamically detect words and preserve old indices!
ACTIONS_FILE = "actions_list.npy"
if os.path.exists(ACTIONS_FILE):
    old_actions = list(np.load(ACTIONS_FILE, allow_pickle=True))
else:
    old_actions = []

word_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

# Append any new words to the end so old integer mappings stay perfectly aligned
for folder in word_folders:
    if folder not in old_actions:
        old_actions.append(folder)

ACTIONS = np.array(old_actions)
np.save(ACTIONS_FILE, ACTIONS)
print(f"Loaded {len(ACTIONS)} words (Preserved old order!): {ACTIONS}")
SEQUENCE_LENGTH = 122
FEATURE_SIZE    = 144
FINAL_FEATURES  = 288
NOISE_FACTOR    = 0.01
TEST_SPLIT      = 0.2   # last 20% of files reserved for test — DO NOT TOUCH

POSE_END  = 18
LH_START  = 18
LH_END    = 81
RH_START  = 81
RH_END    = 144

# ============================================================
# PREPROCESSING FUNCTIONS
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

def mirror(sequence):
    seq = sequence.copy()
    seq[:, 0:POSE_END:3]       = -seq[:, 0:POSE_END:3]
    seq[:, LH_START:LH_END:3]  = -seq[:, LH_START:LH_END:3]
    seq[:, RH_START:RH_END:3]  = -seq[:, RH_START:RH_END:3]
    lh_copy = seq[:, LH_START:LH_END].copy()
    seq[:, LH_START:LH_END] = seq[:, RH_START:RH_END]
    seq[:, RH_START:RH_END] = lh_copy
    return seq

def add_noise(sequence, factor=NOISE_FACTOR):
    return sequence + np.random.normal(0, factor, sequence.shape)

def time_stretch(sequence, rate=0.8):
    original_len = len(sequence)
    new_len      = int(original_len * rate)
    if new_len < 2: return sequence
    old_indices  = np.linspace(0, original_len - 1, new_len)
    return np.array([
        sequence[int(idx)] if idx == int(idx)
        else (1 - (idx % 1)) * sequence[int(idx)] + (idx % 1) * sequence[min(int(idx)+1, original_len-1)]
        for idx in old_indices
    ])

# ============================================================
# MAIN
# ============================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)
label_map  = {label: num for num, label in enumerate(ACTIONS)}
total_loaded    = 0
total_augmented = 0

print("TRAINING SET PREPROCESSING\n")

# ── PASS 1: Count total samples so we can pre-allocate memmap ──
total_samples = 0
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    split_idx = int(len(npy_files) * (1 - TEST_SPLIT))
    total_samples += split_idx * 5  # 5 augmentations per file

print(f"Total training samples (with augmentation): {total_samples}\n")

# ── PASS 2: Write directly to disk via memmap — no RAM spike ──
X_train_path = os.path.join(OUTPUT_PATH, 'X_train.npy')
X_memmap = np.lib.format.open_memmap(
    X_train_path, mode='w+',
    dtype=np.float16,
    shape=(total_samples, SEQUENCE_LENGTH, FINAL_FEATURES)
)

labels  = []
row_idx = 0

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"⚠️  Folder not found: {action_path}")
        continue

    npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    npy_files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))

    # ── TAKE ONLY FIRST 80% FOR TRAINING ──
    split_idx   = int(len(npy_files) * (1 - TEST_SPLIT))
    train_files = npy_files[:split_idx]
    test_files  = npy_files[split_idx:]

    print(f"--- [{action}] Total: {len(npy_files)} | Train: {len(train_files)} | Test (reserved): {len(test_files)} ---")

    for npy_file in train_files:
        raw = np.load(os.path.join(action_path, npy_file))

        seq = interpolate_missing(raw)
        seq = normalize(seq)
        seq = add_velocity(seq)
        seq = pad_or_truncate(seq)

        # Write each augmentation directly to disk
        X_memmap[row_idx] = seq.astype(np.float16);          labels.append(label_map[action]); row_idx += 1
        X_memmap[row_idx] = mirror(seq).astype(np.float16);  labels.append(label_map[action]); row_idx += 1
        X_memmap[row_idx] = add_noise(seq).astype(np.float16); labels.append(label_map[action]); row_idx += 1

        raw_fast = pad_or_truncate(add_velocity(normalize(time_stretch(raw, rate=0.8))))
        X_memmap[row_idx] = raw_fast.astype(np.float16);     labels.append(label_map[action]); row_idx += 1

        raw_slow = pad_or_truncate(add_velocity(normalize(time_stretch(raw, rate=1.2))))
        X_memmap[row_idx] = raw_slow.astype(np.float16);     labels.append(label_map[action]); row_idx += 1

        total_loaded += 1

    total_augmented += len(train_files) * 4
    print(f"  Loaded: {len(train_files)} | After augmentation: {len(train_files) * 5}\n")

# Flush memmap to disk
del X_memmap
gc.collect()

# Save labels
y_train = to_categorical(labels, num_classes=len(ACTIONS)).astype(np.float32)
del labels
gc.collect()
np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)

# ============================================================
# SUMMARY
# ============================================================
print("=" * 50)
print(f"TRAINING SET DONE")
print(f"  Original loaded   : {total_loaded}")
print(f"  Augmented added   : {total_augmented}")
print(f"  Total samples     : {total_samples}")
print(f"  X_train shape     : ({total_samples}, {SEQUENCE_LENGTH}, {FINAL_FEATURES})")
print(f"  y_train shape     : {y_train.shape}")
print(f"  X_train dtype     : float16 on disk (cast to float32 in Colab)")
print(f"  Saved to          : {OUTPUT_PATH}")