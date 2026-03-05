"""
PREPROCESSING PIPELINE
Order of operations:
    1. Load raw .npy files (num_frames, 144)
    2. Interpolate missing frames (fix zeros)
    3. Normalize (position + scale)
    4. Add velocity features (144 → 288 per frame)
    5. Pad/Truncate to fixed sequence length
    6. Augment (mirror, noise, time stretch)
    7. Save final X and y tensors

FINAL OUTPUT SHAPES:
    X: (total_samples, SEQUENCE_LENGTH, 288)
    y: (total_samples, num_classes)  one-hot encoded
"""

import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH       = r"D:\Extracted"   # parent folder with word subfolders
OUTPUT_PATH     = r"D:\Preprocessed"

ACTIONS         = np.array(['I', 'Water', 'Food', 'Room', 'Help'])  # add all your words here

SEQUENCE_LENGTH = 60    # fixed length for all sequences
                        # 60 is good starting point, adjust if avg video is longer/shorter
FEATURE_SIZE    = 144   # raw features per frame (6 pose x3 + 21 lh x3 + 21 rh x3)
FINAL_FEATURES  = 288   # after adding velocity (144 coords + 144 velocity = 288)

# Augmentation settings
NOISE_FACTOR    = 0.01  # small random noise ±0.01
TIME_RATES      = [0.8, 1.2]  # time stretch to 80% and 120% of original

# Pose wrist indices in our 6-landmark pose array
# Our pose order: [l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist]
# Left wrist  = index 4 → flattened: 4*3=12
# Right wrist = index 5 → flattened: 5*3=15
L_WRIST_START = 12  # x index of left wrist in pose section
R_WRIST_START = 15  # x index of right wrist in pose section

# Hand sections in 144-value array
POSE_END  = 18   # pose:  0  to 18
LH_START  = 18   # left hand: 18 to 81
LH_END    = 81
RH_START  = 81   # right hand: 81 to 144
RH_END    = 144

# ============================================================
# STEP 1 — INTERPOLATION
# Fix frames where MediaPipe failed (zeros in hand section)
# ============================================================
def interpolate_missing(sequence):
    """
    Replace zero-padded hand frames with linear interpolation
    from surrounding valid frames.
    If more than 30% frames are missing, return None (discard video).
    """
    sequence = sequence.copy()
    num_frames = len(sequence)

    # A frame is "missing hands" if both hand sections are all zeros
    missing = np.array([
        np.all(sequence[i, LH_START:RH_END] == 0)
        for i in range(num_frames)
    ])

    # Discard if too many missing frames
    missing_pct = np.sum(missing) / num_frames
    if missing_pct > 0.30:
        return None

    # Find valid frame indices
    valid_indices = np.where(~missing)[0]
    if len(valid_indices) == 0:
        return None

    # Interpolate each missing frame
    for i in np.where(missing)[0]:
        # Find nearest valid frames before and after
        before = valid_indices[valid_indices < i]
        after  = valid_indices[valid_indices > i]

        if len(before) > 0 and len(after) > 0:
            b, a = before[-1], after[0]
            # Linear interpolation
            alpha = (i - b) / (a - b)
            sequence[i] = (1 - alpha) * sequence[b] + alpha * sequence[a]
        elif len(before) > 0:
            sequence[i] = sequence[before[-1]]  # copy last known
        elif len(after) > 0:
            sequence[i] = sequence[after[0]]    # copy next known

    return sequence


# ============================================================
# STEP 2 — NORMALIZATION
# Position normalization: subtract wrist from hand landmarks
# Scale normalization: divide by hand size
# ============================================================
def normalize(sequence):
    """
    For each frame:
    - Subtract wrist position from all hand landmarks (position invariance)
    - Divide by hand size (scale invariance)
    """
    sequence = sequence.copy()

    for i in range(len(sequence)):
        frame = sequence[i]

        # --- Left hand normalization ---
        lh = frame[LH_START:LH_END].reshape(21, 3)
        if not np.all(lh == 0):
            wrist = lh[0].copy()          # landmark 0 = wrist
            lh = lh - wrist               # center at wrist
            hand_size = np.linalg.norm(lh[12])  # wrist to middle fingertip
            if hand_size > 0:
                lh = lh / hand_size       # scale normalize
            sequence[i, LH_START:LH_END] = lh.flatten()

        # --- Right hand normalization ---
        rh = frame[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            wrist = rh[0].copy()
            rh = rh - wrist
            hand_size = np.linalg.norm(rh[12])
            if hand_size > 0:
                rh = rh / hand_size
            sequence[i, RH_START:RH_END] = rh.flatten()

        # --- Pose normalization (relative to midpoint of shoulders) ---
        pose = frame[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            mid_shoulder = (pose[0] + pose[1]) / 2  # midpoint of left+right shoulder
            pose = pose - mid_shoulder
            sequence[i, 0:POSE_END] = pose.flatten()

    return sequence


# ============================================================
# STEP 3 — VELOCITY FEATURES
# Add frame-to-frame difference as extra features
# Final size per frame: 144 coords + 144 velocity = 288
# ============================================================
def add_velocity(sequence):
    """
    Compute velocity as difference between consecutive frames.
    Frame 0 velocity = zeros (no previous frame).
    Returns sequence of shape (num_frames, 288)
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]  # frame[i] - frame[i-1]
    return np.concatenate([sequence, velocity], axis=1)  # (frames, 288)


# ============================================================
# STEP 4 — PAD / TRUNCATE to fixed SEQUENCE_LENGTH
# ============================================================
def pad_or_truncate(sequence, length=SEQUENCE_LENGTH):
    """
    Truncate from center if too long (keeps signing portion).
    Pad with zeros at end if too short.
    """
    current_len = len(sequence)
    feature_dim = sequence.shape[1]

    if current_len > length:
        # Center crop — removes equal amount from start and end
        start = (current_len - length) // 2
        return sequence[start: start + length]

    elif current_len < length:
        pad = np.zeros((length - current_len, feature_dim))
        return np.vstack([sequence, pad])

    return sequence


# ============================================================
# STEP 5 — AUGMENTATION
# Mirror, noise, time stretch
# ============================================================
def mirror(sequence):
    """
    Flip x-coordinates of hands and pose.
    Simulates left-handed signing.
    Also swaps left and right hand data.
    """
    seq = sequence.copy()
    # Flip x for pose (every 3rd value starting at 0)
    seq[:, 0:POSE_END:3] = -seq[:, 0:POSE_END:3]
    # Flip x for left hand
    seq[:, LH_START:LH_END:3] = -seq[:, LH_START:LH_END:3]
    # Flip x for right hand
    seq[:, RH_START:RH_END:3] = -seq[:, RH_START:RH_END:3]
    # Swap left and right hand
    lh_copy = seq[:, LH_START:LH_END].copy()
    seq[:, LH_START:LH_END] = seq[:, RH_START:RH_END]
    seq[:, RH_START:RH_END] = lh_copy
    return seq


def add_noise(sequence, factor=NOISE_FACTOR):
    """Add small Gaussian noise to simulate natural variation"""
    noise = np.random.normal(0, factor, sequence.shape)
    return sequence + noise


def time_stretch(sequence, rate=0.8):
    """
    Resample sequence to simulate faster (rate<1) or slower (rate>1) signing.
    rate=0.8 → 80% of frames → faster signing
    rate=1.2 → 120% of frames → slower signing
    """
    original_len = len(sequence)
    new_len = int(original_len * rate)
    if new_len < 2:
        return sequence
    # Resample indices
    old_indices = np.linspace(0, original_len - 1, new_len)
    new_sequence = np.array([
        sequence[int(idx)] if idx == int(idx)
        else (1 - (idx % 1)) * sequence[int(idx)] + (idx % 1) * sequence[min(int(idx)+1, original_len-1)]
        for idx in old_indices
    ])
    return new_sequence


# ============================================================
# MAIN PIPELINE
# ============================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)

label_map = {label: num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

total_loaded    = 0
total_discarded = 0
total_augmented = 0

print("Starting preprocessing pipeline...\n")

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        print(f"⚠️  Folder not found: {action_path} — skipping")
        continue

    npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    npy_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    print(f"--- [{action}] {len(npy_files)} sequences ---")

    for npy_file in npy_files:
        raw = np.load(os.path.join(action_path, npy_file))  # (frames, 144)

        # STEP 1 — Interpolate
        seq = interpolate_missing(raw)
        if seq is None:
            print(f"  ⚠️  Discarded {npy_file} — too many missing frames")
            total_discarded += 1
            continue

        # STEP 2 — Normalize
        seq = normalize(seq)

        # STEP 3 — Velocity
        seq = add_velocity(seq)  # (frames, 288)

        # STEP 4 — Pad/Truncate
        seq = pad_or_truncate(seq)  # (SEQUENCE_LENGTH, 288)

        # Save original
        sequences.append(seq)
        labels.append(label_map[action])
        total_loaded += 1

        # STEP 5 — Augmentation (applied AFTER padding so shapes are consistent)
        # Mirror
        seq_mirror = mirror(seq)
        sequences.append(seq_mirror)
        labels.append(label_map[action])
        total_augmented += 1

        # Noise
        seq_noise = add_noise(seq)
        sequences.append(seq_noise)
        labels.append(label_map[action])
        total_augmented += 1

        # Time stretch fast (0.8) — stretch then re-pad
        raw_fast = time_stretch(raw, rate=0.8)
        raw_fast = normalize(add_velocity(raw_fast))

        # Re-pad after time stretch since length changed
        raw_fast_padded = pad_or_truncate(raw_fast)
        sequences.append(raw_fast_padded)
        labels.append(label_map[action])
        total_augmented += 1

        # Time stretch slow (1.2)
        raw_slow = time_stretch(raw, rate=1.2)
        raw_slow = normalize(add_velocity(raw_slow))
        raw_slow_padded = pad_or_truncate(raw_slow)
        sequences.append(raw_slow_padded)
        labels.append(label_map[action])
        total_augmented += 1

    print(f"  Loaded: {len(npy_files)} | After augmentation: {len(npy_files) * 5} sequences\n")

# ===================z=========================================
# SAVE
# ============================================================
X = np.array(sequences)                          # (total, 60, 288)
y = to_categorical(labels, num_classes=len(ACTIONS)).astype(int)  # (total, num_classes)

x_path = os.path.join(OUTPUT_PATH, 'X.npy')
y_path = os.path.join(OUTPUT_PATH, 'y.npy')

np.save(x_path, X)
np.save(y_path, y)

print("=" * 50)
print(f"DONE.")
print(f"  Original sequences loaded : {total_loaded}")
print(f"  Discarded (too noisy)     : {total_discarded}")
print(f"  Augmented samples added   : {total_augmented}")
print(f"  Total samples in dataset  : {len(sequences)}")
print(f"  X shape : {X.shape}  →  (samples, {SEQUENCE_LENGTH} frames, {FINAL_FEATURES} features)")
print(f"  y shape : {y.shape}  →  (samples, {len(ACTIONS)} classes)")
print(f"  Saved to: {OUTPUT_PATH}")