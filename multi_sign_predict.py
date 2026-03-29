"""
SignBridge — Multi-Sign Video Prediction
Give it one video with multiple signs, it splits and predicts each one.

Usage:
    1. Record a video: idle → sign1 → idle → sign2 → idle → sign3 → idle
    2. Set VIDEO_PATH below
    3. Run script
    4. See each sign predicted + final sentence
"""

import os
import sys
os.environ['GLOG_minloglevel']      = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stderr = open(os.devnull, 'w')

import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Input

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH           = r"D:\Test\Sentence Level Test\2026-03-29 23-41-00.mp4"
WEIGHTS_PATH         = 'model_weights.npy'
SEQUENCE_LENGTH      = 131
ACTIONS              = ['I', 'Need', 'Food', 'Water', 'Nothing', 'Key', 'Room']
CONFIDENCE_THRESHOLD = 0.50
IDLE_THRESHOLD       = 15   # frames of no hands = sign boundary
MIN_SIGN_FRAMES      = 20   # minimum frames to count as a valid sign
IDLE_PADDING = 10  # idle frames to add before and after each sign

# ============================================================

POSE_END    = 18
LH_START    = 18
LH_END      = 81
RH_START    = 81
RH_END      = 144
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 288)),
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(ACTIONS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    weights = np.load(WEIGHTS_PATH, allow_pickle=True)
    model.set_weights(weights)
    return model


# ============================================================
# PREPROCESSING
# ============================================================
def extract_landmarks(results):
    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z]
            for i in USEFUL_POSE
        ]).flatten()
    else:
        pose = np.zeros(18)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])


def interpolate_missing(sequence):
    sequence      = sequence.copy()
    missing       = np.array([np.all(sequence[i, LH_START:RH_END] == 0)
                               for i in range(len(sequence))])
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
        f  = sequence[i]
        lh = f[LH_START:LH_END].reshape(21, 3)
        if not np.all(lh == 0):
            lh = lh - lh[0]
            size = np.linalg.norm(lh[12])
            if size > 0: lh = lh / size
            sequence[i, LH_START:LH_END] = lh.flatten()
        rh = f[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            rh = rh - rh[0]
            size = np.linalg.norm(rh[12])
            if size > 0: rh = rh / size
            sequence[i, RH_START:RH_END] = rh.flatten()
        pose = f[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            pose = pose - (pose[0] + pose[1]) / 2
            sequence[i, 0:POSE_END] = pose.flatten()
    return sequence


def add_velocity(sequence):
    v     = np.zeros_like(sequence)
    v[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, v], axis=1)


def pad_or_truncate(sequence, length=SEQUENCE_LENGTH):
    n, d = len(sequence), sequence.shape[1]
    if n > length:
        s = (n - length) // 2
        return sequence[s: s + length]
    elif n < length:
        return np.vstack([sequence, np.zeros((length - n, d))])
    return sequence


def preprocess_segment(frames):
    seq = np.array(frames)
    seq = interpolate_missing(seq)
    seq = normalize(seq)
    seq = add_velocity(seq)
    seq = pad_or_truncate(seq)
    return seq.astype(np.float32)


# ============================================================
# MAIN — EXTRACT + SPLIT + PREDICT
# ============================================================
print("Loading model...")
model = load_model()
print("Model loaded.\n")

print(f"Processing video: {VIDEO_PATH}")
cap    = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames at {fps:.0f} FPS ({total_frames/fps:.1f} seconds)\n")

# Step 1 — Extract all landmarks + track hand visibility
all_landmarks  = []
all_has_hands  = []

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           model_complexity=0) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame     = cv2.resize(frame, (640, 480))
        results   = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kp        = extract_landmarks(results)
        has_hands = (results.left_hand_landmarks is not None or
                     results.right_hand_landmarks is not None)
        all_landmarks.append(kp)
        all_has_hands.append(has_hands)

cap.release()
print(f"Extracted {len(all_landmarks)} frames")

# Step 2 — Split into signing segments at idle positions
# Add this at top of script with other config

IDLE_PADDING = 10  # idle frames to add before and after each sign
segments    = []
current_seg = []
idle_count  = 0
idle_buffer = []  # keeps last N idle frames

for kp, has_hands in zip(all_landmarks, all_has_hands):
    if has_hands:
        if idle_count >= IDLE_THRESHOLD and len(current_seg) >= MIN_SIGN_FRAMES:
            # Previous segment ended — save it with idle tail
            segments.append(current_seg.copy())
            current_seg = []

        # Add recent idle frames as prefix if starting new segment
        if len(current_seg) == 0 and len(idle_buffer) > 0:
            current_seg.extend(idle_buffer[-IDLE_PADDING:])

        idle_count = 0
        idle_buffer = []
        current_seg.append(kp)
    else:
        idle_count += 1
        idle_buffer.append(kp)
        if current_seg:
            current_seg.append(kp)  # add idle tail to current segment

# Catch last segment
if len(current_seg) >= MIN_SIGN_FRAMES:
    segments.append(current_seg)

print(f"Found {len(segments)} sign segments\n")

if len(segments) == 0:
    print("❌ No signs detected. Check that:")
    print("   - Hands are clearly visible during signs")
    print("   - There is enough idle time between signs (0.5+ seconds)")
    exit()

# Step 3 — Predict each segment
print("=" * 50)
predicted_words = []

for i, seg in enumerate(segments):
    seq        = preprocess_segment(seg)
    pred       = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
    confidence = np.max(pred)
    word       = ACTIONS[np.argmax(pred)]

    # Show all probabilities for this segment
    print(f"Sign {i+1} ({len(seg)} frames):")
    for action, prob in zip(ACTIONS, pred):
        bar = '█' * int(prob * 25)
        print(f"  {action:<12} {prob*100:5.1f}%  {bar}")

    if confidence >= CONFIDENCE_THRESHOLD and word != 'Nothing':
        predicted_words.append(word)
        print(f"  → ✅ ACCEPTED: {word} ({confidence*100:.1f}%)")
    else:
        print(f"  → ❌ REJECTED: {word} ({confidence*100:.1f}%) — below threshold or Nothing")
    print()

# Step 4 — Form sentence
print("=" * 50)
if predicted_words:
    sentence = " ".join(predicted_words)
    print(f"Words detected : {predicted_words}")
    print(f"Sentence       : {sentence}")
else:
    print("No words accepted — try lowering CONFIDENCE_THRESHOLD or check your signs")