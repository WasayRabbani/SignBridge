"""
SignBridge — Multi-Sign Video Prediction
Give it one video with multiple signs, it splits and predicts each one.

Usage:
    1. Record a video: idle → sign1 → idle → sign2 → idle → sign3 → idle
    2. Set VIDEO_PATH below
    3. Set ACTIVE_MODEL below
    4. Run script
"""

from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Input
from keras.models import Sequential
import mediapipe as mp
import numpy as np
import cv2
import os

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================
# ★ CHANGE THESE TWO LINES
# ============================================================
VIDEO_PATH = r"D:\OBS Recordings\2026-05-04 11-22-10.mp4"
ACTIVE_MODEL = 'lstm'   # options: 'lstm' | 'bilstm'

# ============================================================
# MODEL REGISTRY — add new models here, touch nothing else
# ============================================================

ACTIONS_22 = np.array(["Bathroom", "Bill", "Bring", "Broken", "Clean", "Cold",
                      "Dirty", "Find", "Food", "Help", "Hot", "I",
                       "Key", "Luggage", "Need", "No", "Nothing", "Now",
                       "Please", "Room", "Towel", "Water"
                       ])

MODEL_CONFIGS = {
    'lstm': {
        'weights_path':    'lstm_weights.npy',
        'model_type':      'keras',
        'sequence_length': 122,
        'actions':         ACTIONS_22,
        'build_fn': lambda sl, nc: _build_lstm(sl, nc),
    },
    'bilstm': {
        'weights_path':    'bilstm_weights.npy',
        'model_type':      'keras',
        'sequence_length': 122,
        'actions':         ACTIONS_22,
        'build_fn': lambda sl, nc: _build_bilstm(sl, nc),
    },

}

_CFG = MODEL_CONFIGS[ACTIVE_MODEL]
WEIGHTS_PATH = _CFG['weights_path']
SEQUENCE_LENGTH = _CFG['sequence_length']
ACTIONS = _CFG['actions']

# ============================================================
# PIPELINE CONFIG
# ============================================================
CONFIDENCE_THRESHOLD = 0.50
IDLE_THRESHOLD = 15
MIN_SIGN_FRAMES = 20
IDLE_PADDING = 10

POSE_END = 18
LH_START = 18
LH_END = 81
RH_START = 81
RH_END = 144
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic

# ============================================================
# ARCHITECTURE BUILDERS
# ============================================================


def _build_lstm(seq_len, n_classes):
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
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def _build_bilstm(seq_len, n_classes):
    model = Sequential([
        Input(shape=(seq_len, 288)),
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


# ============================================================
# LOAD MODEL
# ============================================================


def load_model():
    if _CFG['model_type'] == 'keras':
        model = _CFG['build_fn'](SEQUENCE_LENGTH, len(ACTIONS))
        weights = np.load(WEIGHTS_PATH, allow_pickle=True)
        model.set_weights(weights)
        return model
    raise ValueError(f"Unknown model_type: {_CFG['model_type']}")

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
    sequence = sequence.copy()
    missing = np.array([np.all(sequence[i, LH_START:RH_END] == 0)
                        for i in range(len(sequence))])
    valid_indices = np.where(~missing)[0]
    if len(valid_indices) == 0:
        return sequence
    for i in np.where(missing)[0]:
        before = valid_indices[valid_indices < i]
        after = valid_indices[valid_indices > i]
        if len(before) > 0 and len(after) > 0:
            b, a = before[-1], after[0]
            alpha = (i - b) / (a - b)
            sequence[i] = (1 - alpha) * sequence[b] + alpha * sequence[a]
    return sequence


def normalize(sequence):
    sequence = sequence.copy()
    for i in range(len(sequence)):
        f = sequence[i]
        lh = f[LH_START:LH_END].reshape(21, 3)
        if not np.all(lh == 0):
            lh = lh - lh[0]
            size = np.linalg.norm(lh[12])
            if size > 0:
                lh = lh / size
            sequence[i, LH_START:LH_END] = lh.flatten()
        rh = f[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            rh = rh - rh[0]
            size = np.linalg.norm(rh[12])
            if size > 0:
                rh = rh / size
            sequence[i, RH_START:RH_END] = rh.flatten()
        pose = f[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            pose = pose - (pose[0] + pose[1]) / 2
            sequence[i, 0:POSE_END] = pose.flatten()
    return sequence


def add_velocity(sequence):
    v = np.zeros_like(sequence)
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
# MAIN — LOAD → EXTRACT → SPLIT → PREDICT
# ============================================================
print(
    f"Loading model: [{ACTIVE_MODEL.upper()}]  weights={WEIGHTS_PATH}  classes={len(ACTIONS)}")
model = load_model()
print("Model loaded.\n")

print(f"Processing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(
    f"Video: {total_frames} frames @ {fps:.0f} FPS ({total_frames/fps:.1f}s)\n")

# Step 1 — Extract landmarks
all_landmarks = []
all_has_hands = []

print("Extracting landmarks...")
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          model_complexity=0) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kp = extract_landmarks(results)
        has_hands = (results.left_hand_landmarks is not None or
                     results.right_hand_landmarks is not None)
        all_landmarks.append(kp)
        all_has_hands.append(has_hands)
        print(f"  Frame {len(all_landmarks)}/{total_frames}", end='\r')

cap.release()
print(f"\nExtracted {len(all_landmarks)} frames")

# Step 2 — Split into sign segments
segments = []
current_seg = []
idle_count = 0
idle_buffer = []

for kp, has_hands in zip(all_landmarks, all_has_hands):
    if has_hands:
        if idle_count >= IDLE_THRESHOLD and len(current_seg) >= MIN_SIGN_FRAMES:
            segments.append(current_seg.copy())
            current_seg = []
        if len(current_seg) == 0 and len(idle_buffer) > 0:
            current_seg.extend(idle_buffer[-IDLE_PADDING:])
        idle_count = 0
        idle_buffer = []
        current_seg.append(kp)
    else:
        idle_count += 1
        idle_buffer.append(kp)
        if current_seg:
            current_seg.append(kp)

if len(current_seg) >= MIN_SIGN_FRAMES:
    segments.append(current_seg)

print(f"Found {len(segments)} sign segment(s)\n")

if len(segments) == 0:
    print("❌ No signs detected. Check that:")
    print("   - Hands are clearly visible during signs")
    print("   - Idle time between signs ≥ 0.5s")
    exit()

# Step 3 — Predict each segment
print("=" * 50)
predicted_words = []

for i, seg in enumerate(segments):
    seq = preprocess_segment(seg)

    probs = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]

    confidence = np.max(probs)
    word = ACTIONS[np.argmax(probs)]

    print(f"Sign {i+1} ({len(seg)} frames):")
    for action, prob in zip(ACTIONS, probs):
        bar = '█' * int(prob * 25)
        print(f"  {action:<12} {prob*100:5.1f}%  {bar}")

    if confidence >= CONFIDENCE_THRESHOLD and word != 'Nothing':
        predicted_words.append(word)
        print(f"  → ACCEPTED: {word} ({confidence*100:.1f}%)")
    else:
        print(
            f"  → REJECTED: {word} ({confidence*100:.1f}%) — below threshold or Nothing")
    print()

# Step 4 — Sentence
print("=" * 50)
if predicted_words:
    sentence = " ".join(predicted_words)
    print(f"Words detected : {predicted_words}")
    print(f"Sentence       : {sentence}")
else:
    print("No words accepted — lower CONFIDENCE_THRESHOLD or check signs")
