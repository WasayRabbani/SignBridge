"""
SignBridge — Single Video Test
Loads model from model_weights.npy — works on any Keras version
"""
import os

import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Input

# ============================================================
# CONFIGURATION
# ============================================================

import os
import sys

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# Redirect stderr to suppress MediaPipe logs
import io
sys.stderr = io.StringIO()

VIDEO_PATH   = r"D:\Test\2026-03-16 15-09-58.mp4"
WEIGHTS_PATH = 'model_weights.npy'           # download this from Colab
SEQUENCE_LENGTH = 142
ACTIONS      = ['I', 'Need', 'Food', 'Water', 'Nothing']
# ============================================================

USEFUL_POSE = [11, 12, 13, 14, 15, 16]
POSE_END = 18
LH_START = 18
LH_END   = 81
RH_START = 81
RH_END   = 144

mp_holistic = mp.solutions.holistic


# ============================================================
# LOAD MODEL FROM NUMPY WEIGHTS
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
    missing       = np.array([np.all(sequence[i, LH_START:RH_END] == 0) for i in range(len(sequence))])
    valid_indices = np.where(~missing)[0]
    if len(valid_indices) == 0:
        return sequence
    for i in np.where(missing)[0]:
        before = valid_indices[valid_indices < i]
        after  = valid_indices[valid_indices > i]
        if len(before) > 0 and len(after) > 0:
            b, a  = before[-1], after[0]
            sequence[i] = (1 - (i-b)/(a-b)) * sequence[b] + ((i-b)/(a-b)) * sequence[a]
    return sequence


def normalize(sequence):
    sequence = sequence.copy()
    for i in range(len(sequence)):
        f  = sequence[i]
        lh = f[LH_START:LH_END].reshape(21, 3)
        if not np.all(lh == 0):
            lh = lh - lh[0]; size = np.linalg.norm(lh[12])
            if size > 0: lh = lh / size
            sequence[i, LH_START:LH_END] = lh.flatten()
        rh = f[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            rh = rh - rh[0]; size = np.linalg.norm(rh[12])
            if size > 0: rh = rh / size
            sequence[i, RH_START:RH_END] = rh.flatten()
        pose = f[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            pose = pose - (pose[0]+pose[1])/2
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


def preprocess_video(path):
    cap, frames = cv2.VideoCapture(path), []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=0) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            frames.append(extract_landmarks(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    if not frames: return None
    seq = np.array(frames)
    seq = interpolate_missing(seq)
    seq = normalize(seq)
    seq = add_velocity(seq)
    seq = pad_or_truncate(seq)
    return seq.astype(np.float32)


# ============================================================
# MAIN
# ============================================================
print("Loading model...")
model = load_model()
print("Model loaded.\n")

print(f"Processing: {VIDEO_PATH}")
seq = preprocess_video(VIDEO_PATH)

if seq is None:
    print("ERROR: No frames extracted")
else:
    pred          = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
    idx           = np.argmax(pred)
    print(f"\n{'='*40}")
    print(f"Predicted : {ACTIONS[idx]}")
    print(f"Confidence: {pred[idx]*100:.1f}%")
    print(f"{'='*40}")
    print("\nAll probabilities:")
    for action, prob in zip(ACTIONS, pred):
        print(f"  {action:<12} {prob*100:5.1f}%  {'█' * int(prob*30)}")