"""
SignBridge — Mode 2: Folder Evaluation
Tests model on a folder of videos and gives proper evaluation.

Expected folder structure:
    D:\TestVideos\
        I\
            I_test_1.mp4
            I_test_2.mp4
        Water\
            Water_test_1.mp4
        ...

Output:
    - Per class accuracy
    - Overall accuracy
    - Confusion matrix
    - Which words are being confused with each other
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization

# ============================================================
# CONFIGURATION — CHANGE THESE
# ============================================================
TEST_FOLDER     = r"D:\TestVideos"
WEIGHTS_PATH    = 'signbridge_weights.weights.h5'
SEQUENCE_LENGTH = 142
ACTIONS         = ['I', 'Need', 'Food', 'Water', 'Nothing']  # exact order as training
# ============================================================

USEFUL_POSE = [11, 12, 13, 14, 15, 16]
POSE_END    = 18
LH_START    = 18
LH_END      = 81
RH_START    = 81
RH_END      = 144

mp_holistic = mp.solutions.holistic


# ============================================================
# LOAD MODEL
# ============================================================
def load_signbridge_model():
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'),
                      input_shape=(SEQUENCE_LENGTH, 288)),
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
    model.load_weights(WEIGHTS_PATH)
    return model


# ============================================================
# PREPROCESSING — same as training, no augmentation
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

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])


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
            if size > 0:
                lh = lh / size
            sequence[i, LH_START:LH_END] = lh.flatten()

        rh = frame[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            wrist = rh[0].copy()
            rh    = rh - wrist
            size  = np.linalg.norm(rh[12])
            if size > 0:
                rh = rh / size
            sequence[i, RH_START:RH_END] = rh.flatten()

        pose = frame[0:POSE_END].reshape(6, 3)
        if not np.all(pose == 0):
            mid           = (pose[0] + pose[1]) / 2
            pose          = pose - mid
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


def preprocess_video(video_path):
    cap    = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=0) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame   = cv2.resize(frame, (640, 480))
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(extract_landmarks(results))

    cap.release()

    if len(frames) == 0:
        return None

    seq = np.array(frames)
    seq = interpolate_missing(seq)
    seq = normalize(seq)
    seq = add_velocity(seq)
    seq = pad_or_truncate(seq)
    return seq.astype(np.float32)


# ============================================================
# CONFUSION MATRIX DISPLAY
# ============================================================
def print_confusion_matrix(matrix, labels):
    col_width = 10
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(" " * col_width + "".join(f"{l:>{col_width}}" for l in labels))
    for i, row_label in enumerate(labels):
        row = "".join(f"{int(matrix[i][j]):>{col_width}}" for j in range(len(labels)))
        print(f"{row_label:>{col_width}}{row}")


# ============================================================
# MAIN EVALUATION
# ============================================================
print("Loading model...")
model = load_signbridge_model()
print("Model loaded.\n")

# Track results
all_true      = []
all_predicted = []
class_results = {action: {'correct': 0, 'total': 0} for action in ACTIONS}

# Confusion matrix
conf_matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)

# Get test folders
test_folders = [f for f in os.listdir(TEST_FOLDER)
                if os.path.isdir(os.path.join(TEST_FOLDER, f)) and f in ACTIONS]

if not test_folders:
    print(f"ERROR: No valid word folders found in {TEST_FOLDER}")
    print(f"Expected folders named: {ACTIONS}")
    exit()

print(f"Found test folders: {test_folders}\n")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           model_complexity=0) as holistic:

    for true_word in test_folders:
        word_path = os.path.join(TEST_FOLDER, true_word)
        videos    = [v for v in os.listdir(word_path)
                     if v.lower().endswith(('.mp4', '.avi', '.mov'))]

        print(f"--- [{true_word}] {len(videos)} test videos ---")
        true_idx = ACTIONS.index(true_word)

        for video_file in videos:
            video_path = os.path.join(word_path, video_file)
            seq        = preprocess_video(video_path)

            if seq is None:
                print(f"  ⚠️  SKIPPED {video_file} — no frames")
                continue

            input_data     = np.expand_dims(seq, axis=0)
            prediction     = model.predict(input_data, verbose=0)[0]
            predicted_idx  = np.argmax(prediction)
            predicted_word = ACTIONS[predicted_idx]
            confidence     = prediction[predicted_idx] * 100
            correct        = predicted_word == true_word

            # Update tracking
            class_results[true_word]['total']   += 1
            all_true.append(true_idx)
            all_predicted.append(predicted_idx)
            conf_matrix[true_idx][predicted_idx] += 1

            if correct:
                class_results[true_word]['correct'] += 1
                print(f"  ✅ {video_file:<30} → {predicted_word} ({confidence:.0f}%)")
            else:
                print(f"  ❌ {video_file:<30} → {predicted_word} ({confidence:.0f}%) [True: {true_word}]")

        print()

# ============================================================
# FINAL RESULTS
# ============================================================
print("=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)

print("\nPer Class Accuracy:")
for action in ACTIONS:
    if action not in class_results or class_results[action]['total'] == 0:
        print(f"  {action:<12} No test videos found")
        continue
    correct = class_results[action]['correct']
    total   = class_results[action]['total']
    acc     = (correct / total) * 100
    bar     = '█' * int(acc / 5)
    print(f"  {action:<12} {correct}/{total}  {acc:5.1f}%  {bar}")

total_correct = sum(r['correct'] for r in class_results.values())
total_videos  = sum(r['total'] for r in class_results.values())
overall_acc   = (total_correct / total_videos * 100) if total_videos > 0 else 0

print(f"\nOverall Accuracy: {total_correct}/{total_videos} = {overall_acc:.1f}%")

print_confusion_matrix(conf_matrix, ACTIONS)

print("\nHow to read confusion matrix:")
print("  Diagonal = correct predictions")
print("  Off-diagonal = what model confused each word with")