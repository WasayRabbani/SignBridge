"""
SignBridge — Real-Time Inference with Clear Visual Feedback

WHAT YOU SEE:
- Top bar shows current state in large text
- Progress bar fills as you sign
- Flash screen when word is detected
- Bottom shows collected words
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
WEIGHTS_PATH         = 'model_weights.npy'
SEQUENCE_LENGTH      = 142
ACTIONS              = ['I', 'Need', 'Food', 'Water', 'Nothing']
PREDICTION_THRESHOLD = 0.85
SIGN_BUFFER_SIZE     = 20
BOUNDARY_THRESHOLD   = 20
SENTENCE_PAUSE       = 60

POSE_END    = 18
LH_START    = 18
LH_END      = 81
RH_START    = 81
RH_END      = 144
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
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
print("Model loaded.\n")

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


# ============================================================
# LANDMARK EXTRACTION
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


def normalize_frame(frame):
    frame = frame.copy()
    lh = frame[LH_START:LH_END].reshape(21, 3)
    if not np.all(lh == 0):
        lh = lh - lh[0]; size = np.linalg.norm(lh[12])
        if size > 0: lh = lh / size
        frame[LH_START:LH_END] = lh.flatten()
    rh = frame[RH_START:RH_END].reshape(21, 3)
    if not np.all(rh == 0):
        rh = rh - rh[0]; size = np.linalg.norm(rh[12])
        if size > 0: rh = rh / size
        frame[RH_START:RH_END] = rh.flatten()
    pose = frame[0:POSE_END].reshape(6, 3)
    if not np.all(pose == 0):
        pose = pose - (pose[0] + pose[1]) / 2
        frame[0:POSE_END] = pose.flatten()
    return frame


def prepare_sequence(frames):
    seq      = np.array([normalize_frame(f) for f in frames])
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    seq      = np.concatenate([seq, velocity], axis=1)
    n        = len(seq)
    if n > SEQUENCE_LENGTH:
        s   = (n - SEQUENCE_LENGTH) // 2
        seq = seq[s: s + SEQUENCE_LENGTH]
    elif n < SEQUENCE_LENGTH:
        seq = np.vstack([seq, np.zeros((SEQUENCE_LENGTH - n, seq.shape[1]))])
    return seq.astype(np.float32)


def hands_visible(results):
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


# ============================================================
# UI HELPER
# ============================================================
def draw_text_with_bg(frame, text, pos, font_scale, color, bg_color, thickness=2):
    """Draw text with background rectangle for readability"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)


# ============================================================
# STATES
# ============================================================
STATE_WAITING   = "WAITING FOR SIGN"
STATE_COLLECTING = "COLLECTING..."
STATE_PREDICTING = "PREDICTING..."
STATE_DETECTED   = "WORD DETECTED!"

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_buffer     = []
word_list        = []
current_word     = None
no_hand_counter  = 0
prediction_text  = ""
sentence_text    = ""
collecting       = False
current_state    = STATE_WAITING
flash_counter    = 0      # frames to show detection flash
flash_word       = ""     # word to show in flash

print("Camera started.")
print("Raise your hands to start signing.")
print("Lower hands to confirm word.")
print("Press Q to quit.\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.resize(frame, (640, 480))
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_landmarks(results)

        # ── STATE MACHINE ──
        if hands_visible(results):
            no_hand_counter = 0
            collecting      = True
            current_state   = STATE_COLLECTING
            frame_buffer.append(keypoints)

        else:
            no_hand_counter += 1

            if collecting:
                frame_buffer.append(keypoints)

            # Word boundary — hands lowered long enough
            if no_hand_counter >= BOUNDARY_THRESHOLD and len(frame_buffer) >= SIGN_BUFFER_SIZE:
                current_state = STATE_PREDICTING

                seq        = prepare_sequence(frame_buffer)
                prediction = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
                confidence = np.max(prediction)
                predicted  = ACTIONS[np.argmax(prediction)]

                if confidence > PREDICTION_THRESHOLD and predicted != 'Nothing':
                    if predicted != current_word:
                        current_word    = predicted
                        word_list.append(predicted)
                        prediction_text = f"{predicted}  {confidence*100:.0f}%"
                        flash_word      = predicted
                        flash_counter   = 30   # show flash for 30 frames
                        current_state   = STATE_DETECTED
                        print(f"✅ Word detected: {predicted} ({confidence*100:.1f}%)")
                else:
                    current_state = STATE_WAITING
                    print(f"   Low confidence or Nothing: {predicted} ({confidence*100:.1f}%) — ignored")

                frame_buffer = []
                collecting   = False

            elif no_hand_counter > BOUNDARY_THRESHOLD:
                current_state = STATE_WAITING

            # Sentence complete
            if no_hand_counter >= SENTENCE_PAUSE and len(word_list) > 0:
                sentence_text = " ".join(word_list)
                print(f"\n📝 Sentence: {sentence_text}\n")
                word_list    = []
                current_word = None

        # ── DRAW HAND LANDMARKS ──
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 180), thickness=1)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(180, 80, 0), thickness=1)
            )

        # ── FLASH SCREEN ON DETECTION ──
        if flash_counter > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (0, 200, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            flash_counter -= 1

        # ── TOP STATUS BAR ──
        bar_color = {
            STATE_WAITING:    (50, 50, 50),
            STATE_COLLECTING: (0, 120, 0),
            STATE_PREDICTING: (0, 120, 200),
            STATE_DETECTED:   (0, 180, 0),
        }.get(current_state, (50, 50, 50))

        cv2.rectangle(frame, (0, 0), (640, 70), bar_color, -1)
        cv2.putText(frame, current_state, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Last detected word + confidence
        if prediction_text:
            cv2.putText(frame, f"Last: {prediction_text}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

        # ── PROGRESS BAR ──
        if collecting and frame_buffer:
            progress = min(len(frame_buffer) / SEQUENCE_LENGTH, 1.0)
            bar_w    = int(620 * progress)
            cv2.rectangle(frame, (10, 72), (630, 82), (50, 50, 50), -1)     # background
            cv2.rectangle(frame, (10, 72), (10 + bar_w, 82), (0, 255, 0), -1)  # fill
            cv2.putText(frame, f"{int(progress*100)}%", (590, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── INSTRUCTION TEXT ──
        if current_state == STATE_WAITING:
            cv2.putText(frame, "Raise hands to start signing",
                        (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        # ── FLASH WORD IN CENTER ──
        if flash_counter > 0 and flash_word:
            font_scale = 2.5
            text_size  = cv2.getTextSize(flash_word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)[0]
            text_x     = (640 - text_size[0]) // 2
            cv2.putText(frame, flash_word, (text_x, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4)

        # ── BOTTOM BAR ──
        cv2.rectangle(frame, (0, 415), (640, 480), (20, 20, 20), -1)

        words_display = "  →  ".join(word_list) if word_list else "No words yet"
        cv2.putText(frame, f"Words: {words_display}", (10, 438),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if sentence_text:
            cv2.putText(frame, f"Sentence: {sentence_text}", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Lower hands 2sec to form sentence", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        cv2.imshow('SignBridge', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()