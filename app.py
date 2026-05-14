import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Input
from keras.models import Sequential
import mediapipe as mp
# ============================================================
# CONFIGURATION & CONSTANTS
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
    },
    'bilstm': {
        'weights_path':    'bilstm_weights.npy',
        'model_type':      'keras',
        'sequence_length': 122,
        'actions':         ACTIONS_22,
    },

}

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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model



@st.cache_resource
def load_model(active_model):
    cfg = MODEL_CONFIGS[active_model]
    weights_path = cfg['weights_path']
    seq_len = cfg['sequence_length']
    actions = cfg['actions']
    
    if not os.path.exists(weights_path):
        st.error(f"Weights file {weights_path} not found!")
        return None, cfg
        
    if cfg['model_type'] == 'keras':
        if active_model == 'lstm':
            model = _build_lstm(seq_len, len(actions))
        elif active_model == 'bilstm':
            model = _build_bilstm(seq_len, len(actions))
        
        weights = np.load(weights_path, allow_pickle=True)
        model.set_weights(weights)
        return model, cfg

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
    missing = np.array([np.all(sequence[i, LH_START:RH_END] == 0) for i in range(len(sequence))])
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

def pad_or_truncate(sequence, length):
    n, d = len(sequence), sequence.shape[1]
    if n > length:
        s = (n - length) // 2
        return sequence[s: s + length]
    elif n < length:
        return np.vstack([sequence, np.zeros((length - n, d))])
    return sequence

def preprocess_segment(frames, length):
    seq = np.array(frames)
    seq = interpolate_missing(seq)
    seq = normalize(seq)
    seq = add_velocity(seq)
    seq = pad_or_truncate(seq, length)
    return seq.astype(np.float32)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="SignBridge Predictor", layout="centered")

st.title("🤟 SignBridge: Video Predictor")
st.markdown("Upload a video containing sign language gestures. The model will extract features and predict the signed sentence.")

col1, col2 = st.columns([1, 2])
with col1:
    active_model_name = st.selectbox("Select Model", ["lstm", "bilstm"])
with col2:
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if video_file is not None:
    st.video(video_file)
    
    if st.button("Extract and Predict", type="primary"):
        # Load Model
        with st.spinner("Loading Model..."):
            model, cfg = load_model(active_model_name)
            if model is None:
                st.stop()
        
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        
        progress_text = "Extracting landmarks..."
        progress_bar = st.progress(0, text=progress_text)
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_landmarks = []
        all_has_hands = []
        
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
                
                # Update progress
                current_frame = len(all_landmarks)
                if current_frame % 5 == 0 or current_frame == total_frames:
                    progress = min(current_frame / float(total_frames), 1.0)
                    progress_bar.progress(progress, text=f"Extracting landmarks... ({current_frame}/{total_frames})")
                    
        cap.release()
        os.unlink(tfile.name)
        
        if len(all_landmarks) == 0:
            st.error("Could not extract frames from video.")
            st.stop()
            
        progress_bar.progress(1.0, text="Segmenting signs...")
        
        # Segment
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
            
        st.write(f"**Found {len(segments)} sign segment(s).**")
        
        if len(segments) == 0:
            st.warning("No signs detected. Check that hands are clearly visible and there are idle pauses between signs.")
            st.stop()
            
        # Predict
        predicted_words = []
        actions = cfg['actions']
        
        st.subheader("Predictions")
        
        for i, seg in enumerate(segments):
            seq = preprocess_segment(seg, cfg['sequence_length'])
            
            probs = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
                
            confidence = np.max(probs)
            word = actions[np.argmax(probs)]
            
            with st.expander(f"Sign {i+1} ({len(seg)} frames) - **{word}** ({confidence*100:.1f}%)"):
                # Top 3 predictions
                top3_idx = np.argsort(probs)[-3:][::-1]
                for idx in top3_idx:
                    st.write(f"- {actions[idx]}: {probs[idx]*100:.1f}%")
                    
            if confidence >= CONFIDENCE_THRESHOLD and word != 'Nothing':
                predicted_words.append(word)
                
        st.divider()
        st.subheader("Final Sentence")
        if predicted_words:
            st.success(" ".join(predicted_words))
        else:
            st.info("No valid words accepted above the confidence threshold.")
