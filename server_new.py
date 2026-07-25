"""
SignBridge — Flask Inference Server
Accepts video upload → preprocess → LSTM predict → return JSON
"""

import os
import time
import uuid
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.models import Sequential
import requests

os.environ['GLOG_minloglevel']      = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================
# CONFIG
# ============================================================
WEIGHTS_PATH  = 'LSTM_weights.npy'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


CACHE_FILE = 'vocab_cache.npy'
VOCAB_CACHE = {}
if os.path.exists(CACHE_FILE):
    try:
        VOCAB_CACHE = np.load(CACHE_FILE, allow_pickle=True).item()
        print(f"✓ Vocab cache loaded: {len(VOCAB_CACHE)} words ready for animation.")
    except Exception as e:
        print(f"Error loading vocab cache: {e}")
else:
    print(f"WARNING: Vocab cache '{CACHE_FILE}' not found! Animation endpoints will fail.")

ACTIONS_FILE = 'actions_list.npy'
if os.path.exists(ACTIONS_FILE):
    try:
        ACTIONS = np.load(ACTIONS_FILE, allow_pickle=True)
        print(f"✓ Vocabulary loaded dynamically: {len(ACTIONS)} words.")
    except Exception as e:
        print(f"Error loading actions list: {e}")
        ACTIONS = np.array(["Bathroom", "Bill", "Bring", "Broken", "Clean", "Cold",
                             "Dirty", "Find", "Food", "Help", "Hot", "I",
                             "Key", "Luggage", "Need", "No", "Nothing", "Now",
                             "Please", "Room", "Towel", "Water"])
else:
    ACTIONS = np.array(["Bathroom", "Bill", "Bring", "Broken", "Clean", "Cold",
                         "Dirty", "Find", "Food", "Help", "Hot", "I",
                         "Key", "Luggage", "Need", "No", "Nothing", "Now",
                         "Please", "Room", "Towel", "Water"])

try:
    NOTHING_IDX = np.where(ACTIONS == 'Nothing')[0][0]
except IndexError:
    NOTHING_IDX = -1
SEQUENCE_LENGTH      = 122
CONFIDENCE_THRESHOLD = 0.45   # lowered from 0.50
IDLE_THRESHOLD       = 12     # increased from 8
MIN_SIGN_FRAMES      = 15     # increased from 10
IDLE_PADDING         = 5
FRAME_SKIP           = 2      # no skip — accuracy > speed for server

POSE_END    = 18
LH_START    = 18
LH_END      = 81
RH_START    = 81
RH_END      = 144
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic


# GROQ GLOSS MAPPING

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def map_text_to_gloss(text):
    vocab_keys = [w.lower() for w in ACTIONS if w.lower() != 'nothing']
    
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found in environment. Using simple keyword fallback.")
        words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split()
        return [w for w in words if w in vocab_keys]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
You are a text-to-sign language translator for a hotel communication system.
Translate the input manager sentence into a sequence of sign language gloss words.
You MUST ONLY use words from this vocabulary list:
{", ".join(vocab_keys)}

Instructions:
- Extract ONLY words that match the meaning of the input sentence.
- Drop all other words (e.g. greetings, pronouns, prepositions, or words not in the list).
- Return ONLY the list of matched words, separated by commas.
- If no words match, return an empty string.

Input: "{text}"
Output:"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip().lower()
            if response_text:
                raw_words = [w.strip() for w in response_text.split(",") if w.strip()]
                return [w for w in raw_words if w in vocab_keys]
        else:
            print(f"Groq API returned status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        
    # Fallback to simple matching
    words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split()
    return [w for w in words if w in vocab_keys]

# ============================================================
# MODEL
# ============================================================
def build_lstm(seq_len, n_classes):
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

print(f"Loading weights: {WEIGHTS_PATH}")
model   = build_lstm(SEQUENCE_LENGTH, len(ACTIONS))
weights = np.load(WEIGHTS_PATH, allow_pickle=True)
model.set_weights(weights)
print("Model ready.\n")

# ============================================================
# PREPROCESSING
# ============================================================
def extract_landmarks(results):
    pose = np.array([
        [results.pose_landmarks.landmark[i].x,
         results.pose_landmarks.landmark[i].y,
         results.pose_landmarks.landmark[i].z]
        for i in USEFUL_POSE
    ]).flatten() if results.pose_landmarks else np.zeros(18)

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
            sz = np.linalg.norm(lh[12])
            if sz > 0: lh = lh / sz
            sequence[i, LH_START:LH_END] = lh.flatten()
        rh = f[RH_START:RH_END].reshape(21, 3)
        if not np.all(rh == 0):
            rh = rh - rh[0]
            sz = np.linalg.norm(rh[12])
            if sz > 0: rh = rh / sz
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


def extract_and_segment(video_path):
    cap           = cv2.VideoCapture(video_path)
    all_landmarks = []
    all_has_hands = []
    frame_idx     = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
        smooth_landmarks=True
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue
            frame   = cv2.resize(frame, (640, 480))
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp      = extract_landmarks(results)
            has_hands = (results.left_hand_landmarks is not None or
                         results.right_hand_landmarks is not None)
            all_landmarks.append(kp)
            all_has_hands.append(has_hands)

    cap.release()

    # Segment
    segments    = []
    current_seg = []
    idle_count  = 0
    idle_buffer = []

    for kp, has_hands in zip(all_landmarks, all_has_hands):
        if has_hands:
            if idle_count >= IDLE_THRESHOLD and len(current_seg) >= MIN_SIGN_FRAMES:
                segments.append(current_seg.copy())
                current_seg = []
            if len(current_seg) == 0 and len(idle_buffer) > 0:
                current_seg.extend(idle_buffer[-IDLE_PADDING:])
            idle_count  = 0
            idle_buffer = []
            current_seg.append(kp)
        else:
            idle_count += 1
            idle_buffer.append(kp)
            if current_seg:
                current_seg.append(kp)

    if len(current_seg) >= MIN_SIGN_FRAMES:
        segments.append(current_seg)

    return segments


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext      = os.path.splitext(video_file.filename)[-1] or '.mp4'
    tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    video_file.save(tmp_path)

    try:
        t_start  = time.time()
        segments = extract_and_segment(tmp_path)

        if not segments:
            return jsonify({
                'sentence': '',
                'words':    [],
                'message':  'No signs detected. Ensure hands are visible with idle gaps between signs.'
            })

        predicted_words = []
        segment_details = []

        for i, seg in enumerate(segments):
            seq   = preprocess_segment(seg)
            probs = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]

            # get raw confidence BEFORE suppressing Nothing
            confidence = float(np.max(probs))

            # suppress Nothing — renormalize
            if NOTHING_IDX != -1:
                probs[NOTHING_IDX] = 0.0
            probs = probs / probs.sum()

            word = ACTIONS[np.argmax(probs)]

            detail = {
                'segment':    i + 1,
                'frames':     len(seg),
                'word':       word,
                'confidence': round(float(np.max(probs)) * 100, 1),
                'accepted':   False,
                'scores':     {a: round(float(p) * 100, 1) for a, p in zip(ACTIONS, probs)}
            }

            if confidence >= CONFIDENCE_THRESHOLD and word != 'Nothing':
                predicted_words.append(word)
                detail['accepted'] = True

            segment_details.append(detail)

        elapsed = round(time.time() - t_start, 1)

        return jsonify({
            'sentence':                 ' '.join(predicted_words),
            'words':                    predicted_words,
            'segments':                 segment_details,
            'processing_time_seconds':  elapsed
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/predict_coordinates', methods=['POST'])
def predict_coordinates():
    """
    Accepts raw JSON coordinates from the Flutter app.
    Expected format: 
    {
      "segments": [
        [ [x,y,z... 144 features], [x,y,z...], ... ],
        [ [x,y,z...], ... ]
      ]
    }
    """
    try:
        t_start = time.time()
        data = request.get_json()
        
        if not data or 'segments' not in data:
            return jsonify({'error': 'No segments provided in JSON payload'}), 400
            
        segments = data['segments']
        
        if not segments:
            return jsonify({
                'sentence': '',
                'words':    [],
                'message':  'No signs detected.'
            })

        predicted_words = []
        segment_details = []

        for i, seg in enumerate(segments):
            # preprocess_segment expects a list of frames, and handles conversion to numpy
            seq   = preprocess_segment(seg)
            probs = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]

            # get raw confidence BEFORE suppressing Nothing
            confidence = float(np.max(probs))

            # suppress Nothing — renormalize
            if NOTHING_IDX != -1:
                probs[NOTHING_IDX] = 0.0
            probs = probs / probs.sum()

            word = ACTIONS[np.argmax(probs)]

            detail = {
                'segment':    i + 1,
                'frames':     len(seg),
                'word':       word,
                'confidence': round(float(np.max(probs)) * 100, 1),
                'accepted':   False,
                'scores':     {a: round(float(p) * 100, 1) for a, p in zip(ACTIONS, probs)}
            }

            if confidence >= CONFIDENCE_THRESHOLD and word != 'Nothing':
                predicted_words.append(word)
                detail['accepted'] = True

            segment_details.append(detail)

        elapsed = round(time.time() - t_start, 3)

        return jsonify({
            'sentence':                 ' '.join(predicted_words),
            'words':                    predicted_words,
            'segments':                 segment_details,
            'processing_time_seconds':  elapsed
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/animate', methods=['POST'])
def animate():
    """
    Accepts raw manager reply text: {"text": "Bringing towel now"}
    1. Maps text to vocabulary using Groq.
    2. Builds interpolated continuous landmark sequence.
    3. Returns JSON frames.
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Map to Glosses using Groq
    matched_glosses = map_text_to_gloss(text)

    # If no matching glosses found, return empty payload (Flutter skips widget)
    if not matched_glosses:
        return jsonify({
            'words': [],
            'frames': [],
            'word_indices': []
        })

    # 2. Build continuous interpolated sequence
    try:
        frames_list = []
        word_indices = []
        TRANSITION_FRAMES = 8

        # Filter out glosses that aren't in the cache
        valid_glosses = [g for g in matched_glosses if g in VOCAB_CACHE]
        if not valid_glosses:
            return jsonify({
                'words': [],
                'frames': [],
                'word_indices': []
            })

        for i, gloss in enumerate(valid_glosses):
            seq = VOCAB_CACHE[gloss]
            for frame in seq:
                frames_list.append(frame.tolist())
                word_indices.append(i)
            
            # Interpolate transition to next word
            if i < len(valid_glosses) - 1:
                next_gloss = valid_glosses[i+1]
                last_frame = seq[-1]
                first_frame = VOCAB_CACHE[next_gloss][0]
                
                for t in range(1, TRANSITION_FRAMES + 1):
                    alpha = t / (TRANSITION_FRAMES + 1)
                    interp = (1 - alpha) * last_frame + alpha * first_frame
                    frames_list.append(interp.tolist())
                    word_indices.append(i)

        return jsonify({
            'words': valid_glosses,
            'frames': frames_list,
            'word_indices': word_indices
        })
    except Exception as e:
        return jsonify({'error': f"Failed to build animation: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'LSTM SignBridge'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)