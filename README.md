# Hotel Sign Language Recognition System
### Final Year Project — Sign Language to Text Communication for Hotel Environment

---

## Project Overview

A real-time sign language recognition system designed for hotel environments. A deaf guest performs sign language gestures in front of a camera. The system detects and classifies the signs, forms a natural sentence using an LLM, and sends it to hotel staff. Staff reply via text which is displayed clearly to the guest.

---

## System Pipeline

```
Camera Feed
    ↓
MediaPipe (Hand + Pose Landmark Extraction)
    ↓
BiLSTM Model (Sign Classification)
    ↓
Word Buffer (Sequence Collection)
    ↓
LLM - Gemini API (Sentence Formation)
    ↓
Manager Screen (Staff Reply via Text)
    ↓
Guest Screen (Text Display)
```

---

## Project Status

| Phase | Status | Notes |
|---|---|---|
| Word finalization | ✅ Done | 15-20 hotel-related signs |
| Dataset recording | 🔄 In Progress | 250 videos per word, 5 signers |
| Video quality checking | ✅ Done | Automated checker script |
| Landmark extraction | ✅ Done | 144 features per frame |
| Preprocessing | ✅ Done | Normalization + augmentation |
| BiLSTM training | ⏳ Pending | |
| Real-time pipeline | ⏳ Pending | |
| LLM integration | ⏳ Pending | Gemini API |
| UI / Manager screen | ⏳ Pending | |

---

## Dataset

### Words (Classes)
Hotel-related signs including: I, Water, Food, Room, Help, Toilet, Key, Bed, Bill, Towel, Nothing (+ more)

The **Nothing** class covers idle gestures, hair touching, random movements — teaches the model what is NOT a sign.

### Recording Protocol
- **People:** 5 signers with different hand sizes and signing speeds
- **Videos per word:** 250 (across all signers)
- **Structure per video:** Hands at sides → Perform sign → Hands back to sides
- **Environment:** Plain background, even front lighting, full upper body in frame
- **Variation:** Fast signs, slow signs, slight left/right position variation

### Folder Structure
```
D:\Signs\
    Water\
        Water_1.mp4
        Water_2.mp4
        ...
    I\
        I_1.mp4
        ...
```

---

## Scripts

### 1. Video Renaming
`rename_videos.py`
Renames raw recorded videos to consistent format: `WordName_1.mp4, WordName_2.mp4`

```python
rename_videos(r"D:\Signs\Water", 'Water')
```

### 2. Video Quality Checker
`video_quality_checker.py`
Checks each video for:
- Body/elbow out of frame
- Head position
- Shoulder visibility
- Hand detection during signing window (middle 60% of video)
- Video too short/long

```python
DATA_PATH = r"D:\Signs"
```

### 3. Landmark Extraction
`extract_landmarks.py`
Extracts MediaPipe landmarks from all videos. Saves one `.npy` file per video.

**Features extracted per frame: 144 values**
- 6 pose landmarks × 3 (x,y,z) = 18 values → shoulders, elbows, wrists only
- 21 left hand landmarks × 3 = 63 values
- 21 right hand landmarks × 3 = 63 values

Face and lower body landmarks are intentionally excluded — irrelevant for sign language.

```python
INPUT_FOLDER  = r"D:\Signs"
OUTPUT_FOLDER = r"D:\Extracted"
```

Output structure:
```
D:\Extracted\
    Water\
        0.npy    # shape: (num_frames, 144)
        1.npy
        ...
```

### 4. Landmark Visualizer
`visualize_landmarks.py`
Plays back extracted `.npy` files as dot animation to verify extraction quality.
- Green dots = pose (shoulders, elbows, wrists)
- Blue dots = left hand
- Red dots = right hand

Idle frames should show only green dots. Signing frames should show green + red/blue.

### 5. Preprocessing
`preprocess.py`
Full preprocessing pipeline. Each original video becomes 5 training samples via augmentation.

**Pipeline order (order matters):**
1. Interpolate missing frames (fix MediaPipe detection gaps)
2. Normalize — position (subtract wrist) + scale (divide by hand size)
3. Add velocity features (frame-to-frame difference) → 144 + 144 = 288 features
4. Pad/Truncate to fixed sequence length (60 frames)
5. Augment — mirror, gaussian noise, time stretch fast (0.8x), time stretch slow (1.2x)

**Output:**
```
X.npy — shape: (total_samples, 60, 288)
y.npy — shape: (total_samples, num_classes)  one-hot encoded
```

250 videos per word × 5 augmentations = **1250 training samples per word**

```python
DATA_PATH       = r"D:\Extracted"
OUTPUT_PATH     = r"D:\Preprocessed"
SEQUENCE_LENGTH = 60
ACTIONS         = np.array(['I', 'Water', 'Food', ...])
```

---

## Model — BiLSTM (Pending)

- Input shape: `(60, 288)` — 60 frames, 288 features per frame
- Architecture: Bidirectional LSTM layers + Dense classification head
- Output: Softmax over N word classes

---

## Real-Time Pipeline — Logic (Pending)

```python
word_buffer = []
current_word = None
side_counter = 0
SIDE_THRESHOLD = 15  # frames of hands-at-sides = word boundary

# When hands return to sides after a sign → word is complete → save to buffer
# When buffer pause exceeds 60 frames → send word list to LLM → display sentence
```

### Word → Sentence (LLM)
Word buffer like `["I", "want", "room"]` sent to Gemini API with hotel-scoped system prompt.
LLM returns natural sentence: `"I would like to book a room."`

---

## Generalization Strategy

| Technique | Where Applied | Benefit |
|---|---|---|
| 5 diverse signers | Recording | Different hand sizes and speeds |
| Speed variation | Recording | Fast + slow signs per word |
| Position normalization | Preprocessing | Signer position in frame irrelevant |
| Scale normalization | Preprocessing | Hand size differences removed |
| Velocity features | Preprocessing | Movement dynamics captured |
| Mirror augmentation | Preprocessing | Left/right hand variation |
| Noise augmentation | Preprocessing | Natural tremor simulation |
| Time stretch augmentation | Preprocessing | Speed variation without re-recording |
| Leave-one-person-out test | Evaluation | True generalization validation |

---

## Tech Stack

| Component | Technology |
|---|---|
| Landmark extraction | MediaPipe Holistic |
| Model | TensorFlow / Keras BiLSTM |
| Sentence formation | Gemini API |
| Real-time capture | OpenCV |
| Data processing | NumPy |
| UI | TBD |

---

## Requirements

```
tensorflow
mediapipe
opencv-python
numpy
google-generativeai
```

---

## To Do (Next Steps)

- [ ] Finish recording all word datasets
- [ ] Run extraction on full dataset
- [ ] Run preprocessing — verify X.npy and y.npy shapes
- [ ] Build and train BiLSTM model
- [ ] Test leave-one-person-out generalization
- [ ] Build real-time detection pipeline
- [ ] Integrate Gemini API for sentence formation
- [ ] Build two-screen UI (guest + manager)
- [ ] End-to-end testing

---

## Known Limitations

- Model trained on specific signers — may need fine-tuning for completely new users
- Hotel vocabulary limited to trained words only
- Manager replies kept short and simple for deaf guest readability
- Requires adequate front lighting for reliable MediaPipe detection

---

*Last updated: Dataset recording and preprocessing phase*