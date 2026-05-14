# SignBridge: Hotel Sign Language Recognition System

A real-time, AI-powered sign language recognition system tailored for hotel environments. SignBridge enables deaf or hard-of-hearing guests to communicate via sign language in front of a camera. The system detects the hand and pose landmarks, classifies the sequence of signs, and reconstructs a sentence.

This repository focuses on the complete **AI Training & Inference Pipeline**—from raw data collection to a functional interactive web application.

---

## ✨ Features

- **End-to-End Pipeline:** Scripts provided for data cleaning, landmark extraction, preprocessing, augmentation, and model training.
- **Robust Feature Extraction:** Leverages MediaPipe Holistic to capture 144 facial, pose, and hand landmarks per frame.
- **Advanced Preprocessing:** Uses interpolation for missing frames, spatial normalization, velocity feature calculation, and dynamic augmentations (Gaussian noise, time-stretching, mirroring).
- **Continuous Sign Parsing:** Can automatically segment and predict multiple consecutive signs from a single continuous video.
- **Interactive UI:** Includes a beautiful Streamlit web application (`app.py`) for drag-and-drop video testing and inference visualization.

---

## 📦 Setup & Installation

> [!IMPORTANT]
> A machine with a reasonably modern CPU is sufficient for inference, but a GPU is heavily recommended if you intend to retrain the models.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SignBridge.git
   cd SignBridge
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start (Testing)

The easiest way to see SignBridge in action is by using the provided Streamlit web app. We provide pre-trained LSTM weights for a 22-word hotel-specific vocabulary.

### 1. Interactive Web UI
Launch the interactive web dashboard to upload a video and see real-time extraction and prediction:
```bash
streamlit run app.py
```
*This will open the application in your default web browser.*

### 2. Command-Line Multi-Sign Prediction
To run a continuous multi-sign video directly from your terminal:
1. Open `multi_sign_predict.py` in your text editor.
2. Edit the `VIDEO_PATH` variable to point to your test video.
3. Run the script:
   ```bash
   python multi_sign_predict.py
   ```

### 3. Command-Line Single Sign Prediction
To test a single, isolated sign:
1. Open `single_vid_test.py`.
2. Edit the `VIDEO_PATH` variable.
3. Run:
   ```bash
   python single_vid_test.py
   ```

---

## 🛠️ Pipeline: Training Your Own Model

If you want to train the model from scratch on your own dataset, follow these sequential steps:

### Phase 1: Data Formatting & QA
Place your raw `.mp4` videos sorted into subfolders by class name.
- **`rename_files.py`**: Standardizes your raw video filenames into a consistent indexed format (e.g., `Water_1.mp4`).
- **`checking_short_vid.py`**: Scans the dataset to identify videos that are too short to contain meaningful sign gestures.
- **`split_nothing_class.py`**: Automatically splits long, continuous idle recordings into multiple short clips for the "Nothing" class.

### Phase 2: Landmark Extraction
- **`extract_raw_coordinates.py`**: A multiprocessing script that runs MediaPipe Holistic across your video dataset, saving the raw landmarks as `.npy` arrays.
- **`checking_raw_coordinates_validity.py`**: (Optional) Re-renders the extracted `.npy` coordinates onto a blank video canvas to visually verify extraction quality.
- **`checking_sequence.py`**: Calculates the average sequence frame length of your dataset to help you set an optimal `SEQUENCE_LENGTH`.

### Phase 3: Preprocessing
> [!WARNING]  
> The preprocessing pipeline rigorously prevents data leakage by separating the test split *before* augmentation.

- **`preprocess_train.py`**: Takes the first 80% of data per class, interpolates missing frames, normalizes coordinates, calculates velocities, and pads/truncates. It then applies a **5x data augmentation** multiplier (noise, fast/slow time stretch, horizontal mirror) and saves `X_train.npy` and `y_train.npy`.
- **`preprocess_test.py`**: Takes the remaining 20% of data, applies only the basic formatting (NO augmentation), and saves `X_test.npy` and `y_test.npy` to ensure realistic validation accuracy.

### Phase 4: Training & Validation
- **`LSTM_Model_Training.ipynb`**: A Jupyter Notebook that loads the preprocessed data, compiles the LSTM/BiLSTM architecture, and trains the model. It also generates evaluation metrics (Confusion Matrix, Classification Report) and exports the final weights (`lstm_weights.npy` and `bilstm_weights.npy`).

---

## 🧠 Model Architecture

The default `LSTM` architecture processes sequences of 122 frames with 288 features each (spatial coordinates + velocities):

- `Input Layer (122, 288)`
- `LSTM (64 units, return_sequences=True)` + `Dropout(0.3)`
- `LSTM (128 units, return_sequences=True)` + `Dropout(0.3)`
- `LSTM (64 units, return_sequences=False)`
- `BatchNormalization` + `Dense (64 units)` + `Dropout(0.3)`
- `Dense (Softmax Output)`

---

## 🔮 Future Roadmap

- **LLM Integration:** Connect the predicted sequence of isolated words (e.g., `["I", "Need", "Room", "Clean"]`) to a Large Language Model (like Google Gemini or GPT-4) to generate grammatically fluid natural language output.
- **Live Camera Feed:** Re-enable live camera Streamlit components once deployed in a controlled physical hotel kiosk.