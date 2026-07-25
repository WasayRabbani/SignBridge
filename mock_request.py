import cv2
import mediapipe as mp
import numpy as np
import requests
import json
import time

# ==========================================
# TEST CONFIGURATION
# ==========================================
VIDEO_PATH = "D:\Signs\Bill\Bill_15.mp4" # REPLACE WITH A VALID VIDEO PATH ON YOUR PC
SERVER_URL = "http://127.0.0.1:5050/predict_coordinates"

# The 6 Pose landmarks we actually care about (Shoulders, Elbows, Wrists)
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """Mimics the extraction logic the Flutter app will use"""
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

    return np.concatenate([pose, lh, rh]).tolist()

def simulate_flutter_app():
    print(f"Simulating Flutter App: Processing {VIDEO_PATH}...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}. Please change VIDEO_PATH to a valid file.")
        return

    frames = []
    has_hands_list = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (640, 480))
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            has_hands = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)
            landmarks = extract_landmarks(results)
            
            frames.append(landmarks)
            has_hands_list.append(has_hands)

    cap.release()
    
    # Simulate Flutter's segmentation (idle gaps)
    segments = []
    current_seg = []
    idle_count = 0
    
    for kp, has_hands in zip(frames, has_hands_list):
        if has_hands:
            if idle_count >= 12 and len(current_seg) >= 15:
                segments.append(current_seg.copy())
                current_seg = []
            idle_count = 0
            current_seg.append(kp)
        else:
            idle_count += 1
            if current_seg:
                current_seg.append(kp)
                
    if len(current_seg) >= 15:
        segments.append(current_seg)

    print(f"Extracted {len(segments)} segments. Sending JSON payload to {SERVER_URL}...")
    
    payload = {
        "segments": segments
    }
    
    # Measure time taken for the API call (this represents the actual wait time for the user)
    start_api = time.time()
    try:
        response = requests.post(SERVER_URL, json=payload)
        end_api = time.time()
        
        print("\n--- SERVER RESPONSE ---")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print(f"\nAPI Call Latency: {round(end_api - start_api, 3)} seconds")
        
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to {SERVER_URL}. Is server.py running?")

if __name__ == "__main__":
    simulate_flutter_app()
