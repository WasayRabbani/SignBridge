"""
Incremental Landmark Extraction
- Skips videos that already have extracted .npy files
- Safe to run multiple times as you add new videos
- Names .npy files after video name (not sequence number)

OUTPUT STRUCTURE:
D:\Extracted\
    Water\
        Water_1.npy   ← from Water_1.mp4
        Water_2.npy   ← from Water_2.mp4
        ...
    I\
        I_1.npy
        ...
"""

import cv2
import numpy as np
import os
import mediapipe as mp

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FOLDER  = r"D:\Signs"
OUTPUT_FOLDER = r"D:\Extracted"

# Only useful pose landmarks
# 11=left shoulder, 12=right shoulder, 13=left elbow,
# 14=right elbow,  15=left wrist,     16=right wrist
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """144 values per frame: 6 pose x3 + 21 lh x3 + 21 rh x3"""
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


def get_video_number(filename):
    """Sort by trailing number: I_10.mp4 → 10"""
    name = os.path.splitext(filename)[0]
    part = name.split('_')[-1]
    return int(part) if part.isdigit() else 0


# ============================================================
# MAIN
# ============================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

word_folders = [f for f in os.listdir(INPUT_FOLDER)
                if os.path.isdir(os.path.join(INPUT_FOLDER, f))]

if not word_folders:
    print("ERROR: No word subfolders found in INPUT_FOLDER")
    exit()

print(f"Found {len(word_folders)} word folders: {word_folders}\n")

total_processed = 0
total_skipped   = 0
total_failed    = 0

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,model_complexity=0) as holistic:

    for word in word_folders:
        word_input_path  = os.path.join(INPUT_FOLDER, word)
        word_output_path = os.path.join(OUTPUT_FOLDER, word)
        os.makedirs(word_output_path, exist_ok=True)

        videos = [v for v in os.listdir(word_input_path)
                  if v.lower().endswith(('.mp4', '.avi', '.mov'))]
        videos.sort(key=get_video_number)

        print(f"--- [{word}] {len(videos)} videos found ---")

        for video_file in videos:
            video_name  = os.path.splitext(video_file)[0]  # e.g. "I_1"
            npy_name    = f"{video_name}.npy"               # e.g. "I_1.npy"
            save_path   = os.path.join(word_output_path, npy_name)

            # ── SKIP if already extracted ──
            if os.path.exists(save_path):
                print(f"  ⏭️  SKIPPED {video_file} — already extracted")
                total_skipped += 1
                continue

            # ── EXTRACT ──
            video_path = os.path.join(word_input_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(extract_landmarks(results))

            cap.release()

            if len(frames) == 0:
                print(f"  ⚠️  FAILED {video_file} — no frames extracted")
                total_failed += 1
                continue

            # Save as (num_frames, 144)
            np.save(save_path, np.array(frames))
            total_processed += 1
            print(f"  ✅ {video_file} → {npy_name} | shape: {np.array(frames).shape}")

        print()

print("=" * 50)
print(f"DONE.")
print(f"  Newly extracted : {total_processed}")
print(f"  Skipped (exist) : {total_skipped}")
print(f"  Failed          : {total_failed}")
print(f"Saved to: {OUTPUT_FOLDER}")