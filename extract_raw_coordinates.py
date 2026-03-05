"""
Landmark Extraction Script
- Extracts selected pose + both hands only (144 values per frame)
- No preprocessing — raw coordinates saved
- One .npy file per video (shape: num_frames x 144)
- Works with renamed files like: Water_1.mp4, Water_2.mp4 etc.

OUTPUT STRUCTURE:
D:\Extracted\
    Water\
        0.npy  ← Water_1.mp4 (shape: frames x 144)
        1.npy  ← Water_2.mp4
        ...
    I\
        0.npy
        ...
"""

import cv2
import numpy as np
import os
import mediapipe as mp

# ============================================================
# CONFIGURATION — CHANGE THESE
# ============================================================
INPUT_FOLDER  = r"D:\Signs"      # Folder containing word subfolders (e.g. D:\Signs\Water, D:\Signs\I)
OUTPUT_FOLDER = r"D:\Extracted"  # Where .npy files will be saved
# ============================================================

# Only these 6 pose landmarks are useful for sign language
# 11=left shoulder, 12=right shoulder, 13=left elbow,
# 14=right elbow,  15=left wrist,     16=right wrist
USEFUL_POSE = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """
    Extracts selected pose + both hands.
    Returns 144 values per frame:
        - 6 pose landmarks x 3 (x,y,z) = 18
        - 21 left hand landmarks x 3    = 63
        - 21 right hand landmarks x 3   = 63
    """
    # Selected pose landmarks
    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z]
            for i in USEFUL_POSE
        ]).flatten()
    else:
        pose = np.zeros(18)

    # Left hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)

    # Right hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])  # 144 values


def get_video_number(filename):
    """Extract trailing number from renamed files like Water_1.mp4 -> 1"""
    name = os.path.splitext(filename)[0]   # remove extension
    part = name.split('_')[-1]             # get part after last underscore
    return int(part) if part.isdigit() else 0


# ============================================================
# MAIN EXTRACTION
# ============================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all word folders (e.g. Water, I, Room, etc.)
word_folders = [f for f in os.listdir(INPUT_FOLDER)
                if os.path.isdir(os.path.join(INPUT_FOLDER, f))]

if not word_folders:
    print("ERROR: No subfolders found in INPUT_FOLDER.")
    print("Expected structure: D:\\Signs\\Water\\Water_1.mp4 etc.")
    exit()

print(f"Found {len(word_folders)} word folders: {word_folders}\n")

total_videos  = 0
total_skipped = 0

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:

    for word in word_folders:
        word_input_path  = os.path.join(INPUT_FOLDER, word)
        word_output_path = os.path.join(OUTPUT_FOLDER, word)
        os.makedirs(word_output_path, exist_ok=True)

        # Get videos and sort by their trailing number
        videos = [v for v in os.listdir(word_input_path)
                  if v.lower().endswith(('.mp4', '.avi', '.mov'))]
        videos.sort(key=get_video_number)

        print(f"--- [{word}] {len(videos)} videos ---")

        for sequence, video_file in enumerate(videos):
            video_path = os.path.join(word_input_path, video_file)
            cap = cv2.VideoCapture(video_path)

            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                keypoints = extract_landmarks(results)
                frames.append(keypoints)

            cap.release()

            # Skip empty videos
            if len(frames) == 0:
                print(f"  ⚠️  SKIPPED {video_file} — no frames found")
                total_skipped += 1
                continue

            # Save as single 2D array: shape (num_frames, 144)
            sequence_array = np.array(frames)
            save_path = os.path.join(word_output_path, f"{sequence}.npy")
            np.save(save_path, sequence_array)

            total_videos += 1
            print(f"  ✅ {video_file} -> {sequence}.npy | shape: {sequence_array.shape}")

        print()

print("=" * 50)
print(f"DONE. {total_videos} videos extracted, {total_skipped} skipped.")
print(f"Each .npy shape: (num_frames, 144)")
print(f"Saved to: {OUTPUT_FOLDER}")