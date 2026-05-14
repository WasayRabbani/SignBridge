"""
Incremental Landmark Extraction — Multiprocessing Version
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FOLDER  = r"D:\Signs"
OUTPUT_FOLDER = r"D:\Extracted"

USEFUL_POSE = [11, 12, 13, 14, 15, 16]
NUM_WORKERS = 6  # use half your cores — MediaPipe is already CPU-heavy per worker


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


def get_video_number(filename):
    name = os.path.splitext(filename)[0]
    part = name.split('_')[-1]
    return int(part) if part.isdigit() else 0


def process_word(word):
    """Each worker processes one word folder independently."""
    mp_holistic = mp.solutions.holistic

    word_input_path  = os.path.join(INPUT_FOLDER, word)
    word_output_path = os.path.join(OUTPUT_FOLDER, word)
    os.makedirs(word_output_path, exist_ok=True)

    videos = [v for v in os.listdir(word_input_path)
              if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    videos.sort(key=get_video_number)

    processed, skipped, failed = 0, 0, 0

    # Each worker creates its OWN Holistic instance — this is key
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=0) as holistic:
        for video_file in videos:
            video_name = os.path.splitext(video_file)[0]
            save_path  = os.path.join(word_output_path, f"{video_name}.npy")

            if os.path.exists(save_path):
                print(f"  [{word}] ⏭️  SKIPPED {video_file}")
                skipped += 1
                continue

            video_path = os.path.join(word_input_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 2 == 0:  # frame skipping
                    frame = cv2.resize(frame, (640, 480))
                    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(extract_landmarks(results))
                frame_idx += 1

            cap.release()

            if len(frames) == 0:
                print(f"  [{word}] ⚠️  FAILED {video_file}")
                failed += 1
                continue

            arr = np.array(frames)
            np.save(save_path, arr)
            processed += 1
            print(f"  [{word}] ✅ {video_file} | shape: {arr.shape}")

    return word, processed, skipped, failed


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    word_folders = [f for f in os.listdir(INPUT_FOLDER)
                    if os.path.isdir(os.path.join(INPUT_FOLDER, f))]

    if not word_folders:
        print("ERROR: No word subfolders found in INPUT_FOLDER")
        exit()

    print(f"Found {len(word_folders)} word folders: {word_folders}")
    print(f"Running with {NUM_WORKERS} workers\n")

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_word, word_folders)

    print("\n" + "=" * 50)
    print("DONE.")
    total_p = total_s = total_f = 0
    for word, p, s, f in results:
        print(f"  {word}: extracted={p}, skipped={s}, failed={f}")
        total_p += p; total_s += s; total_f += f
    print(f"\n  Total extracted : {total_p}")
    print(f"  Total skipped   : {total_s}")
    print(f"  Total failed    : {total_f}")
    print(f"Saved to: {OUTPUT_FOLDER}")