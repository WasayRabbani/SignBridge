"""This check mp4 video whether elbow, hand,landmarks are visible or not"""

import cv2
import mediapipe as mp
import os

# /Claude
mp_holistic = mp.solutions.holistic

def analyze_video_quality(video_path):
    cap = cv2.VideoCapture(video_path)
    issues = []
    frame_count = 0
    all_frames_results = []
    missing_body_frames = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Store result for each frame to analyze signing window later
            all_frames_results.append(results)

            # --- Body Check ---
            if results.pose_landmarks:
                # Elbow boundary check
                for i in [13, 14]:
                    lm = results.pose_landmarks.landmark[i]
                    if lm.x < 0.02 or lm.x > 0.98 or lm.y > 0.98:
                        issues.append("Elbow out of frame")

                # Head too high
                if results.pose_landmarks.landmark[0].y < 0.05:
                    issues.append("Head too high - step back")

                # Shoulder visibility check
                for i in [11, 12]:
                    lm = results.pose_landmarks.landmark[i]
                    if lm.visibility < 0.5:
                        issues.append("Shoulders not clearly visible")
            else:
                missing_body_frames += 1

            # Check hands going out of frame (applies to whole video)
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        if lm.x < 0.02 or lm.x > 0.98 or lm.y < 0.02 or lm.y > 0.98:
                            issues.append("Hand going out of frame")
                            break

    cap.release()

    # --- Post Processing Checks ---
    if frame_count == 0:
        return ["Video is empty or unreadable"]

    # Video length check
    if frame_count < 15:
        issues.append(f"Video too short ({frame_count} frames) - minimum 15 frames needed")
    if frame_count > 250:
        issues.append(f"Video too long ({frame_count} frames) - trim neutral portions")

    # --- Hand Check ONLY on Middle 60% (Signing Window) ---
    # First 20% = hands idle start, Last 20% = hands idle end
    # Only the middle 60% should have active signing
    signing_start = int(frame_count * 0.20)
    signing_end = int(frame_count * 0.80)
    signing_frames = all_frames_results[signing_start:signing_end]

    if len(signing_frames) > 0:
        hand_detected_in_signing = sum(
            1 for r in signing_frames
            if r.left_hand_landmarks is not None or r.right_hand_landmarks is not None
        )
        signing_hand_pct = (hand_detected_in_signing / len(signing_frames)) * 100

        if hand_detected_in_signing == 0:
            issues.append("Hands never detected during signing portion - check lighting")
        elif signing_hand_pct < 20:
            issues.append(f"Hands only detected in {signing_hand_pct:.0f}% of signing window - sign too fast or invisible")

    # Body missing check
    body_missing_pct = (missing_body_frames / frame_count) * 100
    if body_missing_pct > 10:
        issues.append(f"Body not detected in {body_missing_pct:.0f}% of frames")

    return list(set(issues))

# --- CONFIGURATION ---
DATA_PATH = r"D:\Extracted\I\0.npy"

if __name__ == "__main__":
    print(f"Scanning videos in: {DATA_PATH}...\n")
    total = 0
    passed = 0
    failed = 0

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Path not found: {DATA_PATH}")
    else:
        folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
        if not folders:
            folders = ["."]

        for word in folders:
            word_path = os.path.join(DATA_PATH, word)
            vids = [v for v in os.listdir(word_path) if v.lower().endswith(('.mp4', '.mov'))]

            if vids:
                print(f"--- Folder: {word} ---")
                for vid in vids:
                    total += 1
                    problems = analyze_video_quality(os.path.join(word_path, vid))
                    if problems:
                        failed += 1
                        print(f"  ❌ {vid}: {', '.join(problems)}")
                    else:
                        passed += 1
                        print(f"  ✅ {vid}: Perfect")

        print(f"\n📊 Summary: {passed}/{total} passed, {failed} need re-recording")