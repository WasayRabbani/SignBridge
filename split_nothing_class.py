"""
Nothing Class Video Splitter
Records or splits a long video into short random-length clips automatically.
No manual cropping needed.

Two modes:
    1. SPLIT MODE  — split an existing long video into clips
    2. RECORD MODE — record directly from camera and auto-split on the fly
"""

import cv2
import os
import random

# ============================================================
# CONFIGURATION
# ============================================================
MODE            = "SPLIT"                    # "SPLIT" or "RECORD"
INPUT_VIDEO     = r"D:\Signs\Nothing\Long 2.mp4"  # used in SPLIT mode only
OUTPUT_FOLDER   = r"D:\Signs\Nothing\Sign_Cut_"

CLIP_MIN_SEC    = 2      # minimum clip length in seconds
CLIP_MAX_SEC    = 4      # maximum clip length in seconds
TARGET_CLIPS    = 80    # how many clips to generate

START_FROM = 78
# ============================================================


def split_existing_video(video_path, output_folder, min_sec, max_sec, target_clips):
    """Split a long recorded video into random-length short clips"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps         = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec   = total_frames / fps

    print(f"Video loaded: {total_sec:.1f} seconds at {fps} FPS")
    print(f"Target: {target_clips} clips of {min_sec}-{max_sec} seconds each\n")

    # Check if video is long enough
    min_needed = target_clips * min_sec
    if total_sec < min_needed:
        print(f"⚠️  WARNING: Video is {total_sec:.0f}s but you need ~{min_needed}s for {target_clips} clips")
        print(f"   Record a longer video or reduce TARGET_CLIPS")

    os.makedirs(output_folder, exist_ok=True)

    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    clip_count  = 0
    frame_pos   = 0

    while clip_count < target_clips and frame_pos < total_frames:
        # Random clip length in frames
        clip_sec    = random.uniform(min_sec, max_sec)
        clip_frames = int(clip_sec * fps)

        # Stop if not enough frames left
        if frame_pos + clip_frames > total_frames:
            break

        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        # Output file
        clip_name = f"Nothing_{START_FROM + clip_count + 1}.mp4"
        clip_path   = os.path.join(output_folder, clip_name)
        out         = cv2.VideoWriter(clip_path, fourcc, fps, (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))

        # Write frames
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        clip_count += 1
        frame_pos  += clip_frames  # move to next clip start (no overlap)

        print(f"  ✅ {clip_name} — {clip_sec:.1f}s ({clip_frames} frames)")

    cap.release()
    print(f"\n📊 Done. {clip_count} clips saved to {output_folder}")

    if clip_count < target_clips:
        print(f"⚠️  Only got {clip_count}/{target_clips} clips — record more footage")


def record_and_split(output_folder, min_sec, max_sec, target_clips):
    """
    Record directly from webcam.
    Press SPACE to start/stop recording a clip.
    Auto-saves each clip and counts toward target.
    Press Q to quit early.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap     = cv2.VideoCapture(0)
    fps     = 30
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_count  = 0
    recording   = False
    out         = None
    frame_count = 0

    print("RECORD MODE")
    print("Press SPACE to start recording a clip")
    print("Press SPACE again to stop and save")
    print("Press Q to quit\n")

    while clip_count < target_clips:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if recording:
            out.write(frame)
            frame_count += 1
            # Auto-stop if clip exceeds max length
            if frame_count >= max_sec * fps:
                recording = False
                out.release()
                clip_count += 1
                print(f"  ✅ Auto-saved Nothing_{clip_count}.mp4 ({frame_count} frames)")
                frame_count = 0

            cv2.putText(display, f"RECORDING... {frame_count/fps:.1f}s", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"Ready. Clips: {clip_count}/{target_clips}. SPACE to record",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Nothing Class Recorder", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                # Start recording
                clip_name = f"Nothing_{clip_count + 1}.mp4"
                clip_path = os.path.join(output_folder, clip_name)
                out = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
                recording = True
                frame_count = 0
                print(f"  🔴 Started recording {clip_name}")
            else:
                # Stop recording manually
                if frame_count >= min_sec * fps:
                    recording = False
                    out.release()
                    clip_count += 1
                    print(f"  ✅ Saved Nothing_{clip_count}.mp4 ({frame_count} frames)")
                    frame_count = 0
                else:
                    print(f"  ⚠️  Too short ({frame_count/fps:.1f}s) — keep recording")

        elif key == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Done. {clip_count} clips saved to {output_folder}")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    if MODE == "SPLIT":
        split_existing_video(INPUT_VIDEO, OUTPUT_FOLDER, CLIP_MIN_SEC, CLIP_MAX_SEC, TARGET_CLIPS)
    elif MODE == "RECORD":
        record_and_split(OUTPUT_FOLDER, CLIP_MIN_SEC, CLIP_MAX_SEC, TARGET_CLIPS)
    else:
        print("ERROR: MODE must be 'SPLIT' or 'RECORD'")