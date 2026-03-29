"""Visualize extracted landmarks from new format (144 values, one npy per video)
It shows npy coordinates in lamdmarks form"""
# Claude

import cv2
import numpy as np

# --- CONFIGURATION ---
NPY_FILE = r"D:\Extracted\Water\Water_56.npy"
WIDTH, HEIGHT = 640, 480
FPS_DELAY = 30  # milliseconds between frames

# Load full video sequence: shape (num_frames, 144)
sequence = np.load(NPY_FILE)
print(f"Loaded: {sequence.shape[0]} frames, {sequence.shape[1]} values per frame")

for frame_data in sequence:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Pose: indices 0-18 → 6 landmarks x 3 (x,y,z)
    pose = frame_data[0:18].reshape(6, 3)
    for x, y, z in pose:
        if x != 0.0 and y != 0.0:
            cv2.circle(image, (int(x * WIDTH), int(y * HEIGHT)), 5, (0, 255, 0), -1)  # Green

    # Left hand: indices 18-81 → 21 landmarks x 3
    lh = frame_data[18:81].reshape(21, 3)
    for x, y, z in lh:
        if x != 0.0 and y != 0.0:
            cv2.circle(image, (int(x * WIDTH), int(y * HEIGHT)), 3, (255, 0, 0), -1)  # Blue

    # Right hand: indices 81-144 → 21 landmarks x 3
    rh = frame_data[81:144].reshape(21, 3)
    for x, y, z in rh:
        if x != 0.0 and y != 0.0:
            cv2.circle(image, (int(x * WIDTH), int(y * HEIGHT)), 3, (0, 0, 255), -1)  # Red

    cv2.imshow('Landmark Verification', image)
    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()