"""It checks whether a video is short and needs to be removed from training data"""
import numpy as np
import os

DATA_PATH = r"D:\Extracted"

for word in os.listdir(DATA_PATH):
    word_path = os.path.join(DATA_PATH, word)
    if not os.path.isdir(word_path): continue
    for f in os.listdir(word_path):
        if f.endswith('.npy'):
            data = np.load(os.path.join(word_path, f))
            if data.shape[0] < 60:
                print(f"⚠️  SHORT: {word}/{f} — {data.shape[0]} frames")