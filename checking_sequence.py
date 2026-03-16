"""This checks sequence length, like what sequence to set"""

import numpy as np
import os

lengths = []
for word in os.listdir(r"D:\Extracted"):
    word_path = os.path.join(r"D:\Extracted", word)
    if not os.path.isdir(word_path): continue
    for f in os.listdir(word_path):
        if f.endswith('.npy'):
            lengths.append(np.load(os.path.join(word_path, f)).shape[0])

lengths = sorted(lengths)
p95 = lengths[int(len(lengths) * 0.95)]
print(f"95th percentile: {p95} frames  ← use this as SEQUENCE_LENGTH")
print(f"Max: {max(lengths)}, Min: {min(lengths)}, Avg: {int(sum(lengths)/len(lengths))}")