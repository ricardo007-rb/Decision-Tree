# 01_feature_extraction.py

import os
import cv2
import mediapipe as mp
import pandas as pd

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
base_dir = "CW2_dataset_final"
output_rows = []

print(" Starting feature extraction...")

# Loop through folders A to J
for label in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, label)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n Processing folder: {label}")

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        fpath = os.path.join(folder_path, fname)
        print(f"   ➤ Reading image: {fname}")

        image = cv2.imread(fpath)
        if image is None:
            print(f"      Could not read image, SKIPPING.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            print(f"      NO HAND detected.")
            continue

        print(f"      HAND detected!")

        landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        features.append(label)
        output_rows.append(features)

# Save to CSV
if output_rows:
    num_features = len(output_rows[0]) - 1
    columns = [f"f{i}" for i in range(num_features)] + ["label"]
    df = pd.DataFrame(output_rows, columns=columns)

    os.makedirs("data_features", exist_ok=True)
    df.to_csv("data_features/asl_features.csv", index=False)

    print(f"\n DONE! Saved {len(df)} samples with {num_features} features each.")
    print(" File saved to: data_features/asl_features.csv")
else:
    print("\n NO HAND landmarks were detected in ANY image.")
    print("   This means MediaPipe could not find a hand in your dataset.")