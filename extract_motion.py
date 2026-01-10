import cv2
import numpy as np
from tqdm import tqdm

video_path = "crowd_15fps.mp4"
cap = cv2.VideoCapture(video_path)

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

motion_data = []

frame_id = 0

for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    motion_data.append([
        frame_id,
        np.mean(mag),
        np.var(mag)
    ])

    prev_gray = gray
    frame_id += 1

cap.release()

np.save("motion_features.npy", np.array(motion_data))
print("✅ Motion features extracted and saved.")
