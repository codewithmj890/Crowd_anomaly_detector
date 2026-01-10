import numpy as np
import pandas as pd

data = np.load("motion_features.npy")

df = pd.DataFrame(
    data,
    columns=["frame_id", "mean_motion", "variance_motion"]
)

# Thresholds (data-driven)
p60 = np.percentile(df["mean_motion"], 60)
p85 = np.percentile(df["mean_motion"], 85)

# Binary abnormal activity label
df["abnormal_activity"] = (df["mean_motion"] >= p85).astype(int)

# One-hot normal activity
df["normal_activity"] = 1 - df["abnormal_activity"]

# Alert levels
df["normal_alert"] = (df["mean_motion"] < p60).astype(int)
df["medium_alert"] = (
    (df["mean_motion"] >= p60) &
    (df["mean_motion"] < p85)
).astype(int)
df["high_alert"] = df["abnormal_activity"]

df.to_csv("labeled_motion_features.csv", index=False)
print("Labeled dataset saved")

