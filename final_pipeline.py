"""
AI-Based Public Safety Monitoring & Risk Detection System
Production-Ready ML Pipeline

Author: Mrityunjay Singh
Dataset: UMN Crowd Activity Video
Ethical Scope: Crowd-level only (no individual tracking)
"""

# IMPORTS
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# CONFIGURATION
VIDEO_PATH = "crowd_15fps.mp4"
FEATURE_CSV = "labeled_motion_features.csv"
FPS = 15

TRAIN_RATIO = 0.7
THRESHOLD = 0.4          # lower threshold → higher recall
TEMPORAL_WINDOW = 3      # temporal OR smoothing

# HELPER FUNCTIONS
def frame_to_timestamp(frame_id, fps):
    total_seconds = int(frame_id / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# STEP 1: MOTION FEATURE EXTRACTION
def extract_motion_features(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    features = []
    frame_id = 0

    print("🔹 Extracting optical flow features...")
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        features.append([
            frame_id,
            np.mean(mag),
            np.var(mag)
        ])

        prev_gray = gray
        frame_id += 1

    cap.release()
    return pd.DataFrame(
        features,
        columns=["frame_id", "mean_motion", "variance_motion"]
    )

# STEP 2: WEAK SUPERVISED LABELING
def label_motion_data(df):
    print("🔹 Generating rule-based labels...")

    p60 = np.percentile(df["mean_motion"], 60)
    p85 = np.percentile(df["mean_motion"], 85)

    df["normal_alert"] = (df["mean_motion"] < p60).astype(int)
    df["medium_alert"] = (
        (df["mean_motion"] >= p60) & (df["mean_motion"] < p85)
    ).astype(int)
    df["high_alert"] = (df["mean_motion"] >= p85).astype(int)

    df.to_csv(FEATURE_CSV, index=False)
    return df

# STEP 3: TEMPORAL LEAKAGE FIX
def prepare_temporal_data(df):
    df["future_high_alert"] = df["high_alert"].shift(-1)
    return df.dropna()

# STEP 4: TRAIN & PREDICT
def train_and_predict(df):
    split_idx = int(TRAIN_RATIO * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[["mean_motion", "variance_motion"]]
    y_train = train_df["future_high_alert"]

    X_test = test_df[["mean_motion", "variance_motion"]]
    y_test = test_df["future_high_alert"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Probability-based prediction
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_raw = (y_prob >= THRESHOLD).astype(int)

    # Temporal OR smoothing
    y_pred = []
    for i in range(len(y_pred_raw)):
        if i < TEMPORAL_WINDOW:
            y_pred.append(y_pred_raw[i])
        else:
            y_pred.append(int(any(y_pred_raw[i-TEMPORAL_WINDOW:i+1])))

    return y_test.values, np.array(y_pred), test_df

# STEP 5: ALERT GENERATION
def generate_alerts(test_df, predictions):
    print("\n🚨 ALERTS")
    print("------------------------")

    in_event = False
    start_frame = None
    start_time = None

    for idx, pred in zip(test_df["frame_id"], predictions):
        timestamp = frame_to_timestamp(idx, FPS)

        # Event starts
        if pred == 1 and not in_event:
            in_event = True
            start_frame = idx
            start_time = timestamp

        # Event ends
        elif pred == 0 and in_event:
            end_frame = idx - 1
            end_time = frame_to_timestamp(end_frame, FPS)

            print(f"""
🚨 ABNORMAL CROWD ACTIVITY DETECTED
Time Range : {start_time} → {end_time}
Frame Range: {start_frame} → {end_frame}
Risk Level : HIGH
Cause      : Sustained abnormal crowd behavior
------------------------
""")
            in_event = False

    # Handle case where video ends during an event
    if in_event:
        end_frame = test_df["frame_id"].iloc[-1]
        end_time = frame_to_timestamp(end_frame, FPS)

        print(f"""
🚨 ABNORMAL CROWD ACTIVITY DETECTED
Time Range : {start_time} → {end_time}
Frame Range: {start_frame} → {end_frame}
Risk Level : HIGH
Cause      : Sustained abnormal crowd behavior
------------------------
""")

# STEP 6: PERFORMANCE TABLE
def display_performance_table(cm, report_dict):
    tn, fp, fn, tp = cm.ravel()

    accuracy = report_dict["accuracy"]
    precision = report_dict["1.0"]["precision"]
    recall = report_dict["1.0"]["recall"]
    f1 = report_dict["1.0"]["f1-score"]

    false_alert_rate = fp / (fp + tn)

    print("\n📊 PERFORMANCE METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25}{'Value':<12}Meaning")
    print("-" * 70)

    rows = [
        ("True Positives (TP)", tp, "Abnormal events correctly detected"),
        ("False Positives (FP)", fp, "Normal events wrongly flagged as alerts"),
        ("True Negatives (TN)", tn, "Normal events correctly identified"),
        ("False Negatives (FN)", fn, "Abnormal events missed (critical)"),
        ("Accuracy", f"{accuracy*100:.2f}%", "Overall correctness"),
        ("Precision (Abnormal)", f"{precision:.2f}", "Alert reliability"),
        ("Recall (Abnormal)", f"{recall:.2f}", "Ability to catch danger"),
        ("F1-Score (Abnormal)", f"{f1:.2f}", "Precision–Recall balance"),
        ("False Alert Rate", f"{false_alert_rate*100:.2f}%", "Normal events flagged")
    ]

    for r in rows:
        print(f"{r[0]:<25}{str(r[1]):<12}{r[2]}")

    print("=" * 70)

# STEP 7: EVALUATION
def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    display_performance_table(cm, report)

# MAIN PIPELINE
def main():
    df = extract_motion_features(VIDEO_PATH)
    df = label_motion_data(df)
    df = prepare_temporal_data(df)

    y_true, y_pred, test_df = train_and_predict(df)

    generate_alerts(test_df, y_pred)
    evaluate(y_true, y_pred)

    print("\n✅ Pipeline execution completed successfully.")

# RUN
if __name__ == "__main__":
    main()
